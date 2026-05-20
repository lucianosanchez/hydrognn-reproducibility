"""Adaptadores `RunoffModel` para las tres fases del experimento GNN.

El experimento mide **cuánta información aporta la estructura geográfica**
al pasar de Seq2Seq (sin grafo) a HydroGNN. Las tres fases representan
distintos grados de información disponible:

    Fase 1   Información completa: grafo exacto, todas las estaciones, niveles
             de los tres embalses. Acota por arriba la calidad alcanzable —
             no es realista en una cuenca real, pero sí simulable.
    Fase 2.1 Información parcial: misma topología, pero sólo un subconjunto
             de estaciones está sensorizado y los niveles de los embalses son
             latentes. Mide el coste de no tener el cauce completamente
             observado.
    Fase 2.2 Información mínima: pluviosidad parcial y **posiciones de los
             embalses desconocidas**. Hay M embalses formales libres que el
             modelo coloca donde quiere; sin penalización de sparsity (el
             objetivo no es purgar nodos sino observar dónde caen).

Las tres comparten `HydroGNNCore` y la rutina de entrenamiento `_entrenar_core`.
"""

from __future__ import annotations

import math
import pickle
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ..basin import BasinSpec
from ..config import Config
from ..model import Forecast, ForecastScenario, RunoffModel
from .core import HydroGNNCore
from .dataset import GNNWindow, build_training_dataset, build_window
from .graph import BasinGraph, dense_candidate_graph
from .losses import LossParts, total_loss


# ---------------------------------------------------------------------------
# Configuración específica del GNN.
# ---------------------------------------------------------------------------


@dataclass
class GNNConfig:
    """Hiperparámetros del entrenamiento GNN.

    Se mantiene aparte de `Config` porque las dos rutas (TF y PyTorch) tienen
    parámetros distintos y no debe haber acoplamiento entre ellas. Lleva
    siempre un `basin` para no acoplar el código a una cuenca concreta.
    """

    basin: BasinSpec
    historia: int = 20
    horizonte: int = 10
    hidden: int = 64
    node_static_dim: int = 8
    ctx_dim: int = 2

    epochs: int = 50
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 32
    grad_clip: float = 1.0

    # Pesos de las componentes de la pérdida (eq. 6.6).
    # Por defecto NO penalizamos sparsity en ninguna fase: queremos medir
    # pérdida de información, no purgar nodos. Cada fase puede sobreescribirlo.
    lam_smooth: float = 0.01
    lam_sparse: float = 0.0
    lam_phys: float = 0.0
    lam_res: float = 0.1   # MSE de embalse observado (sólo Fase 1)

    # Peso del énfasis en bajo caudal (eq. 6.1). κ alto (>20) hace al modelo
    # mucho más conservador a la hora de predecir caudales por debajo del umbral.
    kappa_low_flow: float = 5.0
    escala_low_flow: float = 5.0

    # Cobertura sensorial efectiva. None ⇒ todas las estaciones del grafo.
    # Una lista (e.g. DEFAULT_OBSERVED_STATIONS_PARTIAL) ⇒ sólo esas;
    # el resto va con mask=0 durante todo el entrenamiento e inferencia.
    observed_stations: Optional[List[str]] = None

    # Fase 2.2: número de embalses latentes libres y sesgo inicial del
    # logit de splitting hacia los embalses (negativo ⇒ poco flujo divertido
    # al inicio). Necesario porque el grafo de candidatos es denso.
    M_latent: int = 6
    logw12_init: float = -3.0

    # Phase 2.2 variant: si True, usa `acyclic_candidate_graph` en lugar de
    # `dense_candidate_graph`. Cada embalse se asocia a un nodo-ancla por
    # `anchor_strategy` y sólo recibe E_12 de sus ancestros / vierte por
    # E_21 a sus descendientes en E_11. Cero back-flow por construcción.
    acyclic_candidates: bool = False
    anchor_strategy: str = "bfs_uniform"

    # Velocidad efectiva del río (km/día). Si está y el BasinGraph aporta
    # longitudes, los logits de routing λ se inicializan informados por
    # length/velocity. Default None (inicialización homogénea).
    river_velocity_km_day: Optional[float] = None

    device: str = "cpu"
    semilla: int = 0


# ---------------------------------------------------------------------------
# Loader y bucle de entrenamiento compartido.
# ---------------------------------------------------------------------------


def _stack_windows(ventanas: List[GNNWindow]) -> dict:
    rain = torch.stack([w.rain for w in ventanas])
    mask = torch.stack([w.mask for w in ventanas])
    ctx = torch.stack([w.ctx for w in ventanas])
    Q = torch.stack([w.Q_obs for w in ventanas])
    S_list = [w.S_obs for w in ventanas if w.S_obs is not None]
    S = torch.stack(S_list) if len(S_list) == len(ventanas) else None
    return {"rain": rain, "mask": mask, "ctx": ctx, "Q": Q, "S": S}


def _iter_minibatches(
    ventanas: List[GNNWindow],
    batch_size: int,
    rng: np.random.Generator,
) -> Iterable[dict]:
    indices = np.arange(len(ventanas))
    rng.shuffle(indices)
    for i in range(0, len(indices), batch_size):
        batch = [ventanas[j] for j in indices[i:i + batch_size]]
        yield _stack_windows(batch)


def _entrenar_core(
    core: HydroGNNCore,
    ventanas: List[GNNWindow],
    cfg: GNNConfig,
    q_min_norm: float,
    obs_to_res_index: Optional[torch.Tensor],
    use_S_obs: bool,
) -> List[dict]:
    """Bucle de entrenamiento minimalista (algoritmo en sec. 11)."""
    device = torch.device(cfg.device)
    core.to(device)
    optim = torch.optim.Adam(core.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    rng = np.random.default_rng(cfg.semilla)

    historico: List[dict] = []
    for ep in range(1, cfg.epochs + 1):
        core.train()
        agreg = {"loss": 0.0, "flow": 0.0, "smooth": 0.0, "sparse": 0.0, "embalse": 0.0}
        nb = 0
        for batch in _iter_minibatches(ventanas, cfg.batch_size, rng):
            rain = batch["rain"].to(device)
            mask = batch["mask"].to(device)
            ctx = batch["ctx"].to(device)
            Q = batch["Q"].to(device)
            S = batch["S"].to(device) if (use_S_obs and batch["S"] is not None) else None

            out = core(rain, mask, ctx)
            partes: LossParts = total_loss(
                out, Q, H=cfg.historia, T=cfg.horizonte, q_min=q_min_norm,
                S_obs=S, obs_to_res_index=obs_to_res_index,
                lam_smooth=cfg.lam_smooth, lam_sparse=cfg.lam_sparse,
                lam_phys=cfg.lam_phys, lam_res=cfg.lam_res if use_S_obs else 0.0,
                kappa_low_flow=cfg.kappa_low_flow,
                escala_low_flow=cfg.escala_low_flow,
            )

            optim.zero_grad()
            partes.total.backward()
            torch.nn.utils.clip_grad_norm_(core.parameters(), cfg.grad_clip)
            optim.step()

            agreg["loss"] += float(partes.total.detach())
            agreg["flow"] += float(partes.flow.detach())
            agreg["smooth"] += float(partes.smooth.detach())
            agreg["sparse"] += float(partes.sparse.detach())
            agreg["embalse"] += float(partes.embalse_obs.detach())
            nb += 1
        for k in agreg:
            agreg[k] /= max(1, nb)
        agreg["epoch"] = ep
        historico.append(agreg)
    return historico


# ---------------------------------------------------------------------------
# Base común para los tres adaptadores RunoffModel.
# ---------------------------------------------------------------------------


class _HydroGNNBase(RunoffModel):
    """Esqueleto compartido. Las subclases deciden grafo, observación y pérdidas."""

    nombre = "hydrognn-base"
    use_gates: str = "none"
    use_S_obs: bool = False
    logw12_init_default: float = 0.0

    def __init__(self, cfg: GNNConfig, graph: BasinGraph):
        self.cfg = cfg
        self.graph = graph
        torch.manual_seed(cfg.semilla)
        self.core = HydroGNNCore(
            self.graph,
            use_gates=self.use_gates,
            node_static_dim=cfg.node_static_dim,
            ctx_dim=cfg.ctx_dim,
            hidden=cfg.hidden,
            river_velocity_km_day=getattr(cfg, "river_velocity_km_day", None),
            logw12_init=getattr(cfg, "logw12_init", self.logw12_init_default),
        )
        self._maximos: Optional[pd.Series] = None

    # --------- helpers internos -------------------------------------------

    def _q_min_normalizado(self, maximos: pd.Series, caudal_minimo_m3s: float) -> float:
        return float(caudal_minimo_m3s / maximos[self.cfg.basin.flow_column])

    def _obs_to_res_index(self) -> Optional[torch.Tensor]:
        idx = []
        for k, name in enumerate(self.graph.res_names):
            if self.graph.res_to_observed.get(name) is not None:
                idx.append(k)
        return torch.tensor(idx, dtype=torch.long) if idx else None

    # --------- API de RunoffModel -----------------------------------------

    def fit(self, df_train: pd.DataFrame, maximos: pd.Series) -> List[dict]:
        cfg = self.cfg
        self._maximos = maximos.copy()
        ventanas = list(build_training_dataset(
            df_train, self.graph,
            H=cfg.historia, T=cfg.horizonte,
            flow_column=cfg.basin.flow_column,
            observed_stations=cfg.observed_stations,
        ))
        if not ventanas:
            raise ValueError("No se han podido construir ventanas (datos insuficientes).")

        q_min_norm = self._q_min_normalizado(maximos, cfg.basin.caudal_minimo_m3s)

        return _entrenar_core(
            self.core, ventanas, cfg, q_min_norm,
            obs_to_res_index=self._obs_to_res_index() if self.use_S_obs else None,
            use_S_obs=self.use_S_obs,
        )

    @torch.no_grad()
    def predict(
        self,
        df: pd.DataFrame,
        hoy: pd.Timestamp,
        maximos: pd.Series,
        escenario: str = ForecastScenario.OBSERVED,
    ) -> Forecast:
        cfg = self.cfg
        device = torch.device(cfg.device)
        self.core.eval()

        df_local = df.copy()
        if escenario == ForecastScenario.WORST:
            # Si el horizonte se sale del DataFrame, prolongamos con filas
            # placeholder: la lluvia se pondrá a cero unas líneas más abajo, y
            # los demás valores no se usan durante la inferencia (`build_window`
            # extrae Q_obs y S_obs pero `predict` no los lee).
            fin_horizonte = hoy + timedelta(days=cfg.horizonte)
            if fin_horizonte not in df_local.index:
                ultimo = df_local.index[-1]
                if fin_horizonte > ultimo:
                    n_extra = (fin_horizonte - ultimo).days
                    new_index = pd.date_range(ultimo + timedelta(days=1),
                                              fin_horizonte, freq="D")
                    pad = pd.DataFrame(
                        np.repeat(df_local.iloc[[-1]].to_numpy(), n_extra, axis=0),
                        index=new_index, columns=df_local.columns,
                    )
                    df_local = pd.concat([df_local, pad])
            futuro = df_local.loc[hoy + timedelta(days=1):fin_horizonte].index
            cols_lluvia = list(self.graph.rain_to_type1.keys()) + [cfg.basin.rain_aggregate_column]
            cols_lluvia = [c for c in cols_lluvia if c in df_local.columns]
            df_local.loc[futuro, cols_lluvia] = 0.0
        elif escenario != ForecastScenario.OBSERVED:
            raise ValueError(f"Escenario desconocido: {escenario!r}")

        ventana = build_window(
            df_local, self.graph, hoy, cfg.historia, cfg.horizonte,
            flow_column=cfg.basin.flow_column,
            observed_stations=cfg.observed_stations,
        )
        rain = ventana.rain.unsqueeze(0).to(device)
        mask = ventana.mask.unsqueeze(0).to(device)
        ctx = ventana.ctx.unsqueeze(0).to(device)

        out = self.core(rain, mask, ctx)
        H, T = cfg.historia, cfg.horizonte
        mu_norm = out.mu_Q[0, H:H + T].cpu().numpy()
        S_norm = out.S_hist[0, H:H + T].cpu().numpy()

        # P(Q ≥ Q_min) — útil aunque no se use directamente en la Forecast.
        sigma = torch.nn.functional.softplus(out.log_sigma[0, H:H + T]).cpu().numpy() + 1e-6
        q_min_norm = self._q_min_normalizado(maximos, cfg.basin.caudal_minimo_m3s)
        p_compliance = 0.5 * (1 + np.vectorize(math.erf)((mu_norm - q_min_norm) / (sigma * math.sqrt(2))))

        fechas = pd.date_range(hoy + timedelta(days=1), periods=cfg.horizonte)
        return Forecast(
            fechas=fechas,
            caudal=mu_norm * maximos[cfg.basin.flow_column],
            embalse=S_norm.sum(axis=-1) * maximos.get(cfg.basin.reservoir_aggregate_column, 1.0),
            caudal_logit=p_compliance,  # reusamos el campo como probabilidad de cumplimiento
        )

    # --------- persistencia -----------------------------------------------

    def save(self, directorio: str | Path) -> None:
        d = Path(directorio)
        d.mkdir(parents=True, exist_ok=True)
        torch.save(self.core.state_dict(), d / "core.pt")
        with open(d / "meta.pkl", "wb") as f:
            pickle.dump({"cfg": self.cfg, "graph": self.graph,
                         "maximos": self._maximos, "nombre": self.nombre}, f)

    @classmethod
    def load(cls, directorio: str | Path, config: Config) -> "_HydroGNNBase":
        d = Path(directorio)
        with open(d / "meta.pkl", "rb") as f:
            meta = pickle.load(f)
        instancia = cls(meta["cfg"], graph=meta["graph"])
        instancia.core.load_state_dict(torch.load(d / "core.pt", map_location="cpu"))
        instancia._maximos = meta["maximos"]
        return instancia


# ---------------------------------------------------------------------------
# Las tres fases.
# ---------------------------------------------------------------------------


class HydroGNNPhase1(_HydroGNNBase):
    """Fase 1 — INFORMACIÓN COMPLETA (cota inferior de error).

    Grafo exacto de la cuenca. Todas las estaciones de pluviosidad
    observadas (mask = 1). Niveles de los embalses supervisados durante el
    entrenamiento mediante el término `lam_res · MSE(S, EACUM)`.

    No es un escenario realista — ninguna cuenca está completamente
    sensorizada — pero sí reproducible con un simulador y por tanto sirve
    para acotar la mejor calidad alcanzable por la familia.
    """
    nombre = "hydrognn-fase1"
    use_gates = "none"
    use_S_obs = True
    logw12_init_default = 0.0

    def __init__(self, cfg: GNNConfig, graph: BasinGraph):
        super().__init__(cfg, graph=graph)


class HydroGNNPhase2_1(_HydroGNNBase):
    """Fase 2.1 — INFORMACIÓN PARCIAL CON POSICIONES CONOCIDAS.

    Grafo exacto (sabemos dónde están los embalses) pero:
      * sólo un subconjunto de estaciones está sensorizado (`cfg.observed_stations`);
      * los niveles de los embalses **no** se observan (lam_res = 0).

    Mide cuánto se pierde respecto a Fase 1 cuando renunciamos a sensorizar
    todo el cauce y a registrar el almacenamiento.
    """
    nombre = "hydrognn-fase2.1"
    use_gates = "none"
    use_S_obs = False
    logw12_init_default = 0.0

    def __init__(self, cfg: GNNConfig, graph: BasinGraph):
        if cfg.observed_stations is None:
            raise ValueError(
                "HydroGNNPhase2_1 requiere `cfg.observed_stations` — la lista "
                "de estaciones realmente sensorizadas en el escenario."
            )
        super().__init__(cfg, graph=graph)


class HydroGNNPhase2_2(_HydroGNNBase):
    """Fase 2.2 — INFORMACIÓN MÍNIMA: TAMPOCO CONOCEMOS POSICIONES.

    Pluviosidad parcial (igual que Fase 2.1) y grafo de candidatos densos:
    `M_latent` embalses formales libres conectados desde y hacia cualquier
    nodo Tipo-1. El reparto del flujo entre embalses lo aprenden los pesos
    de splitting (`logw12`, `logw21`) — sin gates, sin sparsity penalty.

    El experimento es comprobar si:
      (a) el modelo coloca embalses formales en posiciones distintas a las
          reales, y
      (b) la masa total acumulada en esos embalses formales se aproxima a
          la capacidad real de la cuenca.

    Usa `analyze_positions()` después de entrenar para inspeccionar el
    reparto aprendido.
    """
    nombre = "hydrognn-fase2.2"
    use_gates = "none"   # explícitamente sin gates: no es un experimento de pruning
    use_S_obs = False
    logw12_init_default = -3.0

    def __init__(self, cfg: GNNConfig, graph_base: BasinGraph = None,
                 graph: BasinGraph = None):
        """`graph_base` aporta la topología fluvial Tipo-1 ↔ Tipo-1; el grafo
        de candidatos se construye automáticamente con `M_latent` embalses
        formales. Por defecto el grafo es DENSO (toda fuente Type-1 puede
        alimentar a todo embalse, y todo embalse puede verter a todo Type-1).
        Si `cfg.acyclic_candidates=True`, se usa en su lugar
        `acyclic_candidate_graph`, que restringe cada embalse a una
        sub-cuenca conexa con su nodo-ancla (cf. estrategia
        `cfg.anchor_strategy`).

        Para el camino de carga (`_HydroGNNBase.load`), se acepta el alias
        `graph=` con el grafo de candidatos ya construido (almacenado en
        `meta["graph"]`) y se evita reconstruir la topología."""
        if graph is not None and graph_base is None:
            # Camino de carga: el grafo de candidatos ya está construido.
            super().__init__(cfg, graph=graph)
            self.graph_topologia = graph
            return
        if graph_base is None:
            raise TypeError(
                "HydroGNNPhase2_2 requiere `graph_base` (training) o `graph` "
                "(loading from checkpoint)."
            )
        if cfg.observed_stations is None:
            raise ValueError(
                "HydroGNNPhase2_2 requiere `cfg.observed_stations` — la lista "
                "de estaciones realmente sensorizadas en el escenario."
            )
        if getattr(cfg, "acyclic_candidates", False):
            from .graph import acyclic_candidate_graph
            candidatos = acyclic_candidate_graph(
                graph_base, M=cfg.M_latent,
                anchor_strategy=getattr(cfg, "anchor_strategy", "bfs_uniform"),
            )
        else:
            candidatos = dense_candidate_graph(graph_base, M=cfg.M_latent)
        super().__init__(cfg, graph=candidatos)
        # Guardamos la topología original para mostrar nombres en `analyze_positions`.
        self.graph_topologia = graph_base

    def analyze_positions(self) -> dict:
        """Resumen interpretable de la estructura aprendida.

        Devuelve un dict con:
            inflow_share[k, i]   : qué fracción del flujo del nodo Tipo-1 i
                                   entra al embalse formal k.
            outflow_share[k, j]  : a qué Tipo-1 j envía sus sueltas.
            type1_names          : nombres de nodos para etiquetar matrices.
            res_names            : nombres de los embalses formales.
        """
        info = self.core.analyze_positions()
        info["type1_names"] = list(self.graph_topologia.type1_names)
        info["res_names"] = list(self.graph.res_names)
        return info
