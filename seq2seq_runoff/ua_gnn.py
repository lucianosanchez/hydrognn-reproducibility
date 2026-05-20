"""Uncertainty-aware HydroGNN (sec. 4 of paper_methods.tex).

Extiende `HydroGNNCore` con un posterior gaussiano sobre el estado
hidrológico inicial — almacenamiento de cada embalse Tipo-2 y buffer
de routing de cada arista E_11/E_12/E_21. El posterior se construye
por un MLP "encoder" que consume estadísticos de los H días de
burn-in. K muestras Monte-Carlo del posterior se propagan por la
dinámica determinista (preservando el balance de masa) y producen una
distribución predictiva mezcla-de-gaussianas en el aforo.

Combinado con la librería de escenarios de `scenarios.py` y los
criterios de decisión de `decision.py`, el modelo cierra el bucle
descrito en §4: información geográfica × incertidumbre climática ×
incertidumbre hidrológica → operating point cost-aware.

Diferencias clave con `HydroGNNPhase*`:
    * El estado inicial se muestrea (no se inicializa a cero).
    * La pérdida de entrenamiento es log-mixture NLL sobre K muestras.
    * `predict_distribution()` devuelve (M, K, T) caudal en m³/s.
"""

from __future__ import annotations

import math
import pickle
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

from .basin import BasinSpec
from .config import Config
from .gnn.core import HydroGNNCore, _make_mlp
from .gnn.dataset import GNNWindow, build_training_dataset, build_window
from .gnn.graph import BasinGraph
from .gnn.losses import lowflow_weight
from .gnn.model import GNNConfig
from .model import Forecast, ForecastScenario, RunoffModel


# ===========================================================================
# Core: HydroGNNCore + initial-state posterior + MC sampler
# ===========================================================================


class UAHydroGNNCore(nn.Module):
    """`HydroGNNCore` con posterior gaussiano sobre el estado inicial.

    Para cada componente del estado en $t = 0$ — almacenamiento de cada
    embalse y buffer de routing de cada arista — un MLP produce
    parámetros $(\\mu, \\log\\sigma^2)$ a partir de los estadísticos de
    los H días de burn-in. Una `softplus` final garantiza
    no-negatividad. `forward_mc` toma K muestras y devuelve el tensor
    apilado de predicciones del aforo, listo para alimentar tanto la
    pérdida log-mixture como las decisiones cost-aware.
    """

    def __init__(
        self,
        graph: BasinGraph,
        use_gates: str = "none",
        node_static_dim: int = 8,
        ctx_dim: int = 2,
        hidden: int = 64,
        logw12_init: float = 0.0,
        K_train: int = 10,
        K_inference: int = 50,
        rain_bypass: bool = False,
        lam11_init: float = 0.0,
        river_velocity_km_day: Optional[float] = None,
    ):
        """`rain_bypass` y `lam11_init` son OPT-IN.

        Activar `rain_bypass=True` y `lam11_init=2.0` rompe el atractor
        de "predicción constante" en cuencas grandes (N1 ≳ 32) donde el
        core determinista converge a un punto fijo insensible a la
        lluvia futura, pero **degrada** Ebro y N=16 (cf. §4.10 del
        paper, "Remediation as a per-basin hyperparameter"). Default
        OFF para reproducir el comportamiento headline del paper.

        `river_velocity_km_day` activa la inicialización informada de λ
        cuando el `BasinGraph` aporta longitudes fluviales por arista
        (cf. \\Cref{sec:ebro} del paper)."""
        super().__init__()
        self.core = HydroGNNCore(
            graph,
            use_gates=use_gates,
            node_static_dim=node_static_dim,
            ctx_dim=ctx_dim,
            hidden=hidden,
            logw12_init=logw12_init,
            rain_bypass=rain_bypass,
            lam11_init=lam11_init,
            river_velocity_km_day=river_velocity_km_day,
        )
        self.K_train = int(K_train)
        self.K_inference = int(K_inference)

        N1 = self.core.N1
        M, E11, E12, E21 = self.core.M, self.core.E11, self.core.E12, self.core.E21
        d_state = M + E11 + E12 + E21
        self._d_state = d_state
        self._slices = {
            "S":   slice(0, M),
            "x11": slice(M, M + E11),
            "x12": slice(M + E11, M + E11 + E12),
            "x21": slice(M + E11 + E12, d_state),
        }
        # Encoder: mean+std+last-step de la lluvia por nodo en el burn-in.
        in_dim = N1 * 3
        self.initial_encoder = _make_mlp(in_dim, hidden, 2 * d_state, n_layers=3)

    # -------- posterior parameters ------------------------------------------

    def _aggregate_features(self, rain_burn_in: torch.Tensor) -> torch.Tensor:
        """`(B, H, N1)` → `(B, N1 * 3)`."""
        mean = rain_burn_in.mean(dim=1)
        # std puede degenerar a 0 si H==1; clamp para evitar NaN en gradientes.
        std = rain_burn_in.std(dim=1).clamp(min=1e-6)
        last = rain_burn_in[:, -1, :]
        return torch.cat([mean, std, last], dim=-1)

    def posterior_params(self, rain: torch.Tensor, H: int):
        """Devuelve `(mu, log_var)` de shape `(B, d_state)`."""
        feats = self._aggregate_features(rain[:, :H, :])
        out = self.initial_encoder(feats)
        mu = out[..., :self._d_state]
        log_var = out[..., self._d_state:]
        # Clamp log_var para estabilidad numérica
        log_var = log_var.clamp(-10.0, 10.0)
        return mu, log_var

    def kl_to_prior(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """`KL(N(mu, sigma^2) || N(0, I))`, promediado sobre batch."""
        return -0.5 * torch.mean(
            torch.sum(1.0 + log_var - mu * mu - torch.exp(log_var), dim=-1)
        )

    def kl_to_prior_free_bits(
        self, mu: torch.Tensor, log_var: torch.Tensor, free_bits: float
    ) -> torch.Tensor:
        """KL con "free bits" por dimensión.

        Cada dimensión del posterior tiene un presupuesto `free_bits` (en
        nats) de KL "gratis" — el coste de KL sólo se paga por encima de
        ese umbral. Esto evita que el optimizador empuje el posterior al
        prior (colapso) por tomarlo como camino fácil para reducir la
        pérdida total. Cf. Kingma et al. (2016) "Improving Variational
        Inference with Inverse Autoregressive Flow", sec. 6.
        """
        kl_per_dim = -0.5 * (1.0 + log_var - mu * mu - torch.exp(log_var))  # (B, D)
        if free_bits > 0.0:
            kl_per_dim = torch.clamp(kl_per_dim, min=free_bits)
        return kl_per_dim.sum(dim=-1).mean()

    def _sample_initial(self, mu: torch.Tensor, log_var: torch.Tensor):
        """Un sample reparametrizado del estado inicial, devuelto split en
        `(S_0, x11_0, x12_0, x21_0)` con no-negatividad garantizada."""
        sigma = torch.exp(0.5 * log_var)
        eps = torch.randn_like(mu)
        z = mu + sigma * eps
        S_0 = F.softplus(z[..., self._slices["S"]])
        x11_0 = F.softplus(z[..., self._slices["x11"]])
        x12_0 = F.softplus(z[..., self._slices["x12"]])
        x21_0 = F.softplus(z[..., self._slices["x21"]])
        return S_0, x11_0, x12_0, x21_0

    # -------- forward pass MC -----------------------------------------------

    def forward_mc(
        self,
        rain: torch.Tensor,         # (B, L, N1)
        mask: torch.Tensor,         # (B, L, N1)
        ctx: torch.Tensor,          # (B, L, ctx_dim)
        H: int,
        K: Optional[int] = None,
        return_states: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """K pasos por el core determinista. Devuelve dict con:

            mu_Q    (K, B, L)    media de la predictiva por sample
            log_sigma (K, B, L)  log σ aleatórica del head sigma
            mu      (B, d_state) media del posterior inicial
            log_var (B, d_state) log varianza del posterior inicial
            S_hist  (K, B, L, M) opcional, sólo si `return_states`
            O_hist  (K, B, L, M) opcional, sólo si `return_states`
        """
        K = K if K is not None else self.K_inference
        mu, log_var = self.posterior_params(rain, H)
        mu_Q_list, log_sigma_list = [], []
        S_hist_list, O_hist_list = [], []
        for _ in range(K):
            S_0, x11_0, x12_0, x21_0 = self._sample_initial(mu, log_var)
            out = self.core(
                rain, mask, ctx,
                initial_S=S_0, initial_x11=x11_0,
                initial_x12=x12_0, initial_x21=x21_0,
            )
            mu_Q_list.append(out.mu_Q)
            log_sigma_list.append(out.log_sigma)
            if return_states:
                S_hist_list.append(out.S_hist)
                O_hist_list.append(out.O_hist)
        result = {
            "mu_Q":      torch.stack(mu_Q_list, dim=0),       # (K, B, L)
            "log_sigma": torch.stack(log_sigma_list, dim=0),  # (K, B, L)
            "mu":        mu,                                   # (B, d_state)
            "log_var":   log_var,                              # (B, d_state)
        }
        if return_states:
            result["S_hist"] = torch.stack(S_hist_list, dim=0)
            result["O_hist"] = torch.stack(O_hist_list, dim=0)
        return result


# ===========================================================================
# Mixture predictive loss + training step
# ===========================================================================


def log_mixture_nll(
    y_true: torch.Tensor,          # (B, L)
    mu_Q: torch.Tensor,            # (K, B, L)
    log_sigma: torch.Tensor,       # (K, B, L)
    weight: Optional[torch.Tensor] = None,  # (B, L) opcional
    eps: float = 1e-6,
) -> torch.Tensor:
    """Negative log-likelihood de una mezcla uniforme de K gaussianas.

    Si `weight` se proporciona (e.g., `lowflow_weight(y_true)`), se aplica
    multiplicativamente al log-likelihood por paso, igual que en
    `total_loss` (sec. 6 del paper).
    """
    sigma = F.softplus(log_sigma) + eps
    K = mu_Q.shape[0]
    y_exp = y_true.unsqueeze(0).expand(K, -1, -1)
    log_p = (
        -0.5 * torch.log(2.0 * math.pi * sigma * sigma)
        - 0.5 * ((y_exp - mu_Q) / sigma) ** 2
    )                                            # (K, B, L)
    log_p_mix = torch.logsumexp(log_p, dim=0) - math.log(K)  # (B, L)
    if weight is not None:
        log_p_mix = log_p_mix * weight
    return -log_p_mix.mean()


def _train_ua_core(
    core: UAHydroGNNCore,
    ventanas: List[GNNWindow],
    cfg: GNNConfig,
    q_min_norm: float,
    beta_ua: float,
    K_train: int,
    warmup_epochs: int = 0,
    ramp_epochs: int = 1,
    free_bits: float = 0.0,
    verbose_every: int = 0,
    max_windows: Optional[int] = None,
    window_seed: int = 0,
) -> List[dict]:
    """Bucle de entrenamiento UA-HydroGNN: log-mixture NLL + KL prior + smooth.

    Parameters
    ----------
    warmup_epochs : int
        Número de épocas iniciales con β=0 (puro NLL). Permite al core
        determinista aprender a usar la lluvia futura antes de que el KL
        empuje el posterior al prior y bloquee el aprendizaje.
    ramp_epochs : int
        Tras `warmup_epochs`, β crece linealmente desde 0 hasta `beta_ua`
        en estas épocas. Total efectivo: `warmup_epochs + ramp_epochs`.
    free_bits : float
        Si > 0, cada dimensión del posterior puede tener `free_bits` nats
        de KL "gratis" — el optimizador no paga coste hasta que la KL por
        dimensión los supera. Ayuda a evitar colapso global cuando algunas
        dimensiones son útiles y otras no.
    verbose_every : int
        Si > 0, imprime el progreso cada `verbose_every` épocas (loss,
        flow, kl, beta efectivo).
    """
    device = torch.device(cfg.device)
    core.to(device)
    optim = torch.optim.Adam(core.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    rng = np.random.default_rng(cfg.semilla)

    if max_windows is not None and len(ventanas) > max_windows:
        sel_rng = np.random.default_rng(window_seed)
        idx_sel = sel_rng.choice(len(ventanas), size=max_windows, replace=False)
        ventanas = [ventanas[int(i)] for i in sorted(idx_sel)]

    H, T = cfg.historia, cfg.horizonte
    historico = []

    def _beta_at(ep: int) -> float:
        if ep <= warmup_epochs:
            return 0.0
        if ramp_epochs <= 1:
            return beta_ua
        progress = (ep - warmup_epochs) / float(ramp_epochs)
        return beta_ua * min(1.0, progress)

    for ep in range(1, cfg.epochs + 1):
        core.train()
        beta_eff = _beta_at(ep)
        indices = np.arange(len(ventanas))
        rng.shuffle(indices)
        agg = {"loss": 0.0, "flow": 0.0, "kl": 0.0, "kl_fb": 0.0, "smooth": 0.0}
        nb = 0
        for i in range(0, len(indices), cfg.batch_size):
            batch_idx = indices[i:i + cfg.batch_size]
            batch = [ventanas[j] for j in batch_idx]
            rain = torch.stack([w.rain for w in batch]).to(device)
            mask = torch.stack([w.mask for w in batch]).to(device)
            ctx = torch.stack([w.ctx for w in batch]).to(device)
            Q = torch.stack([w.Q_obs for w in batch]).to(device)

            out = core.forward_mc(rain, mask, ctx, H=H, K=K_train,
                                   return_states=True)
            mu_Q = out["mu_Q"][:, :, H:H + T]              # (K, B, T)
            log_sigma = out["log_sigma"][:, :, H:H + T]    # (K, B, T)
            y_true = Q[:, H:H + T]                         # (B, T)
            w = lowflow_weight(y_true, q_min_norm)         # (B, T)
            flow = log_mixture_nll(y_true, mu_Q, log_sigma, weight=w)

            kl_raw = core.kl_to_prior(out["mu"], out["log_var"])
            kl_fb = core.kl_to_prior_free_bits(
                out["mu"], out["log_var"], free_bits=free_bits,
            )

            # Smoothness sobre la media de las soltadas O_k(t) entre samples.
            O_mean = out["O_hist"][:, :, H:H + T, :].mean(dim=0)  # (B, T, M)
            smooth = (O_mean[:, 1:, :] - O_mean[:, :-1, :]).abs().mean() \
                if O_mean.size(1) > 1 else flow.new_tensor(0.0)

            sparse = out.get("expected_l0_total", flow.new_tensor(0.0))
            total = flow + cfg.lam_smooth * smooth + cfg.lam_sparse * sparse \
                + beta_eff * kl_fb

            optim.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(core.parameters(), cfg.grad_clip)
            optim.step()

            agg["loss"] += float(total.detach())
            agg["flow"] += float(flow.detach())
            agg["kl"] += float(kl_raw.detach())
            agg["kl_fb"] += float(kl_fb.detach())
            agg["smooth"] += float(smooth.detach())
            nb += 1
        for k in agg:
            agg[k] /= max(1, nb)
        agg["epoch"] = ep
        agg["beta_eff"] = beta_eff
        historico.append(agg)
        if verbose_every and (ep % verbose_every == 0 or ep == 1):
            print(f"   [ep {ep:4d}] β={beta_eff:.2e}  loss={agg['loss']:.4f}  "
                  f"flow={agg['flow']:.4f}  kl={agg['kl']:.4f}  "
                  f"kl_fb={agg['kl_fb']:.4f}")
    return historico


# ===========================================================================
# RunoffModel adapter
# ===========================================================================


class UAHydroGNNModel(RunoffModel):
    """RunoffModel adapter para UA-HydroGNN (sec. 4 del paper).

    Construido sobre `UAHydroGNNCore`. La interfaz `predict()` devuelve la
    media sobre `K_inference` muestras MC (compatible con `rolling_evaluation`).
    `predict_distribution()` devuelve `(M, K, T)` caudal en m³/s para
    alimentar los criterios de decisión cost-aware sobre dos ejes de
    incertidumbre (sec. §4.5).
    """

    nombre = "ua-hydrognn"
    use_gates = "none"   # Phase 1 por defecto

    def __init__(
        self,
        cfg: GNNConfig,
        graph: BasinGraph,
        K_train: int = 10,
        K_inference: int = 50,
        beta_ua: float = 1e-3,
        warmup_epochs: int = 0,
        ramp_epochs: int = 1,
        free_bits: float = 0.0,
        rain_bypass: bool = False,
        lam11_init: float = 0.0,
        max_windows: Optional[int] = None,
        river_velocity_km_day: Optional[float] = None,
    ):
        """Adapter del modelo UA-HydroGNN.

        Defaults: configuración \"original\" del paper (sin remediación
        N=64). Reproduce los headline de Ebro y synth-N=16 (FN=44 y FN=0
        respectivamente bajo Savage). Para basin sintéticos grandes
        (N1 ≳ 32) que muestran el atractor de predicción constante,
        activar la remediación con
            rain_bypass=True, lam11_init=2.0,
            warmup_epochs=80, ramp_epochs=40, free_bits=0.02.
        Cf. §4.10 del paper, "Remediation as a per-basin hyperparameter".

        `river_velocity_km_day` activa la inicialización informada de λ
        cuando el `BasinGraph` aporta longitudes (cf. \\Cref{sec:ebro})."""
        self.cfg = cfg
        self.graph = graph
        self.K_train = K_train
        self.K_inference = K_inference
        self.beta_ua = beta_ua
        self.warmup_epochs = int(warmup_epochs)
        self.ramp_epochs = max(1, int(ramp_epochs))
        self.free_bits = float(free_bits)
        self.rain_bypass = bool(rain_bypass)
        self.lam11_init = float(lam11_init)
        self.max_windows = max_windows
        self.river_velocity_km_day = river_velocity_km_day
        torch.manual_seed(cfg.semilla)
        self.core = UAHydroGNNCore(
            graph,
            use_gates=self.use_gates,
            node_static_dim=cfg.node_static_dim,
            ctx_dim=cfg.ctx_dim,
            hidden=cfg.hidden,
            K_train=K_train,
            K_inference=K_inference,
            rain_bypass=rain_bypass,
            lam11_init=lam11_init,
            river_velocity_km_day=river_velocity_km_day,
        )
        self._maximos: Optional[pd.Series] = None

    # --- helpers --------------------------------------------------------

    def _q_min_normalizado(self, maximos: pd.Series, caudal_minimo_m3s: float) -> float:
        return float(caudal_minimo_m3s / maximos[self.cfg.basin.flow_column])

    # --- API: fit -------------------------------------------------------

    def fit(self, df_train: pd.DataFrame, maximos: pd.Series) -> List[dict]:
        cfg = self.cfg
        self._maximos = maximos.copy()
        rng = np.random.default_rng(cfg.semilla)
        ventanas = list(build_training_dataset(
            df_train, self.graph,
            H=cfg.historia, T=cfg.horizonte,
            flow_column=cfg.basin.flow_column,
            observed_stations=cfg.observed_stations,
        ))
        if not ventanas:
            raise ValueError("No se han podido construir ventanas (datos insuficientes).")

        q_min_norm = self._q_min_normalizado(maximos, cfg.basin.caudal_minimo_m3s)
        return _train_ua_core(
            self.core, ventanas, cfg, q_min_norm,
            beta_ua=self.beta_ua, K_train=self.K_train,
            warmup_epochs=self.warmup_epochs,
            ramp_epochs=self.ramp_epochs,
            free_bits=self.free_bits,
            verbose_every=max(1, cfg.epochs // 10),
            max_windows=self.max_windows,
            window_seed=cfg.semilla,
        )

    # --- API: predict (compatible con rolling_evaluation) ---------------

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

        out = self.core.forward_mc(rain, mask, ctx, H=cfg.historia,
                                    K=self.K_inference, return_states=True)
        H, T = cfg.historia, cfg.horizonte
        mu_norm = out["mu_Q"][:, 0, H:H + T].mean(dim=0).cpu().numpy()  # (T,)
        S_norm = out["S_hist"][:, 0, H:H + T].mean(dim=(0)).cpu().numpy()  # (T, M)
        sigma = F.softplus(out["log_sigma"][:, 0, H:H + T]).mean(dim=0).cpu().numpy() + 1e-6
        q_min_norm = self._q_min_normalizado(maximos, cfg.basin.caudal_minimo_m3s)
        p_comp = 0.5 * (1 + np.vectorize(math.erf)((mu_norm - q_min_norm) / (sigma * math.sqrt(2))))

        fechas = pd.date_range(hoy + timedelta(days=1), periods=cfg.horizonte)
        return Forecast(
            fechas=fechas,
            caudal=mu_norm * maximos[cfg.basin.flow_column],
            embalse=S_norm.sum(axis=-1) * maximos.get(cfg.basin.reservoir_aggregate_column, 1.0),
            caudal_logit=p_comp,
        )

    # --- API: predict_distribution (sec. §4.5) ---------------------------

    @torch.no_grad()
    def predict_distribution(
        self,
        df: pd.DataFrame,
        hoy: pd.Timestamp,
        maximos: pd.Series,
        pacum_future: np.ndarray,      # (T,) o (M, T): M escenarios de lluvia agregada
        K: Optional[int] = None,
    ) -> np.ndarray:
        """Distribución predictiva de caudal en m³/s.

        Cada escenario s suministra una trayectoria agregada de lluvia
        futura `pacum_future[s, t]`. La trayectoria se distribuye
        uniformemente entre los nodos Tipo-1 con estación observable
        (`graph.rain_to_type1`), y K muestras del posterior inicial
        producen la mezcla predictiva en el aforo.

        Returns
        -------
        np.ndarray shape (M, K, T)
        """
        cfg = self.cfg
        device = torch.device(cfg.device)
        self.core.eval()
        K = K or self.K_inference

        pacum_arr = np.asarray(pacum_future, dtype=np.float32)
        if pacum_arr.ndim == 1:
            pacum_arr = pacum_arr.reshape(1, -1)
        M, T = pacum_arr.shape
        assert T == cfg.horizonte, f"pacum_future debe tener T={cfg.horizonte} pasos"

        # 1. Ventana base (igual para todos los escenarios): construye
        #    rain, mask, ctx con la HISTORIA observada y los T días futuros
        #    en blanco (los rellenaremos por escenario).
        df_local = df.copy()
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

        ventana = build_window(
            df_local, self.graph, hoy, cfg.historia, cfg.horizonte,
            flow_column=cfg.basin.flow_column,
            observed_stations=cfg.observed_stations,
        )
        rain_base = ventana.rain.unsqueeze(0).to(device)   # (1, L, N1)
        mask_base = ventana.mask.unsqueeze(0).to(device)
        ctx_base = ventana.ctx.unsqueeze(0).to(device)
        L = rain_base.shape[1]
        N1 = rain_base.shape[2]
        H = cfg.historia

        # 2. Para cada escenario, sustituye la lluvia FUTURA por la
        #    trayectoria del escenario distribuida entre los nodos Tipo-1
        #    con pluviómetro observable.
        nodos_obs = sorted(set(self.graph.rain_to_type1.values()))
        if cfg.observed_stations is not None:
            nodos_obs = [self.graph.rain_to_type1[s] for s in cfg.observed_stations
                          if s in self.graph.rain_to_type1]
        if not nodos_obs:
            nodos_obs = sorted(set(self.graph.rain_to_type1.values()))
        # Pasa pacum a unidades normalizadas (mismo factor que se usa en
        # entrenamiento).
        pacum_norm = pacum_arr / maximos[cfg.basin.rain_aggregate_column]
        per_station = pacum_norm[:, :, None] / max(len(nodos_obs), 1)   # (M, T, 1)

        rain_scenarios = rain_base.repeat(M, 1, 1)   # (M, L, N1)
        for n_idx in nodos_obs:
            rain_scenarios[:, H:H + T, n_idx] = torch.tensor(
                per_station[:, :, 0], dtype=rain_scenarios.dtype, device=device
            )
        mask_scenarios = mask_base.repeat(M, 1, 1)
        ctx_scenarios = ctx_base.repeat(M, 1, 1)

        # 3. K muestras del posterior por cada uno de los M escenarios.
        #    El posterior depende sólo del burn-in (que es idéntico entre
        #    escenarios), por lo que basta con calcularlo una vez por sample.
        out = self.core.forward_mc(rain_scenarios, mask_scenarios, ctx_scenarios,
                                    H=H, K=K, return_states=False)
        mu_Q = out["mu_Q"][:, :, H:H + T]   # (K, M, T) en unidades normalizadas

        # Reordena a (M, K, T) y reescala a m³/s.
        caudal_m3s = mu_Q.permute(1, 0, 2).cpu().numpy() * float(maximos[cfg.basin.flow_column])
        return caudal_m3s

    # --- persistencia ---------------------------------------------------

    def save(self, directorio) -> None:
        d = Path(directorio)
        d.mkdir(parents=True, exist_ok=True)
        torch.save(self.core.state_dict(), d / "ua_core.pt")
        with open(d / "ua_meta.pkl", "wb") as f:
            pickle.dump({
                "cfg": self.cfg, "graph": self.graph,
                "maximos": self._maximos,
                "K_train": self.K_train, "K_inference": self.K_inference,
                "beta_ua": self.beta_ua,
                "warmup_epochs": self.warmup_epochs,
                "ramp_epochs": self.ramp_epochs,
                "free_bits": self.free_bits,
                "rain_bypass": self.rain_bypass,
                "lam11_init": self.lam11_init,
                "river_velocity_km_day": self.river_velocity_km_day,
            }, f)

    @classmethod
    def load(cls, directorio, config: Config = None) -> "UAHydroGNNModel":
        d = Path(directorio)
        with open(d / "ua_meta.pkl", "rb") as f:
            meta = pickle.load(f)
        instancia = cls(
            meta["cfg"], graph=meta["graph"],
            K_train=meta["K_train"],
            K_inference=meta["K_inference"],
            beta_ua=meta["beta_ua"],
            warmup_epochs=meta.get("warmup_epochs", 0),
            ramp_epochs=meta.get("ramp_epochs", 1),
            free_bits=meta.get("free_bits", 0.0),
            rain_bypass=meta.get("rain_bypass", False),
            lam11_init=meta.get("lam11_init", 0.0),
            river_velocity_km_day=meta.get("river_velocity_km_day", None),
        )
        instancia.core.load_state_dict(torch.load(d / "ua_core.pt", map_location="cpu"))
        instancia._maximos = meta["maximos"]
        return instancia
