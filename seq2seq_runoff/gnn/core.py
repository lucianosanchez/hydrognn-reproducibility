"""Núcleo dinámico del HydroGNN — sigue la sec. 5 de report2.tex.

`HydroGNNCore` es un `nn.Module` que recorre L = H + T pasos de tiempo y
devuelve, para cada paso, la media y la log-σ del caudal en el nodo objetivo
y la trayectoria de las soltadas O_k. La fase del experimento (1, 2.1, 2.2)
sólo configura qué gates se aprenden; el bucle de simulación es el mismo.

Notación interna (dimensiones):
    B  = tamaño del batch
    L  = longitud temporal de la ventana (H + T)
    N1 = número de nodos Tipo-1
    M  = número de embalses (reales o candidatos)
    Eij = número de aristas en E_{ij}
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .gates import HardConcreteGate
from .graph import BasinGraph


def _scatter_add(values: torch.Tensor, index: torch.Tensor, dim: int, dim_size: int) -> torch.Tensor:
    """Equivalente compacto a torch_scatter.scatter_add (sin la dependencia)."""
    out_shape = list(values.shape)
    out_shape[dim] = dim_size
    out = torch.zeros(out_shape, dtype=values.dtype, device=values.device)
    expanded = index
    if values.dim() != index.dim():
        expanded = index.view(*(1,) * dim, -1, *(1,) * (values.dim() - dim - 1))
        expanded = expanded.expand_as(values)
    return out.scatter_add_(dim, expanded, values)


def _make_mlp(in_dim: int, hidden: int, out_dim: int, n_layers: int = 2, dropout: float = 0.0) -> nn.Sequential:
    capas = []
    d = in_dim
    for _ in range(n_layers - 1):
        capas += [nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(dropout)]
        d = hidden
    capas += [nn.Linear(d, out_dim)]
    return nn.Sequential(*capas)


@dataclass
class HydroGNNOutput:
    mu_Q: torch.Tensor          # (B, L)
    log_sigma: torch.Tensor     # (B, L)
    O_hist: torch.Tensor        # (B, L, M) — soltadas controladas
    S_hist: torch.Tensor        # (B, L, M) — almacenamiento
    expected_l0_total: torch.Tensor  # escalar


class HydroGNNCore(nn.Module):
    """Implementación del modelo Tipo-1 + Tipo-2 (sec. 5 / 11).

    Parameters
    ----------
    graph
        Grafo (real o candidato) con los conjuntos V_1, V_2, E_11, E_12, E_21.
    use_gates
        - "none"          : todos los gates fijos en 1 (Fase 1 y 2.1).
        - "edges"         : sólo E_12, E_21 son aprendibles.
        - "nodes_and_edges": gates de nodos y aristas (Fase 2.2).
    node_static_dim, ctx_dim, hidden
        Dimensiones de los embeddings y las MLPs.
    """

    def __init__(
        self,
        graph: BasinGraph,
        use_gates: str = "none",
        node_static_dim: int = 8,
        ctx_dim: int = 2,
        hidden: int = 64,
        logw12_init: float = 0.0,
        rain_bypass: bool = False,
        lam11_init: float = 0.0,
        river_velocity_km_day: Optional[float] = None,
    ):
        """
        Parameters
        ----------
        rain_bypass : bool, default False
            Si `True`, añade una "ruta directa" lluvia→caudal en paralelo a la
            propagación gráfica. Útil cuando la propagación es lenta (e.g.
            cuencas grandes con cadenas largas HM→OUTLET en horizontes
            cortos): rompe el atractor de "predicción constante" garantizando
            un gradiente no-nulo respecto a la lluvia futura desde el paso 0.
            Default `False` para no afectar a corridas pre-existentes.
        lam11_init : float, default 0.0
            Logit inicial homogéneo de los pesos de routing `lam11`.
            `0.0` da λ≈0.5; valores positivos (e.g. `2.0` ⇒ λ≈0.88)
            aceleran la propagación HM→OUTLET en cuencas grandes.
            **Se ignora por arista si `river_velocity_km_day` está dado y
            el grafo aporta `edge_len_km_11`** — en ese caso, λ por arista
            se inicializa de forma informada por la longitud fluvial.
        river_velocity_km_day : float, optional
            Velocidad efectiva del río en km/día. Si se proporciona y el
            `BasinGraph` aporta `edge_len_km_11` (y, opcionalmente,
            `len_12`/`len_21`), los logits de routing se inicializan
            como `logit(1 - exp(-Δt/τ_e))` con `τ_e = length_e / v`,
            Δt = 1 día. Esto reemplaza `lam11_init` por arista. El
            modelo sigue libre de ajustar λ por gradiente. Para Ebro,
            v ≈ 35-70 km/día es razonable.
        """
        super().__init__()
        if use_gates not in ("none", "edges", "nodes_and_edges"):
            raise ValueError(f"use_gates desconocido: {use_gates!r}")
        self.use_gates = use_gates
        self.logw12_init = float(logw12_init)
        self.rain_bypass = bool(rain_bypass)
        self.N1 = graph.N1
        self.M = graph.M
        self.target_idx = graph.target_node_idx

        self.register_buffer("edge_index_11", torch.from_numpy(graph.edge_index_11))
        self.register_buffer("src12", torch.from_numpy(graph.src12))
        self.register_buffer("dst12", torch.from_numpy(graph.dst12))
        self.register_buffer("src21", torch.from_numpy(graph.src21))
        self.register_buffer("dst21", torch.from_numpy(graph.dst21))

        self.E11 = int(graph.E11)
        self.E12 = int(graph.E12)
        self.E21 = int(graph.E21)

        self.node_embed = nn.Parameter(torch.randn(self.N1, node_static_dim) * 0.1)
        self.res_embed = nn.Parameter(torch.randn(self.M, node_static_dim) * 0.1)

        # Tipo-1: redes para runoff (eq. 5.6) y atenuación (eq. 5.8).
        in_type1 = 1 + 1 + 1 + node_static_dim + ctx_dim  # P, mask, F_in agregado, embed, ctx
        self.runoff_net = _make_mlp(in_type1, hidden, 1, n_layers=3)
        self.alpha_net = _make_mlp(in_type1, hidden, 1, n_layers=3)

        # Tipo-2: fracción de soltada β (eq. 5.12) con dependencia monótona en A.
        self.beta_ctx = _make_mlp(ctx_dim + node_static_dim, hidden, 1, n_layers=2)
        self.beta_wA = nn.Parameter(torch.zeros(self.M))
        self.gamma_logit = nn.Parameter(torch.full((self.M,), -4.0))

        # Cabezal de incertidumbre.
        self.sigma_head = _make_mlp(1 + ctx_dim, hidden, 1, n_layers=2)

        # Ruta directa lluvia→caudal (opt-in). Toma lluvia agregada en una
        # ventana causal corta (últimos `bypass_lookback` pasos) más ctx, y
        # predice una contribución aditiva al caudal del outlet. Se inicializa
        # con un sesgo positivo pequeño para arrancar con gradiente no-nulo.
        self.bypass_lookback = 4
        if self.rain_bypass:
            self.bypass_head = _make_mlp(self.bypass_lookback + ctx_dim,
                                          hidden, 1, n_layers=2)
            # Sesgo positivo pequeño en la última capa para activar gradiente.
            with torch.no_grad():
                self.bypass_head[-1].bias.fill_(0.05)
                self.bypass_head[-1].weight.mul_(0.1)
        else:
            self.bypass_head = None

        # Parámetros de routing y splitting.
        # Si tenemos longitudes fluviales y una velocidad efectiva, los
        # logits de routing arrancan con valor físicamente plausible por
        # arista: λ_e^(0) = 1 - exp(-Δt/τ_e), τ_e = length_e / v.
        def _logit_from_lengths(length_km: np.ndarray) -> torch.Tensor:
            tau_days = np.maximum(length_km / float(river_velocity_km_day), 1e-3)
            lam = 1.0 - np.exp(-1.0 / tau_days)               # Δt = 1 día
            lam = np.clip(lam, 1e-3, 1.0 - 1e-3)
            return torch.tensor(np.log(lam / (1.0 - lam)), dtype=torch.float32)

        if river_velocity_km_day is not None and getattr(graph, "edge_len_km_11", None) is not None:
            self.lam11_logit = nn.Parameter(_logit_from_lengths(graph.edge_len_km_11))
        else:
            self.lam11_logit = nn.Parameter(torch.full((self.E11,), float(lam11_init)))

        if river_velocity_km_day is not None and getattr(graph, "len_12", None) is not None:
            self.lam12_logit = nn.Parameter(_logit_from_lengths(graph.len_12))
        else:
            self.lam12_logit = nn.Parameter(torch.zeros(self.E12))

        if river_velocity_km_day is not None and getattr(graph, "len_21", None) is not None:
            self.lam21_logit = nn.Parameter(_logit_from_lengths(graph.len_21))
        else:
            self.lam21_logit = nn.Parameter(torch.zeros(self.E21))
        self.logw11 = nn.Parameter(torch.zeros(self.E11))
        # logw12 puede arrancar negativo cuando E_12 es denso, para evitar
        # que cada Tipo-1 envíe la mayor parte de su flujo a los embalses.
        self.logw12 = nn.Parameter(torch.full((self.E12,), float(logw12_init)))
        self.logw21 = nn.Parameter(torch.zeros(self.E21))

        # Hard-Concrete gates: se crean siempre, pero pueden quedar congelados.
        self.gate_res = HardConcreteGate((self.M,))
        self.gate_12 = HardConcreteGate((self.E12,))
        self.gate_21 = HardConcreteGate((self.E21,))
        if self.use_gates == "none":
            for g in (self.gate_res, self.gate_12, self.gate_21):
                g.log_alpha.requires_grad_(False)
                # Forzamos log_alpha grande para que la sigmoide sature en 1.
                g.log_alpha.data.fill_(10.0)
        elif self.use_gates == "edges":
            self.gate_res.log_alpha.requires_grad_(False)
            self.gate_res.log_alpha.data.fill_(10.0)

    # ---------------------------------------------------------------------
    # Penalización de sparsity para la pérdida total (eq. 5.4 + sec. 6.3).
    # ---------------------------------------------------------------------

    # ---------------------------------------------------------------------
    # Análisis post-entrenamiento (útil para Fase 2.2).
    # ---------------------------------------------------------------------

    @torch.no_grad()
    def analyze_positions(self) -> dict:
        """Devuelve un resumen interpretable de la estructura aprendida.

        Para Fase 2.2 con grafo de candidatos densos, esto muestra:
            * `inflow_share[k, i]`  : fracción del flujo de Tipo-1 i que
              entra al embalse latente k (softmax de logw12 + logw11 sobre
              las salidas del nodo i).
            * `outflow_share[k, j]` : a qué Tipo-1 va cada embalse k.
            * `lambda12, lambda21` : retardos de routing aprendidos.
        """
        self.eval()
        src11 = self.edge_index_11[0]

        # logits relativos al nodo origen para cada arista saliente.
        w11 = torch.exp(self.logw11)
        w12 = torch.exp(self.logw12)
        denom = _scatter_add(w11, src11, dim=0, dim_size=self.N1) \
              + _scatter_add(w12, self.src12, dim=0, dim_size=self.N1) + 1e-8

        share12 = (w12 / denom[self.src12])  # (E12,)
        inflow_share = torch.zeros(self.M, self.N1)
        for e in range(self.E12):
            i = int(self.src12[e]); k = int(self.dst12[e])
            inflow_share[k, i] += float(share12[e])

        w21 = torch.exp(self.logw21)
        denomR = _scatter_add(w21, self.src21, dim=0, dim_size=self.M) + 1e-8
        share21 = (w21 / denomR[self.src21])
        outflow_share = torch.zeros(self.M, self.N1)
        for e in range(self.E21):
            k = int(self.src21[e]); j = int(self.dst21[e])
            outflow_share[k, j] += float(share21[e])

        return {
            "inflow_share": inflow_share.numpy(),
            "outflow_share": outflow_share.numpy(),
            "lambda12": torch.sigmoid(self.lam12_logit).cpu().numpy(),
            "lambda21": torch.sigmoid(self.lam21_logit).cpu().numpy(),
            "z_res": self.gate_res.sample(training=False).cpu().numpy(),
        }

    def expected_l0_total(self) -> torch.Tensor:
        """Suma del L0 esperado sobre los gates aprendibles."""
        suma = self.edge_index_11.new_tensor(0.0, dtype=torch.float32)
        if self.use_gates == "nodes_and_edges":
            suma = suma + self.gate_res.expected_l0().sum()
        if self.use_gates in ("edges", "nodes_and_edges"):
            suma = suma + self.gate_12.expected_l0().sum() + self.gate_21.expected_l0().sum()
        return suma

    # ---------------------------------------------------------------------
    # Forward — bucle de simulación L pasos.
    # ---------------------------------------------------------------------

    def forward(
        self,
        rain: torch.Tensor,         # (B, L, N1)
        mask: torch.Tensor,         # (B, L, N1)
        ctx: torch.Tensor,          # (B, L, ctx_dim)
        initial_S: torch.Tensor = None,    # (B, M)   override del estado inicial
        initial_x11: torch.Tensor = None,  # (B, E11) override del estado inicial
        initial_x12: torch.Tensor = None,  # (B, E12) override del estado inicial
        initial_x21: torch.Tensor = None,  # (B, E21) override del estado inicial
    ) -> HydroGNNOutput:
        device = rain.device
        B, L, N1 = rain.shape
        assert N1 == self.N1, f"N1 esperado={self.N1}, recibido={N1}"
        eps = 1e-8

        z_res = self.gate_res.sample(self.training).to(device)
        z_12 = self.gate_12.sample(self.training).to(device)
        z_21 = self.gate_21.sample(self.training).to(device)

        # Estados iniciales: cero por defecto, opcionalmente sobreescribibles
        # por un muestreo del posterior (UA-HydroGNN, sec. 4 del paper).
        S = initial_S if initial_S is not None else rain.new_zeros(B, self.M)
        x11 = initial_x11 if initial_x11 is not None else rain.new_zeros(B, self.E11)
        x12 = initial_x12 if initial_x12 is not None else rain.new_zeros(B, self.E12)
        x21 = initial_x21 if initial_x21 is not None else rain.new_zeros(B, self.E21)

        src11 = self.edge_index_11[0]
        dst11 = self.edge_index_11[1]

        lam11 = torch.sigmoid(self.lam11_logit).clamp(1e-4, 1.0)
        lam12 = torch.sigmoid(self.lam12_logit).clamp(1e-4, 1.0)
        lam21 = torch.sigmoid(self.lam21_logit).clamp(1e-4, 1.0)

        node_emb = self.node_embed.unsqueeze(0).expand(B, -1, -1)
        res_emb = self.res_embed.unsqueeze(0).expand(B, -1, -1)

        mu_list, ls_list, O_list, S_list = [], [], [], []

        for t in range(L):
            P_t = rain[:, t, :]                 # (B, N1)
            M_t = mask[:, t, :]                 # (B, N1)
            C_t = ctx[:, t, :]                  # (B, ctx_dim)
            P_eff = torch.where(M_t > 0.5, P_t, torch.zeros_like(P_t))

            # Flujos entrantes (eq. 5.7).
            inflow_11 = _scatter_add(x11, dst11, dim=1, dim_size=self.N1)
            inflow_21 = _scatter_add(x21, self.dst21, dim=1, dim_size=self.N1)
            inflow_total = inflow_11 + inflow_21

            C_node = C_t.unsqueeze(1).expand(B, self.N1, C_t.size(-1))
            type1_in = torch.cat(
                [
                    P_eff.unsqueeze(-1),
                    M_t.unsqueeze(-1),
                    inflow_total.unsqueeze(-1),
                    node_emb,
                    C_node,
                ],
                dim=-1,
            )

            # Runoff (eq. 5.6) y outflow Tipo-1 (eq. 5.8).
            r = F.softplus(self.runoff_net(type1_in)).squeeze(-1)
            F_i = inflow_total + r
            alpha = torch.sigmoid(self.alpha_net(type1_in)).squeeze(-1)
            q_out = alpha * F_i  # 0 ≤ q_i ≤ F_i por construcción

            # Splitting (eq. 5.9): pesos ponderados por gates.
            w11 = torch.exp(self.logw11)
            w12 = torch.exp(self.logw12) * z_12 * z_res[self.dst12]

            denom_11 = _scatter_add(w11, src11, dim=0, dim_size=self.N1)
            denom_12 = _scatter_add(w12, self.src12, dim=0, dim_size=self.N1)
            denom = denom_11 + denom_12 + eps

            pi11 = w11 / denom[src11]                       # (E11,)
            pi12 = w12 / denom[self.src12]                  # (E12,)
            msg11 = pi11.unsqueeze(0) * q_out[:, src11]     # (B, E11)
            msg12 = pi12.unsqueeze(0) * q_out[:, self.src12]

            # Routing (eq. 5.10).
            x11 = (1 - lam11).unsqueeze(0) * x11 + lam11.unsqueeze(0) * msg11
            x12 = (1 - lam12).unsqueeze(0) * x12 + lam12.unsqueeze(0) * msg12

            # Tipo-2 dynamics (sec. 5.11–5.12).
            I_k = _scatter_add(x12, self.dst12, dim=1, dim_size=self.M)
            A_k = S + I_k
            C_res = C_t.unsqueeze(1).expand(B, self.M, C_t.size(-1))
            beta_ctx_term = self.beta_ctx(torch.cat([C_res, res_emb], dim=-1)).squeeze(-1)
            wA = F.softplus(self.beta_wA).unsqueeze(0)
            beta_k = torch.sigmoid(beta_ctx_term + wA * torch.log1p(A_k + 1e-6))
            gamma_k = torch.sigmoid(self.gamma_logit).unsqueeze(0)
            O_k = (z_res.unsqueeze(0) * beta_k) * A_k
            L_k = (z_res.unsqueeze(0) * gamma_k) * S
            S = F.softplus(S + I_k - O_k - L_k)
            O_list.append(O_k)
            S_list.append(S)

            # Routing sueltas a Type-1 (sec. 5.13).
            w21 = torch.exp(self.logw21) * z_21 * z_res[self.src21]
            denomR = _scatter_add(w21, self.src21, dim=0, dim_size=self.M) + eps
            pi21 = w21 / denomR[self.src21]
            msg21 = pi21.unsqueeze(0) * O_k[:, self.src21]
            x21 = (1 - lam21).unsqueeze(0) * x21 + lam21.unsqueeze(0) * msg21

            # Salida en el nodo objetivo (sec. 5.14).
            mu_t = q_out[:, self.target_idx]

            # Ruta directa lluvia→caudal: garantiza gradiente respecto a la
            # lluvia (especialmente futura) sin pasar por toda la cadena de
            # routing. Toma la suma por-step de los últimos K pasos de la
            # lluvia agregada sobre la cuenca, donde K = bypass_lookback.
            if self.bypass_head is not None:
                K_bp = self.bypass_lookback
                # lluvia agregada efectiva por nodo (P_eff promedio sobre N1)
                # para los últimos K_bp pasos (incluido t).
                start = max(0, t + 1 - K_bp)
                P_window = rain[:, start:t + 1, :]
                M_window = mask[:, start:t + 1, :]
                P_eff_window = torch.where(M_window > 0.5, P_window,
                                            torch.zeros_like(P_window))
                # Agrega por nodo: mean sobre N1 con mask normalizada.
                m_count = M_window.sum(dim=-1).clamp(min=1.0)        # (B, len)
                P_agg_seq = P_eff_window.sum(dim=-1) / m_count        # (B, len)
                # Pad a K_bp si la ventana es más corta (relleno con ceros al
                # principio para mantener la dimensionalidad fija).
                Lw = P_agg_seq.shape[1]
                if Lw < K_bp:
                    pad = P_agg_seq.new_zeros(B, K_bp - Lw)
                    P_agg_seq = torch.cat([pad, P_agg_seq], dim=1)
                bypass_in = torch.cat([P_agg_seq, C_t], dim=-1)       # (B, K_bp+ctx)
                mu_bp = F.softplus(self.bypass_head(bypass_in).squeeze(-1))
                mu_t = mu_t + mu_bp

            mu_list.append(mu_t)
            sigma_in = torch.cat([mu_t.unsqueeze(-1), C_t], dim=-1)
            ls_t = self.sigma_head(sigma_in).squeeze(-1)
            ls_list.append(ls_t)

        return HydroGNNOutput(
            mu_Q=torch.stack(mu_list, dim=1),
            log_sigma=torch.stack(ls_list, dim=1),
            O_hist=torch.stack(O_list, dim=1),
            S_hist=torch.stack(S_list, dim=1),
            expected_l0_total=self.expected_l0_total(),
        )
