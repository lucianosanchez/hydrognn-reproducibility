"""Pérdidas del modelo GNN — implementan la sec. 6 de report2.tex.

Componentes:
    L_flow   Eq. 6.1 — NLL gaussiana ponderada hacia caudales bajos.
    L_smooth Eq. 6.2 — penalización de oscilaciones en O_k(t).
    L_sparse Eq. 6.3 — L0 esperado de los gates.
    L_phys   Eq. 6.4 — residuo numérico de balance de masa (opcional).
    L_res    Extra de Fase 1 — MSE entre S_k predicho y EACUM observado.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from .core import HydroGNNOutput


def lowflow_weight(y: torch.Tensor, q_min: float, kappa: float = 5.0, escala: float = 5.0) -> torch.Tensor:
    """Peso w(y) que enfatiza errores cerca y por debajo del umbral (eq. 6.1)."""
    return 1.0 + kappa * torch.sigmoid((q_min - y) / escala)


def gaussian_nll(mu: torch.Tensor, log_sigma: torch.Tensor, y: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """NLL gaussiana por paso, sin reducir."""
    sigma = F.softplus(log_sigma) + eps
    return 0.5 * ((y - mu) / sigma) ** 2 + torch.log(sigma)


@dataclass
class LossParts:
    total: torch.Tensor
    flow: torch.Tensor
    smooth: torch.Tensor
    sparse: torch.Tensor
    phys: torch.Tensor
    embalse_obs: torch.Tensor


def total_loss(
    output: HydroGNNOutput,
    Q_obs: torch.Tensor,                  # (B, L)
    H: int,
    T: int,
    q_min: float,
    *,
    S_obs: Optional[torch.Tensor] = None,  # (B, L, M_obs) — sólo Fase 1
    obs_to_res_index: Optional[torch.Tensor] = None,  # (M_obs,) — índices en V_2
    lam_smooth: float = 0.01,
    lam_sparse: float = 1e-3,
    lam_phys: float = 0.0,
    lam_res: float = 0.0,
    kappa_low_flow: float = 5.0,
    escala_low_flow: float = 5.0,
) -> LossParts:
    """Pérdida total (eq. 6.6) con piezas que pueden activarse o no."""
    y_true = Q_obs[:, H:H + T]
    mu = output.mu_Q[:, H:H + T]
    ls = output.log_sigma[:, H:H + T]

    nll = gaussian_nll(mu, ls, y_true)
    w = lowflow_weight(y_true, q_min, kappa=kappa_low_flow, escala=escala_low_flow)
    flow = (w * nll).mean()

    O_f = output.O_hist[:, H:H + T, :]
    smooth = (O_f[:, 1:, :] - O_f[:, :-1, :]).abs().mean() if O_f.size(1) > 1 else flow.new_tensor(0.0)

    sparse = output.expected_l0_total

    phys = flow.new_tensor(0.0)
    if lam_phys > 0:
        # Residuo simple: |dS - I + O + L| (con L despreciado en este modelo mínimo).
        S = output.S_hist
        dS = S[:, 1:, :] - S[:, :-1, :]
        dO = output.O_hist[:, 1:, :]
        phys = dS.add(dO).abs().mean()

    embalse_obs = flow.new_tensor(0.0)
    if S_obs is not None and obs_to_res_index is not None and lam_res > 0:
        S_pred = output.S_hist[:, H:H + T, :].index_select(-1, obs_to_res_index)
        embalse_obs = F.mse_loss(S_pred, S_obs[:, H:H + T, :])

    total = flow + lam_smooth * smooth + lam_sparse * sparse + lam_phys * phys + lam_res * embalse_obs
    return LossParts(total=total, flow=flow, smooth=smooth, sparse=sparse, phys=phys, embalse_obs=embalse_obs)
