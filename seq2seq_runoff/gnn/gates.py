"""Hard-Concrete gates (sec. 5.4 de report2.tex).

La distribución Hard-Concrete es una relajación continua de Bernoulli que
permite penalizar el L0 esperado (número de unidades activas) de manera
diferenciable. Implementación literal del pseudocódigo (sec. 11).

Cada gate produce un escalar `z ∈ [0, 1]` por cada elemento de su shape.
En el modelo se usa para:
    z_k    decide si el embalse latente k está activo (Fase 2.2).
    z_e    decide si una arista candidata sobrevive.

Durante entrenamiento se muestrea con ruido (estocástico, derivable);
durante inferencia se usa la media determinista.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
from torch import nn


class HardConcreteGate(nn.Module):
    """Gate Hard-Concrete con penalización L0 esperada."""

    def __init__(
        self,
        shape: Tuple[int, ...],
        init_log_alpha: float = -2.0,
        beta: float = 2.0 / 3.0,
        gamma: float = -0.1,
        zeta: float = 1.1,
    ):
        super().__init__()
        if not (gamma < 0.0 < 1.0 < zeta):
            raise ValueError(f"Se requiere gamma<0<1<zeta; recibido gamma={gamma}, zeta={zeta}")
        self.log_alpha = nn.Parameter(torch.full(shape, float(init_log_alpha)))
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.zeta = float(zeta)

    def sample(self, training: bool) -> torch.Tensor:
        if training:
            u = torch.rand_like(self.log_alpha).clamp(1e-6, 1.0 - 1e-6)
            s = torch.sigmoid(
                (torch.log(u) - torch.log1p(-u) + self.log_alpha) / self.beta
            )
        else:
            s = torch.sigmoid(self.log_alpha)
        s_estirada = s * (self.zeta - self.gamma) + self.gamma
        return s_estirada.clamp(0.0, 1.0)

    def expected_l0(self) -> torch.Tensor:
        """Surrogate diferenciable de P(z > 0)."""
        const = -self.beta * math.log(-self.gamma / self.zeta)
        return torch.sigmoid(self.log_alpha + const)
