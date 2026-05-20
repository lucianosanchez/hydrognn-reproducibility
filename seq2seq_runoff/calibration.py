"""Calibrador monótono entre la salida del decodificador de caudal y el caudal real.

El decodificador de caudal produce una "logit" en [0, 1] que representa la
confianza en superar el umbral. Para visualizar y para pasarla a m³/s
ajustamos un spline monótono PCHIP que mapee logit → caudal observado.

Cuando la pérdida del caudal es identidad (regresión), este calibrador no
hace nada (`forward = identidad`); por eso el constructor admite el modo
`identity` que no requiere ajuste.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import minimize


class MonotonicCalibrator:
    """Spline monótono PCHIP de [0, 1] en [0, 1].

    Se ajusta minimizando el ECM contra los pares (logit, observado), con un
    término de penalización que castiga las violaciones de monotonía.
    """

    def __init__(self, n_puntos: int = 50, peso_monotonia: float = 0.01):
        self.n_puntos = n_puntos
        self.peso_monotonia = peso_monotonia
        self._spline: Optional[PchipInterpolator] = None

    @classmethod
    def identity(cls) -> "MonotonicCalibrator":
        cal = cls()
        x = np.linspace(0.0, 1.0, 2)
        cal._spline = PchipInterpolator(x, x)
        return cal

    def fit(self, logits: np.ndarray, observados: np.ndarray) -> "MonotonicCalibrator":
        """Ajusta el spline a partir de los pares (logit, observado)."""
        logits = np.asarray(logits).reshape(-1)
        observados = np.asarray(observados).reshape(-1)
        x_eval = np.linspace(0.0, 1.0, self.n_puntos)
        t_eval = np.ones_like(x_eval) * 0.5

        def objetivo(t):
            par = np.cumsum(t)
            par = par - par[0]
            par = par / par[-1]
            spline = PchipInterpolator(x_eval, par)
            ecm = np.mean((observados - spline(logits)) ** 2)
            monotonia = np.sum(np.minimum(par[1:] - par[:-1], 0))
            return ecm - self.peso_monotonia * monotonia

        resultado = minimize(
            objetivo,
            t_eval,
            method="L-BFGS-B",
            bounds=[(0.0, np.inf)] * len(t_eval),
        )
        par = np.cumsum(resultado.x)
        par = par - par[0]
        par = par / par[-1]
        self._spline = PchipInterpolator(x_eval, par)
        return self

    def __call__(self, logit: np.ndarray) -> np.ndarray:
        if self._spline is None:
            raise RuntimeError("MonotonicCalibrator no está ajustado.")
        return self._spline(np.asarray(logit))
