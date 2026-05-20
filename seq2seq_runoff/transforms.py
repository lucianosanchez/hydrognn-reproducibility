"""Transformaciones invertibles aplicadas al caudal antes de modelar.

El notebook original ofrecía siete modos numerados (TRANSFORMACION=0..6) que
se mezclaban entre sí. Aquí cada modo es una clase pequeña que cumple un
mismo `Protocol`:

    forward:  caudal real (m³/s)  →  espacio del modelo
    inverse:  espacio del modelo   →  caudal real (m³/s)

`IdentityTransform` es la opción por defecto y la usada en el modelo baseline.
Se incluye `BoxCoxTransform` como ejemplo de cómo añadir nuevas; otras
opciones (PIT, log, clip, discretización) se pueden implementar en el mismo
patrón.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd
from scipy import special, stats


@runtime_checkable
class FlowTransform(Protocol):
    """Transformación monótona e invertible del caudal."""

    def forward(self, q: np.ndarray | pd.Series) -> np.ndarray | pd.Series: ...
    def inverse(self, z: np.ndarray | pd.Series) -> np.ndarray | pd.Series: ...


class IdentityTransform:
    """Sin transformación. Útil como baseline y para depurar."""

    def forward(self, q):
        return q

    def inverse(self, z):
        return z


class BoxCoxTransform:
    """Box-Cox seguida de estandarización a media 0 y desviación 1.

    Se ajusta el parámetro lambda sobre el periodo de entrenamiento que se
    pase al constructor; la transformación se aplica luego a toda la serie.
    """

    def __init__(self, q_train: np.ndarray | pd.Series):
        q_train = np.asarray(q_train)
        transformed, self.lambda_ = stats.boxcox(q_train)
        self.media = float(np.mean(transformed))
        self.std = float(np.std(transformed))
        self.minimo = float((np.min(transformed) - self.media) / self.std)

    def forward(self, q):
        z = special.boxcox(np.asarray(q), self.lambda_)
        return (z - self.media) / self.std - self.minimo

    def inverse(self, z):
        z = (np.asarray(z) + self.minimo) * self.std + self.media
        return special.inv_boxcox(z, self.lambda_)
