"""Configuración central del experimento.

`Config` agrupa los hiperparámetros de la tubería; la información de la
cuenca vive en `BasinSpec`. Una instancia de `Config` siempre lleva un
`basin`, y las columnas de las variables del modelo se derivan de él.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .basin import BasinSpec


@dataclass
class Config:
    """Hiperparámetros del modelo y de la tubería.

    Sólo contiene parámetros estables del experimento. Lo específico de la
    cuenca (estaciones, embalses, columnas) llega vía `basin`.
    """

    basin: BasinSpec

    # Ventana del codificador y horizonte de predicción.
    historia: int = 20
    horizonte: int = 10

    # Capacidad del LSTM del baseline Seq2Seq.
    latent_dim: int = 25

    # Entrenamiento del baseline.
    batch_size: int = 64
    epochs: int = 2000
    fraccion_test: float = 0.2
    fraccion_validacion: float = 0.5
    paciencia_factor: float = 0.25

    # Pérdida del embalse: peso del término de suavidad.
    peso_suavidad_embalse: float = 5.0

    # Pérdida del caudal: peso de los días anómalos relativo a los normales.
    # `desbalance` alto sesga el clasificador a declarar alarma con más frecuencia,
    # reduciendo FN a costa de FP. Es el equivalente Seq2Seq a `kappa_low_flow`
    # del GNN — sweepable para optimizar el coste asimétrico.
    desbalance: float = 10.0

    # Umbral operacional. Por defecto el de la cuenca; se puede sobreescribir.
    caudal_minimo_m3s: Optional[float] = None

    def __post_init__(self):
        if self.caudal_minimo_m3s is None:
            self.caudal_minimo_m3s = self.basin.caudal_minimo_m3s

    # --- columnas de las variables (derivadas del basin) ------------------

    @property
    def variables_codificador(self) -> List[str]:
        return self.basin.encoder_columns

    @property
    def variables_decoder_embalse(self) -> List[str]:
        return self.basin.decoder_embalse_columns

    @property
    def variables_decoder_caudal(self) -> List[str]:
        return self.basin.decoder_caudal_columns

    @property
    def num_variables_codificador(self) -> int:
        return len(self.variables_codificador)
