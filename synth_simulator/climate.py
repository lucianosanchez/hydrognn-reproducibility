"""Generador estocástico de pluviosidad.

Modelo deliberadamente sencillo, con tres ingredientes que dan series
realistas:

  1. **Estacionalidad**: la probabilidad de día lluvioso varía de forma
     sinusoidal entre un valor invernal (p_wet_winter) y uno estival
     (p_wet_summer), con máximo en torno al 1 de enero.
  2. **Persistencia (Markov)**: una cadena binaria regional gobierna si
     "el día es lluvioso en la cuenca" (parámetros p_wet_given_wet y
     p_wet_given_dry). Esto produce rachas creíbles.
  3. **Correlación espacial**: cada estación copia el estado regional con
     probabilidad `spatial_corr`; en otro caso elige independientemente.
     Cuando llueve, la cantidad sigue una Gamma de media `rainfall_mm_mean`.

Devuelve un DataFrame indexado por fecha y una columna por estación.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from .config import ClimateConfig


def _seasonal_p_wet(doy: np.ndarray, p_winter: float, p_summer: float) -> np.ndarray:
    """Sinusoide centrada en el solsticio invernal del hemisferio norte."""
    factor = 0.5 * (1 + np.cos(2 * np.pi * (doy - 15) / 365.25))  # 1 en invierno, 0 en verano
    return p_summer + (p_winter - p_summer) * factor


def generate_rainfall(climate: ClimateConfig, station_ids: List[str]) -> pd.DataFrame:
    """Genera la matriz pluviométrica del periodo del clima."""
    rng = np.random.default_rng(climate.seed)
    fechas = pd.date_range(climate.start_date, climate.end_date, freq="D")
    n_days = len(fechas)
    n_stations = len(station_ids)

    p_wet = _seasonal_p_wet(fechas.dayofyear.to_numpy(), climate.p_wet_winter, climate.p_wet_summer)

    # Cadena de Markov regional: gobierna si el día es lluvioso a nivel de cuenca.
    regional = np.zeros(n_days, dtype=np.int8)
    state = 1 if rng.random() < p_wet[0] else 0
    for t in range(n_days):
        if t > 0:
            p = climate.p_wet_given_wet if state == 1 else climate.p_wet_given_dry
            # Modulamos un poco la persistencia con la estacionalidad.
            p = 0.5 * (p + p_wet[t])
            state = 1 if rng.random() < p else 0
        regional[t] = state

    # Por estación: copia regional con prob spatial_corr; si no, decisión local.
    rain = np.zeros((n_days, n_stations), dtype=np.float32)
    scale = climate.rainfall_mm_mean / climate.rainfall_mm_shape
    for j in range(n_stations):
        sigue_regional = rng.random(n_days) < climate.spatial_corr
        decision_local = rng.random(n_days) < p_wet
        wet_today = np.where(sigue_regional, regional == 1, decision_local)
        amount = rng.gamma(shape=climate.rainfall_mm_shape, scale=scale, size=n_days).astype(np.float32)
        rain[:, j] = np.where(wet_today, amount, 0.0)

    return pd.DataFrame(rain, index=fechas, columns=station_ids)
