"""Librería de escenarios de lluvia futura (sec. 3.4 de paper_methods.tex).

Cada escenario es una terna de factores multiplicativos sobre las
estadísticas del clima generativo:

    p_wet_scale     prevalencia de días lluviosos
    mu_p_scale      intensidad media por evento (mm)
    shape_scale     parámetro de forma Gamma (más bajo ⇒ cola más pesada)

El cambio climático en el Mediterráneo se proyecta como una combinación
de menor frecuencia de eventos, menor cantidad acumulada y mayor
concentración en eventos extremos (cf. CMIP6, IPCC AR6). La librería
incluye cinco escenarios canónicos que cubren ese espectro más un
ancla de peor caso (lluvia nula).

Dos aplicaciones de escenarios:

  * Sobre la cuenca SINTÉTICA: se generan trayectorias de pluviosidad
    desde cero con `apply_scenario_to_climate`, que produce un
    `ClimateConfig` modificado, y se usa el generador estándar para
    obtener M muestras Monte Carlo.

  * Sobre la cuenca REAL (Ebro): la pluviosidad futura es un registro
    histórico ya conocido. `apply_scenario_to_historical` aplica las
    transformaciones multiplicativas + redistribución temporal a la
    serie observada para construir M variantes del registro.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .basin import BasinSpec


# ===========================================================================
# Definición de escenarios
# ===========================================================================


@dataclass(frozen=True)
class RainfallScenario:
    """Conjunto de factores multiplicativos que parametrizan una proyección
    climática sobre la distribución de pluviosidad futura.

    Los factores se aplican a los parámetros de la cadena de Markov regional
    (`p_wet`) y a la distribución Gamma de cantidades por evento.
    """
    name: str
    p_wet_scale: float = 1.0       # multiplica p_wet_winter, p_wet_summer
    mu_p_scale: float = 1.0        # multiplica rainfall_mm_mean
    shape_scale: float = 1.0       # multiplica shape de Gamma (1 = sin cambio)
    description: str = ""

    def __post_init__(self):
        # Validaciones suaves.
        for f in (self.p_wet_scale, self.mu_p_scale, self.shape_scale):
            if f < 0:
                raise ValueError(f"Factor multiplicativo negativo en {self.name}: {f}")


# Librería canónica usada en la sec. 3.4 del paper.
SCENARIO_LIBRARY: Dict[str, RainfallScenario] = {
    "baseline": RainfallScenario(
        name="baseline",
        description="Distribución de entrenamiento; estacionario.",
    ),
    "mild_drought": RainfallScenario(
        name="mild_drought",
        p_wet_scale=0.85, mu_p_scale=0.95,
        description="Tendencia mediterránea moderada (RCP-4.5 like).",
    ),
    "severe_drought": RainfallScenario(
        name="severe_drought",
        p_wet_scale=0.60, mu_p_scale=0.80,
        description="Tendencia mediterránea severa (RCP-8.5 like).",
    ),
    "flashy": RainfallScenario(
        name="flashy",
        p_wet_scale=0.70, mu_p_scale=1.70, shape_scale=0.60,
        description="Eventos menos frecuentes pero más intensos (colas pesadas).",
    ),
    "no_rain": RainfallScenario(
        name="no_rain", p_wet_scale=0.0,
        description="Ancla de peor caso — usado como worst-case en sec. 1.5.",
    ),
}


def default_library() -> List[RainfallScenario]:
    """Devuelve los 5 escenarios canónicos en orden de aparición."""
    return list(SCENARIO_LIBRARY.values())


# ===========================================================================
# Aplicación a la cuenca SINTÉTICA: modifica el ClimateConfig
# ===========================================================================


def apply_scenario_to_climate(climate, scenario: RainfallScenario):
    """Devuelve un `ClimateConfig` derivado aplicando los factores del escenario.

    El import de `ClimateConfig` se hace tarde para no introducir un
    requisito circular con `synth_simulator`.
    """
    return replace(
        climate,
        p_wet_winter=climate.p_wet_winter * scenario.p_wet_scale,
        p_wet_summer=climate.p_wet_summer * scenario.p_wet_scale,
        rainfall_mm_mean=climate.rainfall_mm_mean * scenario.mu_p_scale,
        rainfall_mm_shape=max(0.1, climate.rainfall_mm_shape * scenario.shape_scale),
    )


def sample_synthetic_rainfall_trajectories(
    base_climate,
    scenario: RainfallScenario,
    n_days: int,
    n_samples: int,
    station_ids: List[str],
    seed_offset: int = 0,
) -> np.ndarray:
    """Genera `n_samples` trayectorias de pluviosidad de `n_days` días bajo el
    escenario, agregadas sobre las estaciones (suma).

    Returns
    -------
    np.ndarray shape (n_samples, n_days)
    """
    from synth_simulator.climate import generate_rainfall
    from synth_simulator.config import ClimateConfig

    out = np.zeros((n_samples, n_days), dtype=np.float32)
    for m in range(n_samples):
        clima = apply_scenario_to_climate(base_climate, scenario)
        # Distinta semilla por muestra para diferentes realizaciones.
        clima = replace(clima, seed=clima.seed + seed_offset + 1000 * m)
        # Generamos sólo n_days; el generador necesita fechas inicio/fin coherentes.
        end_date = pd.to_datetime(clima.start_date) + pd.Timedelta(days=n_days - 1)
        clima_short = replace(clima, end_date=end_date.strftime("%Y-%m-%d"))
        rainfall = generate_rainfall(clima_short, station_ids)
        # Agregamos sobre estaciones para tener un PACUM total
        out[m] = rainfall.sum(axis=1).to_numpy()[:n_days]
    return out


# ===========================================================================
# Aplicación a la cuenca REAL (Ebro): perturbación del registro histórico
# ===========================================================================


def apply_scenario_to_historical(
    historical: np.ndarray,
    scenario: RainfallScenario,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Aplica las transformaciones del escenario a una serie histórica de
    pluviosidad (agregada sobre la cuenca, shape (T,)).

    Procedimiento:
      1. `no_rain`: devuelve ceros.
      2. Escala la cantidad por evento (`mu_p_scale`).
      3. Diezma eventos: cada día lluvioso se conserva con probabilidad
         `p_wet_scale` (≤1) o se duplica con prob `p_wet_scale - 1` (>1).
      4. Redistribución por "flashiness" (`shape_scale < 1`): concentra la
         cantidad agregada en los días con mayor intensidad original.

    El resultado es **una** trayectoria. Para obtener M trayectorias se
    repite con distintas seeds del rng.
    """
    rng = rng or np.random.default_rng()
    T = len(historical)

    if scenario.p_wet_scale <= 0:
        return np.zeros(T, dtype=np.float32)

    perturbed = historical.copy().astype(np.float32) * scenario.mu_p_scale

    # (3) diezmar/clonar eventos
    if scenario.p_wet_scale != 1.0:
        wet_mask = perturbed > 0
        if scenario.p_wet_scale < 1.0:
            keep = rng.random(T) < scenario.p_wet_scale
            perturbed = np.where(wet_mask & ~keep, 0.0, perturbed)
        else:
            extra_factor = scenario.p_wet_scale - 1.0
            extra = rng.random(T) < extra_factor
            # En días secos, duplicamos un evento aleatorio de la serie.
            mean_event = perturbed[wet_mask].mean() if wet_mask.any() else 0.0
            perturbed = np.where(~wet_mask & extra, mean_event, perturbed)

    # (4) concentrar en eventos más grandes si la cola es pesada
    if scenario.shape_scale < 1.0 and perturbed.sum() > 0:
        # Reparte la masa total dando más peso a los días ya intensos
        intensity_order = np.argsort(perturbed)[::-1]
        n_keep = max(1, int(T * scenario.shape_scale))
        concentrated = np.zeros(T, dtype=np.float32)
        total = perturbed.sum()
        # Asigna toda la masa a los n_keep días más intensos, proporcional
        # a su intensidad original (peso suave).
        weights = perturbed[intensity_order[:n_keep]]
        weights = weights / weights.sum() if weights.sum() > 0 else None
        if weights is not None:
            concentrated[intensity_order[:n_keep]] = total * weights
        perturbed = concentrated

    return perturbed


def sample_historical_trajectories(
    df: pd.DataFrame,
    basin: BasinSpec,
    scenario: RainfallScenario,
    hoy: pd.Timestamp,
    horizonte: int,
    n_samples: int,
    rng_seed: int = 0,
) -> np.ndarray:
    """Construye `n_samples` variantes de pluviosidad futura agregada para el
    Ebro (o cualquier cuenca con datos históricos), aplicando el escenario.

    Estrategia: cada muestra arranca tomando T días consecutivos del registro
    histórico (resampleados con reposición de "ventanas" para introducir
    estocasticidad) y le aplica las transformaciones del escenario.

    Returns
    -------
    np.ndarray shape (n_samples, horizonte) con pluviosidad agregada en mm/día.
    """
    rng = np.random.default_rng(rng_seed)
    rain_col = basin.rain_aggregate_column
    if rain_col not in df.columns:
        raise KeyError(f"Falta la columna agregada '{rain_col}' en el DataFrame.")

    # Ventanas históricas posibles: cualquier inicio donde quepan `horizonte` días.
    posibles = df.index[: -horizonte]
    if len(posibles) == 0:
        raise ValueError(f"El DataFrame es demasiado corto para horizonte={horizonte}.")

    out = np.zeros((n_samples, horizonte), dtype=np.float32)
    for m in range(n_samples):
        i = int(rng.integers(0, len(posibles)))
        start = posibles[i]
        slice_ = df.loc[start:start + pd.Timedelta(days=horizonte - 1), rain_col]
        out[m] = apply_scenario_to_historical(
            slice_.to_numpy(dtype=np.float32), scenario, rng,
        )
    return out
