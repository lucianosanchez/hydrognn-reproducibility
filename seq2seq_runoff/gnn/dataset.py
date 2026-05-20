"""Conversión del DataFrame al formato tensorial que espera HydroGNNCore.

Cada ventana produce una tupla:
    rain   : (L, N1)        — pluviosidad por nodo Tipo-1.
    mask   : (L, N1)        — 1 donde hay observación, 0 si no.
    ctx    : (L, ctx_dim)   — sin/cos del día del año.
    Q_obs  : (L,)           — caudal observado en m³/s normalizado.
    S_obs  : (L, M_obs)     — almacenamiento observado normalizado (Fase 1).

La máscara representa **una configuración de despliegue fija**, no una
augmentación: si una estación no existe en el escenario, su mask permanece
en 0 durante todo el entrenamiento y la inferencia. El `Phase2_*` se
configura pasando `observed_stations` en `GNNConfig`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
import torch

from .graph import BasinGraph


@dataclass
class GNNWindow:
    rain: torch.Tensor
    mask: torch.Tensor
    ctx: torch.Tensor
    Q_obs: torch.Tensor
    S_obs: Optional[torch.Tensor]
    fecha_fin_burn_in: pd.Timestamp


def _ctx_features(fechas: pd.DatetimeIndex) -> np.ndarray:
    """sin/cos del día del año, vector ctx_dim=2."""
    doy = fechas.dayofyear.to_numpy(dtype=np.float32)
    ang = 2 * np.pi * doy / 365.25
    return np.stack([np.sin(ang), np.cos(ang)], axis=-1)


def _rain_per_node(df: pd.DataFrame, graph: BasinGraph) -> np.ndarray:
    """(T, N1) con la pluviosidad de cada nodo Tipo-1; 0 si no tiene estación."""
    out = np.zeros((len(df), graph.N1), dtype=np.float32)
    for col, nodo in graph.rain_to_type1.items():
        if col in df.columns:
            out[:, nodo] = df[col].to_numpy(dtype=np.float32)
    return out


def _S_observed(df: pd.DataFrame, graph: BasinGraph) -> tuple[np.ndarray, np.ndarray]:
    """(T, M_obs) con almacenamiento observado y los índices en V_2."""
    obs_indices: List[int] = []
    cols: List[str] = []
    for k, name in enumerate(graph.res_names):
        col = graph.res_to_observed.get(name)
        if col and col in df.columns:
            obs_indices.append(k)
            cols.append(col)
    if not obs_indices:
        return np.empty((len(df), 0), dtype=np.float32), np.empty((0,), dtype=np.int64)
    arr = df[cols].to_numpy(dtype=np.float32)
    return arr, np.asarray(obs_indices, dtype=np.int64)


def _resolver_mascara_estaciones(
    graph: BasinGraph,
    observed_stations: Optional[List[str]],
) -> Optional[np.ndarray]:
    """Devuelve un vector (N1,) de 0/1 con las estaciones observables, o None
    para "todas observables".

    `observed_stations` se interpreta como nombres de columna del DataFrame
    (e.g. "EM01-PACUM"). Estaciones no listadas quedan en 0 durante toda la
    ventana — modela la cobertura sensorial real, no una augmentación.
    """
    if observed_stations is None:
        return None
    activos = np.zeros(graph.N1, dtype=np.float32)
    for col in observed_stations:
        if col not in graph.rain_to_type1:
            raise KeyError(
                f"Estación {col!r} no aparece en el grafo. "
                f"Disponibles: {sorted(graph.rain_to_type1)}"
            )
        activos[graph.rain_to_type1[col]] = 1.0
    return activos


def build_window(
    df: pd.DataFrame,
    graph: BasinGraph,
    fin_burn_in: pd.Timestamp,
    H: int,
    T: int,
    *,
    flow_column: str,
    observed_stations: Optional[List[str]] = None,
) -> GNNWindow:
    """Construye una ventana centrada en `fin_burn_in`.

    El paso 0 corresponde a `fin_burn_in - (H-1)`. Los primeros H pasos son
    burn-in (no se puntúan); los T siguientes son los que el modelo debe
    pronosticar.

    Si `observed_stations` se pasa, sólo esas estaciones aparecen como
    observadas (mask = 1); el resto va a 0. Esta máscara es **fija**: la
    misma en train y en inferencia, porque modela una cobertura real.
    """
    inicio = fin_burn_in - pd.Timedelta(days=H - 1)
    fin = fin_burn_in + pd.Timedelta(days=T)
    if inicio not in df.index or fin not in df.index:
        raise KeyError(f"La ventana {inicio} – {fin} se sale del DataFrame.")
    sub = df.loc[inicio:fin]
    rain = _rain_per_node(sub, graph)
    mask = np.ones_like(rain, dtype=np.float32)

    activos = _resolver_mascara_estaciones(graph, observed_stations)
    if activos is not None:
        mask[:, :] = activos[None, :]
        # Las estaciones no observadas no deben filtrar pluviosidad.
        rain = rain * activos[None, :]

    ctx = _ctx_features(sub.index)
    Q = sub[flow_column].to_numpy(dtype=np.float32)
    S, _ = _S_observed(sub, graph)

    return GNNWindow(
        rain=torch.from_numpy(rain),
        mask=torch.from_numpy(mask),
        ctx=torch.from_numpy(ctx),
        Q_obs=torch.from_numpy(Q),
        S_obs=torch.from_numpy(S) if S.size > 0 else None,
        fecha_fin_burn_in=fin_burn_in,
    )


def build_training_dataset(
    df: pd.DataFrame,
    graph: BasinGraph,
    H: int,
    T: int,
    *,
    flow_column: str,
    observed_stations: Optional[List[str]] = None,
):
    """Generador de todas las ventanas posibles (sin barajar)."""
    fechas = df.index
    if len(fechas) < H + T + 1:
        return
    primer = fechas[H - 1]
    ultimo = fechas[-T - 1]
    for fin_bi in pd.date_range(primer, ultimo):
        try:
            yield build_window(
                df, graph, fin_bi, H, T,
                flow_column=flow_column,
                observed_stations=observed_stations,
            )
        except KeyError:
            continue
