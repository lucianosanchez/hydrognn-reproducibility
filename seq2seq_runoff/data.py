"""Carga genérica del dataframe de una cuenca.

El módulo lee un conjunto de CSVs descritos en un `BasinSpec` y los unifica
en un único DataFrame con índice diario y columnas:

    <rain_columns ...>, <rain_aggregate_column>,
    <reservoir_columns ...>, <reservoir_aggregate_column>,
    <flow_column>

donde los nombres concretos los aporta el `BasinSpec`. Para una cuenca
nueva, basta con escribir su factoría en `seq2seq_runoff.basins`.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

import pandas as pd

from .basin import BasinSpec, StationSpec


def _read_station_csv(path: Path, basin: BasinSpec, station: StationSpec) -> pd.DataFrame:
    """Lee un CSV individual y devuelve un DataFrame con una sola columna
    cuyo nombre es `station.name` y cuyo índice es la fecha."""
    df = pd.read_csv(
        path,
        sep=basin.csv_sep,
        decimal=basin.csv_decimal,
        encoding=basin.csv_encoding,
        skipinitialspace=True,
    )
    df[station.csv_date_column] = pd.to_datetime(
        df[station.csv_date_column], format="%Y-%m-%d %H:%M:%S"
    )
    df = df.set_index(station.csv_date_column)
    df = df.rename(columns={station.csv_value_column: station.name})
    return df[[station.name]]


def _ruta(basin: BasinSpec, directorio: Path, firma: str, station: StationSpec) -> Path:
    return directorio / basin.file_pattern.format(firma=firma, code=station.file_code)


def load_basin_dataframe(
    basin: BasinSpec,
    directorio: str | os.PathLike,
    firma: str,
    fecha_corte_inicial: str | None = None,
) -> pd.DataFrame:
    """Lee todos los CSVs de la cuenca y los combina en un único DataFrame.

    Parameters
    ----------
    basin
        Configuración de la cuenca (qué series, cómo se llaman, formato CSV).
    directorio
        Carpeta que contiene los CSVs.
    firma
        Identificador numérico que aparece en cada nombre de fichero.
    fecha_corte_inicial
        Filas anteriores se descartan. Si es `None` se toma de
        `basin.fecha_corte_inicial`.
    """
    d = Path(directorio)
    series = []

    for s in basin.rainfall_stations:
        series.append(_read_station_csv(_ruta(basin, d, firma, s), basin, s))
    for s in basin.reservoirs:
        series.append(_read_station_csv(_ruta(basin, d, firma, s), basin, s))
    series.append(_read_station_csv(_ruta(basin, d, firma, basin.flow_station), basin, basin.flow_station))

    df = pd.concat(series, axis=1).sort_index()

    # Imputación: huecos de pluviosidad → 0 (no llovió);
    # huecos de embalse y caudal → interpolación lineal.
    df[basin.rain_columns] = df[basin.rain_columns].fillna(0.0)
    df = df.interpolate(method="linear")

    df[basin.rain_aggregate_column] = df[basin.rain_columns].sum(axis=1)
    df[basin.reservoir_aggregate_column] = df[basin.reservoir_columns].sum(axis=1)

    df = df.bfill().fillna(df.mean(numeric_only=True))

    corte = fecha_corte_inicial or basin.fecha_corte_inicial
    df = df.loc[corte:]
    return df


def split_train_test(df: pd.DataFrame, fraccion_test: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Particiona la serie en train/test conservando el orden temporal."""
    n = len(df)
    n_train = int(n * (1 - fraccion_test))
    return df.iloc[:n_train].copy(), df.iloc[n_train:].copy()


def scale_to_unit(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Escala las columnas dividiendo por su máximo (todas son positivas).

    Devuelve el DataFrame escalado y los máximos para poder revertir.
    """
    maximos = df.max(axis=0)
    return df.divide(maximos, axis=1), maximos
