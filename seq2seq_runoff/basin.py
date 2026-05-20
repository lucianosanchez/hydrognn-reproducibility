"""Especificación genérica de una cuenca hidrográfica.

`BasinSpec` agrupa todo lo que el código necesita saber sobre una cuenca:
qué series temporales existen (estaciones, embalses, aforo), cómo se llaman
las columnas en los CSV originales y cuál es la convención de nombres de
ficheros. Para añadir una nueva cuenca, basta con escribir una función
factoría análoga a `seq2seq_runoff.basins.ebro.ebro_basin()`.

`StationSpec` describe una serie temporal individual.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal


StationKind = Literal["rain", "reservoir", "flow"]


@dataclass(frozen=True)
class StationSpec:
    """Una serie temporal de la cuenca (lluvia, embalse o aforo).

    name
        Identificador canónico que se usará como columna en el DataFrame y
        como clave en `BasinGraph.rain_to_type1`. Conviene que sea estable
        (e.g. "EM01-PACUM" o "ALAGON-CAUDAL"), no el nombre del operador.
    file_code
        Código que aparece en el nombre del fichero CSV
        (e.g. "EM01L84PACUM"). Se inserta en `BasinSpec.file_pattern`.
    csv_value_column
        Nombre exacto de la columna del valor en el CSV original (los
        ficheros reales suelen traer espacios al final como "Acumulado").
    csv_date_column
        Nombre exacto de la columna de fecha en el CSV original.
    kind
        Tipo de serie. Determina qué papel juega en la tubería.
    """

    name: str
    file_code: str
    csv_value_column: str
    csv_date_column: str
    kind: StationKind


@dataclass
class BasinSpec:
    """Configuración completa de una cuenca hidrográfica.

    Atributos clave
    ---------------
    rainfall_stations / reservoirs / flow_station
        Las series que componen la cuenca.
    file_pattern
        Plantilla del nombre de fichero CSV. Las llaves disponibles son
        `{firma}` y `{code}` (file_code de la `StationSpec`).
    rain_aggregate_column / reservoir_aggregate_column
        Nombre de las columnas derivadas que suman pluviosidad / volumen
        embalsado sobre toda la cuenca. Se construyen automáticamente en
        `data.load_basin_dataframe`.
    caudal_minimo_m3s
        Umbral operacional por defecto. La planta concreta puede
        sobreescribirlo desde `Config`, pero ponerlo aquí permite a una
        cuenca traer su propia política regulatoria.
    """

    name: str
    rainfall_stations: List[StationSpec]
    reservoirs: List[StationSpec]
    flow_station: StationSpec

    file_pattern: str = "DatosHistoricos_{firma}_{code}.csv"
    csv_sep: str = ";"
    csv_decimal: str = ","
    csv_encoding: str = "ISO-8859-1"

    rain_aggregate_column: str = "PACUM"
    reservoir_aggregate_column: str = "EACUM"
    caudal_minimo_m3s: float = 30.0

    # Fecha mínima a partir de la cual el dataset es de calidad suficiente.
    # Por defecto la dejamos abierta; cada cuenca puede pinar la suya.
    fecha_corte_inicial: str = "1900-01-01"

    # Validación elemental
    def __post_init__(self):
        if self.flow_station.kind != "flow":
            raise ValueError("flow_station.kind debe ser 'flow'.")
        if any(s.kind != "rain" for s in self.rainfall_stations):
            raise ValueError("Todas las rainfall_stations deben tener kind='rain'.")
        if any(s.kind != "reservoir" for s in self.reservoirs):
            raise ValueError("Todos los reservoirs deben tener kind='reservoir'.")

    # --- accesos rápidos por columna ---------------------------------------

    @property
    def rain_columns(self) -> List[str]:
        return [s.name for s in self.rainfall_stations]

    @property
    def reservoir_columns(self) -> List[str]:
        return [s.name for s in self.reservoirs]

    @property
    def flow_column(self) -> str:
        return self.flow_station.name

    @property
    def encoder_columns(self) -> List[str]:
        """Variables de entrada al codificador del Seq2Seq (en orden)."""
        return [self.rain_aggregate_column, self.reservoir_aggregate_column, self.flow_column]

    @property
    def decoder_embalse_columns(self) -> List[str]:
        return [self.rain_aggregate_column]

    @property
    def decoder_caudal_columns(self) -> List[str]:
        return [self.rain_aggregate_column, self.reservoir_aggregate_column]
