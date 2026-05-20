"""Lectura del YAML de configuración a dataclasses tipadas.

El formato del YAML está pensado para escribirse a mano. Mira
`example_basin.yaml` para una plantilla comentada.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

import yaml


@dataclass
class NodeConfig:
    id: str
    rain_station: Optional[str] = None     # None → este nodo no tiene pluviómetro
    catchment_km2: float = 0.0             # área de drenaje (modula la respuesta a la lluvia)
    runoff_coef: float = 0.3               # fracción de la lluvia que se convierte en escorrentía
    flow_station: Optional[str] = None     # nombre de la columna de caudal (sólo el aforo)
    is_major_river: bool = False           # ¿forma parte del cauce principal? (para grafo simplificado)
    position: Optional[List[float]] = None  # [x, y] para los plots de topología (opcional)


@dataclass
class EdgeConfig:
    src: str
    dst: str
    length_km: float


@dataclass
class ReservoirConfig:
    id: str
    name: str                              # nombre que aparecerá en los CSV
    capacity_hm3: float
    inflow_from: str                       # nodo Tipo-1 que aporta agua
    release_to: str                        # nodo Tipo-1 al que se suelta
    release_fraction_per_day: float = 0.04
    initial_storage_hm3: float = 0.0
    is_biggest: bool = False               # para grafo simplificado (modelos 3 y 4)
    position: Optional[List[float]] = None  # [x, y] para los plots de topología (opcional)


@dataclass
class ClimateConfig:
    start_date: str
    end_date: str
    seed: int = 42
    # Probabilidad de día lluvioso por estación (sinusoide día del año).
    p_wet_winter: float = 0.45
    p_wet_summer: float = 0.10
    # Persistencia (Markov chain) sobre el estado regional húmedo/seco.
    p_wet_given_wet: float = 0.6
    p_wet_given_dry: float = 0.2
    # Distribución de la lluvia en mm para días lluviosos (Gamma).
    rainfall_mm_mean: float = 8.0
    rainfall_mm_shape: float = 1.5
    # Correlación espacial: probabilidad de que la estación siga el evento regional.
    spatial_corr: float = 0.7


@dataclass
class VisibilityConfig:
    """Define una vista del dataset: qué se ve y qué se oculta.

    `ALL` (string) ⇒ todo visible.
    Lista de ids ⇒ sólo esos elementos están visibles.
    """
    name: str
    visible_rain_stations: Union[str, List[str]] = "ALL"
    visible_reservoirs: Union[str, List[str]] = "ALL"


@dataclass
class BasinSimConfig:
    name: str
    river_velocity_km_per_day: float
    nodes: List[NodeConfig]
    edges_11: List[EdgeConfig]
    reservoirs: List[ReservoirConfig]
    climate: ClimateConfig
    output_configurations: List[VisibilityConfig]
    output_directory: str
    firma: str
    file_pattern: str = "DatosHistoricos_{firma}_{code}.csv"
    caudal_minimo_m3s: float = 30.0
    # Amplitud de la deriva temporal en la fracción de soltada de los embalses.
    # 0 = manejo perfectamente estacionario; valores típicos 0.05–0.3.
    # Usado por hydro.simulate_hydrology para inducir no-estacionariedad operativa.
    nonstationarity_amplitude: float = 0.0

    # --- helpers ----------------------------------------------------------

    def find_node(self, node_id: str) -> NodeConfig:
        for n in self.nodes:
            if n.id == node_id:
                return n
        raise KeyError(node_id)

    def outlet(self) -> NodeConfig:
        for n in self.nodes:
            if n.flow_station is not None:
                return n
        raise ValueError("Ningún nodo tiene flow_station; falta declarar el aforo.")

    def major_river_node_ids(self) -> List[str]:
        return [n.id for n in self.nodes if n.is_major_river]

    def biggest_reservoir_ids(self) -> List[str]:
        return [r.id for r in self.reservoirs if r.is_biggest]


# --------------------------- loader ----------------------------------------


def load_basin_config(yaml_path: Union[str, Path]) -> BasinSimConfig:
    """Lee y valida el YAML de configuración."""
    raw = yaml.safe_load(Path(yaml_path).read_text())

    nodes = [NodeConfig(**n) for n in raw["nodes"]]
    edges = [EdgeConfig(**e) for e in raw["edges_11"]]
    reservoirs = [ReservoirConfig(**r) for r in raw["reservoirs"]]
    climate = ClimateConfig(**raw["climate"])

    # output_configurations se acepta tanto como dict {nombre: {...}} como lista de dicts con name.
    raw_oc = raw["output_configurations"]
    if isinstance(raw_oc, dict):
        configs = [VisibilityConfig(name=k, **v) for k, v in raw_oc.items()]
    else:
        configs = [VisibilityConfig(**v) for v in raw_oc]

    return BasinSimConfig(
        name=raw["basin"]["name"],
        river_velocity_km_per_day=float(raw["basin"]["river_velocity_km_per_day"]),
        nodes=nodes,
        edges_11=edges,
        reservoirs=reservoirs,
        climate=climate,
        output_configurations=configs,
        output_directory=raw["output"]["directory"],
        firma=raw["output"]["firma"],
        file_pattern=raw["output"].get("file_pattern", "DatosHistoricos_{firma}_{code}.csv"),
        caudal_minimo_m3s=float(raw["basin"].get("caudal_minimo_m3s", 30.0)),
    )
