"""Simulador hidrográfico sintético.

Genera CSVs en el mismo formato que consume `seq2seq_runoff` a partir de un
YAML de configuración con (a) topología, (b) clima y (c) reglas de
visibilidad (qué ocultar a la salida). Es deliberadamente externo al modelo
GNN: aquí se simula la "verdad terreno" y la sensibilización; allí se
modela.

Uso típico:

    python -m synth_simulator path/to/basin.yaml

genera, en `output.directory/<config>/`, los CSVs visibles y un
`manifest.yaml` que `seq2seq_runoff.basins.synth` consume para construir
`BasinSpec` y `BasinGraph`.
"""

from .config import BasinSimConfig, load_basin_config
from .climate import generate_rainfall
from .hydro import simulate_hydrology
from .output import write_visibility_outputs
from .topology_generator import random_basin
from . import viz

__all__ = [
    "BasinSimConfig",
    "load_basin_config",
    "generate_rainfall",
    "simulate_hydrology",
    "write_visibility_outputs",
    "random_basin",
    "viz",
]
