"""Modelo Tipo-1 + Tipo-2 GNN — sec. 5/11 de report2.tex.

Tres fases:

    HydroGNNPhase1     grafo exacto + estaciones completas + embalses observados.
    HydroGNNPhase2_1   grafo exacto + estaciones parciales + embalses latentes.
    HydroGNNPhase2_2   grafo de candidatos + estaciones parciales + estructura aprendida.

Las tres implementan `RunoffModel` y comparten `HydroGNNCore`.
"""

from .core import HydroGNNCore, HydroGNNOutput
from .gates import HardConcreteGate
from .graph import BasinGraph, dense_candidate_graph
from .losses import gaussian_nll, lowflow_weight, total_loss
from .model import (
    GNNConfig,
    HydroGNNPhase1,
    HydroGNNPhase2_1,
    HydroGNNPhase2_2,
)

# `canonical_ebro_graph` y otros constructores de cuencas concretas viven
# en `seq2seq_runoff.basins.<cuenca>` (e.g. `from seq2seq_runoff.basins import ebro_graph`).
# El subpaquete gnn se mantiene agnóstico a la cuenca.

__all__ = [
    "HydroGNNCore",
    "HydroGNNOutput",
    "HardConcreteGate",
    "BasinGraph",
    "dense_candidate_graph",
    "gaussian_nll",
    "lowflow_weight",
    "total_loss",
    "GNNConfig",
    "HydroGNNPhase1",
    "HydroGNNPhase2_1",
    "HydroGNNPhase2_2",
]
