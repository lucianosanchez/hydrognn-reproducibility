"""Catálogo de cuencas pre-configuradas.

Para añadir una cuenca nueva (Duero, Tajo, Mississippi, …) crea un fichero
análogo a `ebro.py` con dos funciones públicas:

    def <cuenca>_basin() -> BasinSpec: ...
    def <cuenca>_graph() -> BasinGraph: ...

Y vuelve a exportarlas aquí. El resto del paquete es genérico.
"""

from .ebro import ebro_basin, ebro_graph
from .synth import synth_basin, synth_graph_full, synth_graph_simplified

__all__ = [
    "ebro_basin",
    "ebro_graph",
    "synth_basin",
    "synth_graph_full",
    "synth_graph_simplified",
]
