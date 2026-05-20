"""Modelo de escorrentía: baseline Seq2Seq y GNN físico-guiado.

Para usar el paquete con una cuenca distinta del Ebro, escribe una factoría
en `seq2seq_runoff.basins.<nombre>` que devuelva un `BasinSpec` (y, si vas a
usar el GNN, también un `BasinGraph`). El resto de la tubería es genérica.
"""

from .basin import BasinSpec, StationSpec
from .config import Config
from .data import load_basin_dataframe, scale_to_unit, split_train_test
from .windows import build_windows, WindowSet
from .transforms import FlowTransform, IdentityTransform, BoxCoxTransform
from .calibration import MonotonicCalibrator
from .model import RunoffModel, Seq2SeqRunoffModel, Forecast, ForecastScenario

# El VAE depende de TensorFlow/Keras 3. Si esa cadena de dependencias no
# está disponible (e.g. máquinas que solo entrenan UA-HydroGNN, que es
# PyTorch puro), se reemplaza por un placeholder que falla con un mensaje
# claro al instanciarse. Esto evita que un `import seq2seq_runoff` rompa
# scripts que no usan VAE.
try:
    from .vae import VAESeq2SeqRunoffModel
    _VAE_AVAILABLE = True
    _VAE_IMPORT_ERROR = None
except Exception as _e:    # pragma: no cover — depende del entorno
    _VAE_AVAILABLE = False
    _VAE_IMPORT_ERROR = _e
    class VAESeq2SeqRunoffModel:    # type: ignore
        """Placeholder cuando TensorFlow/Keras 3 no está disponible."""
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "VAESeq2SeqRunoffModel requiere TensorFlow + Keras 3, "
                "que no están disponibles en este entorno. Causa raíz: "
                f"{_VAE_IMPORT_ERROR}"
            )

from .ua_gnn import UAHydroGNNModel
from .scenarios import (
    RainfallScenario, SCENARIO_LIBRARY, default_library,
    apply_scenario_to_climate, apply_scenario_to_historical,
)
from .decision import (
    CriterionResult, cost_grid_per_scenario,
    maximin_delta, maximax_delta, savage_delta, naive_delta,
    evaluate_all_criteria, format_criterion_report,
)
from .evaluation import (
    per_lag_mse,
    low_flow_classification,
    nash_sutcliffe,
    kling_gupta,
    summary,
    rolling_evaluation,
)

__all__ = [
    "BasinSpec",
    "StationSpec",
    "Config",
    "load_basin_dataframe",
    "scale_to_unit",
    "split_train_test",
    "build_windows",
    "WindowSet",
    "FlowTransform",
    "IdentityTransform",
    "BoxCoxTransform",
    "MonotonicCalibrator",
    "RunoffModel",
    "Seq2SeqRunoffModel",
    "Forecast",
    "ForecastScenario",
    "per_lag_mse",
    "low_flow_classification",
    "nash_sutcliffe",
    "kling_gupta",
    "summary",
    "rolling_evaluation",
    # VAE extension (sec. 3)
    "VAESeq2SeqRunoffModel",
    # Uncertainty-aware GNN extension (sec. 4)
    "UAHydroGNNModel",
    "RainfallScenario", "SCENARIO_LIBRARY", "default_library",
    "apply_scenario_to_climate", "apply_scenario_to_historical",
    "CriterionResult", "cost_grid_per_scenario",
    "maximin_delta", "maximax_delta", "savage_delta", "naive_delta",
    "evaluate_all_criteria", "format_criterion_report",
]
