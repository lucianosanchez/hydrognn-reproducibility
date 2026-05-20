"""Criterios de decisión bajo incertidumbre escenario (sec. 3.5).

La salida del modelo variacional es una distribución predictiva de caudal
para cada combinación (escenario s, muestra latente z). Combinada con un
caudal observado de referencia y una acción `δ` (offset del umbral),
produce un coste L(δ, s) por escenario.

Esta módulo expone:
    - `cost_grid_per_scenario(...)` → matriz L[δ, s] dado:
            * distribución predictiva por escenario (M, K, T) en m³/s,
            * caudal observado de referencia (T,) o (M, T) en m³/s,
            * grid de δ a evaluar.
    - `maximin_delta`, `maximax_delta`, `savage_delta`, `naive_delta`
      → escogen el δ óptimo bajo cada criterio.
    - `evaluate_all_criteria` → devuelve `CriterionResult` para los
      cuatro criterios.

Toda la lógica es pura NumPy/Pandas; el modelo entrenado se llama desde
`scripts/run_vae_experiment.py` y se le pide `predict_distribution(...)`
para construir el tensor predictivo.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ===========================================================================
# Resultado de un criterio
# ===========================================================================


@dataclass
class CriterionResult:
    """Resultado de aplicar un criterio sobre la matriz L[δ, s]."""

    name: str                         # "maximin", "maximax", "savage", "naive"
    delta_star: float                 # δ óptimo elegido
    cost_per_scenario: Dict[str, float]   # L(δ*, s) para cada escenario
    regret_per_scenario: Dict[str, float] # R(δ*, s) = L(δ*, s) − min_δ L(δ, s)
    fn_per_scenario: Dict[str, float]     # FN(δ*, s) — días de parada
    fp_per_scenario: Dict[str, float]     # FP(δ*, s)

    @property
    def worst_case_cost(self) -> float:
        return max(self.cost_per_scenario.values()) if self.cost_per_scenario else float("nan")

    @property
    def expected_cost(self) -> float:
        v = list(self.cost_per_scenario.values())
        return float(sum(v) / len(v)) if v else float("nan")

    @property
    def max_regret(self) -> float:
        return max(self.regret_per_scenario.values()) if self.regret_per_scenario else float("nan")


# ===========================================================================
# Construcción del grid de costes
# ===========================================================================


def cost_grid_per_scenario(
    predicted_distribution: np.ndarray,   # (M, K, T) en m³/s
    observed: np.ndarray,                  # (T,) o (M, T) en m³/s
    deltas: np.ndarray,                    # (D,) en m³/s — grid de δ
    q_min: float,
    coste_falsa_alarma: float,
    coste_omision: float,
    scenario_names: List[str],
    quantile_alpha: float = 0.5,
    quantile_mode: str = "predictor",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute L[D, M], FN[D, M], FP[D, M] sobre el grid de δ.

    Hay dos formas de "tomar α-cuantil" en la decisión cost-aware, según
    el plano donde se aplica el cuantil. Ambas coinciden cuando $\\alpha=0.5$
    y la pérdida es lineal en la predicción, pero divergen en los extremos
    (cf. \\Cref{eq:uagnn_Lalpha} del paper).

    Parameters
    ----------
    quantile_alpha : float in (0, 1)
        α-cuantil de las K muestras del posterior.
        - 0.5 → mediana (baseline).
        - >0.5 → operador conservador respecto a FN.
        - <0.5 → operador optimista.
    quantile_mode : {"predictor", "cost"}
        - "predictor" (default): toma el α-cuantil de $\\mu_Q^{(k)}$ a
          nivel de (m, t), y decide alarma comparándolo con $q_{\\min}+\\delta$.
          Una sola decisión por paso. Computacionalmente $O(K + D)$.
        - "cost": para cada k computa $\\mathrm{FP}^{(k)}$ y
          $\\mathrm{FN}^{(k)}$ sobre todo el horizonte, agrega coste
          $c_{\\mathrm{FP}}\\mathrm{FP}^{(k)} + c_{\\mathrm{FN}}\\mathrm{FN}^{(k)}$
          y toma el α-cuantil sobre los K costes. Esta es la versión
          estricta de \\Cref{eq:uagnn_Lalpha} del paper. Computacionalmente
          $O(K \\cdot D)$.

    Returns
    -------
    L  : (D, M) coste total por (δ, escenario)
    FN : (D, M)
    FP : (D, M)

    Notas
    -----
    En "cost" mode, `FN[d_idx, m]` y `FP[d_idx, m]` retornados son los
    α-cuantiles sobre k, no la media — de modo que `c_FP*FP + c_FN*FN`
    NO es exactamente igual al α-cuantil del coste. Para usos
    visuales/diagnósticos `FN`/`FP` son representativos; para reproducir
    exactamente $L_\\alpha$ debe usarse el `L` retornado.
    """
    M, K, T = predicted_distribution.shape
    if observed.ndim == 1:
        observed = np.broadcast_to(observed, (M, T))
    assert observed.shape == (M, T), \
        f"observed debe tener shape (T,) o (M, T); recibido {observed.shape}"
    assert len(scenario_names) == M
    if not (0.0 < quantile_alpha < 1.0):
        raise ValueError(f"quantile_alpha debe estar en (0,1); recibido {quantile_alpha}")
    if quantile_mode not in ("predictor", "cost"):
        raise ValueError(f"quantile_mode debe ser 'predictor' o 'cost'; recibido {quantile_mode!r}")

    D = len(deltas)
    L = np.zeros((D, M), dtype=np.float32)
    FN = np.zeros((D, M), dtype=np.float32)
    FP = np.zeros((D, M), dtype=np.float32)
    obs_alarma = observed <= q_min  # (M, T)

    if quantile_mode == "predictor":
        # Cuantil α sobre las K muestras del CAUDAL, luego una decisión por paso.
        pred_stat = np.quantile(predicted_distribution, q=quantile_alpha, axis=1)  # (M, T)
        for d_idx, delta in enumerate(deltas):
            pred_alarma = pred_stat <= (q_min + delta)
            fp = np.sum(pred_alarma & ~obs_alarma, axis=1).astype(np.float32)
            fn = np.sum(~pred_alarma & obs_alarma, axis=1).astype(np.float32)
            FN[d_idx] = fn
            FP[d_idx] = fp
            L[d_idx] = coste_falsa_alarma * fp + coste_omision * fn

    else:  # quantile_mode == "cost" — versión estricta de eq:uagnn_Lalpha
        # Para cada (delta, k) computamos FP^(k)(δ, s), FN^(k)(δ, s) por
        # acumulación sobre el horizonte; el coste resultante L^(k)(δ, s)
        # se reduce sobre k mediante el α-cuantil.
        # Vectorización: cada delta da pred_alarma de shape (M, K, T).
        # Para mantenerlo eficiente, iteramos sobre delta.
        for d_idx, delta in enumerate(deltas):
            pred_alarma_k = predicted_distribution <= (q_min + delta)         # (M, K, T)
            obs_alarma_b = np.broadcast_to(obs_alarma[:, None, :], pred_alarma_k.shape)
            fp_k = np.sum(pred_alarma_k & ~obs_alarma_b, axis=2)              # (M, K)
            fn_k = np.sum(~pred_alarma_k & obs_alarma_b, axis=2)              # (M, K)
            cost_k = coste_falsa_alarma * fp_k + coste_omision * fn_k         # (M, K)
            L[d_idx] = np.quantile(cost_k, q=quantile_alpha, axis=1)
            FN[d_idx] = np.quantile(fn_k.astype(np.float32),
                                     q=quantile_alpha, axis=1)
            FP[d_idx] = np.quantile(fp_k.astype(np.float32),
                                     q=quantile_alpha, axis=1)

    return L, FN, FP


# ===========================================================================
# Criterios
# ===========================================================================


def _result_from_idx(
    name: str, idx: int, deltas: np.ndarray,
    L: np.ndarray, FN: np.ndarray, FP: np.ndarray,
    scenario_names: List[str],
) -> CriterionResult:
    """Empaqueta `CriterionResult` para un índice de δ ya elegido."""
    delta_star = float(deltas[idx])
    min_per_scenario = L.min(axis=0)              # (M,)
    cost_at = L[idx]                              # (M,)
    regret = cost_at - min_per_scenario
    return CriterionResult(
        name=name, delta_star=delta_star,
        cost_per_scenario={s: float(cost_at[m]) for m, s in enumerate(scenario_names)},
        regret_per_scenario={s: float(regret[m]) for m, s in enumerate(scenario_names)},
        fn_per_scenario={s: float(FN[idx, m]) for m, s in enumerate(scenario_names)},
        fp_per_scenario={s: float(FP[idx, m]) for m, s in enumerate(scenario_names)},
    )


def maximin_delta(
    L: np.ndarray, FN: np.ndarray, FP: np.ndarray,
    deltas: np.ndarray, scenario_names: List[str],
) -> CriterionResult:
    """Wald, pesimista: minimiza max_s L(δ, s)."""
    max_per_delta = L.max(axis=1)                  # (D,)
    idx = int(np.argmin(max_per_delta))
    return _result_from_idx("maximin", idx, deltas, L, FN, FP, scenario_names)


def maximax_delta(
    L: np.ndarray, FN: np.ndarray, FP: np.ndarray,
    deltas: np.ndarray, scenario_names: List[str],
) -> CriterionResult:
    """Hurwicz α=1, optimista: minimiza min_s L(δ, s)."""
    min_per_delta = L.min(axis=1)                  # (D,)
    idx = int(np.argmin(min_per_delta))
    return _result_from_idx("maximax", idx, deltas, L, FN, FP, scenario_names)


def savage_delta(
    L: np.ndarray, FN: np.ndarray, FP: np.ndarray,
    deltas: np.ndarray, scenario_names: List[str],
) -> CriterionResult:
    """Savage min-max regret: minimiza max_s [L(δ, s) − min_δ' L(δ', s)]."""
    min_per_scenario = L.min(axis=0, keepdims=True)     # (1, M)
    regret = L - min_per_scenario                        # (D, M)
    max_regret_per_delta = regret.max(axis=1)            # (D,)
    idx = int(np.argmin(max_regret_per_delta))
    return _result_from_idx("savage", idx, deltas, L, FN, FP, scenario_names)


def naive_delta(
    L: np.ndarray, FN: np.ndarray, FP: np.ndarray,
    deltas: np.ndarray, scenario_names: List[str],
    baseline_scenario: str = "baseline",
) -> CriterionResult:
    """Criterio naive: minimiza L(δ, s_baseline) ignorando los demás escenarios.

    Es el comportamiento por defecto del baseline determinista de sec. 1.5
    cuando se trabaja con un único escenario futuro.
    """
    if baseline_scenario not in scenario_names:
        raise ValueError(f"baseline_scenario={baseline_scenario!r} no está "
                         f"en {scenario_names}.")
    m_baseline = scenario_names.index(baseline_scenario)
    idx = int(np.argmin(L[:, m_baseline]))
    return _result_from_idx("naive", idx, deltas, L, FN, FP, scenario_names)


# ===========================================================================
# Evaluación conjunta
# ===========================================================================


def evaluate_all_criteria(
    predicted_distribution: np.ndarray,
    observed: np.ndarray,
    deltas: np.ndarray,
    q_min: float,
    coste_falsa_alarma: float,
    coste_omision: float,
    scenario_names: List[str],
    baseline_scenario: str = "baseline",
    quantile_alpha: float = 0.5,
    quantile_mode: str = "predictor",
) -> Dict[str, CriterionResult]:
    """Aplica los 4 criterios sobre el mismo grid y devuelve resultados.

    Parameters
    ----------
    quantile_alpha : float, default 0.5
        Cuantil α de las K muestras usado como estadístico de decisión.
    quantile_mode : {"predictor", "cost"}, default "predictor"
        Plano donde se toma el cuantil. Cf. `cost_grid_per_scenario`.
    """
    L, FN, FP = cost_grid_per_scenario(
        predicted_distribution, observed, deltas, q_min,
        coste_falsa_alarma, coste_omision, scenario_names,
        quantile_alpha=quantile_alpha,
        quantile_mode=quantile_mode,
    )
    return {
        "naive":    naive_delta(L, FN, FP, deltas, scenario_names, baseline_scenario),
        "maximin":  maximin_delta(L, FN, FP, deltas, scenario_names),
        "maximax":  maximax_delta(L, FN, FP, deltas, scenario_names),
        "savage":   savage_delta(L, FN, FP, deltas, scenario_names),
    }


def format_criterion_report(results: Dict[str, CriterionResult]) -> str:
    """Tabla legible para consola: una fila por criterio, columna por
    escenario; al final, las métricas headline (worst-case, expected, max regret)."""
    if not results:
        return "(sin resultados)"
    scenarios = list(next(iter(results.values())).cost_per_scenario.keys())
    header = f"{'criterio':10s}  {'δ*':>8s}  " + "  ".join(f"{s:>14s}" for s in scenarios) \
             + "    |  " + "  ".join(f"{m:>10s}" for m in
                                      ("worst", "expected", "maxRegret"))
    rows = [header, "-" * len(header)]
    for name, r in results.items():
        scen_costs = "  ".join(f"{r.cost_per_scenario[s]:>14.0f}" for s in scenarios)
        head_metrics = "  ".join(f"{v:>10.0f}" for v in
                                  (r.worst_case_cost, r.expected_cost, r.max_regret))
        rows.append(f"{name:10s}  {r.delta_star:>+8.2f}  {scen_costs}    |  {head_metrics}")
    return "\n".join(rows)
