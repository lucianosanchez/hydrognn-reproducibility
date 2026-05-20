"""Métricas de evaluación, alineadas con la sec. 7.2 de report2.tex.

Lo importante para la decisión operativa es la clasificación binaria
"caudal por debajo del umbral mínimo" en el horizonte completo. Por eso las
métricas centrales son precision/recall/F1 sobre esa decisión, agregadas por
lag y, opcionalmente, globales. Se incluyen también NSE/KGE como medidas
secundarias de regresión.

`rolling_evaluation` recorre la serie día a día aplicando un `RunoffModel`
arbitrario y acumula los resultados; es la pieza que comparten todos los
modelos del benchmark (Seq2Seq, GNN tipo-1, GNN tipo-1+2, ablaciones).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Dict, List, Optional, TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .basin import BasinSpec
    from .model import RunoffModel  # solo para anotación de tipo


@dataclass
class LowFlowMetrics:
    """Resumen binario para la decisión "el caudal estará bajo Q_min".

    Convención (positivo = "almacenar agua"; negativo = "situación segura"):
        aciertos          (TP) — predijo alarma y la hubo: bien, almacenamos.
        aciertos_normales (TN) — predijo seguro y lo fue: bien, producimos.
        falsas_alarmas    (FP) — predijo alarma sin necesidad. *Coste económico moderado*.
        omisiones         (FN) — predijo seguro pero faltó agua. *Coste catastrófico:
                                 hay que detener la producción de la central.*

    Tasas:
        tasa_falsas_alarmas = FP / (FP + TN)   — fracción de días normales mal clasificados.
        tasa_omisiones      = FN / (FN + TP)   — fracción de alarmas reales no detectadas.
                                                  (es 1 − recall.)
        especificidad       = 1 − tasa_falsas_alarmas.
    """
    precision: float
    recall: float
    f1: float
    aciertos: int
    falsas_alarmas: int
    omisiones: int
    aciertos_normales: int
    tasa_falsas_alarmas: float = 0.0
    tasa_omisiones: float = 0.0
    especificidad: float = 0.0


def per_lag_mse(y_true: np.ndarray, y_pred: np.ndarray) -> List[float]:
    """MSE por horizonte. Espera arrays con forma (N, T) o (N, T, 1)."""
    yt = y_true.squeeze(-1) if y_true.ndim == 3 else y_true
    yp = y_pred.squeeze(-1) if y_pred.ndim == 3 else y_pred
    return [float(np.mean((yt[:, lag] - yp[:, lag]) ** 2)) for lag in range(yt.shape[1])]


def low_flow_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    umbral: float,
) -> LowFlowMetrics:
    """Métricas binarias para la decisión "caudal por debajo del umbral".

    Convención (positivo = caudal por debajo del umbral, situación de alarma):
        aciertos     : pred y obs están por debajo
        falsas_alarmas: pred dice debajo, obs está por encima
        omisiones    : pred dice encima, obs está debajo
    """
    obs_alarma = y_true.reshape(-1) <= umbral
    pred_alarma = y_pred.reshape(-1) <= umbral

    aciertos = int(np.sum(pred_alarma & obs_alarma))
    falsas = int(np.sum(pred_alarma & ~obs_alarma))
    omisiones = int(np.sum(~pred_alarma & obs_alarma))
    aciertos_normales = int(np.sum(~pred_alarma & ~obs_alarma))

    precision = aciertos / max(aciertos + falsas, 1)
    recall = aciertos / max(aciertos + omisiones, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)

    tasa_fa = falsas / max(falsas + aciertos_normales, 1)
    tasa_om = omisiones / max(omisiones + aciertos, 1)
    return LowFlowMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        aciertos=aciertos,
        falsas_alarmas=falsas,
        omisiones=omisiones,
        aciertos_normales=aciertos_normales,
        tasa_falsas_alarmas=tasa_fa,
        tasa_omisiones=tasa_om,
        especificidad=1.0 - tasa_fa,
    )


def nash_sutcliffe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """NSE: 1 = perfecto, 0 = no mejora la media, <0 = peor que la media."""
    yt = y_true.reshape(-1)
    yp = y_pred.reshape(-1)
    num = np.sum((yt - yp) ** 2)
    den = np.sum((yt - np.mean(yt)) ** 2)
    return float(1.0 - num / den) if den > 0 else float("nan")


def kling_gupta(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """KGE: descompuesta en correlación, sesgo y variabilidad."""
    yt = y_true.reshape(-1)
    yp = y_pred.reshape(-1)
    r = float(np.corrcoef(yt, yp)[0, 1])
    alpha = float(np.std(yp) / np.std(yt)) if np.std(yt) > 0 else float("nan")
    beta = float(np.mean(yp) / np.mean(yt)) if np.mean(yt) > 0 else float("nan")
    return float(1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))


def summary(y_true: np.ndarray, y_pred: np.ndarray, umbral: float) -> Dict[str, float]:
    cls = low_flow_classification(y_true, y_pred, umbral)
    return {
        "precision_alarma": cls.precision,
        "recall_alarma": cls.recall,
        "f1_alarma": cls.f1,
        "nse": nash_sutcliffe(y_true, y_pred),
        "kge": kling_gupta(y_true, y_pred),
    }


# ---------------------------------------------------------------------------
# Análisis económico de la decisión (coste asimétrico FP vs FN).
# ---------------------------------------------------------------------------


@dataclass
class CostBreakdown:
    coste_total: float
    coste_por_dia: float
    coste_falsas_alarmas: float
    coste_omisiones: float
    n_dias: int


def expected_cost(
    metrics: LowFlowMetrics,
    coste_falsa_alarma: float,
    coste_omision: float,
) -> CostBreakdown:
    """Coste total esperado asumiendo costes lineales por evento.

    `coste_falsa_alarma` y `coste_omision` están en las unidades que tú
    decidas (€, k€, días de parada, etc.). El cociente entre ambos es lo
    importante: coste_omision >> coste_falsa_alarma para una térmica que
    no puede captar agua si el caudal cae por debajo del mínimo.
    """
    c_fa = coste_falsa_alarma * metrics.falsas_alarmas
    c_om = coste_omision * metrics.omisiones
    n_dias = (metrics.aciertos + metrics.falsas_alarmas
              + metrics.omisiones + metrics.aciertos_normales)
    total = c_fa + c_om
    return CostBreakdown(
        coste_total=total,
        coste_por_dia=total / max(n_dias, 1),
        coste_falsas_alarmas=c_fa,
        coste_omisiones=c_om,
        n_dias=n_dias,
    )


def _metrics_at_delta(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    umbral_alarma_m3s: float,
    delta: float,
    coste_falsa_alarma: float,
    coste_omision: float,
):
    """Reconstruye `LowFlowMetrics` y `CostBreakdown` aplicando un δ concreto."""
    pred_alarma = np.asarray(y_pred).reshape(-1) <= (umbral_alarma_m3s + delta)
    obs_alarma = np.asarray(y_true).reshape(-1) <= umbral_alarma_m3s
    tp = int(np.sum(pred_alarma & obs_alarma))
    fp = int(np.sum(pred_alarma & ~obs_alarma))
    fn = int(np.sum(~pred_alarma & obs_alarma))
    tn = int(np.sum(~pred_alarma & ~obs_alarma))
    tasa_fa = fp / max(fp + tn, 1)
    tasa_om = fn / max(fn + tp, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    metrics = LowFlowMetrics(
        precision=precision, recall=recall, f1=f1,
        aciertos=tp, falsas_alarmas=fp, omisiones=fn, aciertos_normales=tn,
        tasa_falsas_alarmas=tasa_fa, tasa_omisiones=tasa_om,
        especificidad=1.0 - tasa_fa,
    )
    breakdown = expected_cost(metrics, coste_falsa_alarma, coste_omision)
    return metrics, breakdown


def compare_models_at_operating_points(
    predicciones: Dict[str, "tuple[np.ndarray, np.ndarray]"],
    q_min: float,
    coste_falsa_alarma: float,
    coste_omision: float,
    max_fn: int = 0,
) -> pd.DataFrame:
    """Tabla comparativa de varios modelos a tres puntos de operación.

    Parameters
    ----------
    predicciones
        Diccionario `{etiqueta_modelo: (observado, predicho)}`. Ambos arrays
        deben tener la misma longitud y estar en m³/s.
    q_min, coste_falsa_alarma, coste_omision, max_fn
        Igual que en `metrics_at_optimal_threshold` y `metrics_with_max_fn`.

    Returns
    -------
    DataFrame en formato largo con una fila por (modelo, punto de operación).
    Puntos de operación:
        "natural"    — decidir alarma cuando pred ≤ q_min (δ = 0).
        "optimal"    — δ que minimiza coste esperado.
        "safe"       — δ que minimiza coste sujeto a FN ≤ max_fn.

    Las columnas incluyen: modelo, operating_point, delta, umbral_decision,
    tp, fp, fn, tn, precision, recall, f1, tasa_falsas_alarmas,
    tasa_omisiones, coste_total, coste_por_dia, nse, kge, factible.
    """
    filas = []
    for label, (obs, pred) in predicciones.items():
        obs = np.asarray(obs).reshape(-1)
        pred = np.asarray(pred).reshape(-1)
        nse = nash_sutcliffe(obs, pred)
        kge = kling_gupta(obs, pred)

        # Tres puntos de operación.
        operativos = []
        m, br = _metrics_at_delta(obs, pred, q_min, 0.0, coste_falsa_alarma, coste_omision)
        operativos.append(("natural", 0.0, m, br, True))
        d_opt, m_opt, br_opt = metrics_at_optimal_threshold(
            obs, pred, q_min, coste_falsa_alarma, coste_omision)
        operativos.append(("optimal", d_opt, m_opt, br_opt, True))
        d_safe, m_safe, br_safe, factible = metrics_with_max_fn(
            obs, pred, q_min, coste_falsa_alarma, coste_omision, max_fn=max_fn)
        operativos.append(("safe", d_safe, m_safe, br_safe, factible))

        for nombre, delta, m, br, fact in operativos:
            filas.append({
                "modelo": label,
                "operating_point": nombre,
                "delta": float(delta),
                "umbral_decision": float(q_min + delta),
                "tp": m.aciertos, "fp": m.falsas_alarmas,
                "fn": m.omisiones, "tn": m.aciertos_normales,
                "precision": m.precision, "recall": m.recall, "f1": m.f1,
                "tasa_falsas_alarmas": m.tasa_falsas_alarmas,
                "tasa_omisiones": m.tasa_omisiones,
                "coste_total": br.coste_total, "coste_por_dia": br.coste_por_dia,
                "nse": nse, "kge": kge,
                "factible": fact,
            })
    return pd.DataFrame(filas)


def metrics_with_max_fn(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    umbral_alarma_m3s: float,
    coste_falsa_alarma: float,
    coste_omision: float,
    max_fn: int = 0,
    deltas_m3s: Optional[np.ndarray] = None,
):
    """Punto operacional con restricción dura `FN ≤ max_fn`.

    Devuelve `(δ, metrics, breakdown, factible)` donde `factible` es False
    si ni el δ máximo del barrido cumple la restricción (en ese caso se
    devuelve el δ con menor FN posible).

    Para `max_fn=0` se obtiene el "operating point conservador": jamás se
    apaga la central (a costa de muchas falsas alarmas).
    """
    if deltas_m3s is None:
        # Barrido asimétrico: para forzar FN bajo suele hacer falta δ grande.
        amp_neg = max(umbral_alarma_m3s * 0.3, 5.0)
        amp_pos = max(umbral_alarma_m3s * 3.0, 50.0)
        deltas_m3s = np.linspace(-amp_neg, amp_pos, 121)

    curve = cost_curve(
        y_true, y_pred, umbral_alarma_m3s,
        coste_falsa_alarma, coste_omision, deltas_m3s,
    )
    factibles = curve[curve["fn"] <= max_fn]
    es_factible = not factibles.empty
    if not es_factible:
        # Coge el δ con menor FN absoluto y, dentro de esos, el menor coste.
        min_fn = int(curve["fn"].min())
        factibles = curve[curve["fn"] == min_fn]

    idx = factibles["coste_total"].idxmin()
    d = float(curve.loc[idx, "delta"])
    metrics, breakdown = _metrics_at_delta(
        y_true, y_pred, umbral_alarma_m3s, d,
        coste_falsa_alarma, coste_omision,
    )
    return d, metrics, breakdown, es_factible


def metrics_at_optimal_threshold(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    umbral_alarma_m3s: float,
    coste_falsa_alarma: float,
    coste_omision: float,
    deltas_m3s: Optional[np.ndarray] = None,
):
    """Encuentra δ que minimiza el coste y devuelve (δ_opt, LowFlowMetrics, CostBreakdown).

    Útil para reportar "qué pasaría si tomáramos la decisión a Q_min + δ_opt en
    vez de a Q_min". El modelo no se reentrena: sólo se desplaza el umbral
    operacional.
    """
    curve = cost_curve(
        y_true, y_pred, umbral_alarma_m3s,
        coste_falsa_alarma, coste_omision, deltas_m3s,
    )
    d_opt = float(curve.iloc[curve["coste_total"].idxmin()]["delta"])
    metrics, breakdown = _metrics_at_delta(
        y_true, y_pred, umbral_alarma_m3s, d_opt,
        coste_falsa_alarma, coste_omision,
    )
    return d_opt, metrics, breakdown


def cost_curve(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    umbral_alarma_m3s: float,
    coste_falsa_alarma: float,
    coste_omision: float,
    deltas_m3s: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Cómo cambian FP, FN y coste cuando movemos el umbral de decisión.

    El modelo predice un caudal continuo y la decisión "almacenar" se toma
    si `pred ≤ umbral_alarma + δ`. Un δ positivo hace al modelo más
    conservador (más alarmas, más FP, menos FN). Un δ negativo lo hace
    más permisivo (menos alarmas, más FN, menos FP).

    Devuelve un DataFrame con una fila por δ y columnas: `delta`, `tp`,
    `fp`, `fn`, `tn`, `precision`, `recall`, `tasa_falsas_alarmas`,
    `tasa_omisiones`, `coste_total`. El δ óptimo es el que minimiza el
    coste esperado dado tu par (coste_falsa_alarma, coste_omision).
    """
    if deltas_m3s is None:
        # Por defecto barre ±50% del umbral en 41 puntos (paso fino cerca de 0).
        amp = max(umbral_alarma_m3s * 0.5, 5.0)
        deltas_m3s = np.linspace(-amp, amp, 41)

    obs = np.asarray(y_true).reshape(-1)
    pred = np.asarray(y_pred).reshape(-1)
    obs_alarma = obs <= umbral_alarma_m3s

    filas = []
    for d in deltas_m3s:
        pred_alarma = pred <= (umbral_alarma_m3s + d)
        tp = int(np.sum(pred_alarma & obs_alarma))
        fp = int(np.sum(pred_alarma & ~obs_alarma))
        fn = int(np.sum(~pred_alarma & obs_alarma))
        tn = int(np.sum(~pred_alarma & ~obs_alarma))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        coste = coste_falsa_alarma * fp + coste_omision * fn
        filas.append({
            "delta": float(d),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision,
            "recall": recall,
            "tasa_falsas_alarmas": fp / max(fp + tn, 1),
            "tasa_omisiones": fn / max(fn + tp, 1),
            "coste_total": coste,
        })
    return pd.DataFrame(filas)


def format_decision_report(
    metrics: LowFlowMetrics,
    breakdown: Optional[CostBreakdown] = None,
) -> str:
    """Resumen multilínea legible con etiquetas operativas."""
    n_total = (metrics.aciertos + metrics.falsas_alarmas
               + metrics.omisiones + metrics.aciertos_normales)
    n_alarmas = metrics.aciertos + metrics.omisiones
    n_normales = metrics.falsas_alarmas + metrics.aciertos_normales
    bloque = [
        f"  Días totales evaluados: {n_total}",
        f"  Días de alarma reales (caudal < umbral): {n_alarmas}  ({n_alarmas/max(n_total,1)*100:.1f}%)",
        f"  Días normales reales:                    {n_normales}  ({n_normales/max(n_total,1)*100:.1f}%)",
        "",
        "  Decisiones del modelo:",
        f"    [TP] Predijo alarma y la hubo  → almacenamos bien:    {metrics.aciertos:6d}",
        f"    [FP] Predijo alarma y no hacía falta (coste leve):    {metrics.falsas_alarmas:6d}"
        f"   ({metrics.tasa_falsas_alarmas*100:5.1f}% de los días normales)",
        f"    [FN] No predijo alarma pero faltó agua (CRÍTICO):     {metrics.omisiones:6d}"
        f"   ({metrics.tasa_omisiones*100:5.1f}% de las alarmas reales)",
        f"    [TN] Predijo normal y lo era → producimos bien:       {metrics.aciertos_normales:6d}",
        "",
        f"  Recall (sensibilidad)        = {metrics.recall:.3f}   (1 − tasa de omisiones)",
        f"  Especificidad                = {metrics.especificidad:.3f}   (1 − tasa de falsas alarmas)",
        f"  Precisión                    = {metrics.precision:.3f}",
        f"  F1                           = {metrics.f1:.3f}",
    ]
    if breakdown is not None:
        bloque += [
            "",
            "  Coste económico esperado:",
            f"    Por falsas alarmas:   {breakdown.coste_falsas_alarmas:>12.1f}",
            f"    Por omisiones:        {breakdown.coste_omisiones:>12.1f}",
            f"    TOTAL:                {breakdown.coste_total:>12.1f}",
            f"    Por día evaluado:     {breakdown.coste_por_dia:>12.4f}",
        ]
    return "\n".join(bloque)


def rolling_evaluation(
    modelo: "RunoffModel",
    basin: "BasinSpec",
    df: pd.DataFrame,
    maximos: pd.Series,
    fecha_inicio: pd.Timestamp,
    fecha_fin: pd.Timestamp,
    horizonte: int,
    caudal_minimo_m3s: float,
    escenario: str = "worst",
    paso_dias: int = 1,
) -> Dict[str, np.ndarray]:
    """Aplica el modelo cada `paso_dias` y acumula predicción y observación.

    Parameters
    ----------
    modelo
        Cualquier `RunoffModel` ya entrenado.
    basin
        Especificación de la cuenca (define `flow_column` y
        `reservoir_aggregate_column`).
    df
        Serie escalada (mismas columnas que en entrenamiento). El caudal
        observado se reconstruye como `df[basin.flow_column] * maximos[...]`.
    maximos
        Vector de máximos usado para escalar el dataframe.
    fecha_inicio, fecha_fin
        Rango de "días HOY" sobre el que iterar.
    horizonte
        Número de pasos de predicción.
    caudal_minimo_m3s
        Umbral operacional para la clasificación de alarma.
    escenario
        Escenario de pluviosidad futura (`"observed"` o `"worst"`).
    paso_dias
        Frecuencia con que se lanza el modelo (1 = todos los días).

    Returns
    -------
    Dict con vectores `caudal_obs`, `caudal_pred`, `embalse_obs`, `embalse_pred`,
    todos de longitud `n_dias * horizonte`.
    """
    flow = basin.flow_column
    eacum = basin.reservoir_aggregate_column
    fechas_hoy = pd.date_range(fecha_inicio, fecha_fin, freq=f"{paso_dias}D")
    caudal_obs, caudal_pred = [], []
    embalse_obs, embalse_pred = [], []
    for hoy in fechas_hoy:
        manana = hoy + timedelta(days=1)
        fin = hoy + timedelta(days=horizonte)
        if fin not in df.index:
            break
        f = modelo.predict(df, hoy, maximos, escenario=escenario)
        caudal_pred.extend(f.caudal)
        embalse_pred.extend(f.embalse)
        caudal_obs.extend(df.loc[manana:fin, flow].to_numpy() * maximos[flow])
        embalse_obs.extend(df.loc[manana:fin, eacum].to_numpy() * maximos[eacum])
    return {
        "caudal_obs": np.asarray(caudal_obs),
        "caudal_pred": np.asarray(caudal_pred),
        "embalse_obs": np.asarray(embalse_obs),
        "embalse_pred": np.asarray(embalse_pred),
    }
