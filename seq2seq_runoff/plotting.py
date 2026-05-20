"""Gráficos sencillos para inspección visual.

El módulo es opcional: la tubería numérica funciona sin matplotlib.

Funciones disponibles:
    plot_forecast              Forecast puntual (caudal y embalse) con observación.
    plot_loss_history          Curva de pérdida del entrenamiento.
    plot_rolling_predictions   Series temporales pred-vs-obs sobre todo el rolling.
    plot_pred_obs_scatter      Diagrama 1:1 con la línea Q_min.
    save_baseline_plots        Helper que serializa todas las figuras a disco.
"""

from __future__ import annotations

import os
from datetime import timedelta
from pathlib import Path
from typing import Dict, Optional, TYPE_CHECKING, Union

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .basin import BasinSpec
    from .model import Forecast


def plot_forecast(
    forecast_observado: Optional["Forecast"],
    forecast_peor_caso: Optional["Forecast"],
    df_real: pd.DataFrame,
    maximos: pd.Series,
    hoy: pd.Timestamp,
    horizonte: int,
    caudal_minimo_m3s: float,
    basin: "BasinSpec",
):
    """Dibuja en dos paneles las predicciones de caudal y embalse.

    `df_real` debe estar en escala normalizada (la misma con la que se
    entrenó el modelo); las observaciones se reescalan internamente con los
    nombres de columna del `basin`.
    """
    import matplotlib.pyplot as plt

    flow = basin.flow_column
    eacum = basin.reservoir_aggregate_column

    manana = hoy + timedelta(days=1)
    fin = hoy + timedelta(days=horizonte)
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Panel izquierdo: caudal en m³/s.
    if forecast_observado is not None:
        ax[0].plot(forecast_observado.fechas, forecast_observado.caudal,
                   label="Estimación caudal", lw=2)
    if forecast_peor_caso is not None:
        ax[0].plot(forecast_peor_caso.fechas, forecast_peor_caso.caudal,
                   label="Estimación caudal peor caso", lw=2)
    obs_disponible = (manana in df_real.index and fin in df_real.index)
    if obs_disponible:
        ax[0].plot(df_real.loc[manana:fin].index,
                   df_real.loc[manana:fin, flow].to_numpy() * maximos[flow],
                   ls="dotted", label="Caudal observado")
    ax[0].axhline(caudal_minimo_m3s, c="r", ls="dotted", label="Umbral mínimo")
    ax[0].set_title(f"Caudal {basin.name} — {flow}  (hoy={hoy.date()})")
    ax[0].tick_params(axis="x", labelrotation=45)
    ax[0].legend(fontsize=8)

    # Panel derecho: volumen embalsado en Hm³.
    if forecast_observado is not None:
        ax[1].plot(forecast_observado.fechas, forecast_observado.embalse,
                   label="Modelo embalse", lw=2)
    if forecast_peor_caso is not None:
        ax[1].plot(forecast_peor_caso.fechas, forecast_peor_caso.embalse,
                   label="Modelo embalse peor caso", lw=2)
    if obs_disponible:
        ax[1].plot(df_real.loc[manana:fin].index,
                   df_real.loc[manana:fin, eacum].to_numpy() * maximos[eacum],
                   ls="dotted", label="Embalse observado")
    ax[1].set_title("Volumen embalsado total")
    ax[1].tick_params(axis="x", labelrotation=45)
    ax[1].legend(fontsize=8)

    plt.tight_layout()
    return fig


def plot_loss_history(history, ax=None):
    """Curva de pérdida de entrenamiento y validación."""
    import matplotlib.pyplot as plt
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    else:
        fig = ax.figure
    h = history.history if hasattr(history, "history") else history
    if "loss" in h:
        ax.plot(h["loss"], label="loss")
    if "val_loss" in h:
        ax.plot(h["val_loss"], label="val_loss")
    ax.set_xlabel("epoch")
    # Escala log sólo si todos los valores son positivos (la NLL del GNN puede ser negativa).
    todos_valores = []
    for v in h.values():
        try:
            todos_valores.extend(v)
        except TypeError:
            pass
    if todos_valores and min(todos_valores) > 0:
        ax.set_yscale("log")
        ax.set_title("Pérdida (escala log)")
    else:
        ax.set_title("Pérdida")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig


def plot_rolling_predictions(
    resultados: Dict[str, np.ndarray],
    horizonte: int,
    caudal_minimo_m3s: float,
    titulo: str = "Rolling: caudal observado vs predicho",
    ax=None,
):
    """Series temporales del caudal observado y predicho a lo largo del rolling.

    `resultados` es el dict que devuelve `evaluation.rolling_evaluation`. Como
    cada fecha de origen genera `horizonte` puntos, el plot muestra todos los
    puntos en orden — útil para detectar sesgos sistemáticos a lo largo del
    tiempo.
    """
    import matplotlib.pyplot as plt
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))
    else:
        fig = ax.figure
    obs = resultados["caudal_obs"]
    pred = resultados["caudal_pred"]
    x = np.arange(len(obs))
    ax.plot(x, obs, lw=0.6, color="tab:blue", label="Observado", alpha=0.8)
    ax.plot(x, pred, lw=0.6, color="tab:orange", label="Predicho", alpha=0.8)
    ax.axhline(caudal_minimo_m3s, c="red", ls="--", lw=0.8, label=f"Q_min = {caudal_minimo_m3s}")
    ax.set_xlabel(f"Muestra (horizontes concatenados, {horizonte} pasos por origen)")
    ax.set_ylabel("Caudal (m³/s)")
    ax.set_title(titulo)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    return fig


def plot_pred_obs_scatter(
    resultados: Dict[str, np.ndarray],
    caudal_minimo_m3s: float,
    titulo: str = "Diagrama 1:1 — predicho vs observado",
    ax=None,
):
    """Scatter pred-vs-obs con la línea identidad y el umbral Q_min."""
    import matplotlib.pyplot as plt
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.figure
    obs = resultados["caudal_obs"]
    pred = resultados["caudal_pred"]
    lim = max(np.max(obs), np.max(pred)) * 1.05
    ax.scatter(obs, pred, s=4, alpha=0.3, color="tab:blue")
    ax.plot([0, lim], [0, lim], color="black", lw=0.7, label="y = x")
    ax.axvline(caudal_minimo_m3s, c="red", ls=":", lw=0.8, alpha=0.6)
    ax.axhline(caudal_minimo_m3s, c="red", ls=":", lw=0.8, alpha=0.6)
    ax.set_xlabel("Observado (m³/s)")
    ax.set_ylabel("Predicho (m³/s)")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect("equal")
    ax.set_title(titulo)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    return fig


def plot_confusion_costs(metrics, coste_falsa_alarma=1.0, coste_omision=10.0, ax=None):
    """Matriz de confusión 2×2 con etiquetas operativas y coste anotado.

    Cada celda muestra:
      n.º de días | (coste = c · n)

    El color refleja el coste de la celda, no el número de eventos: el FN
    aparece en rojo intenso aunque sea poco frecuente.
    """
    import matplotlib.pyplot as plt
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))
    else:
        fig = ax.figure

    n = np.array([
        [metrics.aciertos_normales, metrics.falsas_alarmas],     # obs=normal | pred=normal, alarma
        [metrics.omisiones,         metrics.aciertos],            # obs=alarma | pred=normal, alarma
    ])
    coste = np.array([
        [0.0,                       coste_falsa_alarma * metrics.falsas_alarmas],
        [coste_omision * metrics.omisiones, 0.0],
    ])

    im = ax.imshow(coste, cmap="Reds", aspect="equal")
    fig.colorbar(im, ax=ax, label="coste de la celda")

    etiquetas = [
        ["TN  produces bien",        "FP  almacenas de más\n(coste leve)"],
        ["FN  TE QUEDAS SIN AGUA\n(crítico — parar central)", "TP  almacenas con razón"],
    ]
    for i in range(2):
        for j in range(2):
            txt = f"{etiquetas[i][j]}\n\nn = {n[i, j]}"
            if coste[i, j] > 0:
                txt += f"\ncoste = {coste[i, j]:.1f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=9,
                    color="white" if coste[i, j] > coste.max()*0.5 else "black")

    ax.set_xticks([0, 1]); ax.set_xticklabels(["pred normal", "pred alarma"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["obs normal", "obs alarma"])
    ax.set_xlabel("Decisión del modelo")
    ax.set_ylabel("Realidad")
    ax.set_title(f"Matriz de confusión con coste\n(c_FA={coste_falsa_alarma}, c_FN={coste_omision})")
    return fig


def plot_cost_curve(curve_df, q_min, ax=None):
    """Curva de coste vs umbral de decisión.

    `curve_df` es lo que devuelve `evaluation.cost_curve`. Dibuja FP, FN y
    coste total (eje doble) en función de δ. Marca el δ óptimo.
    """
    import matplotlib.pyplot as plt
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig = ax.figure

    deltas = curve_df["delta"].to_numpy()
    ax.plot(deltas, curve_df["fp"], color="tab:orange", lw=1.5, label="Falsas alarmas (FP)")
    ax.plot(deltas, curve_df["fn"], color="tab:red",    lw=1.5, label="Omisiones (FN, crítico)")
    ax.set_xlabel(f"δ (m³/s) — umbral efectivo = Q_min + δ ; Q_min = {q_min}")
    ax.set_ylabel("Número de días")
    ax.grid(True, alpha=0.3)

    ax2 = ax.twinx()
    ax2.plot(deltas, curve_df["coste_total"], color="black", lw=2.0, label="Coste total")
    ax2.set_ylabel("Coste total")

    # δ óptimo (mínimo coste)
    i_opt = int(curve_df["coste_total"].idxmin())
    d_opt = curve_df["delta"].iloc[i_opt]
    c_opt = curve_df["coste_total"].iloc[i_opt]
    ax.axvline(d_opt, color="green", linestyle="--", lw=1.0,
               label=f"δ óptimo = {d_opt:+.1f}  (coste {c_opt:.0f})")

    # Leyendas combinadas
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper center", fontsize=8)
    ax.set_title("Trade-off FP vs FN al mover el umbral de decisión")
    return fig


def plot_model_comparison(
    df_comp,
    coste_omision: float = None,
    ax=None,
):
    """Barplot comparando modelos en los tres puntos de operación.

    `df_comp` es lo que devuelve `evaluation.compare_models_at_operating_points`.
    Se dibujan dos paneles: FN (top, log scale para que el FN=0 se distinga
    del FN=1, etc.) y coste_total (bottom).
    """
    import matplotlib.pyplot as plt
    if ax is None:
        fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=False)
    else:
        fig = ax.figure
        axes = [ax, ax.figure.add_subplot(2, 1, 2)]
    ax_fn, ax_cost = axes

    modelos = list(df_comp["modelo"].unique())
    operating_points = ["natural", "optimal", "safe"]
    width = 0.25
    x = np.arange(len(modelos))

    colores = {"natural": "tab:blue", "optimal": "tab:green", "safe": "tab:orange"}
    for i, op in enumerate(operating_points):
        sub = df_comp[df_comp["operating_point"] == op].set_index("modelo")
        fn_values = [sub.loc[m, "fn"] if m in sub.index else 0 for m in modelos]
        cost_values = [sub.loc[m, "coste_total"] if m in sub.index else 0 for m in modelos]
        offset = (i - 1) * width
        ax_fn.bar(x + offset, fn_values, width, label=op, color=colores[op])
        ax_cost.bar(x + offset, cost_values, width, label=op, color=colores[op])

    ax_fn.set_yscale("symlog", linthresh=1)
    ax_fn.set_ylabel("FN (símlog) — días que paramos producción")
    ax_fn.set_xticks(x)
    ax_fn.set_xticklabels(modelos)
    ax_fn.set_title("Omisiones (FN) — la métrica crítica")
    ax_fn.legend(title="Punto de operación", fontsize=8)
    ax_fn.grid(True, alpha=0.3, axis="y")

    ax_cost.set_ylabel("Coste total")
    ax_cost.set_xticks(x)
    ax_cost.set_xticklabels(modelos)
    titulo = "Coste total"
    if coste_omision is not None:
        titulo += f" (FN = {coste_omision} × omision)"
    ax_cost.set_title(titulo)
    ax_cost.legend(title="Punto de operación", fontsize=8)
    ax_cost.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    return fig


def save_baseline_plots(
    output_dir: Union[str, Path],
    *,
    history=None,
    rolling_resultados: Optional[Dict[str, np.ndarray]] = None,
    forecast_observado: Optional["Forecast"] = None,
    forecast_peor_caso: Optional["Forecast"] = None,
    df_real: Optional[pd.DataFrame] = None,
    maximos: Optional[pd.Series] = None,
    hoy: Optional[pd.Timestamp] = None,
    horizonte: int = 10,
    caudal_minimo_m3s: float = 30.0,
    basin: Optional["BasinSpec"] = None,
    coste_falsa_alarma: float = 1.0,
    coste_omision: float = 100.0,
    dpi: int = 150,
) -> None:
    """Genera y guarda hasta 4 PNG (los argumentos opcionales se omiten si faltan).

        loss_history.png            Curva loss/val_loss en escala log.
        rolling_predictions.png     Serie temporal pred vs obs.
        rolling_scatter.png         Scatter 1:1.
        forecast_<dia>.png          Forecast a horizonte para el día concreto.
    """
    os.environ.setdefault("MPLBACKEND", "Agg")
    import matplotlib.pyplot as plt

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if history is not None:
        plot_loss_history(history).savefig(out / "loss_history.png", dpi=dpi, bbox_inches="tight")
        plt.close("all")

    if rolling_resultados is not None:
        plot_rolling_predictions(
            rolling_resultados, horizonte, caudal_minimo_m3s,
        ).savefig(out / "rolling_predictions.png", dpi=dpi, bbox_inches="tight")
        plt.close("all")
        plot_pred_obs_scatter(
            rolling_resultados, caudal_minimo_m3s,
        ).savefig(out / "rolling_scatter.png", dpi=dpi, bbox_inches="tight")
        plt.close("all")
        # Análisis de coste asimétrico
        from .evaluation import low_flow_classification, cost_curve
        m = low_flow_classification(
            rolling_resultados["caudal_obs"],
            rolling_resultados["caudal_pred"],
            caudal_minimo_m3s,
        )
        plot_confusion_costs(
            m, coste_falsa_alarma, coste_omision,
        ).savefig(out / "confusion_costs.png", dpi=dpi, bbox_inches="tight")
        plt.close("all")
        curve = cost_curve(
            rolling_resultados["caudal_obs"], rolling_resultados["caudal_pred"],
            caudal_minimo_m3s, coste_falsa_alarma, coste_omision,
        )
        plot_cost_curve(curve, caudal_minimo_m3s).savefig(
            out / "cost_curve.png", dpi=dpi, bbox_inches="tight"
        )
        curve.to_csv(out / "cost_curve.csv", index=False)
        plt.close("all")

    if (forecast_observado is not None or forecast_peor_caso is not None) \
            and hoy is not None and df_real is not None and maximos is not None and basin is not None:
        plot_forecast(
            forecast_observado, forecast_peor_caso, df_real, maximos,
            hoy, horizonte, caudal_minimo_m3s, basin,
        ).savefig(out / f"forecast_{hoy.date()}.png", dpi=dpi, bbox_inches="tight")
        plt.close("all")
