"""Tubería completa: carga datos, entrena Seq2Seq, evalúa y predice.

Este script reproduce el experimento del notebook original `Modelo-V0.0.ipynb`,
ahora estructurado en pasos atómicos. Aunque la cuenca por defecto es la del
Ebro, basta con cambiar la línea ``basin = ebro_basin()`` por otra factoría
de ``seq2seq_runoff.basins`` para ejecutar el mismo pipeline en otra cuenca.

Tres bloques claramente separados:

    1. Datos     : leer CSVs y construir dataframe escalado.
    2. Modelo    : construir, entrenar, guardar.
    3. Evaluación: rolling sobre un rango histórico + forecast a futuro.

El mismo script sirve de plantilla para evaluar otros `RunoffModel` (e.g.
los baselines GNN previstos en `report2.tex`): basta con cambiar la línea
de instanciación del modelo.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Permite ejecutar el script directamente sin instalar el paquete:
# inserta la raíz del proyecto (el directorio padre de `scripts/`) en sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from seq2seq_runoff import (
    Config,
    Seq2SeqRunoffModel,
    ForecastScenario,
    load_basin_dataframe,
)
from seq2seq_runoff.basins import ebro_basin, synth_basin
from seq2seq_runoff.data import scale_to_unit, split_train_test
from seq2seq_runoff.evaluation import (
    rolling_evaluation, summary,
    low_flow_classification, expected_cost, format_decision_report,
    metrics_at_optimal_threshold, metrics_with_max_fn,
)


def _autodetect_basin_and_firma(directorio: Path, firma_arg):
    """Si el directorio contiene `manifest.yaml`, usa la cuenca sintética
    (firma leída del manifest); en caso contrario, usa la cuenca del Ebro
    (firma obligatoria desde la línea de comandos)."""
    manifest = directorio / "manifest.yaml"
    if manifest.exists():
        import yaml
        m = yaml.safe_load(manifest.read_text())
        firma = firma_arg or m["basin"]["firma"]
        return synth_basin(directorio), firma
    if not firma_arg:
        raise SystemExit(
            "--firma es obligatorio cuando los datos son del operador del Ebro "
            "(no se ha encontrado manifest.yaml en el directorio)."
        )
    return ebro_basin(), firma_arg


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Entrena y evalúa el Seq2Seq baseline.")
    p.add_argument("--directorio-datos", required=True, type=Path,
                   help="Carpeta con los CSVs (e.g. ./datos-06-07-2023).")
    p.add_argument("--firma", required=False, default=None, type=str,
                   help="Identificador en cada CSV (e.g. 580734). Sólo necesario "
                        "para la cuenca del Ebro; en datos sintéticos se lee del "
                        "`manifest.yaml`.")
    p.add_argument("--dia-prediccion", required=True, type=str,
                   help="Día de la predicción a futuro, formato YYYY-MM-DD.")
    p.add_argument("--directorio-modelo", default=Path("./modelo"), type=Path)
    p.add_argument("--epochs", default=2000, type=int)
    p.add_argument("--rolling-inicio", default="2016-05-01", type=str,
                   help="Inicio del periodo de evaluación rolling.")
    p.add_argument("--rolling-fin", default=None, type=str,
                   help="Fin del periodo de evaluación rolling. Si se omite se usa "
                        "la última fecha posible del dataset.")
    p.add_argument("--escenario", choices=["observed", "worst", "both"], default="both",
                   help="Escenario para la predicción puntual del día seleccionado. "
                        "`observed` requiere datos reales para el horizonte futuro; "
                        "`worst` supone pluviosidad cero (decisión operativa).")
    p.add_argument("--plot", type=Path, default=None,
                   help="Si se especifica, guarda en ese directorio: loss_history.png, "
                        "rolling_predictions.png, rolling_scatter.png, "
                        "confusion_costs.png, cost_curve.png, cost_curve.csv, "
                        "forecast_<dia>.png.")
    p.add_argument("--coste-falsa-alarma", type=float, default=1.0,
                   help="Coste por día con falsa alarma (€ o unidad propia). Default 1.")
    p.add_argument("--coste-omision", type=float, default=100.0,
                   help="Coste por día con omisión (parar la central). Default 100.")
    p.add_argument("--max-fn", type=int, default=0,
                   help="Restricción dura para el operating point conservador: "
                        "número máximo de FN tolerados. 0 = jamás se apaga la "
                        "central (a costa de muchos FP).")
    p.add_argument("--desbalance", type=float, default=10.0,
                   help="Peso de los días de alarma relativo a los normales en "
                        "la pérdida del caudal. Equivalente Seq2Seq a "
                        "`kappa-low-flow` del GNN — sweepable.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Auto-detecta Ebro vs sintética según presencia de manifest.yaml.
    basin, firma = _autodetect_basin_and_firma(args.directorio_datos, args.firma)
    cfg = Config(basin=basin, epochs=args.epochs, desbalance=args.desbalance)

    # ----- 1. datos -------------------------------------------------------
    df = load_basin_dataframe(basin, args.directorio_datos, firma)
    df_escalado, maximos = scale_to_unit(df)
    train, _ = split_train_test(df_escalado, fraccion_test=cfg.fraccion_test)
    print(f"[datos] cuenca={basin.name} (firma={firma}), "
          f"{len(df_escalado)} pasos {df_escalado.index[0].date()} → "
          f"{df_escalado.index[-1].date()}, train={len(train)}")

    # ----- 2. modelo ------------------------------------------------------
    modelo = Seq2SeqRunoffModel(cfg)
    print("[modelo] entrenando...")
    history = modelo.fit(train, maximos)
    modelo.save(args.directorio_modelo)
    print(f"[modelo] guardado en {args.directorio_modelo}")

    # ----- 3. evaluación rolling -----------------------------------------
    fin = pd.Timestamp(args.rolling_fin) if args.rolling_fin else (
        df_escalado.index[-cfg.horizonte - 1]
    )
    resultados = rolling_evaluation(
        modelo, basin, df_escalado, maximos,
        fecha_inicio=pd.Timestamp(args.rolling_inicio),
        fecha_fin=fin,
        horizonte=cfg.horizonte,
        caudal_minimo_m3s=cfg.caudal_minimo_m3s,
        escenario=ForecastScenario.WORST,
    )
    metricas = summary(resultados["caudal_obs"], resultados["caudal_pred"], cfg.caudal_minimo_m3s)
    print("[eval] métricas peor-caso:")
    for k, v in metricas.items():
        print(f"    {k}: {v:.3f}")

    # Persistir predictions.csv inmediatamente después del rolling para que
    # el resultado del experimento sobreviva a fallos posteriores en el
    # forecast puntual o en los plots.
    if args.plot is not None:
        args.plot.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({
            "obs": resultados["caudal_obs"],
            "pred": resultados["caudal_pred"],
        }).to_csv(args.plot / "predictions.csv", index=False)
        print(f"[plot] predictions.csv guardado en {args.plot}/")

    # Reporte explícito con etiquetas operativas y coste asimétrico.
    m = low_flow_classification(
        resultados["caudal_obs"], resultados["caudal_pred"], cfg.caudal_minimo_m3s,
    )
    breakdown = expected_cost(m, args.coste_falsa_alarma, args.coste_omision)
    print("\n[decisión] reporte operativo "
          f"(coste falsa alarma={args.coste_falsa_alarma}, coste omisión={args.coste_omision}):")
    print(format_decision_report(m, breakdown))

    # Umbral óptimo post-hoc.
    d_opt, m_opt, br_opt = metrics_at_optimal_threshold(
        resultados["caudal_obs"], resultados["caudal_pred"],
        cfg.caudal_minimo_m3s, args.coste_falsa_alarma, args.coste_omision,
    )
    print(f"\n[decisión] umbral óptimo post-hoc: δ = {d_opt:+.2f} m³/s "
          f"→ decidir alarma cuando pred ≤ {cfg.caudal_minimo_m3s + d_opt:.2f}")
    print(format_decision_report(m_opt, br_opt))
    if br_opt.coste_total < breakdown.coste_total:
        ahorro = breakdown.coste_total - br_opt.coste_total
        print(f"\n[decisión] aplicar el umbral óptimo reduce el coste total en "
              f"{ahorro:.0f} ({ahorro/breakdown.coste_total*100:.1f}%) sin reentrenar.")

    # Operating point conservador: FN ≤ max_fn como restricción dura.
    d_safe, m_safe, br_safe, factible = metrics_with_max_fn(
        resultados["caudal_obs"], resultados["caudal_pred"],
        cfg.caudal_minimo_m3s, args.coste_falsa_alarma, args.coste_omision,
        max_fn=args.max_fn,
    )
    estado = "alcanzable" if factible else f"NO alcanzable (FN mínimo posible = {m_safe.omisiones})"
    print(f"\n[decisión] operating point conservador (FN ≤ {args.max_fn}, {estado}): "
          f"δ = {d_safe:+.2f} m³/s "
          f"→ decidir alarma cuando pred ≤ {cfg.caudal_minimo_m3s + d_safe:.2f}")
    print(format_decision_report(m_safe, br_safe))

    # ----- 4. predicción puntual al "día de hoy" -------------------------
    hoy = pd.Timestamp(args.dia_prediccion)
    f_obs = None
    f_worst = None
    if args.escenario in ("observed", "both"):
        try:
            f_obs = modelo.predict(df_escalado, hoy, maximos, ForecastScenario.OBSERVED)
            print(f"[forecast] Caudal observado-PACUM:    {f_obs.caudal.round(1)}")
        except ValueError as e:
            if args.escenario == "observed":
                raise
            print(f"[forecast] (escenario observed saltado: {e})")
    if args.escenario in ("worst", "both"):
        f_worst = modelo.predict(df_escalado, hoy, maximos, ForecastScenario.WORST)
        print(f"[forecast] Caudal peor-caso (lluvia 0): {f_worst.caudal.round(1)}")

    # ----- 5. plots opcionales -------------------------------------------
    if args.plot is not None:
        from seq2seq_runoff.plotting import save_baseline_plots
        save_baseline_plots(
            args.plot,
            history=history,
            rolling_resultados=resultados,
            forecast_observado=f_obs,
            forecast_peor_caso=f_worst,
            df_real=df_escalado,
            maximos=maximos,
            hoy=hoy,
            horizonte=cfg.horizonte,
            caudal_minimo_m3s=cfg.caudal_minimo_m3s,
            basin=basin,
            coste_falsa_alarma=args.coste_falsa_alarma,
            coste_omision=args.coste_omision,
        )
        # predictions.csv ya se guardó al terminar el rolling (más arriba).
        print(f"[plot] figuras guardadas en {args.plot}/")


if __name__ == "__main__":
    main()
