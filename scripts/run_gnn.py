"""Tubería de entrenamiento + evaluación de las tres fases del HydroGNN.

Auto-detecta la cuenca a partir de `--directorio-datos`:

    * si contiene `manifest.yaml` → cuenca sintética (`synth_basin`/`synth_graph_full`).
    * si no                       → cuenca del Ebro (`ebro_basin`/`ebro_graph`).

`--fase` selecciona el modelo:
    1    HydroGNNPhase1     todos los embalses observados (supervisión MSE).
    2.1  HydroGNNPhase2_1   posiciones conocidas, embalses latentes.
    2.2  HydroGNNPhase2_2   posiciones aprendidas (grafo de candidatos densos).

Llamadas típicas (los tres problemas en los que ya has lanzado el baseline):

    # Ebro
    python scripts/run_gnn.py --fase 1 \\
        --directorio-datos ../datos-06-07-2023 --firma 580734 \\
        --dia-prediccion 2023-06-25 --epochs 30 --plot ../figs-gnn-ebro

    # Sintética completa (configuración A)
    python scripts/run_gnn.py --fase 1 \\
        --directorio-datos ../datos-synth/full \\
        --dia-prediccion 2024-12-15 --epochs 30 --plot ../figs-gnn-synth-full

    # Sintética parcial (configuración B)
    python scripts/run_gnn.py --fase 2.1 \\
        --directorio-datos ../datos-synth/partial \\
        --dia-prediccion 2024-12-15 --epochs 30 --plot ../figs-gnn-synth-partial
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Permite ejecutar el script directamente sin instalar el paquete.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from seq2seq_runoff import Config, ForecastScenario, load_basin_dataframe
from seq2seq_runoff.basins import ebro_basin, ebro_graph, synth_basin, synth_graph_full
from seq2seq_runoff.data import scale_to_unit, split_train_test
from seq2seq_runoff.evaluation import (
    rolling_evaluation, summary,
    low_flow_classification, expected_cost, format_decision_report,
    metrics_at_optimal_threshold, metrics_with_max_fn,
)
from seq2seq_runoff.gnn import (
    GNNConfig,
    HydroGNNPhase1,
    HydroGNNPhase2_1,
    HydroGNNPhase2_2,
)


_FASES = {
    "1":   HydroGNNPhase1,
    "2.1": HydroGNNPhase2_1,
    "2.2": HydroGNNPhase2_2,
}


# --------------------------------------------------------------------- helpers


def _autodetect_basin_and_graph(directorio: Path, firma_arg):
    """Devuelve (basin, graph_full, firma) detectando Ebro vs sintética."""
    manifest = directorio / "manifest.yaml"
    if manifest.exists():
        import yaml
        m = yaml.safe_load(manifest.read_text())
        firma = firma_arg or m["basin"]["firma"]
        return synth_basin(directorio), synth_graph_full(directorio), firma
    if not firma_arg:
        raise SystemExit(
            "--firma es obligatorio cuando los datos son del operador del Ebro "
            "(no se ha encontrado manifest.yaml en el directorio)."
        )
    return ebro_basin(), ebro_graph(), firma_arg


def _history_to_dict(historico):
    """Convierte la lista de dicts de la historia GNN al formato que espera
    `plot_loss_history` (`{key: [values]}`)."""
    if not historico:
        return {}
    keys = historico[0].keys()
    return {k: [d[k] for d in historico if k in d] for k in keys}


# --------------------------------------------------------------------- args


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Entrena y evalúa una fase del HydroGNN.")
    p.add_argument("--fase", required=True, choices=list(_FASES.keys()))
    p.add_argument("--directorio-datos", required=True, type=Path)
    p.add_argument("--firma", default=None, type=str,
                   help="Sólo necesario para datos del Ebro; en sintéticos se "
                        "lee del manifest.yaml.")
    p.add_argument("--dia-prediccion", required=True, type=str)
    p.add_argument("--directorio-modelo", default=Path("./modelo_gnn"), type=Path)
    p.add_argument("--epochs", default=30, type=int)
    p.add_argument("--observed-stations", nargs="*", default=None,
                   help="Estaciones observables (e.g. EM01-PACUM ...). En Fase 2.x "
                        "se requiere; si se omite se usa la lista completa de "
                        "estaciones del basin.")
    p.add_argument("--m-latent", default=6, type=int,
                   help="Sólo Fase 2.2: número de embalses formales libres.")
    p.add_argument("--device", default="cpu")
    p.add_argument("--rolling-inicio", default="2016-05-01", type=str)
    p.add_argument("--rolling-fin", default=None, type=str)
    p.add_argument("--escenario", choices=["observed", "worst", "both"], default="both",
                   help="Escenario de la predicción puntual.")
    p.add_argument("--plot", type=Path, default=None,
                   help="Si se especifica, guarda en ese directorio: "
                        "loss_history.png, rolling_predictions.png, "
                        "rolling_scatter.png, confusion_costs.png, "
                        "cost_curve.png, forecast_<dia>.png.")
    p.add_argument("--coste-falsa-alarma", type=float, default=1.0)
    p.add_argument("--coste-omision", type=float, default=100.0)
    p.add_argument("--max-fn", type=int, default=0,
                   help="Restricción dura para el operating point conservador: "
                        "número máximo de FN tolerados. 0 = jamás se apaga la "
                        "central (a costa de muchos FP).")
    p.add_argument("--kappa-low-flow", type=float, default=5.0,
                   help="Peso adicional sobre errores en bajo caudal durante el "
                        "entrenamiento. Valores altos (20–50) hacen al GNN más "
                        "conservador a costa de más falsas alarmas.")
    p.add_argument("--river-velocity", type=float, default=None,
                   help="Velocidad efectiva del río (km/día) para inicializar "
                        "λ informado por longitudes (Ebro: 35-70). Default None.")
    p.add_argument("--acyclic-candidates", action="store_true",
                   help="(Solo Phase 2.2) Usar grafo de candidatos acíclico "
                        "en lugar del dense_candidate_graph. Cada embalse "
                        "formal queda asociado a un nodo-ancla y sólo puede "
                        "recibir/verter dentro de su sub-cuenca conexa. "
                        "Mejora la identificabilidad estructural a costa de "
                        "flexibilidad expresiva.")
    p.add_argument("--anchor-strategy", choices=["bfs_uniform", "headwaters"],
                   default="bfs_uniform",
                   help="(Solo si --acyclic-candidates) Estrategia para elegir "
                        "los nodos-ancla. bfs_uniform = uno por capa BFS "
                        "espaciados; headwaters = M cabeceras con más "
                        "descendientes.")
    p.add_argument("--s-low-flow", type=float, default=5.0,
                   help="Anchura (m³/s) de la zona alrededor de Q_min donde se "
                        "aplica el peso `kappa-low-flow`.")
    return p.parse_args()


# --------------------------------------------------------------------- main


def main() -> None:
    args = parse_args()
    cls = _FASES[args.fase]

    # ----- 1. cuenca + grafo (auto-detect) --------------------------------
    basin, graph, firma = _autodetect_basin_and_graph(args.directorio_datos, args.firma)

    base_cfg = Config(basin=basin)
    # En Fase 2.x se necesitan observed_stations; default = todas las del basin.
    observed = args.observed_stations
    if observed is None and args.fase != "1":
        observed = list(basin.rain_columns)

    gnn_cfg = GNNConfig(
        basin=basin,
        historia=base_cfg.historia,
        horizonte=base_cfg.horizonte,
        epochs=args.epochs,
        observed_stations=observed,
        M_latent=args.m_latent,
        device=args.device,
        kappa_low_flow=args.kappa_low_flow,
        escala_low_flow=args.s_low_flow,
        river_velocity_km_day=args.river_velocity,
        acyclic_candidates=args.acyclic_candidates,
        anchor_strategy=args.anchor_strategy,
    )

    # ----- 2. datos -------------------------------------------------------
    df = load_basin_dataframe(basin, args.directorio_datos, firma)
    df_escalado, maximos = scale_to_unit(df)
    train, _ = split_train_test(df_escalado, fraccion_test=base_cfg.fraccion_test)
    print(f"[datos] cuenca={basin.name} (firma={firma}), "
          f"{len(df_escalado)} pasos {df_escalado.index[0].date()} → "
          f"{df_escalado.index[-1].date()}, train={len(train)}")

    # ----- 3. modelo ------------------------------------------------------
    modelo = cls(gnn_cfg, graph)
    print(f"[modelo] {modelo.nombre} — entrenando {args.epochs} épocas...")
    historico = modelo.fit(train, maximos)
    print(f"[modelo] última época: {historico[-1]}")
    modelo.save(args.directorio_modelo)
    print(f"[modelo] guardado en {args.directorio_modelo}")

    # ----- 4. evaluación rolling -----------------------------------------
    fin = pd.Timestamp(args.rolling_fin) if args.rolling_fin else (
        df_escalado.index[-base_cfg.horizonte - 1]
    )
    resultados = rolling_evaluation(
        modelo, basin, df_escalado, maximos,
        fecha_inicio=pd.Timestamp(args.rolling_inicio),
        fecha_fin=fin,
        horizonte=base_cfg.horizonte,
        caudal_minimo_m3s=base_cfg.caudal_minimo_m3s,
        escenario=ForecastScenario.WORST,
    )
    metricas = summary(resultados["caudal_obs"], resultados["caudal_pred"], base_cfg.caudal_minimo_m3s)
    print(f"[eval] métricas peor-caso de {modelo.nombre}:")
    for k, v in metricas.items():
        print(f"    {k}: {v:.3f}")

    # Persistir predictions.csv inmediatamente después del rolling. Si lo que
    # viene a continuación (forecast puntual o plots) falla, el resultado
    # del experimento no se pierde y el cache de tune.py funciona.
    if args.plot is not None:
        args.plot.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({
            "obs": resultados["caudal_obs"],
            "pred": resultados["caudal_pred"],
        }).to_csv(args.plot / "predictions.csv", index=False)
        print(f"[plot] predictions.csv guardado en {args.plot}/")

    # Reporte FP/FN con coste asimétrico (mismo formato que el baseline).
    m = low_flow_classification(
        resultados["caudal_obs"], resultados["caudal_pred"], base_cfg.caudal_minimo_m3s,
    )
    breakdown = expected_cost(m, args.coste_falsa_alarma, args.coste_omision)
    print("\n[decisión] reporte operativo "
          f"(coste falsa alarma={args.coste_falsa_alarma}, coste omisión={args.coste_omision}):")
    print(format_decision_report(m, breakdown))

    # Umbral óptimo post-hoc: ¿qué pasa si decidimos a Q_min + δ_opt?
    d_opt, m_opt, br_opt = metrics_at_optimal_threshold(
        resultados["caudal_obs"], resultados["caudal_pred"],
        base_cfg.caudal_minimo_m3s, args.coste_falsa_alarma, args.coste_omision,
    )
    print(f"\n[decisión] umbral óptimo post-hoc: δ = {d_opt:+.2f} m³/s "
          f"→ decidir alarma cuando pred ≤ {base_cfg.caudal_minimo_m3s + d_opt:.2f}")
    print(format_decision_report(m_opt, br_opt))
    if br_opt.coste_total < breakdown.coste_total:
        ahorro = breakdown.coste_total - br_opt.coste_total
        print(f"\n[decisión] aplicar el umbral óptimo reduce el coste total en "
              f"{ahorro:.0f} ({ahorro/breakdown.coste_total*100:.1f}%) sin reentrenar.")

    # Operating point conservador: FN ≤ max_fn como restricción dura.
    d_safe, m_safe, br_safe, factible = metrics_with_max_fn(
        resultados["caudal_obs"], resultados["caudal_pred"],
        base_cfg.caudal_minimo_m3s, args.coste_falsa_alarma, args.coste_omision,
        max_fn=args.max_fn,
    )
    estado = "alcanzable" if factible else f"NO alcanzable (FN mínimo posible = {m_safe.omisiones})"
    print(f"\n[decisión] operating point conservador (FN ≤ {args.max_fn}, {estado}): "
          f"δ = {d_safe:+.2f} m³/s "
          f"→ decidir alarma cuando pred ≤ {base_cfg.caudal_minimo_m3s + d_safe:.2f}")
    print(format_decision_report(m_safe, br_safe))

    # ----- 5. predicción puntual -----------------------------------------
    hoy = pd.Timestamp(args.dia_prediccion)
    f_obs = None
    f_worst = None
    if args.escenario in ("observed", "both"):
        try:
            f_obs = modelo.predict(df_escalado, hoy, maximos, ForecastScenario.OBSERVED)
            print(f"\n[forecast] Caudal observado:    {f_obs.caudal.round(1)}")
        except (ValueError, KeyError) as e:
            if args.escenario == "observed":
                raise
            print(f"\n[forecast] (escenario observed saltado: {e})")
    if args.escenario in ("worst", "both"):
        try:
            f_worst = modelo.predict(df_escalado, hoy, maximos, ForecastScenario.WORST)
            print(f"[forecast] Caudal peor-caso:     {f_worst.caudal.round(1)}")
            print(f"[forecast] P(Q ≥ {base_cfg.caudal_minimo_m3s}) peor caso: {f_worst.caudal_logit.round(3)}")
        except (ValueError, KeyError) as e:
            if args.escenario == "worst":
                raise
            print(f"[forecast] (escenario worst saltado: {e})")

    # ----- 6. fase 2.2: análisis de posiciones aprendidas ---------------
    if hasattr(modelo, "analyze_positions"):
        info = modelo.analyze_positions()
        cap_total = maximos[basin.reservoir_aggregate_column]
        print(f"\n[fase2.2] capacidad real total embalsada = {cap_total:.1f}")
        if f_worst is not None:
            print(f"[fase2.2] suma media de S_k en peor caso  ≈ {f_worst.embalse.mean():.1f}")
        print("[fase2.2] mejor (origen, destino) por embalse formal:")
        import numpy as np
        for k, name in enumerate(info["res_names"]):
            i_best = int(np.argmax(info["inflow_share"][k]))
            j_best = int(np.argmax(info["outflow_share"][k]))
            print(f"    {name:14s}  origen ≈ {info['type1_names'][i_best]:14s}"
                  f"  destino ≈ {info['type1_names'][j_best]}")

    # ----- 7. plots opcionales -------------------------------------------
    if args.plot is not None:
        from seq2seq_runoff.plotting import save_baseline_plots
        # GNN history es lista de dicts; lo envolvemos para reutilizar el helper
        # que espera `history.history` o un dict.
        class _HistoryLike:
            def __init__(self, h_dict): self.history = h_dict
        hist_dict = _history_to_dict(historico)
        save_baseline_plots(
            args.plot,
            history=_HistoryLike(hist_dict),
            rolling_resultados=resultados,
            forecast_observado=f_obs,
            forecast_peor_caso=f_worst,
            df_real=df_escalado,
            maximos=maximos,
            hoy=hoy,
            horizonte=base_cfg.horizonte,
            caudal_minimo_m3s=base_cfg.caudal_minimo_m3s,
            basin=basin,
            coste_falsa_alarma=args.coste_falsa_alarma,
            coste_omision=args.coste_omision,
        )
        # predictions.csv ya se guardó al terminar el rolling (más arriba).
        print(f"\n[plot] figuras guardadas en {args.plot}/")


if __name__ == "__main__":
    main()
