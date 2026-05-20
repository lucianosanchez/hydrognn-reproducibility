"""Experimento §4.5 del informe técnico: UA-HydroGNN + escenarios × criterios.

Análogo a `run_vae_experiment.py` pero usando `UAHydroGNNModel` en
lugar de `VAESeq2SeqRunoffModel`. La estructura de salida es
idéntica (mismos CSVs y misma resolución de criterios), lo que permite
comparar directamente las dos extensiones (\\S3 y \\S4) y validar las
hipótesis W1--W4 pre-registradas en \\S4.5.

Para Ebro usa el grafo canónico (`ebro_graph`). Para datasets
sintéticos, usa `synth_graph_full` (Phase 1) por defecto. Si quieres
testear Phase 2.2 (grafo de candidatos densos + reservorios formales),
añade `--use-simplified-graph`.

Llamada típica:

    python run_ua_gnn_experiment.py \\
        --directorio-datos ../datos-synth/full \\
        --dia-prediccion 2024-12-15 \\
        --epochs 200 --K-train 10 --K-inference 50 \\
        --n-rain-samples 20 \\
        --rolling-inicio 2020-01-01 \\
        --output ../uagnn-synth-full
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from seq2seq_runoff import Config, ForecastScenario, load_basin_dataframe
from seq2seq_runoff.basins import (
    ebro_basin, ebro_graph,
    synth_basin, synth_graph_full, synth_graph_simplified,
)
from seq2seq_runoff.data import scale_to_unit, split_train_test
from seq2seq_runoff.decision import evaluate_all_criteria
from seq2seq_runoff.gnn import GNNConfig
from seq2seq_runoff.scenarios import default_library, apply_scenario_to_historical
from seq2seq_runoff.ua_gnn import UAHydroGNNModel


# ===========================================================================
# Carga de basin + grafo
# ===========================================================================


def _autodetect(directorio: Path, firma_arg, use_simplified: bool):
    manifest = directorio / "manifest.yaml"
    if manifest.exists():
        import yaml
        m = yaml.safe_load(manifest.read_text())
        firma = firma_arg or m["basin"]["firma"]
        basin = synth_basin(directorio)
        graph_fn = synth_graph_simplified if use_simplified else synth_graph_full
        return basin, firma, graph_fn(directorio)
    if not firma_arg:
        raise SystemExit("--firma es obligatorio para el Ebro (sin manifest.yaml).")
    return ebro_basin(), firma_arg, ebro_graph()


# ===========================================================================
# CLI
# ===========================================================================


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--directorio-datos", required=True, type=Path)
    p.add_argument("--firma", default=None, type=str)
    p.add_argument("--dia-prediccion", required=True, type=str)
    p.add_argument("--rolling-inicio", default="2020-01-01")
    p.add_argument("--rolling-fin", default=None)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--K-train", type=int, default=10)
    p.add_argument("--K-inference", type=int, default=50)
    p.add_argument("--beta-ua", type=float, default=1e-3)
    p.add_argument("--kappa", type=float, default=30.0,
                   help="Peso lowflow durante el entrenamiento (alias de "
                        "`kappa_low_flow` en `GNNConfig`).")
    p.add_argument("--warmup-epochs", type=int, default=0,
                   help="Épocas iniciales con β=0 (puro NLL). Recomendado "
                        "≈ epochs/3 para cuencas grandes (N1 ≳ 32).")
    p.add_argument("--ramp-epochs", type=int, default=1,
                   help="Tras el warmup, β crece linealmente hasta β_UA "
                        "en estas épocas.")
    p.add_argument("--free-bits", type=float, default=0.0,
                   help="KL gratis (nats) por dimensión del posterior. "
                        "0.02–0.05 evita colapso global del posterior.")
    p.add_argument("--rain-bypass", action="store_true",
                   help="Activa la ruta directa lluvia→caudal (opt-in). "
                        "Sólo recomendado en cuencas grandes que muestren "
                        "el atractor de predicción constante (cf. §4.10 "
                        "'Remediation as a per-basin hyperparameter').")
    p.add_argument("--lam11-init", type=float, default=0.0,
                   help="Logit inicial de los routings lam11 (default 0.0 "
                        "⇒ λ≈0.5). Activar valores altos (2.0 ⇒ λ≈0.88) "
                        "solo en cuencas grandes con cadenas largas "
                        "(HM→OUTLET) que muestren predicción constante.")
    p.add_argument("--max-windows", type=int, default=None,
                   help="Submuestrear las ventanas de entrenamiento a este "
                        "número, sampleo aleatorio reproducible. Útil para "
                        "acotar el coste por época en cuencas grandes.")
    p.add_argument("--batch-size", type=int, default=32,
                   help="Tamaño de batch (default GNNConfig: 32).")
    p.add_argument("--seed", type=int, default=42,
                   help="Semilla aleatoria global (numpy + torch). Default 42.")
    p.add_argument("--river-velocity", type=float, default=None,
                   help="Velocidad efectiva del río (km/día) para inicializar "
                        "λ por arista de forma informada cuando el BasinGraph "
                        "trae longitudes fluviales. Para Ebro, valores típicos "
                        "35-70 km/día. Si se omite, λ se inicializa con el "
                        "valor homogéneo de --lam11-init.")
    p.add_argument("--n-rain-samples", type=int, default=20)
    p.add_argument("--coste-falsa-alarma", type=float, default=1.0)
    p.add_argument("--coste-omision", type=float, default=100.0)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--device", default="cpu")
    p.add_argument("--use-simplified-graph", action="store_true",
                   help="Usar `synth_graph_simplified` (Phase 2.2-like) en lugar "
                        "del grafo completo para datasets sintéticos.")
    return p.parse_args()


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    basin, firma, graph = _autodetect(args.directorio_datos, args.firma,
                                       args.use_simplified_graph)
    base_cfg = Config(basin=basin, epochs=args.epochs)
    gnn_cfg = GNNConfig(
        basin=basin,
        historia=base_cfg.historia, horizonte=base_cfg.horizonte,
        epochs=args.epochs, kappa_low_flow=args.kappa,
        device=args.device, batch_size=args.batch_size,
        semilla=args.seed,
    )

    df = load_basin_dataframe(basin, args.directorio_datos, firma)
    df_scaled, maximos = scale_to_unit(df)
    train, _ = split_train_test(df_scaled, fraccion_test=base_cfg.fraccion_test)
    print(f"[datos] cuenca={basin.name} | firma={firma} | "
          f"{len(df_scaled)} pasos {df_scaled.index[0].date()} → "
          f"{df_scaled.index[-1].date()}")
    print(f"[grafo] N1={graph.N1}, M={graph.M}, "
          f"E11={graph.E11}, E12={graph.E12}, E21={graph.E21}")

    # 1. Entrenamiento.
    print(f"[ua-gnn] entrenando (K_train={args.K_train}, "
          f"K_inference={args.K_inference}, β_UA={args.beta_ua}, "
          f"epochs={args.epochs})")
    modelo = UAHydroGNNModel(
        gnn_cfg, graph,
        K_train=args.K_train, K_inference=args.K_inference,
        beta_ua=args.beta_ua,
        warmup_epochs=args.warmup_epochs,
        ramp_epochs=args.ramp_epochs,
        free_bits=args.free_bits,
        rain_bypass=args.rain_bypass,
        lam11_init=args.lam11_init,
        max_windows=args.max_windows,
        river_velocity_km_day=args.river_velocity,
    )
    historico = modelo.fit(train, maximos)
    print(f"[ua-gnn] última época: loss={historico[-1]['loss']:.3f}, "
          f"flow={historico[-1]['flow']:.3f}, kl={historico[-1]['kl']:.3f}, "
          f"kl_fb={historico[-1].get('kl_fb', 0.0):.3f}, "
          f"β_eff={historico[-1].get('beta_eff', 0.0):.2e}")
    modelo.save(args.output / "modelo_uagnn")

    # 2. Rolling + escenarios + criterios.
    scenarios = default_library()
    scenario_names = [s.name for s in scenarios]
    flow_col = basin.flow_column
    rain_col = basin.rain_aggregate_column

    rolling_start = pd.Timestamp(args.rolling_inicio)
    rolling_end = (pd.Timestamp(args.rolling_fin) if args.rolling_fin else
                    df_scaled.index[-base_cfg.horizonte - 1])
    rolling_days = pd.date_range(rolling_start, rolling_end, freq="D")
    rolling_days = [d for d in rolling_days
                    if (d - pd.Timedelta(days=base_cfg.historia - 1)) in df_scaled.index
                    and (d + pd.Timedelta(days=base_cfg.horizonte)) in df_scaled.index]
    print(f"[eval] {len(rolling_days)} días; {len(scenarios)} escenarios.")

    q_min = base_cfg.caudal_minimo_m3s
    # Grid amplio para evitar saturación en los extremos (especialmente
    # con cuantiles bajos del operador en L_α): -2·q_min..3·q_min, 161 pts.
    deltas = np.linspace(-2.0 * q_min, 3.0 * q_min, 161)

    # Diagnóstico: confirmamos que los escenarios sí diferencian la lluvia
    # futura para una ventana representativa.
    diag_hoy = rolling_days[len(rolling_days) // 2]
    print(f"\n[diag] pluviosidad por escenario en torno a {diag_hoy.date()}:")
    diag_manana = diag_hoy + pd.Timedelta(days=1)
    diag_fin = diag_hoy + pd.Timedelta(days=base_cfg.horizonte)
    diag_baseline = df.loc[diag_manana:diag_fin, rain_col].to_numpy(dtype=np.float32)
    for s in scenarios:
        rng_diag = np.random.default_rng(42 + hash(s.name) % 10000)
        pert = apply_scenario_to_historical(diag_baseline, s, rng_diag)
        print(f"   {s.name:18s}  Σ = {pert.sum():7.2f}    "
              f"max = {pert.max():6.2f}   "
              f"nº lluviosos = {(pert > 0.1).sum()}")
    print()

    agg = {name: {c: [] for c in ("naive", "maximin", "maximax", "savage")}
           for name in scenario_names}
    headline_rows = []

    for hoy in rolling_days:
        manana = hoy + pd.Timedelta(days=1)
        fin = hoy + pd.Timedelta(days=base_cfg.horizonte)
        observed_pacum = df.loc[manana:fin, rain_col].to_numpy(dtype=np.float32)

        pacum_per_scenario = []
        for s in scenarios:
            trajs = np.stack([
                apply_scenario_to_historical(
                    observed_pacum, s,
                    rng=np.random.default_rng(
                        int(hoy.toordinal()) + 1000 * m + hash(s.name) % 10000
                    ),
                )
                for m in range(args.n_rain_samples)
            ], axis=0)
            pacum_per_scenario.append(trajs.mean(axis=0))
        pacum_arr = np.stack(pacum_per_scenario, axis=0)  # (S, T) mm/día

        # UA-HydroGNN: distribución predictiva (M, K, T) en m³/s.
        dist = modelo.predict_distribution(
            df_scaled, hoy, maximos,
            pacum_future=pacum_arr.astype(np.float32),
            K=args.K_inference,
        )

        obs = df_scaled.loc[manana:fin, flow_col].to_numpy() * maximos[flow_col]
        resultados = evaluate_all_criteria(
            predicted_distribution=dist,
            observed=obs,
            deltas=deltas, q_min=q_min,
            coste_falsa_alarma=args.coste_falsa_alarma,
            coste_omision=args.coste_omision,
            scenario_names=scenario_names,
        )

        for c_name, r in resultados.items():
            headline_rows.append({
                "fecha": hoy.date().isoformat(),
                "criterio": c_name,
                "delta_star": r.delta_star,
                "worst_case_cost": r.worst_case_cost,
                "expected_cost": r.expected_cost,
                "max_regret": r.max_regret,
                **{f"cost_{s}": r.cost_per_scenario[s] for s in scenario_names},
                **{f"fn_{s}": r.fn_per_scenario[s] for s in scenario_names},
                **{f"fp_{s}": r.fp_per_scenario[s] for s in scenario_names},
            })
            for s in scenario_names:
                agg[s][c_name].append(r.cost_per_scenario[s])

    # 3. Persistencia + headlines.
    headline_df = pd.DataFrame(headline_rows)
    headline_csv = args.output / "headline_per_day.csv"
    headline_df.to_csv(headline_csv, index=False)
    print(f"\n[ua-gnn] headline_per_day.csv ({len(headline_rows)}) → {headline_csv}")

    summary_rows = []
    for s in scenario_names:
        for c in ("naive", "maximin", "maximax", "savage"):
            arr = np.asarray(agg[s][c])
            summary_rows.append({
                "scenario": s, "criterion": c,
                "median_cost": float(np.median(arr)),
                "q25_cost": float(np.quantile(arr, 0.25)),
                "q75_cost": float(np.quantile(arr, 0.75)),
                "q95_cost": float(np.quantile(arr, 0.95)),
                "max_cost": float(np.max(arr)),
                "mean_cost": float(np.mean(arr)),
                "total_cost": float(np.sum(arr)),
                "n_days": int(len(arr)),
            })
    summary_csv = args.output / "summary_by_scenario_criterion.csv"
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)
    print(f"[ua-gnn] summary_by_scenario_criterion.csv → {summary_csv}")

    crit_rows = []
    for c in ("naive", "maximin", "maximax", "savage"):
        sub = headline_df[headline_df["criterio"] == c]
        crit_rows.append({
            "criterion": c,
            "n_days": int(len(sub)),
            "worst_case_max":  float(sub["worst_case_cost"].max()),
            "worst_case_p95":  float(sub["worst_case_cost"].quantile(0.95)),
            "worst_case_mean": float(sub["worst_case_cost"].mean()),
            "worst_case_total": float(sub["worst_case_cost"].sum()),
            "max_regret_max":  float(sub["max_regret"].max()),
            "max_regret_p95":  float(sub["max_regret"].quantile(0.95)),
            "max_regret_mean": float(sub["max_regret"].mean()),
            **{f"total_cost_{s}": float(sub[f"cost_{s}"].sum()) for s in scenario_names},
            **{f"total_fn_{s}":   float(sub[f"fn_{s}"].sum())   for s in scenario_names},
        })
    headline_metrics_csv = args.output / "headline_metrics.csv"
    pd.DataFrame(crit_rows).to_csv(headline_metrics_csv, index=False)
    print(f"[ua-gnn] headline_metrics.csv → {headline_metrics_csv}")

    print("\n=== HEADLINE V1 (Savage minimiza max regret) ===")
    for r in sorted(crit_rows, key=lambda r: r["max_regret_max"]):
        print(f"  {r['criterion']:10s}  max_regret max = {r['max_regret_max']:7.1f}   "
              f"p95 = {r['max_regret_p95']:6.1f}   mean = {r['max_regret_mean']:6.1f}")

    print("\n=== HEADLINE V2 (Maximin minimiza worst-case) ===")
    for r in sorted(crit_rows, key=lambda r: r["worst_case_max"]):
        print(f"  {r['criterion']:10s}  worst_case max = {r['worst_case_max']:7.1f}   "
              f"p95 = {r['worst_case_p95']:6.1f}   mean = {r['worst_case_mean']:6.1f}")

    print("\n=== HEADLINE V3 (naive dominado en no-baseline) ===")
    for r in crit_rows:
        scen_totals = "  ".join(
            f"{s.split('_')[0]:>10s}={r[f'total_cost_{s}']:6.0f}"
            for s in scenario_names
        )
        print(f"  {r['criterion']:10s}  {scen_totals}")


if __name__ == "__main__":
    main()
