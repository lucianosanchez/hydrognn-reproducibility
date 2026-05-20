"""Experimento de la sec. 3 del paper: VAE-Seq2Seq + 5 escenarios × 4 criterios.

Para cada (dataset, β, d_z):
    1. Entrena un VAE-Seq2Seq con `n_latent_samples = 100`.
    2. Para una ventana de evaluación (rolling, día por día):
        a. Para cada escenario s ∈ {baseline, mild_drought, severe_drought,
           flashy, no_rain}, genera M trayectorias futuras de pluviosidad.
        b. Para cada s, llama a `predict_distribution(...)` para obtener una
           distribución (M, K, T) sobre el caudal.
        c. Construye L[δ, s] sobre un grid de δ y aplica los 4 criterios
           (naive, maximin, maximax, savage).
    3. Acumula los resultados y los serializa en CSV + tabla resumen.

El dataset acepta el manifest del simulador (cuencas sintéticas) o la
detección automática Ebro/synth ya usada por `run_baseline.py`.

Llamada típica desde `seq2seq_runoff/scripts/`:

    python run_vae_experiment.py \\
        --directorio-datos ../datos-synth/full \\
        --dia-prediccion 2024-12-15 \\
        --epochs 200 --beta 1e-2 --latent-dim-z 16 \\
        --output ../vae-results/synth-full
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

from seq2seq_runoff import (
    Config, ForecastScenario, load_basin_dataframe,
)
from seq2seq_runoff.basins import ebro_basin, synth_basin
from seq2seq_runoff.data import scale_to_unit, split_train_test
from seq2seq_runoff.decision import (
    evaluate_all_criteria, format_criterion_report,
)
from seq2seq_runoff.scenarios import (
    default_library, apply_scenario_to_historical, sample_historical_trajectories,
)
from seq2seq_runoff.vae import VAESeq2SeqRunoffModel


# ===========================================================================
# Detección de cuenca (replica run_baseline.py)
# ===========================================================================


def _autodetect_basin_and_firma(directorio: Path, firma_arg):
    manifest = directorio / "manifest.yaml"
    if manifest.exists():
        import yaml
        m = yaml.safe_load(manifest.read_text())
        firma = firma_arg or m["basin"]["firma"]
        return synth_basin(directorio), firma
    if not firma_arg:
        raise SystemExit(
            "--firma es obligatorio para los datos del Ebro "
            "(no se encontró manifest.yaml)."
        )
    return ebro_basin(), firma_arg


# ===========================================================================
# Bucle principal
# ===========================================================================


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--directorio-datos", required=True, type=Path)
    p.add_argument("--firma", default=None, type=str)
    p.add_argument("--dia-prediccion", required=True, type=str,
                   help="Día central del rolling de evaluación.")
    p.add_argument("--rolling-inicio", default="2020-01-01")
    p.add_argument("--rolling-fin", default=None)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--latent-dim-z", type=int, default=16)
    p.add_argument("--beta", type=float, default=1e-2)
    p.add_argument("--n-latent-samples", type=int, default=100)
    p.add_argument("--n-rain-samples", type=int, default=20,
                   help="M trayectorias por escenario.")
    p.add_argument("--coste-falsa-alarma", type=float, default=1.0)
    p.add_argument("--coste-omision", type=float, default=100.0)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    basin, firma = _autodetect_basin_and_firma(args.directorio_datos, args.firma)
    cfg = Config(basin=basin, epochs=args.epochs)
    df = load_basin_dataframe(basin, args.directorio_datos, firma)
    df_scaled, maximos = scale_to_unit(df)
    train, _ = split_train_test(df_scaled, fraccion_test=cfg.fraccion_test)
    print(f"[datos] cuenca={basin.name} | firma={firma} | {len(df_scaled)} pasos "
          f"{df_scaled.index[0].date()} → {df_scaled.index[-1].date()}")

    # 1. Entrenamiento del VAE-Seq2Seq.
    print(f"[vae] entrenando (latent_dim_z={args.latent_dim_z}, β={args.beta}, "
          f"epochs={args.epochs})")
    modelo = VAESeq2SeqRunoffModel(
        cfg,
        latent_dim_z=args.latent_dim_z,
        beta=args.beta,
        n_latent_samples=args.n_latent_samples,
    )
    modelo.fit(train, maximos)
    modelo.save(args.output / "modelo_vae")
    print(f"[vae] guardado en {args.output}/modelo_vae")

    # 2. Bucle de evaluación bajo escenarios y criterios.
    scenarios = default_library()
    scenario_names = [s.name for s in scenarios]
    flow_col = basin.flow_column
    rain_col = basin.rain_aggregate_column

    rolling_start = pd.Timestamp(args.rolling_inicio)
    rolling_end = (pd.Timestamp(args.rolling_fin) if args.rolling_fin else
                    df_scaled.index[-cfg.horizonte - 1])
    rolling_days = pd.date_range(rolling_start, rolling_end, freq="D")
    # Filtramos a las fechas con historia + futuro disponibles
    rolling_days = [d for d in rolling_days
                    if (d - pd.Timedelta(days=cfg.historia - 1)) in df_scaled.index
                    and (d + pd.Timedelta(days=cfg.horizonte)) in df_scaled.index]
    print(f"[eval] {len(rolling_days)} días de evaluación; {len(scenarios)} escenarios.")

    # Grid de δ en m³/s, asimétrico hacia el lado conservador.
    # Grid amplio: −2·Q_min..+3·Q_min, 161 puntos. El extremo inferior se
    # extendió desde −Q_min para evitar saturación con cuantiles bajos
    # del operador (caso L_α en §4.9).
    q_min = cfg.caudal_minimo_m3s
    deltas = np.linspace(-2.0 * q_min, 3.0 * q_min, 161)

    # ---- Diagnóstico de escenarios: imprime estadísticos por escenario sobre
    # una ventana representativa para confirmar que SÍ son distinguibles.
    diag_hoy = rolling_days[len(rolling_days) // 2]
    print(f"\n[diag] pluviosidad por escenario en torno a {diag_hoy.date()} "
          f"(suma sobre {cfg.horizonte} días, mm):")
    diag_manana = diag_hoy + pd.Timedelta(days=1)
    diag_fin = diag_hoy + pd.Timedelta(days=cfg.horizonte)
    diag_baseline = df.loc[diag_manana:diag_fin, rain_col].to_numpy(dtype=np.float32)
    for s in scenarios:
        rng_diag = np.random.default_rng(seed=42 + hash(s.name) % 10000)
        perturbed = apply_scenario_to_historical(diag_baseline, s, rng_diag)
        print(f"   {s.name:18s}  Σ = {perturbed.sum():7.2f}    "
              f"max = {perturbed.max():6.2f}   "
              f"nº días lluviosos = {(perturbed > 0.1).sum()}")
    print()

    # Acumuladores para el resumen agregado.
    agg = {name: {"naive": [], "maximin": [], "maximax": [], "savage": []}
           for name in scenario_names}
    headline_rows = []

    for hoy in rolling_days:
        # Construcción de la pluviosidad futura por escenario:
        # tomamos la realización observada de los próximos T días como
        # "pronóstico base" y le aplicamos las transformaciones de cada
        # escenario. Esto produce S trayectorias **claramente distintas
        # entre sí** (a diferencia del bootstrap aleatorio, que en periodo
        # seco colapsa al mismo cero).
        manana = hoy + pd.Timedelta(days=1)
        fin = hoy + pd.Timedelta(days=cfg.horizonte)
        observed_pacum = df.loc[manana:fin, rain_col].to_numpy(dtype=np.float32)

        pacum_per_scenario = []
        for s in scenarios:
            # M trayectorias estocásticas alrededor del baseline observado
            # (la única fuente de variación es el RNG dentro de
            # apply_scenario_to_historical: clonado/diezmado de eventos +
            # redistribución para `flashy`).
            traj = np.stack([
                apply_scenario_to_historical(
                    observed_pacum, s,
                    rng=np.random.default_rng(
                        int(hoy.toordinal()) + 1000 * m + hash(s.name) % 10000
                    ),
                )
                for m in range(args.n_rain_samples)
            ], axis=0)  # (M_traj, T)
            # Promedio (no mediana) sobre M_traj — promediar conserva la
            # diferencia media entre escenarios mejor que la mediana cuando
            # algunas trayectorias quedan a cero.
            pacum_per_scenario.append(traj.mean(axis=0))
        pacum_arr = np.stack(pacum_per_scenario, axis=0)  # (S, T) en mm/día
        # Normaliza al rango del modelo
        pacum_norm = pacum_arr / maximos[rain_col]

        # Distribución predictiva por escenario.
        dist = modelo.predict_distribution(
            df_scaled, hoy, maximos,
            pacum_future=pacum_norm.astype(np.float32),
        )  # (S, K, T) en m³/s

        # Caudal observado de referencia (igual para todos los escenarios:
        # es la realidad que materializó).
        fin = hoy + pd.Timedelta(days=cfg.horizonte)
        manana = hoy + pd.Timedelta(days=1)
        obs = df_scaled.loc[manana:fin, flow_col].to_numpy() * maximos[flow_col]

        # Aplica los 4 criterios.
        resultados = evaluate_all_criteria(
            predicted_distribution=dist,
            observed=obs,
            deltas=deltas, q_min=q_min,
            coste_falsa_alarma=args.coste_falsa_alarma,
            coste_omision=args.coste_omision,
            scenario_names=scenario_names,
        )

        # Persistir headline por día.
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

    # 3. Resumen agregado.
    print("\n[eval] resumen agregado (mediana del coste por criterio/escenario):")
    print(f"{'escenario':15s}  " + "  ".join(f"{c:>10s}" for c in
                                              ("naive", "maximin", "maximax", "savage")))
    for s in scenario_names:
        medians = "  ".join(f"{float(np.median(agg[s][c])):>10.0f}"
                            for c in ("naive", "maximin", "maximax", "savage"))
        print(f"{s:15s}  {medians}")

    # 4. Persistir.
    headline_csv = args.output / "headline_per_day.csv"
    with headline_csv.open("w") as fh:
        w = csv.DictWriter(fh, fieldnames=list(headline_rows[0].keys()))
        w.writeheader()
        w.writerows(headline_rows)
    print(f"\n[vae] headline_per_day.csv ({len(headline_rows)} filas) → {headline_csv}")

    # ---- (A) Tabla por (escenario, criterio) con cuantiles + total acumulado.
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
    with summary_csv.open("w") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)
    print(f"[vae] summary_by_scenario_criterion.csv → {summary_csv}")

    # ---- (B) Tabla "headline" por criterio: validación directa de V1/V2/V3.
    # Para cada criterio, agregamos a lo largo del rolling el worst_case_cost
    # y el max_regret (que ya se calcularon por día).  El resultado son las
    # métricas que el paper reporta como hallazgos principales.
    headline_df = pd.DataFrame(headline_rows)
    crit_rows = []
    for c in ("naive", "maximin", "maximax", "savage"):
        sub = headline_df[headline_df["criterio"] == c]
        crit_rows.append({
            "criterion": c,
            "n_days": int(len(sub)),
            # V2: cost worst-case across scenarios at the chosen δ
            "worst_case_max":  float(sub["worst_case_cost"].max()),
            "worst_case_p95":  float(sub["worst_case_cost"].quantile(0.95)),
            "worst_case_mean": float(sub["worst_case_cost"].mean()),
            "worst_case_total": float(sub["worst_case_cost"].sum()),
            # V1: max regret across scenarios at the chosen δ
            "max_regret_max":  float(sub["max_regret"].max()),
            "max_regret_p95":  float(sub["max_regret"].quantile(0.95)),
            "max_regret_mean": float(sub["max_regret"].mean()),
            # Coste acumulado por escenario (V3)
            **{f"total_cost_{s}": float(sub[f"cost_{s}"].sum()) for s in scenario_names},
            # FN acumulados por escenario (días que la central pararía)
            **{f"total_fn_{s}": float(sub[f"fn_{s}"].sum()) for s in scenario_names},
        })
    headline_csv = args.output / "headline_metrics.csv"
    with headline_csv.open("w") as fh:
        w = csv.DictWriter(fh, fieldnames=list(crit_rows[0].keys()))
        w.writeheader()
        w.writerows(crit_rows)
    print(f"[vae] headline_metrics.csv → {headline_csv}")

    # ---- (C) Resumen impreso de las hipótesis V1/V2/V3.
    print("\n=== HEADLINE V1 (Savage minimises max regret) ===")
    by_regret = sorted(crit_rows, key=lambda r: r["max_regret_max"])
    for r in by_regret:
        print(f"  {r['criterion']:10s}  max_regret max = {r['max_regret_max']:7.1f}   "
              f"p95 = {r['max_regret_p95']:6.1f}   mean = {r['max_regret_mean']:6.1f}")

    print("\n=== HEADLINE V2 (Maximin minimises worst-case cost) ===")
    by_worst = sorted(crit_rows, key=lambda r: r["worst_case_max"])
    for r in by_worst:
        print(f"  {r['criterion']:10s}  worst_case max = {r['worst_case_max']:7.1f}   "
              f"p95 = {r['worst_case_p95']:6.1f}   mean = {r['worst_case_mean']:6.1f}")

    print("\n=== HEADLINE V3 (Naive dominated in non-baseline scenarios) ===")
    for r in crit_rows:
        scen_totals = "  ".join(
            f"{s.split('_')[0]:>10s}={r[f'total_cost_{s}']:6.0f}"
            for s in scenario_names
        )
        print(f"  {r['criterion']:10s}  {scen_totals}")


if __name__ == "__main__":
    main()
