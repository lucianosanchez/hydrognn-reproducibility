"""Re-optimise δ for each cost ratio on the UA-HydroGNN Ebro headline.

Background
----------
Table tab:uagnn_cost_ratio_sensitivity of §5.3 currently holds δ* fixed
at the canonical-ratio value and reports the cost at that offset under
varying ratios. This is a conservative read: re-optimising δ for each
ratio could only further improve the maximin and Savage rules. This
script performs that re-optimisation by reusing the trained
UA-HydroGNN posterior of outputs/uagnn-ebro-headline.

For each cost ratio in {10, 50, 100, 200, 500} we:
  (1) Load the UA-HydroGNN checkpoint without retraining.
  (2) Replicate the rolling evaluation of run_ua_gnn_experiment.py
      (5 rainfall scenarios × Monte Carlo posterior).
  (3) Call evaluate_all_criteria on the predictive distribution with
      the chosen ratio (coste_falsa_alarma = 1, coste_omision = ratio).
  (4) Aggregate total cost, FN and FP across rolling days.

Output
------
    outputs/uagnn_cost_ratio_sensitivity_reopt.csv
        one row per ratio, columns include criterion totals and the
        re-optimised δ*(criterion, ratio).

Usage
-----
    cd hydrognn
    PYTHON=/path/to/python python scripts/analyze_cost_ratio_resweep.py \
        --ckpt outputs/uagnn-ebro-headline/modelo_uagnn \
        --directorio-datos datos-06-07-2023 --firma 580734 \
        --rolling-inicio 2020-01-01 \
        --output outputs/uagnn_cost_ratio_sensitivity_reopt.csv

Wall time
---------
~30–45 min on a single-process CPU. The bottleneck is the Monte Carlo
inference over the rolling window; the cost-surface evaluation per ratio
is cheap once predictions are cached.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from seq2seq_runoff.config import Config
from seq2seq_runoff.data import load_basin_dataframe, scale_to_unit, split_train_test
from seq2seq_runoff.ua_gnn import UAHydroGNNModel
from seq2seq_runoff.scenarios import default_library, apply_scenario_to_historical
from seq2seq_runoff.decision import evaluate_all_criteria
from seq2seq_runoff.basins.ebro import ebro_basin, ebro_graph


CRITERIA = ("naive", "maximin", "maximax", "savage")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True,
                        help="Directory containing modelo_uagnn/")
    parser.add_argument("--directorio-datos", required=True)
    parser.add_argument("--firma", required=True)
    parser.add_argument("--rolling-inicio", default="2020-01-01")
    parser.add_argument("--rolling-fin", default=None)
    parser.add_argument("--K-inference", type=int, default=50)
    parser.add_argument("--n-rain-samples", type=int, default=20)
    parser.add_argument("--ratios", nargs="+", type=float,
                        default=[10, 50, 100, 200, 500])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    # 1. Basin and data
    basin = ebro_basin()
    base_cfg = Config(basin=basin)
    directorio = Path(args.directorio_datos)
    df = load_basin_dataframe(basin, directorio, args.firma)
    df_scaled, maximos = scale_to_unit(df)

    # 2. Load checkpoint (no retraining)
    print(f"[cost-ratio] loading checkpoint {args.ckpt}")
    modelo = UAHydroGNNModel.load(args.ckpt)

    # 3. Rolling window
    flow_col = basin.flow_column
    rain_col = basin.rain_aggregate_column
    rolling_start = pd.Timestamp(args.rolling_inicio)
    rolling_end = (pd.Timestamp(args.rolling_fin) if args.rolling_fin else
                   df_scaled.index[-base_cfg.horizonte - 1])
    rolling_days = pd.date_range(rolling_start, rolling_end, freq="D")
    rolling_days = [d for d in rolling_days
                    if (d - pd.Timedelta(days=base_cfg.historia - 1)) in df_scaled.index
                    and (d + pd.Timedelta(days=base_cfg.horizonte)) in df_scaled.index]
    scenarios = default_library()
    scenario_names = [s.name for s in scenarios]
    q_min = base_cfg.caudal_minimo_m3s
    deltas = np.linspace(-2.0 * q_min, 3.0 * q_min, 161)
    print(f"[cost-ratio] rolling: {len(rolling_days)} days × {len(scenarios)} scenarios")

    # 4. Cache predictive distributions per day; for each ratio
    #    re-evaluate all four criteria.
    # We accumulate per-day FN/FP/cost per criterion, then sum.
    accumulators = {ratio: {c: {"fn": 0, "fp": 0, "cost": 0.0,
                                "delta_star_sum": 0.0, "n": 0}
                            for c in CRITERIA}
                    for ratio in args.ratios}

    for di, hoy in enumerate(rolling_days):
        if di % 100 == 0:
            print(f"[cost-ratio]   day {di+1}/{len(rolling_days)}: {hoy.date()}")
        manana = hoy + pd.Timedelta(days=1)
        fin = hoy + pd.Timedelta(days=base_cfg.horizonte)
        observed_pacum = df.loc[manana:fin, rain_col].to_numpy(dtype=np.float32)

        # Build the scenario-perturbed rainfall trajectories
        pacum_per_scenario = []
        for s in scenarios:
            trajs = np.stack([
                apply_scenario_to_historical(
                    observed_pacum, s,
                    rng=np.random.default_rng(
                        int(hoy.toordinal()) + 1000 * m
                        + hash(s.name) % 10000),
                )
                for m in range(args.n_rain_samples)
            ], axis=0)
            pacum_per_scenario.append(trajs.mean(axis=0))
        pacum_arr = np.stack(pacum_per_scenario, axis=0)

        # Predict once; reuse across ratios.
        dist = modelo.predict_distribution(
            df_scaled, hoy, maximos,
            pacum_future=pacum_arr.astype(np.float32),
            K=args.K_inference,
        )
        obs = df_scaled.loc[manana:fin, flow_col].to_numpy() * maximos[flow_col]

        for ratio in args.ratios:
            resultados = evaluate_all_criteria(
                predicted_distribution=dist, observed=obs,
                deltas=deltas, q_min=q_min,
                coste_falsa_alarma=1.0, coste_omision=float(ratio),
                scenario_names=scenario_names,
            )
            for c in CRITERIA:
                r = resultados[c]
                acc = accumulators[ratio][c]
                acc["fn"] += int(np.sum([r.fn_per_scenario[s]
                                         for s in scenario_names]))
                acc["fp"] += int(np.sum([r.fp_per_scenario[s]
                                         for s in scenario_names]))
                acc["cost"] += float(r.worst_case_cost)
                acc["delta_star_sum"] += float(r.delta_star)
                acc["n"] += 1

    # 5. Persist aggregate
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["c_fn_over_c_fp"]
    for c in CRITERIA:
        fieldnames += [f"delta_star_mean_{c}", f"fn_{c}", f"fp_{c}", f"cost_{c}"]
    with out.open("w") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for ratio in args.ratios:
            row = {"c_fn_over_c_fp": ratio}
            for c in CRITERIA:
                acc = accumulators[ratio][c]
                row[f"delta_star_mean_{c}"] = acc["delta_star_sum"] / max(1, acc["n"])
                row[f"fn_{c}"] = acc["fn"]
                row[f"fp_{c}"] = acc["fp"]
                row[f"cost_{c}"] = acc["cost"]
            w.writerow(row)

    print(f"[cost-ratio] CSV written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
