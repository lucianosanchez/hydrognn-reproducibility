"""Resume múltiples corridas con semillas distintas.

Consume `headline_metrics.csv` de cada corrida y produce un CSV
agregado con mean ± std de cada métrica importante. Útil para producir
las tablas de robustez (§4.9 del paper).

Uso típico para auditar el headline (sin remediación):

    python summarize_seed_robustness.py \\
        --inputs outputs/uagnn-ebro-headline-seed0 \\
                 outputs/uagnn-ebro-headline-seed7 \\
                 outputs/uagnn-ebro-headline-seed123 \\
        --output outputs/ebro_headline_seed_robustness.csv

O para auditar la regresión con remediación bypass:

    python summarize_seed_robustness.py \\
        --inputs outputs/uagnn-ebro-remediated-seed0 \\
                 outputs/uagnn-ebro-remediated-seed7 \\
                 outputs/uagnn-ebro-remediated-seed123 \\
        --output outputs/ebro_remediated_seed_robustness.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd


METRICS = [
    "worst_case_max", "worst_case_p95", "worst_case_mean", "worst_case_total",
    "max_regret_max", "max_regret_p95", "max_regret_mean",
    "total_cost_baseline", "total_cost_mild_drought", "total_cost_severe_drought",
    "total_cost_flashy", "total_cost_no_rain",
    "total_fn_baseline", "total_fn_mild_drought", "total_fn_severe_drought",
    "total_fn_flashy", "total_fn_no_rain",
]


def _parse():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inputs", type=Path, nargs="+", required=True,
                   help="Directorios de las corridas (con headline_metrics.csv dentro).")
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def main():
    args = _parse()
    runs = []
    for d in args.inputs:
        csv = d / "headline_metrics.csv"
        if not csv.exists():
            print(f"[summarize] {csv} no existe — saltando.")
            continue
        df = pd.read_csv(csv)
        df["run"] = d.name
        runs.append(df)
    if not runs:
        raise SystemExit("[summarize] sin corridas válidas. Aborta.")
    all_runs = pd.concat(runs, ignore_index=True)
    print(f"[summarize] {len(runs)} corridas, {len(all_runs)} filas totales")

    rows = []
    for criterion in ("naive", "maximin", "maximax", "savage"):
        sub = all_runs[all_runs["criterion"] == criterion]
        row = {"criterion": criterion, "n_runs": int(len(sub))}
        for m in METRICS:
            if m not in sub.columns:
                row[f"{m}_mean"] = float("nan")
                row[f"{m}_std"] = float("nan")
                continue
            vals = sub[m].to_numpy(dtype=float)
            row[f"{m}_mean"] = float(vals.mean())
            row[f"{m}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        rows.append(row)

    df_out = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(args.output, index=False)
    print(f"[summarize] → {args.output}")

    print("\n=== Resumen seed-robustness: Savage (mean ± std sobre seeds) ===")
    sav = df_out[df_out["criterion"] == "savage"].iloc[0]
    for col in ("worst_case_max", "max_regret_max",
                "total_cost_flashy", "total_fn_flashy", "total_fn_baseline"):
        m = sav[f"{col}_mean"]
        s = sav[f"{col}_std"]
        print(f"  {col:30s}  {m:>10.1f} ± {s:>8.1f}")


if __name__ == "__main__":
    main()
