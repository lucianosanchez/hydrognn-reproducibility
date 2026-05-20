"""Operating-cost evaluation for the 8 Phase 2.2 checkpoints.

Background
----------
`outputs/phase22_grid/grid_summary.csv` reports training loss and post-hoc
physicalisation metrics for the dense/acyclic × M_latent grid, but does
NOT include operating cost (FN, FP, conservative C†) under the
deterministic worst-case-rainfall convention used by tab:winners_phases.

This script closes that gap: for each checkpoint it (i) re-builds the
basin and graph from the checkpoint metadata, (ii) loads the trained
Phase 2.2 model, (iii) runs the deterministic rolling evaluation with
zero future rainfall, (iv) computes the conservative operating point
under c_FN/c_FP = 100/1 and writes one row per checkpoint to
outputs/phase22_grid/operating_cost.csv.

The deterministic convention matches §5.1 tab:winners_phases:
ForecastScenario.WORST (no future rainfall), no scenario library, no
posterior — consistent with the way Seq2Seq and Phase 1/2.1/dense Phase
2.2 are reported there.

Usage
-----
    cd hydrognn
    PYTHON=/path/to/python bash scripts/evaluate_phase22_operating_cost.sh
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from seq2seq_runoff.config import Config
from seq2seq_runoff.data import load_basin_dataframe, scale_to_unit, split_train_test
from seq2seq_runoff.evaluation import (
    low_flow_classification,
    expected_cost,
    metrics_with_max_fn,
    rolling_evaluation,
)
from seq2seq_runoff.model import ForecastScenario
from seq2seq_runoff.gnn.model import HydroGNNPhase2_2
from seq2seq_runoff.basins.synth import synth_basin, synth_graph_full


C_FP = 1.0
C_FN = 100.0


def evaluate_checkpoint(ckpt_dir: Path, basin_dir: Path, firma: str,
                       rolling_inicio: str, rolling_fin: str) -> dict:
    """Load a Phase 2.2 checkpoint and compute operating cost under the
    deterministic worst-case convention."""
    basin = synth_basin(basin_dir)
    graph = synth_graph_full(basin_dir)
    base_cfg = Config(basin=basin)

    df = load_basin_dataframe(basin, basin_dir, firma)
    df_scaled, maximos = scale_to_unit(df)
    _, _ = split_train_test(df_scaled, fraccion_test=base_cfg.fraccion_test)

    modelo = HydroGNNPhase2_2.load(ckpt_dir, base_cfg)

    fecha_inicio = pd.Timestamp(rolling_inicio)
    fecha_fin = pd.Timestamp(rolling_fin) if rolling_fin else (
        df_scaled.index[-base_cfg.horizonte - 1]
    )
    resultados = rolling_evaluation(
        modelo, basin, df_scaled, maximos,
        fecha_inicio=fecha_inicio, fecha_fin=fecha_fin,
        horizonte=base_cfg.horizonte,
        caudal_minimo_m3s=base_cfg.caudal_minimo_m3s,
        escenario=ForecastScenario.WORST,
    )

    obs = resultados["caudal_obs"]
    pred = resultados["caudal_pred"]
    q_min = base_cfg.caudal_minimo_m3s

    # Conservative operating point (FN = 0 if feasible; else min-FN, least-cost)
    d_safe, m_safe, br_safe, feasible_zero = metrics_with_max_fn(
        obs, pred, q_min, C_FP, C_FN, max_fn=0,
    )
    if not feasible_zero:
        # find min FN reachable, take least-cost among those
        min_fn = m_safe.omisiones
        d_safe, m_safe, br_safe, _ = metrics_with_max_fn(
            obs, pred, q_min, C_FP, C_FN, max_fn=min_fn,
        )

    return {
        "n_obs": int(len(obs)),
        "delta_dagger": float(d_safe),
        "fn": int(m_safe.omisiones),
        "fp": int(m_safe.falsas_alarmas),
        "cost": float(br_safe.coste_total),
        "fn_zero_feasible": bool(feasible_zero),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--dataset", required=True, choices=("synth-N16", "synth-N64"))
    parser.add_argument("--m-latent", type=int, required=True, choices=(3, 6))
    parser.add_argument("--mode", required=True, choices=("dense", "acyclic"))
    parser.add_argument("--rolling-inicio", default="2022-01-01")
    parser.add_argument("--rolling-fin", default="2024-12-01")
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()

    basin_map = {
        "synth-N16": (ROOT / "datos-synth/full", "SYNTH001"),
        "synth-N64": (ROOT / "datos-synth-N64/full", "SYNTH-N64"),
    }
    basin_dir, firma = basin_map[args.dataset]

    metrics = evaluate_checkpoint(
        ckpt_dir=Path(args.ckpt),
        basin_dir=basin_dir,
        firma=firma,
        rolling_inicio=args.rolling_inicio,
        rolling_fin=args.rolling_fin,
    )

    out = Path(args.output_csv)
    new = not out.exists()
    fieldnames = ["dataset", "M_latent", "mode",
                  "n_obs", "delta_dagger", "fn", "fp", "cost",
                  "fn_zero_feasible"]
    row = {
        "dataset": args.dataset, "M_latent": args.m_latent, "mode": args.mode,
        **metrics,
    }
    with out.open("a") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if new:
            w.writeheader()
        w.writerow(row)

    print(f"[phase22-cost] {args.dataset} M={args.m_latent} {args.mode}: "
          f"δ†={metrics['delta_dagger']:+.1f}, "
          f"FN={metrics['fn']}, FP={metrics['fp']}, "
          f"cost={metrics['cost']:.0f}"
          + ("" if metrics["fn_zero_feasible"] else " (FN=0 infeasible)"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
