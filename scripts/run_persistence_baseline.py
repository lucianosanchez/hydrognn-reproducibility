r"""Persistence baseline on all evaluation basins.

The persistence forecaster sets $\widehat Q(t+\tau) = Q(t)$ for every lead
time $\tau$. This script applies the conservative-operating-point
convention of the paper (§3.1) to the persistence forecast and outputs the
resulting metrics (NSE, KGE, conservative offset, FN, FP, total cost) to
a CSV with one row per basin.

The script is intentionally self-contained: it does not load any model or
PyTorch state. It only needs the discharge time series and basin config
(Q_min, rolling window). It can therefore be run with the minimal Python
stack required by the rest of the repository.

The CSV produced (outputs/persistence_baseline.csv) feeds the persistence
rows of tab:winners_phases in §5.1 of the paper, and the persistence floor
referenced in §5.3 "Attribution".

Usage
-----
    python scripts/run_persistence_baseline.py
"""

from __future__ import annotations

import csv
import math
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Per-basin configuration. Q_min and the rolling window match those used
# in the headline deterministic experiments of §5.1.
BASINS = [
    {
        "name": "ebro",
        "discharge_csv": ROOT / "datos-06-07-2023"
                              / "DatosHistoricos_580734_A284Z65QRIO1.csv",
        "encoding": "latin1",
        "delimiter": ";",
        "date_col": 5,
        "value_col": 2,
        "skip_header": True,
        "rolling_inicio": datetime(2020, 1, 1),
        "rolling_fin": datetime(2023, 6, 23),
        "q_min": 30.0,
    },
    {
        "name": "synth-N16",
        "discharge_csv": ROOT / "datos-synth" / "full"
                              / "DatosHistoricos_SYNTH001_SQ-CAUDAL.csv",
        "encoding": "utf-8",
        "delimiter": ";",
        "date_col": 0,
        "value_col": 1,
        "skip_header": True,
        "rolling_inicio": datetime(2022, 1, 1),
        "rolling_fin": datetime(2024, 12, 1),
        "q_min": 30.0,
    },
    {
        "name": "synth-N64",
        "discharge_csv": ROOT / "datos-synth-N64" / "full"
                              / "DatosHistoricos_SYNTH-N64_SQ-CAUDAL.csv",
        "encoding": "utf-8",
        "delimiter": ";",
        "date_col": 0,
        "value_col": 1,
        "skip_header": True,
        "rolling_inicio": datetime(2022, 1, 1),
        "rolling_fin": datetime(2024, 12, 1),
        "q_min": 30.0,
    },
]

HORIZON = 10
C_FN, C_FP = 100.0, 1.0


def load_discharge(cfg: dict) -> dict[datetime, float]:
    """Parse a per-basin discharge CSV into a {date: Q} dict.

    Handles the two formats present in the repository: SAIH-Ebro
    (semicolon, Latin-1, value at column 2) and the synthetic
    simulator output (semicolon, UTF-8, value at column 1).
    """
    discharge = {}
    with cfg["discharge_csv"].open(encoding=cfg["encoding"]) as f:
        reader = csv.reader(f, delimiter=cfg["delimiter"])
        if cfg["skip_header"]:
            next(reader)
        for row in reader:
            if len(row) <= max(cfg["date_col"], cfg["value_col"]):
                continue
            try:
                date_str = row[cfg["date_col"]].strip().split()[0]
                fecha = datetime.strptime(date_str, "%Y-%m-%d")
                q = float(row[cfg["value_col"]].strip().replace(",", "."))
                discharge[fecha] = q
            except (ValueError, IndexError):
                continue
    return discharge


def conservative_operating_point(
    discharge: dict[datetime, float],
    rolling_inicio: datetime,
    rolling_fin: datetime,
    q_min: float,
) -> dict[str, float]:
    r"""Sweep δ and return the conservative operating point.

    Builds (forecast, realised) pairs over the rolling window using
    persistence: $\widehat Q(t+\tau) = Q(t)$. Sweeps δ in
    {−50, …, +120} m³/s and returns the smallest-FN offset with
    tie-breaking by lowest cost, mirroring the convention of §3.1.
    """
    eval_days = sorted([d for d in discharge
                        if rolling_inicio <= d <= rolling_fin])
    pairs = []
    for t in eval_days:
        q_t = discharge.get(t)
        if q_t is None:
            continue
        for tau in range(1, HORIZON + 1):
            q_real = discharge.get(t + timedelta(days=tau))
            if q_real is None:
                continue
            pairs.append((q_t, q_real))

    if not pairs:
        return {}

    realized = [r for _, r in pairs]
    preds = [p for p, _ in pairs]
    n = len(pairs)
    mean_r = sum(realized) / n
    mean_p = sum(preds) / n
    ss_res = sum((r - p) ** 2 for r, p in zip(realized, preds))
    ss_tot = sum((r - mean_r) ** 2 for r in realized)
    nse = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    sx = math.sqrt(sum((r - mean_r) ** 2 for r in realized) / n)
    sy = math.sqrt(sum((p - mean_p) ** 2 for p in preds) / n)
    cov = sum((r - mean_r) * (p - mean_p) for r, p in zip(realized, preds)) / n
    r_corr = cov / (sx * sy) if sx > 0 and sy > 0 else 0.0
    alpha = sy / sx if sx > 0 else float("nan")
    beta = mean_p / mean_r if mean_r > 0 else float("nan")
    kge = 1.0 - math.sqrt((r_corr - 1) ** 2
                          + (alpha - 1) ** 2
                          + (beta - 1) ** 2)

    results = {}
    for delta_int in range(-50, 121):
        delta = float(delta_int)
        thresh = q_min + delta
        fn = fp = tp = tn = 0
        for q_hat, q_real in pairs:
            alarm = q_hat <= thresh
            low = q_real <= q_min
            if alarm and low:
                tp += 1
            elif alarm and not low:
                fp += 1
            elif not alarm and low:
                fn += 1
            else:
                tn += 1
        cost = C_FN * fn + C_FP * fp
        results[delta] = {"fn": fn, "fp": fp, "tp": tp,
                          "tn": tn, "cost": cost}

    min_fn = min(r["fn"] for r in results.values())
    candidates = [(d, r) for d, r in results.items() if r["fn"] == min_fn]
    best_delta, best = min(candidates, key=lambda x: x[1]["cost"])

    return {
        "n_issue_days": len(eval_days),
        "n_pairs": n,
        "nse": nse,
        "kge": kge,
        "pearson_r": r_corr,
        "delta_dagger": best_delta,
        "fn": best["fn"],
        "fp": best["fp"],
        "cost": best["cost"],
        "fn_zero_feasible": min_fn == 0,
    }


def main() -> int:
    out_path = ROOT / "outputs" / "persistence_baseline.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for cfg in BASINS:
        if not cfg["discharge_csv"].exists():
            print(f"[persistence] {cfg['name']}: discharge file missing, skip")
            continue
        discharge = load_discharge(cfg)
        if not discharge:
            print(f"[persistence] {cfg['name']}: empty discharge, skip")
            continue
        m = conservative_operating_point(
            discharge=discharge,
            rolling_inicio=cfg["rolling_inicio"],
            rolling_fin=cfg["rolling_fin"],
            q_min=cfg["q_min"],
        )
        if not m:
            print(f"[persistence] {cfg['name']}: empty evaluation window, skip")
            continue
        row = {"basin": cfg["name"], **m}
        rows.append(row)
        feasible = "" if m["fn_zero_feasible"] else " (FN=0 infeasible)"
        print(f"[persistence] {cfg['name']}: "
              f"δ†={m['delta_dagger']:+.0f}{feasible}, "
              f"FN={m['fn']}, FP={m['fp']}, "
              f"cost={m['cost']:.0f}, NSE={m['nse']:.3f}")

    fieldnames = ["basin", "n_issue_days", "n_pairs",
                  "nse", "kge", "pearson_r",
                  "delta_dagger", "fn", "fp", "cost",
                  "fn_zero_feasible"]
    with out_path.open("w") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"\n[persistence] CSV written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
