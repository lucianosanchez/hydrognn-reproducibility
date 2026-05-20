"""Resumen del grid search de configuraciones de remediación.

Consume `outputs/grid/<config>/uagnn-<dataset>/headline_metrics.csv` y
produce:
  1. Una tabla wide (config × dataset → métricas Savage) en CSV.
  2. Un análisis de "configuración homogénea": ¿hay alguna config que
     mantenga las métricas headline en Ebro/N=16 (defaults) y mejore
     N=64 (full remediation)?

Criterios de "mantenido" en Ebro/N=16 (relajados, son screenings de 100ep):
   * Ebro Savage FN total ≤ 200  (vs 44 del paper headline a 200ep)
   * Ebro Savage worst_case_max ≤ 500 (vs 405 del paper headline)
   * N=16 Savage FN total ≤ 50   (vs 0 del paper headline)

Criterios de "mejorado" en N=64:
   * Savage max_regret_max ≥ 1   (criterios diferencian — vs 0 del colapso)
   * Savage FN total ≤ 5050      (vs 6000 del colapso, ~16% mejora)

Output: `outputs/grid/grid_summary.csv` + reporte por stdout.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


DATASETS = ["synth-N16", "synth-N64", "ebro"]
CRITERIA = ("naive", "maximin", "maximax", "savage")

# Thresholds del "homogeneous winner" (relajados por usar epochs=100 en
# screening; el paper headline usa epochs=200).
PASS_EBRO   = {"fn_total_max": 200,  "worst_case_max_max": 500}
PASS_N16    = {"fn_total_max": 50,   "worst_case_max_max": 50}
PASS_N64    = {"savage_regret_min": 1, "fn_total_max": 5050}


def _parse():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--grid-root", type=Path, default=Path("outputs/grid"))
    p.add_argument("--output", type=Path, default=Path("outputs/grid/grid_summary.csv"))
    return p.parse_args()


def _read_metrics(csv_path: Path) -> dict | None:
    if not csv_path.exists():
        return None
    out = {}
    with open(csv_path) as fh:
        for r in csv.DictReader(fh):
            crit = r["criterion"]
            fn_total = sum(float(r[k]) for k in r if k.startswith("total_fn_"))
            cost_total = sum(float(r[k]) for k in r if k.startswith("total_cost_"))
            out[crit] = {
                "fn_total": fn_total,
                "cost_total": cost_total,
                "worst_case_max": float(r["worst_case_max"]),
                "worst_case_p95": float(r["worst_case_p95"]),
                "max_regret_max": float(r["max_regret_max"]),
                "max_regret_mean": float(r["max_regret_mean"]),
            }
    return out


def _passes_ebro(m: dict) -> bool:
    if "savage" not in m:
        return False
    s = m["savage"]
    return (s["fn_total"]        <= PASS_EBRO["fn_total_max"] and
            s["worst_case_max"]  <= PASS_EBRO["worst_case_max_max"])


def _passes_n16(m: dict) -> bool:
    if "savage" not in m:
        return False
    s = m["savage"]
    return (s["fn_total"]        <= PASS_N16["fn_total_max"] and
            s["worst_case_max"]  <= PASS_N16["worst_case_max_max"])


def _passes_n64(m: dict) -> bool:
    if "savage" not in m:
        return False
    s = m["savage"]
    return (s["max_regret_max"]  >= PASS_N64["savage_regret_min"] and
            s["fn_total"]        <= PASS_N64["fn_total_max"])


def main():
    args = _parse()
    root = args.grid_root
    if not root.exists():
        raise SystemExit(f"{root} no existe — primero lanza run_remediation_grid.sh")

    config_ids = sorted([d.name for d in root.iterdir() if d.is_dir() and not d.name.startswith("_")])
    if not config_ids:
        raise SystemExit(f"{root} está vacío.")
    print(f"[summary] {len(config_ids)} configuraciones detectadas: {' '.join(config_ids)}")

    # Lee todas las métricas a memoria.
    all_metrics: dict[str, dict[str, dict]] = {}
    for cid in config_ids:
        all_metrics[cid] = {}
        for ds in DATASETS:
            csv_path = root / cid / f"uagnn-{ds}" / "headline_metrics.csv"
            m = _read_metrics(csv_path)
            if m is not None:
                all_metrics[cid][ds] = m

    # Tabla wide con Savage por config × dataset.
    rows = []
    for cid in config_ids:
        row = {"config": cid}
        for ds in DATASETS:
            m = all_metrics[cid].get(ds, {})
            for crit in ("savage", "naive"):
                k = m.get(crit, {})
                row[f"{ds}_{crit}_fn"]      = k.get("fn_total", float("nan"))
                row[f"{ds}_{crit}_cost"]    = k.get("cost_total", float("nan"))
                row[f"{ds}_{crit}_worst"]   = k.get("worst_case_max", float("nan"))
                row[f"{ds}_{crit}_regret"]  = k.get("max_regret_max", float("nan"))
        # Pass-fail por dataset
        row["pass_ebro"]    = _passes_ebro(all_metrics[cid].get("ebro", {}))
        row["pass_n16"]     = _passes_n16(all_metrics[cid].get("synth-N16", {}))
        row["pass_n64"]     = _passes_n64(all_metrics[cid].get("synth-N64", {}))
        row["pass_all"]     = row["pass_ebro"] and row["pass_n16"] and row["pass_n64"]
        rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with open(args.output, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print(f"[summary] {len(rows)} filas → {args.output}")

    # Reporte por stdout.
    print()
    print("=" * 96)
    print("RESUMEN GRID SEARCH — Savage en cada (config, dataset)")
    print("=" * 96)
    print(f"{'config':<8} | {'N=16 FN':>9}  {'N=16 reg':>9} | "
          f"{'N=64 FN':>9}  {'N=64 reg':>9} | {'Ebro FN':>9}  {'Ebro reg':>9} | passes")
    print("-" * 96)
    for r in rows:
        p_n16 = "✓" if r["pass_n16"] else "✗"
        p_n64 = "✓" if r["pass_n64"] else "✗"
        p_ebro = "✓" if r["pass_ebro"] else "✗"
        p_all = "★" if r["pass_all"] else " "
        print(f"{r['config']:<8} | {r['synth-N16_savage_fn']:>9.0f}  {r['synth-N16_savage_regret']:>9.1f} | "
              f"{r['synth-N64_savage_fn']:>9.0f}  {r['synth-N64_savage_regret']:>9.1f} | "
              f"{r['ebro_savage_fn']:>9.0f}  {r['ebro_savage_regret']:>9.1f} | "
              f"N16={p_n16} N64={p_n64} Eb={p_ebro} {p_all}")
    print("-" * 96)

    winners = [r["config"] for r in rows if r["pass_all"]]
    print()
    if winners:
        print(f"** GANADOR(ES) HOMOGÉNEO(S): {', '.join(winners)} **")
        print("→ Estas configuraciones cumplen los tres criterios simultáneamente:")
        print(f"    Ebro:   Savage FN ≤ {PASS_EBRO['fn_total_max']}, worst ≤ {PASS_EBRO['worst_case_max_max']}")
        print(f"    N=16:   Savage FN ≤ {PASS_N16['fn_total_max']}, worst ≤ {PASS_N16['worst_case_max_max']}")
        print(f"    N=64:   Savage max_regret ≥ {PASS_N64['savage_regret_min']}, FN ≤ {PASS_N64['fn_total_max']}")
        print()
        print("Recomendación: si los ganadores son configuraciones intermedias")
        print("(B, C, D, E, F, G; NO A ni H), el paper puede reescribirse")
        print("eliminando §4.10 'Remediation as a per-basin hyperparameter' y")
        print("reportando esos flags como nuevos defaults universales.")
    else:
        print("** NO HAY CONFIGURACIÓN HOMOGÉNEA **")
        print("→ Ninguna config cumple los tres pass-criteria a la vez.")
        print("La narrativa actual del paper (§4.10, remediación como")
        print("hiperparámetro per-basin) queda confirmada por evidencia")
        print("empírica del grid search.")

    # Pareto front sobre (Ebro FN, N=64 FN) — útil aunque no haya ganador
    # absoluto: muestra el trade-off.
    print()
    print("Trade-off Pareto (Ebro Savage FN  vs  N=64 Savage FN):")
    pareto = []
    valid = [r for r in rows if not (r["ebro_savage_fn"] != r["ebro_savage_fn"] or
                                       r["synth-N64_savage_fn"] != r["synth-N64_savage_fn"])]
    for r in valid:
        dominated = False
        for r2 in valid:
            if r2 is r:
                continue
            if (r2["ebro_savage_fn"] <= r["ebro_savage_fn"] and
                r2["synth-N64_savage_fn"] <= r["synth-N64_savage_fn"] and
                (r2["ebro_savage_fn"] < r["ebro_savage_fn"] or
                 r2["synth-N64_savage_fn"] < r["synth-N64_savage_fn"])):
                dominated = True
                break
        if not dominated:
            pareto.append(r)
    pareto.sort(key=lambda r: r["ebro_savage_fn"])
    print(f"{'config':<8}  {'Ebro FN':>9}  {'N=64 FN':>9}  {'N=64 regret':>11}")
    for r in pareto:
        print(f"{r['config']:<8}  {r['ebro_savage_fn']:>9.0f}  "
              f"{r['synth-N64_savage_fn']:>9.0f}  {r['synth-N64_savage_regret']:>11.1f}")


if __name__ == "__main__":
    main()
