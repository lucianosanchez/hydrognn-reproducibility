"""Análisis estadístico de los bloques H1, H2, H3, H5.

Lee `sweep_summary.csv` (producido por `run_synth_sweep.py`) y, para cada
bloque, evalúa la hipótesis correspondiente:

    H1   ratio coste(P2.2) / coste(P1)              vs  nonstationarity_amp
    H2   log ratio coste(Seq2Seq) / coste(GNN-P1)   vs  n_type1
    H3   log ratio coste(Seq2Seq) / coste(GNN-best) vs  station_coverage
    H5   log ratio coste(Seq2Seq) / coste(GNN-best) vs  years

Reporta:
    * Pendiente de la regresión lineal en log-ratio.
    * Bootstrap CI 95 % de la pendiente (10 000 muestras).
    * R² y un p-valor empírico (fracción de bootstraps con pendiente
      del signo opuesto al observado).
    * Tabla por familia × nivel con mediana e IQR del coste.

El output queda en `<output>/analysis_<bloque>.csv` y, si se pasa
`--plot`, también `<output>/analysis_<bloque>.png` con scatter+ajuste.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd


# Configuración por bloque: (eje X, familias a comparar, descripción).
_BLOCK_AXIS = {
    "H1": "nonstationarity_amp",
    "H2": "n_type1",
    "H3": "station_coverage",
    "H5": "years",
}

_BLOCK_HYPOTHESIS = {
    "H1": "ratio coste(P2.2) / coste(P1) crece con sigma_op",
    "H2": "GNN-P1 / Seq2Seq decrece (GNN gana más) con n_type1",
    "H3": "GNN-best / Seq2Seq decrece con station_coverage menor",
    "H5": "GNN-best / Seq2Seq decrece (GNN gana más) con menos datos",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bootstrap_slope(x: np.ndarray, y: np.ndarray, n_boot: int = 10_000,
                     seed: int = 0) -> Tuple[float, float, float, float]:
    """Pendiente OLS + CI 95 % bootstrap + R² + p-valor empírico."""
    rng = np.random.default_rng(seed)
    n = len(x)
    if n < 3:
        return float("nan"), float("nan"), float("nan"), float("nan")

    def slope(xx, yy):
        sxx = np.var(xx)
        if sxx == 0:
            return 0.0
        return float(np.cov(xx, yy, ddof=0)[0, 1] / sxx)

    obs = slope(x, y)
    samples = np.zeros(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        samples[b] = slope(x[idx], y[idx])
    lo, hi = np.quantile(samples, [0.025, 0.975])
    # R² del ajuste original
    yhat = obs * (x - x.mean()) + y.mean()
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    # p-valor empírico (one-sided, contra el signo observado)
    if obs >= 0:
        p_emp = float(np.mean(samples <= 0))
    else:
        p_emp = float(np.mean(samples >= 0))
    return obs, lo, hi, r2, p_emp


def _winners_per_seed(df: pd.DataFrame, families: List[str], group_cols: List[str]):
    """De `df` (rows = todas las variantes ganadoras por config), devuelve un
    DataFrame con una fila por (config, familia) tomando la variante con menor
    coste seguro."""
    df = df[df["familia"].isin(families)].copy()
    df["coste_total"] = pd.to_numeric(df["coste_total"], errors="coerce")
    # `winners_by_family.csv` ya da una fila por (config, familia); aquí
    # sólo nos aseguramos de quedarnos con esa fila.
    return df.groupby(group_cols + ["familia"], as_index=False)["coste_total"].min()


def _ratio(df: pd.DataFrame, num_family: str, den_family: str,
           level_col: str, group_cols: List[str]) -> pd.DataFrame:
    """Por cada combinación de `group_cols`, calcula
    coste(num_family) / coste(den_family) y lo etiqueta con `level_col`."""
    pivot = df.pivot_table(index=group_cols + [level_col], columns="familia",
                           values="coste_total", aggfunc="min").reset_index()
    if num_family not in pivot.columns or den_family not in pivot.columns:
        raise SystemExit(f"Faltan columnas en el pivot: {pivot.columns.tolist()}.")
    pivot["ratio"] = pivot[num_family] / pivot[den_family]
    pivot["log_ratio"] = np.log(pivot["ratio"])
    return pivot.dropna(subset=["ratio"])


# ---------------------------------------------------------------------------
# Análisis por bloque
# ---------------------------------------------------------------------------


def analyze_block(df_block: pd.DataFrame, block: str, output: Path,
                  do_plot: bool):
    axis = _BLOCK_AXIS[block]
    print(f"\n=== Bloque {block}: {_BLOCK_HYPOTHESIS[block]} ===")

    if block == "H1":
        ratios = _ratio(df_block, "gnn-fase2.2", "gnn-fase1",
                        level_col=axis, group_cols=["seed"])
        ratios["x"] = ratios[axis]
    elif block in ("H2", "H3", "H5"):
        # Identifica la mejor familia GNN por config (menor coste entre P1/2.1/2.2).
        gnn_fams = ["gnn-fase1", "gnn-fase2.1", "gnn-fase2.2"]
        gnn_only = df_block[df_block["familia"].isin(gnn_fams)].copy()
        gnn_only["coste_total"] = pd.to_numeric(gnn_only["coste_total"], errors="coerce")
        gnn_best = (gnn_only.groupby(["seed", axis], as_index=False)["coste_total"].min()
                    .assign(familia="gnn-best"))
        seq = df_block[df_block["familia"] == "baseline"].copy()
        seq["coste_total"] = pd.to_numeric(seq["coste_total"], errors="coerce")
        merged = pd.merge(seq[["seed", axis, "coste_total"]],
                          gnn_best[["seed", axis, "coste_total"]],
                          on=["seed", axis], suffixes=("_seq", "_gnn"))
        merged["ratio"] = merged["coste_total_seq"] / merged["coste_total_gnn"]
        merged["log_ratio"] = np.log(merged["ratio"])
        merged["x"] = merged[axis]
        ratios = merged
    else:
        raise SystemExit(f"Bloque desconocido: {block}")

    if ratios.empty:
        print(f"[!!] sin datos para {block}.")
        return

    x = ratios["x"].to_numpy(dtype=float)
    y = ratios["log_ratio"].to_numpy(dtype=float)
    slope, lo, hi, r2, p_emp = _bootstrap_slope(x, y)
    print(f"  pendiente OLS log-ratio vs {axis}: {slope:+.3f}")
    print(f"  bootstrap 95 % CI:                 [{lo:+.3f}, {hi:+.3f}]")
    print(f"  R²:                                {r2:.3f}")
    print(f"  p-valor empírico (signo):          {p_emp:.4f}")
    sign_ok = (lo > 0 and slope > 0) or (hi < 0 and slope < 0)
    print(f"  conclusión:                        "
          f"{'efecto SIGNIFICATIVO al 95 %' if sign_ok else 'efecto NO significativo'}")

    output.mkdir(parents=True, exist_ok=True)
    out_csv = output / f"analysis_{block}.csv"
    ratios.to_csv(out_csv, index=False)
    print(f"  → {out_csv}")

    if do_plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(x, y, alpha=0.6)
        xline = np.linspace(x.min(), x.max(), 50)
        yline = slope * (xline - x.mean()) + y.mean()
        ax.plot(xline, yline, "r-", lw=2,
                label=f"slope={slope:+.3f}\nCI 95% = [{lo:+.3f}, {hi:+.3f}]")
        ax.axhline(0, color="gray", linestyle="--", lw=0.7)
        ax.set_xlabel(axis)
        ax.set_ylabel("log(coste num / coste den)")
        ax.set_title(f"Bloque {block}: {_BLOCK_HYPOTHESIS[block]}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plot_path = output / f"analysis_{block}.png"
        fig.savefig(plot_path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"  → {plot_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--summary", required=True, type=Path,
                   help="Ruta a sweep_summary.csv producido por run_synth_sweep.py.")
    p.add_argument("--output", type=Path, default=Path("../sweep-analysis"))
    p.add_argument("--bloques", nargs="+", default=None,
                   help="Subconjunto de bloques (default: todos los presentes).")
    p.add_argument("--plot", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.summary)
    if "block" not in df.columns:
        raise SystemExit("El summary no tiene columna 'block'.")
    bloques = args.bloques or sorted(df["block"].unique())
    for b in bloques:
        if b not in _BLOCK_AXIS:
            print(f"[skip] bloque desconocido: {b}")
            continue
        analyze_block(df[df["block"] == b].copy(), b, args.output, args.plot)


if __name__ == "__main__":
    main()
