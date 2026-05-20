"""Comparativa de varios modelos sobre el mismo problema.

Lee los `predictions.csv` que `run_baseline.py` y `run_gnn.py` generan
cuando se les pasa `--plot DIR`, y produce:

    1. Una tabla comparativa con tres puntos de operación por modelo
       (natural δ=0, óptimo cost-aware, seguro con FN ≤ max_fn).
    2. Por familia (baseline, gnn, …): selecciona la variante con menor
       coste seguro — éste es el comparativo justo entre métodos.
    3. Un barplot que destaca FN (la métrica crítica) y coste total.
    4. CSVs con la tabla larga y los ganadores por familia.

Etiquetas de los `--predictions`:

    * `familia=path`               — sólo una variante; familia = etiqueta.
    * `familia:variante=path`      — múltiples variantes; el script elige la
                                      mejor variante por familia bajo la
                                      restricción `FN ≤ max_fn`.

Llamada típica para comparar el ÓPTIMO de cada familia:

    python scripts/compare_models.py \\
        --predictions baseline:d1=../sweep/baseline_d1 \\
                      baseline:d2=../sweep/baseline_d2 \\
                      baseline:d10=../sweep/baseline_d10 \\
                      gnn:k5=../sweep/gnn_k5 \\
                      gnn:k30=../sweep/gnn_k30 \\
                      gnn:k100=../sweep/gnn_k100 \\
        --q-min 20 --coste-falsa-alarma 1 --coste-omision 100 --max-fn 0 \\
        --output ../figs-compare-synth-full
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Permite ejecutar el script directamente sin instalar el paquete.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from seq2seq_runoff.evaluation import compare_models_at_operating_points
from seq2seq_runoff.plotting import plot_model_comparison


def _parse_predictions_arg(items):
    """`baseline=path1 gnn=path2` → {"baseline": Path1, "gnn": Path2}."""
    out = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--predictions espera 'label=path', recibido {item!r}")
        label, p = item.split("=", 1)
        out[label] = Path(p)
    return out


def _load_predictions(label_to_path):
    """`{label: dir_or_csv}` → `{label: (obs, pred)}` arrays NumPy."""
    out = {}
    for label, path in label_to_path.items():
        csv = path if path.is_file() else path / "predictions.csv"
        if not csv.exists():
            raise SystemExit(f"No existe {csv}. ¿Has ejecutado run_*.py con --plot?")
        df = pd.read_csv(csv)
        if "obs" not in df.columns or "pred" not in df.columns:
            raise SystemExit(f"{csv} debe tener columnas 'obs' y 'pred'.")
        out[label] = (df["obs"].to_numpy(), df["pred"].to_numpy())
    return out


def parse_args():
    p = argparse.ArgumentParser(description="Compara modelos en tres puntos de operación.")
    p.add_argument("--predictions", nargs="+", required=True,
                   help="Pares 'etiqueta=ruta' donde la ruta es un directorio que "
                        "contiene predictions.csv o el propio CSV.")
    p.add_argument("--q-min", type=float, required=True, help="Umbral Q_min (m³/s).")
    p.add_argument("--coste-falsa-alarma", type=float, default=1.0)
    p.add_argument("--coste-omision", type=float, default=100.0)
    p.add_argument("--max-fn", type=int, default=0,
                   help="Restricción dura para el operating point seguro.")
    p.add_argument("--output", type=Path, default=None,
                   help="Si se especifica, guarda comparison.csv y comparison.png "
                        "en ese directorio.")
    return p.parse_args()


def _split_family(label: str):
    """Devuelve (familia, variante). 'baseline:d10' → ('baseline', 'd10')."""
    if ":" in label:
        familia, variante = label.split(":", 1)
        return familia, variante
    return label, label


def _winners_by_family(df: pd.DataFrame, max_fn: int) -> pd.DataFrame:
    """Selecciona la variante con menor coste seguro por cada familia.

    "Coste seguro" = coste_total al operating_point 'safe' (FN ≤ max_fn).
    """
    safe = df[df["operating_point"] == "safe"].copy()
    safe[["familia", "variante"]] = pd.DataFrame(
        safe["modelo"].map(_split_family).tolist(), index=safe.index
    )
    # Para cada familia, fila con coste mínimo (priorizando factibles).
    safe["_factible_int"] = safe["factible"].astype(int)
    safe = safe.sort_values(
        ["familia", "_factible_int", "coste_total"], ascending=[True, False, True]
    )
    ganadores = safe.groupby("familia").head(1).drop(columns="_factible_int")
    return ganadores.reset_index(drop=True)


def _format_table(df: pd.DataFrame) -> str:
    """Formato monoespaciado legible en consola."""
    cols = ["modelo", "operating_point", "delta", "fn", "fp", "coste_total",
            "precision", "recall", "f1", "nse"]
    sub = df[cols].copy()
    sub["delta"] = sub["delta"].map(lambda v: f"{v:+.2f}")
    sub["coste_total"] = sub["coste_total"].map(lambda v: f"{v:>10.0f}")
    for c in ("precision", "recall", "f1", "nse"):
        sub[c] = sub[c].map(lambda v: f"{v:.3f}")
    return sub.to_string(index=False)


def main() -> None:
    args = parse_args()
    paths = _parse_predictions_arg(args.predictions)
    predicciones = _load_predictions(paths)
    print(f"[compare] modelos cargados: {list(predicciones)}")

    df = compare_models_at_operating_points(
        predicciones,
        q_min=args.q_min,
        coste_falsa_alarma=args.coste_falsa_alarma,
        coste_omision=args.coste_omision,
        max_fn=args.max_fn,
    )
    df = df.sort_values(["modelo", "operating_point"]).reset_index(drop=True)

    print("\n=== TABLA COMPARATIVA ===")
    print(_format_table(df))

    # Resumen específico para "no parar ningún día":
    print(f"\n=== OPERATING POINT SEGURO (FN ≤ {args.max_fn}) ===")
    seguro = df[df["operating_point"] == "safe"].sort_values("coste_total")
    for _, r in seguro.iterrows():
        flag = "" if r["factible"] else " (NO factible — FN mín posible)"
        print(f"  {r['modelo']:25s}  FN={r['fn']:3d}  FP={r['fp']:5d}  "
              f"coste={r['coste_total']:>10.0f}  δ={r['delta']:+.2f}{flag}")

    # Ganador por familia: la pregunta que de verdad importa.
    winners = _winners_by_family(df, args.max_fn)
    print(f"\n=== GANADOR POR FAMILIA (mínimo coste seguro, FN ≤ {args.max_fn}) ===")
    for _, r in winners.sort_values("coste_total").iterrows():
        flag = "" if r["factible"] else " (NO factible)"
        print(f"  {r['familia']:12s} → variante {r['variante']:12s}  "
              f"FN={r['fn']:3d}  FP={r['fp']:5d}  "
              f"coste={r['coste_total']:>10.0f}  δ={r['delta']:+.2f}{flag}")
    if len(winners) >= 2:
        ordenado = winners.sort_values("coste_total").reset_index(drop=True)
        mejor = ordenado.iloc[0]
        peor = ordenado.iloc[-1]
        if peor["coste_total"] > 0:
            mejora = (peor["coste_total"] - mejor["coste_total"]) / peor["coste_total"] * 100
            print(f"\n  → '{mejor['familia']}' es {mejora:.1f}% más barato que "
                  f"'{peor['familia']}' al óptimo de cada uno.")

    if args.output is not None:
        args.output.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output / "comparison.csv", index=False)
        winners.to_csv(args.output / "winners_by_family.csv", index=False)
        import os
        os.environ.setdefault("MPLBACKEND", "Agg")
        fig = plot_model_comparison(df, coste_omision=args.coste_omision)
        fig.savefig(args.output / "comparison.png", dpi=150, bbox_inches="tight")
        print(f"\n[compare] tabla, ganadores y plot guardados en {args.output}/")


if __name__ == "__main__":
    main()
