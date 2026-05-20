"""Resumen del grid `phase22_acyclic`.

Lee los 8 checkpoints (synth-N16/N64 × M={3,6} × {dense, acyclic}) y
sus correspondientes `physicalization_metrics.json`, produce una tabla
CSV con las métricas que necesita el paper:

  * NSE-outlet del modelo (de la salida del entrenamiento).
  * back-flow share del grafo aprendido (de physicalize).
  * n_kept tras physicalize (cuántos embalses son físicamente realizables).
  * NSE-rel tras physicalize (cuánto se preserva el caudal al físicalizar).

La tabla resultante responde directamente a la pregunta:
"¿restringir el grafo a topologías acíclicas mejora la identificabilidad
sin degradar el comportamiento operacional?"
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


_TAG_RE = re.compile(r"^(synth-N\d+)-M(\d+)-(dense|acyclic)$")


def _parse():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--grid-root", type=Path, default=Path("outputs/phase22_grid"))
    p.add_argument("--output", type=Path, default=Path("outputs/phase22_grid/grid_summary.csv"))
    return p.parse_args()


def _read_last_loss(log_file: Path) -> dict:
    """Intenta extraer NSE, RMSE, loss del log del entrenamiento.

    El run_gnn.py imprime el historico final como `[modelo] última época:
    {...}`. Hacemos un parseo best-effort.
    """
    out = {}
    if not log_file.exists():
        return out
    txt = log_file.read_text()
    m = re.search(r"ú?ltima [ée]poca:\s*(.+)$", txt, re.MULTILINE)
    if m:
        try:
            txt_rest = m.group(1).strip()
            # heurística: si parece dict-like, eval safe via literal_eval
            import ast
            d = ast.literal_eval(txt_rest)
            if isinstance(d, dict):
                out.update({f"train_{k}": v for k, v in d.items()})
        except Exception:
            pass
    return out


def main():
    args = _parse()
    root = args.grid_root
    rows = []
    tags = []
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        if d.name.endswith("-phys"):
            continue
        m = _TAG_RE.match(d.name)
        if not m:
            continue
        tags.append(d.name)
    print(f"[summary] {len(tags)} checkpoints detectados: {' '.join(tags)}")

    for tag in tags:
        m = _TAG_RE.match(tag)
        dataset, M_str, mode = m.group(1), m.group(2), m.group(3)
        M = int(M_str)
        ckpt = root / tag
        phys_dir = root / f"{tag}-phys"
        phys_json = phys_dir / "physicalization_metrics.json"
        log_file = root.parent / "_logs" / f"phase22_grid-{tag}.log"

        row = {
            "dataset": dataset,
            "M_latent": M,
            "mode": mode,
            "checkpoint_exists": (ckpt / "core.pt").exists(),
            "phys_exists": phys_json.exists(),
        }
        row.update(_read_last_loss(log_file))

        if phys_json.exists():
            with open(phys_json) as fh:
                pd_ = json.load(fh)
            for k in ("n_reservoirs_active", "n_reservoirs_kept",
                       "n_reservoirs_discarded", "n_e12_old", "n_e21_old",
                       "n_e12_new", "n_e21_new", "n_e21_kept_downstream",
                       "n_e21_collapsed_fallback",
                       "backflow_share_original", "downstream_share_original",
                       "verify_nse_relative", "verify_rmse",
                       "verify_pearson_correlation",
                       "verify_diff_max_abs"):
                if k in pd_:
                    row[k] = pd_[k]
        rows.append(row)

    if rows:
        keys = sorted({k for r in rows for k in r.keys()})
        # ordena con dataset/M/mode al frente
        head = ["dataset", "M_latent", "mode"]
        keys = head + [k for k in keys if k not in head]
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print(f"[summary] {len(rows)} filas → {args.output}")

    print()
    print("=" * 110)
    print(f"{'dataset':<10}  {'M':>2}  {'mode':<8}  "
          f"{'backflow':>10}  {'n_kept':>8}  "
          f"{'NSE-rel':>10}  {'Pearson':>8}  {'|E12|new':>9}  {'|E21|new':>9}")
    print("-" * 110)
    rows.sort(key=lambda r: (r["dataset"], r["M_latent"], r["mode"]))
    for r in rows:
        bf = r.get("backflow_share_original", float("nan"))
        nk = r.get("n_reservoirs_kept", "-")
        na = r.get("n_reservoirs_active", "-")
        nr = r.get("verify_nse_relative", float("nan"))
        pr = r.get("verify_pearson_correlation", float("nan"))
        e12 = r.get("n_e12_new", "-")
        e21 = r.get("n_e21_new", "-")
        print(f"{r['dataset']:<10}  {r['M_latent']:>2}  {r['mode']:<8}  "
              f"{bf if isinstance(bf, float) else 'n.m.':>10}  "
              f"{str(nk)+'/'+str(na):>8}  "
              f"{nr if isinstance(nr, float) else 'n.m.':>10}  "
              f"{pr if isinstance(pr, float) else 'n.m.':>8}  "
              f"{e12!s:>9}  {e21!s:>9}")
    print("=" * 110)


if __name__ == "__main__":
    main()
