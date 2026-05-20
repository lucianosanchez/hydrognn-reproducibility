"""Diagnóstico: dump de las aristas E_21 aprendidas vs reales.

Para responder a la observación "las flechas naranja no tienen el mismo
sentido": vuelca de forma textual lo que el modelo ha aprendido para
las sueltas de embalse, y lo compara con el ground-truth del basin
sintético. Si los IDs de origen y destino son consistentes (origen
∈ embalses, destino ∈ Type-1), entonces lo que se ve en la figura es
geométrico (rutado denso, curvas) y no un bug; si están al revés es
un bug en viz.py.

Uso:
    python scripts/diag_e21_directions.py \\
        --ckpt outputs/hydrognn-phase22-synth-N16 \\
        --basin-dir datos-synth/full
"""

from __future__ import annotations
import argparse, sys, pickle
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from seq2seq_runoff.basins import synth_graph_full
from seq2seq_runoff.gnn.core import HydroGNNCore


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--basin-dir", type=Path, required=True)
    p.add_argument("--threshold", type=float, default=0.10)
    args = p.parse_args()

    # Carga el checkpoint baseline (core.pt + meta.pkl)
    with open(args.ckpt / "meta.pkl", "rb") as fh:
        meta = pickle.load(fh)
    graph_learned = meta["graph"]
    core = HydroGNNCore(
        graph_learned,
        use_gates=getattr(meta["cfg"], "use_gates", "none"),
        node_static_dim=meta["cfg"].node_static_dim,
        ctx_dim=meta["cfg"].ctx_dim,
        hidden=meta["cfg"].hidden,
        logw12_init=getattr(meta["cfg"], "logw12_init", 0.0),
    )
    core.load_state_dict(torch.load(args.ckpt / "core.pt", map_location="cpu"))
    info = core.analyze_positions()
    inflow = np.asarray(info["inflow_share"])    # (M, N1): node → reservoir
    outflow = np.asarray(info["outflow_share"])  # (M, N1): reservoir → node

    truth = synth_graph_full(args.basin_dir)
    type1_names = truth.type1_names

    print(f"\n=== Ground truth ({len(truth.res_names)} reservoirs) ===")
    print(f"   E_12 (Type-1 → reservoir):")
    for e in range(truth.E12):
        s = int(truth.src12[e]); d = int(truth.dst12[e])
        print(f"     {type1_names[s]:>12s}  →  {truth.res_names[d]:>6s}")
    print(f"   E_21 (reservoir → Type-1):")
    for e in range(truth.E21):
        s = int(truth.src21[e]); d = int(truth.dst21[e])
        print(f"     {truth.res_names[s]:>6s}      →  {type1_names[d]:>12s}")

    print(f"\n=== Learned ({inflow.shape[0]} candidate reservoirs, "
          f"threshold={args.threshold}) ===")
    print(f"   In/out totals per reservoir (>= 0.10 considered active):")
    print(f"   {'k':>3s}  {'in_total':>10s}  {'out_total':>10s}  active?")
    for k in range(inflow.shape[0]):
        ti, to = float(inflow[k].sum()), float(outflow[k].sum())
        act = "YES" if (ti > 0.10 and to > 0.10) else "no"
        print(f"   {k:>3d}  {ti:>10.3f}  {to:>10.3f}  {act}")

    print(f"\n   Top-K inflow arcs (Type-1 → reservoir, share >= {args.threshold}):")
    print(f"   {'reservoir':>10s}  {'from Type-1':>14s}  {'share':>7s}")
    arcs_in = []
    for k in range(inflow.shape[0]):
        for j in range(inflow.shape[1]):
            if inflow[k, j] >= args.threshold:
                arcs_in.append((float(inflow[k, j]), k, j))
    for share, k, j in sorted(arcs_in, reverse=True)[:30]:
        print(f"   {('R*' + str(k)):>10s}  {type1_names[j]:>14s}  {share:>7.3f}")

    print(f"\n   Top-K outflow arcs (reservoir → Type-1, share >= {args.threshold}):")
    print(f"   {'reservoir':>10s}  {'to Type-1':>14s}  {'share':>7s}")
    arcs_out = []
    for k in range(outflow.shape[0]):
        for j in range(outflow.shape[1]):
            if outflow[k, j] >= args.threshold:
                arcs_out.append((float(outflow[k, j]), k, j))
    for share, k, j in sorted(arcs_out, reverse=True)[:30]:
        print(f"   {('R*' + str(k)):>10s}  {type1_names[j]:>14s}  {share:>7.3f}")


if __name__ == "__main__":
    main()
