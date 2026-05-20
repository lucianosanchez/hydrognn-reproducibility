"""Genera la figura comparativa: topología real vs aprendida.

Funciona con cualquier checkpoint del proyecto (Fase 2.2 del HydroGNN
baseline, UA-HydroGNN, etc.). Lo único que importa para que el plot
tenga contenido es que el checkpoint haya sido entrenado con
*grafo de candidatos densos* (la API de `analyze_positions()` sólo
distingue posiciones si el modelo tenía libertad para colocarlas).

Llamadas típicas:

    # Synth-N16, baseline HydroGNN Phase 2.2 (gates="none", logw12_init=-3)
    python plot_learned_vs_truth.py \\
        --ckpt outputs/uagnn-synth-N16/modelo_uagnn \\
        --basin-dir datos-synth/full --firma SYNTH001 \\
        --output figs/topology_synth_N16.pdf

    # Synth-N64
    python plot_learned_vs_truth.py \\
        --ckpt outputs/uagnn-synth-N64/modelo_uagnn \\
        --basin-dir datos-synth-N64/full --firma SYNTH-N64 \\
        --output figs/topology_synth_N64.pdf

Si quieres incluir una figura con `figs/topology_synth_N16.{pdf,png}`
en el paper, asegúrate de descomentar la `\\includegraphics` que
añade este mismo módulo en `paper_methods.tex`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from seq2seq_runoff.basins import synth_basin, synth_graph_full
from seq2seq_runoff.gnn.graph import BasinGraph
from seq2seq_runoff.gnn.viz import plot_comparison, topology_recovery_metrics


def _parse():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", type=Path, required=True,
                   help="Directorio del checkpoint (con ua_core.pt o core.pt + meta.pkl).")
    p.add_argument("--basin-dir", type=Path, required=True,
                   help="Directorio de datos del basin (con manifest.yaml).")
    p.add_argument("--firma", type=str, required=True)
    p.add_argument("--output", type=Path, required=True,
                   help="Fichero PDF/PNG/SVG de salida.")
    p.add_argument("--threshold", type=float, default=0.10,
                   help="Umbral de share para considerar una arista E_12/E_21 activa.")
    p.add_argument("--title-suffix", default="",
                   help="Cadena que se añade al suptitle.")
    return p.parse_args()


def _load_any_core(ckpt: Path):
    """Carga un core HydroGNN, sea del baseline (core.pt + meta.pkl) o
    del UA-HydroGNN (ua_core.pt + ua_meta.pkl). Devuelve `core` (instancia
    de HydroGNNCore) y `graph_base` si está disponible (None en otro caso)."""
    import torch
    import pickle

    if (ckpt / "ua_core.pt").exists():
        # UA-HydroGNN: hereda HydroGNNCore como atributo .core
        from seq2seq_runoff.ua_gnn import UAHydroGNNModel
        model = UAHydroGNNModel.load(ckpt)
        return model.core.core, getattr(model, "graph", None)
    if (ckpt / "core.pt").exists():
        # HydroGNN baseline
        with open(ckpt / "meta.pkl", "rb") as fh:
            meta = pickle.load(fh)
        from seq2seq_runoff.gnn.core import HydroGNNCore
        graph = meta["graph"]
        core = HydroGNNCore(
            graph,
            use_gates=getattr(meta["cfg"], "use_gates", "none"),
            node_static_dim=meta["cfg"].node_static_dim,
            ctx_dim=meta["cfg"].ctx_dim,
            hidden=meta["cfg"].hidden,
            logw12_init=getattr(meta["cfg"], "logw12_init", 0.0),
        )
        core.load_state_dict(torch.load(ckpt / "core.pt", map_location="cpu"))
        return core, graph
    raise FileNotFoundError(
        f"No encuentro core.pt ni ua_core.pt en {ckpt}; ¿es el checkpoint correcto?"
    )


def main():
    args = _parse()
    print(f"[viz] cargando checkpoint {args.ckpt}")
    core, _ = _load_any_core(args.ckpt)

    print(f"[viz] reconstruyendo grafo 'truth' desde {args.basin_dir}")
    truth = synth_graph_full(args.basin_dir)

    # El core puede haber sido entrenado sobre un grafo de candidatos
    # densos (Phase 2.2) con M = M_latent embalses formales; el truth
    # tiene M = nº real de embalses. La función `topology_recovery_metrics`
    # mapea aprendido → real con un greedy IoU sobre inflow_share.
    print(f"[viz] truth: N1={truth.N1}  M={truth.M}  E11={truth.E11}  "
          f"E12={truth.E12}  E21={truth.E21}")

    suptitle = f"{args.basin_dir.parent.name} ({args.firma})"
    if args.title_suffix:
        suptitle = f"{suptitle} — {args.title_suffix}"

    metrics = plot_comparison(
        truth, core, args.output,
        threshold=args.threshold,
        title_left=f"Ground-truth ({truth.M} embalses)",
        title_right=f"Learned (z_res/share>{args.threshold:g})",
        suptitle=suptitle,
    )
    print(f"[viz] figura → {args.output}")
    print("[viz] métricas de recuperación:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"   {k:32s}  {v:.3f}")
        else:
            print(f"   {k:32s}  {v}")


if __name__ == "__main__":
    main()
