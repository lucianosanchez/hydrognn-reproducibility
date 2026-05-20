"""Post-procesado: convierte el grafo aprendido en uno físicamente coherente.

Carga un checkpoint Phase 2.2 entrenado con `dense_candidate_graph`, llama
a `physicalize_topology` para reorganizar las aristas E_12/E_21 buscando
una topología sin flujos hacia arriba ni bucles, transfiere los pesos
aprendidos al nuevo grafo (sin reentrenar), verifica empíricamente que
el caudal del outlet apenas cambia, y produce:

  * outputs/<tag>/physicalized_graph.pkl
      → BasinGraph nuevo + plan de transferencia (serializado).
  * outputs/<tag>/physicalization_metrics.json
      → métricas de calidad (% backflow original, ΔNSE, ΔRMSE…).
  * figs/topology_<tag>_physicalized.pdf
      → figura side-by-side ground-truth vs aprendido vs físicalizado.

Llamadas típicas:

    python scripts/physicalize_topology.py \\
        --ckpt outputs/hydrognn-phase22-synth-N16 \\
        --basin-dir datos-synth/full --firma SYNTH001 \\
        --output-dir outputs/physicalized-synth-N16 \\
        --fig figs/topology_synth-N16_physicalized.pdf

    python scripts/physicalize_topology.py \\
        --ckpt outputs/hydrognn-phase22-synth-N64 \\
        --basin-dir datos-synth-N64/full --firma SYNTH-N64 \\
        --output-dir outputs/physicalized-synth-N64 \\
        --fig figs/topology_synth-N64_physicalized.pdf
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from seq2seq_runoff.basins import (
    synth_basin, synth_graph_full, ebro_basin, ebro_graph,
)
from seq2seq_runoff.data import load_basin_dataframe, scale_to_unit
from seq2seq_runoff.gnn.core import HydroGNNCore
from seq2seq_runoff.gnn.dataset import build_training_dataset
from seq2seq_runoff.gnn.physicalize import (
    physicalize_topology, transfer_core_weights, verify_equivalence,
)
from seq2seq_runoff.gnn.viz import (
    plot_basin_graph, plot_learned_topology, topology_recovery_metrics,
)


def _load_core(ckpt: Path):
    """Carga HydroGNN baseline (core.pt + meta.pkl) o UA-HydroGNN
    (ua_core.pt + ua_meta.pkl). Devuelve (core, cfg, graph_used_in_training)."""
    if (ckpt / "ua_core.pt").exists():
        from seq2seq_runoff.ua_gnn import UAHydroGNNModel
        model = UAHydroGNNModel.load(ckpt)
        return model.core.core, model.cfg, model.graph
    with open(ckpt / "meta.pkl", "rb") as fh:
        meta = pickle.load(fh)
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
    core.eval()
    return core, meta["cfg"], graph


def _parse():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--basin-dir", type=Path, required=True,
                   help="Directorio del dataset (con manifest.yaml en synth, "
                        "o con los CSV en Ebro).")
    p.add_argument("--firma", required=True)
    p.add_argument("--basin-type", choices=["synth", "ebro"], default="synth")
    p.add_argument("--threshold-in", type=float, default=0.10)
    p.add_argument("--threshold-out", type=float, default=0.10)
    p.add_argument("--mode", choices=["strict", "soft"], default="strict",
                   help="strict: descarta E_21 backflow. soft: las reasigna "
                        "al canonical destination (preservando masa total).")
    p.add_argument("--n-windows", type=int, default=50,
                   help="Cuántas ventanas usar para la verificación empírica.")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--fig", type=Path, default=None,
                   help="Si se da, genera figura 3-paneles ground/learned/physical.")
    return p.parse_args()


def main():
    import matplotlib.pyplot as plt
    args = _parse()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[phys] cargando checkpoint {args.ckpt}")
    old_core, cfg, train_graph = _load_core(args.ckpt)

    print(f"[phys] cargando grafo físico de referencia (tipo={args.basin_type})")
    if args.basin_type == "synth":
        physical_ref = synth_graph_full(args.basin_dir)
    else:
        physical_ref = ebro_graph()

    # ----- Validación: N1 del checkpoint y del grafo de referencia deben coincidir
    if old_core.N1 != physical_ref.N1:
        raise SystemExit(
            f"[phys] MISMATCH: el checkpoint tiene N1={old_core.N1} Type-1 nodes "
            f"pero el grafo físico de referencia tiene N1={physical_ref.N1}. "
            f"Probablemente has cruzado un checkpoint de una cuenca con el "
            f"basin-dir de otra. Sugerencia:\n"
            f"  * si --ckpt es hydrognn-phase22-synth-N16, usa --basin-dir datos-synth/full\n"
            f"  * si --ckpt es hydrognn-phase22-synth-N64, usa --basin-dir datos-synth-N64/full\n"
            f"  * si --ckpt es uagnn-ebro*,                usa --basin-dir datos-06-07-2023 --basin-type ebro"
        )
    print(f"       N1={physical_ref.N1}  M={physical_ref.M}  "
          f"E11={physical_ref.E11}  E12={physical_ref.E12}  E21={physical_ref.E21}")

    print(f"[phys] reorganizando topología (τ_in={args.threshold_in}, "
          f"τ_out={args.threshold_out})")
    plan = physicalize_topology(
        old_core, physical_ref,
        threshold_in=args.threshold_in,
        threshold_out=args.threshold_out,
        mode=args.mode,
    )
    new_graph = plan.new_graph
    print(f"[phys] new graph: N1={new_graph.N1}  M={new_graph.M}  "
          f"E12={new_graph.E12}  E21={new_graph.E21}")
    for k, v in plan.metrics.items():
        print(f"       {k:35s}  {v}")

    # Caso degenerado: ningún embalse activo en el plan.
    if new_graph.M == 0:
        # Persistimos las métricas y salimos sin construir el nuevo core
        # (no tendría sentido: M=0 significa que NINGÚN embalse formal del
        # checkpoint admitió destino canónico, e.g. porque todos los
        # source-sets caen en nodos sin descendientes en el grafo físico).
        out_json = args.output_dir / "physicalization_metrics.json"
        with open(out_json, "w") as fh:
            json.dump(plan.metrics, fh, indent=2)
        print(f"[phys] (AVISO) no hay embalses activos físicamente "
              f"realizables — métricas → {out_json}")
        print(f"       Causas posibles: threshold (--threshold-in/out) "
              f"demasiado alto, modelo poco entrenado, o todos los "
              f"source-sets son outlets sin descendientes.")
        return

    print("[phys] transfiriendo pesos al nuevo core")
    new_core = transfer_core_weights(old_core, plan)

    # ----- Verificación empírica -----
    print(f"[phys] verificando equivalencia sobre {args.n_windows} ventanas")
    basin = synth_basin(args.basin_dir) if args.basin_type == "synth" else ebro_basin()
    df = load_basin_dataframe(basin, args.basin_dir, args.firma)
    df_scaled, maximos = scale_to_unit(df)
    H, T = cfg.historia, cfg.horizonte
    windows = list(build_training_dataset(
        df_scaled, train_graph,
        H=H, T=T,
        flow_column=cfg.basin.flow_column,
        observed_stations=getattr(cfg, "observed_stations", None),
    ))
    rng = np.random.default_rng(0)
    if len(windows) > args.n_windows:
        idx = rng.choice(len(windows), size=args.n_windows, replace=False)
        windows = [windows[int(i)] for i in idx]
    rain = torch.stack([w.rain for w in windows])
    mask = torch.stack([w.mask for w in windows])
    ctx = torch.stack([w.ctx for w in windows])

    eq = verify_equivalence(old_core, new_core, rain, mask, ctx, H=H, T=T)
    print("[phys] equivalencia empírica:")
    for k, v in eq.items():
        if isinstance(v, float):
            print(f"       {k:25s}  {v:.6f}")
        else:
            print(f"       {k:25s}  {v}")

    # ----- Persistencia -----
    out_pkl = args.output_dir / "physicalized_graph.pkl"
    with open(out_pkl, "wb") as fh:
        pickle.dump({"plan": plan, "verify": eq}, fh)
    print(f"[phys] grafo + plan → {out_pkl}")

    out_json = args.output_dir / "physicalization_metrics.json"
    payload = {**plan.metrics, **{f"verify_{k}": v for k, v in eq.items()}}
    with open(out_json, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"[phys] métricas → {out_json}")

    # ----- Figura 3 paneles ----------
    if args.fig is not None:
        print(f"[phys] generando figura {args.fig}")
        fig, axes = plt.subplots(1, 3, figsize=(20, 6.5))
        plot_basin_graph(physical_ref, ax=axes[0],
                          title=f"Ground truth  ({physical_ref.M} reservoirs)")
        plot_learned_topology(old_core, ax=axes[1],
                               title=f"Learned Phase 2.2  "
                                      f"(M_lat={old_core.M})")
        # Para el panel 3, construimos un grafo "etiquetado" como físicalizado
        # y lo dibujamos con la misma rutina genérica de plot_basin_graph,
        # marcando los embalses como latentes (color naranja).
        plot_basin_graph(new_graph, ax=axes[2],
                          title=f"Physicalised  ({new_graph.M} reservoirs, "
                                 f"backflow share={plan.metrics.get('backflow_share_original',0.0):.2f})",
                          latent_res_idx=set(range(new_graph.M)))
        suptitle = (
            f"Topology recovery + post-hoc physicalisation ({args.mode} mode)"
            f"  |  backflow={plan.metrics.get('backflow_share_original',0.0):.2f}"
            f"  |  NSE_rel={eq['nse_relative']:.3f}"
            f"  |  RMSE={eq['rmse']:.4f}"
            f"  |  Pearson={eq['pearson_correlation']:.3f}"
        )
        fig.suptitle(suptitle, fontsize=10)
        args.fig.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout(rect=(0, 0, 1, 0.94))
        fig.savefig(args.fig, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[phys] figura → {args.fig}")


if __name__ == "__main__":
    main()
