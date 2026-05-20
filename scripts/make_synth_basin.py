"""Genera una cuenca sintética bajo demanda y escribe los CSVs en formato del modelo.

Pensado para crear los datasets N=8 y N=64 que la sección 3 del paper
necesita además del N=16 (`datos-synth/full`). Usa `random_basin` con
los mismos parámetros climáticos que `example_basin.yaml` para que la
distribución de pluviosidad sea comparable entre tamaños.

Llamada típica:

    python make_synth_basin.py --n-type1 8  --branching 4.0 \\
        --seed 0 --output ../synth-N8
    python make_synth_basin.py --n-type1 64 --branching 1.5 \\
        --seed 0 --output ../synth-N64

Los CSVs y `manifest.yaml` quedan en `<output>/full/`, listos para ser
consumidos por `run_vae_experiment.py --directorio-datos <output>/full`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from synth_simulator import (
    random_basin, generate_rainfall, simulate_hydrology, write_visibility_outputs,
)
from synth_simulator.config import ClimateConfig, VisibilityConfig


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-type1", type=int, required=True)
    p.add_argument("--branching", type=float, default=2.5)
    p.add_argument("--n-reservoirs", type=int, default=3)
    p.add_argument("--reservoir-strategy", default="headwater",
                   choices=["headwater", "midstream", "scattered", "random"])
    p.add_argument("--station-coverage", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--firma", default=None)
    p.add_argument("--start-date", default="2010-01-01")
    p.add_argument("--end-date", default="2024-12-31")
    return p.parse_args()


def main():
    args = parse_args()
    firma = args.firma or f"SYNTH-N{args.n_type1}"
    cfg = random_basin(
        n_type1=args.n_type1,
        branching_factor=args.branching,
        n_reservoirs=args.n_reservoirs,
        reservoir_strategy=args.reservoir_strategy,
        station_coverage=args.station_coverage,
        output_dir=str(args.output),
        firma=firma,
        seed=args.seed,
        rainfall_climate=ClimateConfig(
            start_date=args.start_date, end_date=args.end_date, seed=args.seed,
        ),
        visibility=[VisibilityConfig(name="full")],
    )
    station_ids = [n.rain_station for n in cfg.nodes if n.rain_station]
    print(f"[gen] {cfg.name}")
    print(f"      nodos Tipo-1 : {len(cfg.nodes)}")
    print(f"      embalses     : {len(cfg.reservoirs)}")
    print(f"      estaciones   : {len(station_ids)} de {sum(1 for n in cfg.nodes if not [e for e in cfg.edges_11 if e.src == n.id] and not any(r.inflow_from == n.id for r in cfg.reservoirs))}")
    rainfall = generate_rainfall(cfg.climate, station_ids)
    sim = simulate_hydrology(cfg, rainfall)
    print(f"[sim] caudal medio aforo : {sim.flow[cfg.outlet().id].mean():.2f} m³/s")
    print(f"      Q_min calibrado    : {cfg.caudal_minimo_m3s:.2f} m³/s")
    print(f"      días alarma reales : {(sim.flow[cfg.outlet().id] < cfg.caudal_minimo_m3s).sum()} de {len(sim.flow)}")
    write_visibility_outputs(cfg, rainfall, sim)
    print(f"[gen] CSVs + manifest → {args.output}/full/")


if __name__ == "__main__":
    main()
