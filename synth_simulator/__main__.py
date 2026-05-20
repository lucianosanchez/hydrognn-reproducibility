"""CLI: `python -m synth_simulator path/to/basin.yaml`."""

from __future__ import annotations

import argparse
from pathlib import Path

from .climate import generate_rainfall
from .config import load_basin_config
from .hydro import simulate_hydrology
from .output import write_visibility_outputs


def main() -> None:
    p = argparse.ArgumentParser(description="Simulador hidrográfico sintético.")
    p.add_argument("yaml", type=Path, help="Fichero de configuración de la cuenca.")
    p.add_argument("--plot", type=Path, default=None,
                   help="Si se especifica, guarda en ese directorio: topology.png, "
                        "rainfall.png, reservoirs.png, outlet_flow.png, summary.png.")
    args = p.parse_args()

    cfg = load_basin_config(args.yaml)
    print(f"[sim] cuenca={cfg.name}, periodo={cfg.climate.start_date} → {cfg.climate.end_date}")

    station_ids = [n.rain_station for n in cfg.nodes if n.rain_station]
    rainfall = generate_rainfall(cfg.climate, station_ids)
    print(f"[sim] generada lluvia: {rainfall.shape[0]} días, {rainfall.shape[1]} estaciones")

    sim = simulate_hydrology(cfg, rainfall)
    print(f"[sim] caudal medio en aforo {cfg.outlet().flow_station}: "
          f"{sim.flow[cfg.outlet().id].mean():.2f} m³/s")
    print(f"[sim] embalse total medio: {sim.storage.sum(axis=1).mean():.1f} Hm³")

    write_visibility_outputs(cfg, rainfall, sim)
    base = Path(cfg.output_directory)
    print(f"[sim] salidas escritas en {base}")
    for vis in cfg.output_configurations:
        print(f"    └── {vis.name}/  ({vis.visible_reservoirs} embalses, "
              f"{vis.visible_rain_stations} estaciones)")

    if args.plot is not None:
        from . import viz
        viz.save_all(cfg, rainfall, sim, args.plot)
        print(f"[sim] plots escritos en {args.plot}/")


if __name__ == "__main__":
    main()
