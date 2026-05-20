"""Sweep controlado sobre cuencas sintéticas para los bloques H1-H5.

Cada bloque corresponde a una hipótesis del paper:

    H1  no-estacionariedad     varía nonstationarity_amp ∈ {0, .1, .2, .3, .5}
    H2  profundidad topológica  varía (n_type1, branching) → distintos D
    H3  cobertura de estaciones varía station_coverage ∈ {.25, .5, .75, 1.0}
    H5  tamaño del dataset      varía duración del registro climático

Por defecto se ejecuta H1; añade `--bloques H1 H2 H3 H5` para más.
Cada `(bloque, nivel, seed)` produce:

    out/<bloque>/<config_id>/
        manifest.yaml             topología + visibilidad
        DatosHistoricos_*.csv     CSVs en formato del modelo
        tune-fase1/  tune-fase2.1/  tune-fase2.2/   sweep apples-to-apples
        compare-phases/           tabla cross-phase

Y al final, un fichero global `out/sweep_summary.csv` consolida los winners
de todas las configuraciones en formato largo (una fila por
familia × bloque × nivel × seed) listo para análisis estadístico.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from synth_simulator import (
    random_basin,
    generate_rainfall,
    simulate_hydrology,
    write_visibility_outputs,
)
from synth_simulator.config import ClimateConfig, VisibilityConfig

_HERE = Path(__file__).resolve().parent
_RUN_ALL_PHASES = _HERE / "run_all_phases.py"
_TUNE = _HERE / "tune.py"
_COMPARE = _HERE / "compare_models.py"


# ===========================================================================
# Parametrización de cada bloque.  Cada bloque define:
#   - levels: lista de configuraciones-base (params que cambian en el bloque)
#   - default_params: el resto de parámetros (fijos a sus valores nominales)
# ===========================================================================


_NOMINAL = dict(
    n_type1=16,
    branching_factor=2.5,
    n_reservoirs=3,
    reservoir_strategy="headwater",
    station_coverage=0.5,
    nonstationarity_amp=0.0,
    catchment_total_km2=5000.0,
)


def _block_h1(seeds: List[int]):
    """H1: no-estacionariedad. 5 niveles × N seeds."""
    levels = [
        {"nonstationarity_amp": amp}
        for amp in (0.0, 0.10, 0.20, 0.30, 0.50)
    ]
    for level in levels:
        for seed in seeds:
            params = {**_NOMINAL, **level, "seed": seed}
            yield ("H1", f"sigma{level['nonstationarity_amp']:.2f}_s{seed}", params)


def _block_h2(seeds: List[int]):
    """H2: profundidad topológica. Variamos (n_type1, branching) para barrer D."""
    levels = [
        {"n_type1": 8,  "branching_factor": 4.0},   # ancho, poco profundo
        {"n_type1": 16, "branching_factor": 2.5},   # default
        {"n_type1": 32, "branching_factor": 2.0},
        {"n_type1": 64, "branching_factor": 1.5},   # estrecho, profundo
    ]
    for level in levels:
        for seed in seeds:
            params = {**_NOMINAL, **level, "seed": seed}
            label = f"N{level['n_type1']}_b{level['branching_factor']:.1f}_s{seed}"
            yield ("H2", label, params)


def _block_h3(seeds: List[int]):
    """H3: cobertura de estaciones."""
    for cov in (0.25, 0.50, 0.75, 1.00):
        for seed in seeds:
            params = {**_NOMINAL, "station_coverage": cov, "seed": seed}
            yield ("H3", f"cov{cov:.2f}_s{seed}", params)


def _block_h5(seeds: List[int]):
    """H5: tamaño del dataset (años de simulación)."""
    for years in (2, 5, 10, 15):
        for seed in seeds:
            params = {**_NOMINAL, "seed": seed, "_years": years}
            yield ("H5", f"yr{years}_s{seed}", params)


_BLOCK_FUNCS = {
    "H1": _block_h1,
    "H2": _block_h2,
    "H3": _block_h3,
    "H5": _block_h5,
}


# ===========================================================================
# Generación + simulación + sweep + comparativo por config.
# ===========================================================================


def _years_to_climate(years: int, seed: int) -> ClimateConfig:
    """ClimateConfig con `years` años, terminando en 2024-12-31."""
    end_year = 2024
    start_year = end_year - years + 1
    return ClimateConfig(
        start_date=f"{start_year}-01-01",
        end_date=f"{end_year}-12-31",
        seed=seed,
    )


def _generate_and_simulate(params: Dict, out_dir: Path) -> Dict:
    """Genera la cuenca, simula los datos, escribe los CSVs.

    Devuelve un dict con metadata útil para la fila final del summary.
    """
    years = params.pop("_years", 10)
    seed = params["seed"]
    climate = _years_to_climate(years, seed)

    cfg = random_basin(
        rainfall_climate=climate,
        output_dir=str(out_dir),
        firma=f"SYN{seed}",
        visibility=[VisibilityConfig(name="full")],
        **params,
    )

    station_ids = [n.rain_station for n in cfg.nodes if n.rain_station]
    rainfall = generate_rainfall(cfg.climate, station_ids)
    sim = simulate_hydrology(cfg, rainfall)
    write_visibility_outputs(cfg, rainfall, sim)

    return {
        "n_type1": params.get("n_type1", 16),
        "branching_factor": params.get("branching_factor", 2.5),
        "n_reservoirs": params.get("n_reservoirs", 3),
        "station_coverage": params.get("station_coverage", 0.5),
        "nonstationarity_amp": params.get("nonstationarity_amp", 0.0),
        "years": years,
        "seed": seed,
        "n_stations_observed": len(station_ids),
        "n_alarm_days": int((sim.flow["OUTLET"] < cfg.caudal_minimo_m3s).sum()),
        "n_total_days": len(sim.flow),
        "q_min": cfg.caudal_minimo_m3s,
        "q_mean": float(sim.flow["OUTLET"].mean()),
    }


def _run_all_phases_for_dataset(config_dir: Path, args) -> Path:
    """Llama a run_all_phases.py sobre el dataset acabado de generar.

    Define un dataset on-the-fly: directorio = full/, key = config_id.
    """
    full_dir = config_dir / "full"
    cmd = [
        sys.executable, str(_RUN_ALL_PHASES),
        "--output-base", str(config_dir),
        "--phases", *args.phases,
        # observed_stations para Fase 2.x: lo deja decidir run_all_phases.py
        # según el subset de estaciones del dataset; aquí pasamos el dataset
        # como una entrada custom mediante un override por --datasets-config.
        # Pero run_all_phases.py no acepta un dataset arbitrario, sólo los
        # tres canónicos; así que usamos tune.py directamente:
    ]
    # En lugar de run_all_phases.py, invocamos tune.py por fase directamente.
    return _run_tune_per_phase(config_dir, args)


def _run_tune_per_phase(config_dir: Path, args, block: str) -> Path:
    """Lanza tune.py para cada fase sobre el dataset full/ del config_dir."""
    full_dir = config_dir / "full"
    common = [
        "--directorio-datos", str(full_dir),
        "--dia-prediccion", "2024-12-15",
        "--baseline-desbalance", *map(str, args.baseline_desbalance),
        "--gnn-kappa", *map(str, args.gnn_kappa),
        "--epochs", str(args.epochs),
        "--epochs-gnn", str(args.epochs_gnn),
        "--max-fn", str(args.max_fn),
        "--coste-falsa-alarma", str(args.coste_falsa_alarma),
        "--coste-omision", str(args.coste_omision),
        "--device", args.device,
    ]
    # observed_stations para Fase 2.x: leemos el manifest
    import yaml
    manifest = full_dir / "manifest.yaml"
    m = yaml.safe_load(manifest.read_text())
    all_stations = [s["name"] for s in m["visible_stations"]]
    # Política de masking por bloque:
    #   H3: la cobertura real ES el eje del bloque; Fase 2.x usa todas las
    #       estaciones existentes (no aplicamos masking adicional).
    #   resto: Fase 2.x ve la mitad de las estaciones existentes, replicando
    #          el régimen de "información parcial" usado en el Ebro/synth.
    if block == "H3":
        observed = all_stations
    else:
        n_obs = max(1, len(all_stations) // 2)
        observed = all_stations[:n_obs]

    for phase in args.phases:
        out = config_dir / f"tune-fase{phase}"
        cmd = [sys.executable, str(_TUNE),
               *common, "--gnn-fase", phase,
               "--output", str(out)]
        if phase != "1":
            cmd += ["--skip-baseline", "--observed-stations", *observed]
        print(f"\n  >> {config_dir.name} / fase {phase}")
        subprocess.run(cmd)

    # Comparativo cross-phase final
    pred_args = []
    for d in args.baseline_desbalance:
        tag = f"d{d:g}".replace(".", "_")
        sub = config_dir / "tune-fase1" / f"baseline_{tag}"
        if (sub / "predictions.csv").exists():
            pred_args.append(f"baseline:{tag}={sub}")
    for phase in args.phases:
        ph_dir = config_dir / f"tune-fase{phase}"
        familia = "gnn-fase1" if phase == "1" else f"gnn-fase{phase}"
        for k in args.gnn_kappa:
            tag = f"k{k:g}".replace(".", "_")
            sub = ph_dir / f"gnn_{tag}"
            if (sub / "predictions.csv").exists():
                pred_args.append(f"{familia}:{tag}={sub}")
    if not pred_args:
        return config_dir / "compare-phases"
    out_cmp = config_dir / "compare-phases"
    q_min = float(m["basin"]["caudal_minimo_m3s"])
    cmd = [sys.executable, str(_COMPARE),
           "--predictions", *pred_args,
           "--q-min", str(q_min),
           "--coste-falsa-alarma", str(args.coste_falsa_alarma),
           "--coste-omision", str(args.coste_omision),
           "--max-fn", str(args.max_fn),
           "--output", str(out_cmp)]
    subprocess.run(cmd)
    return out_cmp


def _read_winners(compare_dir: Path) -> List[Dict]:
    """Lee winners_by_family.csv y devuelve filas como dicts."""
    f = compare_dir / "winners_by_family.csv"
    if not f.exists():
        return []
    with f.open() as fh:
        return list(csv.DictReader(fh))


# ===========================================================================
# CLI
# ===========================================================================


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bloques", nargs="+", default=["H1"],
                   choices=list(_BLOCK_FUNCS.keys()),
                   help="Bloques de hipótesis a ejecutar.")
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2],
                   help="Semillas; más = más potencia estadística (recomendado 5).")
    p.add_argument("--output", type=Path, default=Path("../sweep-results"),
                   help="Directorio raíz de salidas.")
    p.add_argument("--phases", nargs="+", default=["1", "2.1", "2.2"])
    p.add_argument("--baseline-desbalance", nargs="+", type=float,
                   default=[1, 5, 20])  # grid reducido para sweep masivo
    p.add_argument("--gnn-kappa", nargs="+", type=float, default=[5, 30, 100])
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--epochs-gnn", type=int, default=150)
    p.add_argument("--max-fn", type=int, default=0)
    p.add_argument("--coste-falsa-alarma", type=float, default=1.0)
    p.add_argument("--coste-omision", type=float, default=100.0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--skip-existing", action="store_true",
                   help="Si una config ya tiene compare-phases/winners_by_family.csv, no la rehace.")
    return p.parse_args()


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    summary_rows: List[Dict] = []

    for block_name in args.bloques:
        block_func = _BLOCK_FUNCS[block_name]
        for block, config_id, params in block_func(args.seeds):
            config_dir = args.output / block / config_id
            print(f"\n{'='*78}\n=== {block} / {config_id}\n{'='*78}")

            winners_csv = config_dir / "compare-phases" / "winners_by_family.csv"
            if args.skip_existing and winners_csv.exists():
                print(f"[skip] ya existe {winners_csv}")
            else:
                # 1. Generar + simular
                config_dir.mkdir(parents=True, exist_ok=True)
                meta = _generate_and_simulate(dict(params), config_dir)
                # 2. Sweep + comparativo
                _run_tune_per_phase(config_dir, args, block)

            # 3. Leer winners y añadir filas al summary
            for row in _read_winners(config_dir / "compare-phases"):
                # Re-leer params para meta (el dict params ya fue modificado)
                summary_rows.append({
                    "block": block,
                    "config_id": config_id,
                    "n_type1": params.get("n_type1", 16),
                    "branching_factor": params.get("branching_factor", 2.5),
                    "n_reservoirs": params.get("n_reservoirs", 3),
                    "station_coverage": params.get("station_coverage", 0.5),
                    "nonstationarity_amp": params.get("nonstationarity_amp", 0.0),
                    "years": params.get("_years", 10),
                    "seed": params["seed"],
                    **{k: row[k] for k in
                       ["familia", "variante", "delta", "tp", "fp", "fn", "tn",
                        "precision", "recall", "f1", "coste_total", "nse", "kge",
                        "factible"] if k in row},
                })

    # 4. Escribir summary global
    if summary_rows:
        summary_path = args.output / "sweep_summary.csv"
        keys = list(summary_rows[0].keys())
        with summary_path.open("w") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(summary_rows)
        print(f"\n[done] summary global escrito en {summary_path} "
              f"({len(summary_rows)} filas).")
    else:
        print("\n[warn] no hay summary rows; quizá ningún winners_by_family.csv se generó.")


if __name__ == "__main__":
    main()
