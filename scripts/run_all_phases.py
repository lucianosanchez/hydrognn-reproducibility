"""Lanza el sweep COMPLETO de fases (1, 2.1, 2.2) sobre los tres datasets.

Este script orquesta `tune.py` para cada combinación (dataset, fase) y, al
final, llama a `compare_models.py` con etiquetas `familia:variante`
distintas por fase para producir un comparativo cross-phase por dataset.

Estructura de salida en `--output-base` (default `..`):

    tune-fase1-<dataset>/        baselines + GNN Fase 1 (cache hit si existe)
    tune-fase2.1-<dataset>/      sólo GNN Fase 2.1
    tune-fase2.2-<dataset>/      sólo GNN Fase 2.2
    compare-phases-<dataset>/    comparison.csv + winners + plot cross-phase

Reuso de trabajo: cada `tune.py` cachea por `predictions.csv`, así que
relanzar el script no reentrena lo que ya hay. Si una variante falla, se
loguea y el barrido continúa.

Llamada típica desde `seq2seq_runoff/scripts/`:

    python run_all_phases.py
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ============================================================================
# Configuración por dataset.
# ============================================================================
#
# `observed_partial`: subconjunto de estaciones que se consideran observables
# en Fases 2.1 y 2.2. Debe ser un subconjunto estricto de las estaciones
# disponibles en el BasinSpec del dataset.
#
# Si añades un dataset nuevo (cuenca distinta, etc.), añade su entrada aquí
# o pásalo individualmente con --datasets ... y --observed-stations ...

DATASETS = [
    {
        "key": "datos-06-07-2023",            # cuenca real Ebro a Tudela
        "directorio": "../datos-06-07-2023",
        "firma": "580734",
        "dia": "2023-06-25",
        # 3 de 9 estaciones (~33 % cobertura), una por sub-cuenca principal.
        "observed_partial": ["EM01-PACUM", "EM29-PACUM", "EM30-PACUM"],
    },
    {
        "key": "datos-synth-full",             # sintética, visibilidad completa
        "directorio": "../datos-synth/full",
        "firma": None,                          # se lee del manifest
        "dia": "2024-12-15",
        # 2 de 4 estaciones (50 %): cabecera del río principal y un afluente.
        "observed_partial": ["SM-PACUM", "ST1-PACUM"],
    },
    {
        "key": "datos-synth-partial",          # sintética, sólo R0 visible
        "directorio": "../datos-synth/partial",
        "firma": None,
        "dia": "2024-12-15",
        "observed_partial": ["SM-PACUM", "ST1-PACUM"],
    },
]

# Localización de los scripts hermanos.
_HERE = Path(__file__).resolve().parent
_TUNE = _HERE / "tune.py"
_COMPARE = _HERE / "compare_models.py"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-base", default="..", type=Path,
                   help="Directorio padre donde se crean tune-fase*-<dataset>/ y "
                        "compare-phases-<dataset>/.")
    p.add_argument("--phases", nargs="+", default=["1", "2.1", "2.2"],
                   choices=["1", "2.1", "2.2"],
                   help="Fases a ejecutar (default: las tres).")
    p.add_argument("--datasets", nargs="+", default=None,
                   help="Subconjunto de datasets por su 'key'; default: todos.")
    # Hiperparámetros del sweep
    p.add_argument("--baseline-desbalance", nargs="+", type=float,
                   default=[1, 2, 5, 10, 20])
    p.add_argument("--gnn-kappa", nargs="+", type=float, default=[5, 30, 100])
    p.add_argument("--gnn-s-low-flow", type=float, default=5.0)
    p.add_argument("--epochs", type=int, default=300,
                   help="Épocas del baseline (sólo se usa en Fase 1).")
    p.add_argument("--epochs-gnn", type=int, default=200)
    # Análisis económico
    p.add_argument("--coste-falsa-alarma", type=float, default=1.0)
    p.add_argument("--coste-omision", type=float, default=100.0)
    p.add_argument("--max-fn", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--skip-compare", action="store_true",
                   help="No lanzar la comparación final cross-phase por dataset.")
    return p.parse_args()


def _resolve_q_min(ds, args):
    """Q_min en m³/s del dataset (lee manifest si existe; si no, basin del Ebro)."""
    manifest = Path(ds["directorio"]) / "manifest.yaml"
    if manifest.exists():
        import yaml
        m = yaml.safe_load(manifest.read_text())
        return float(m["basin"].get("caudal_minimo_m3s", 30.0))
    # Ebro
    return 30.0


def _run_phase(ds, phase: str, args):
    """Lanza tune.py para (dataset, phase). Devuelve el directorio de salida."""
    output = args.output_base / f"tune-fase{phase}-{ds['key']}"
    cmd = [
        sys.executable, str(_TUNE),
        "--directorio-datos", ds["directorio"],
        "--dia-prediccion", ds["dia"],
        "--gnn-fase", phase,
        "--baseline-desbalance", *map(str, args.baseline_desbalance),
        "--gnn-kappa", *map(str, args.gnn_kappa),
        "--gnn-s-low-flow", str(args.gnn_s_low_flow),
        "--epochs", str(args.epochs),
        "--epochs-gnn", str(args.epochs_gnn),
        "--max-fn", str(args.max_fn),
        "--coste-falsa-alarma", str(args.coste_falsa_alarma),
        "--coste-omision", str(args.coste_omision),
        "--device", args.device,
        "--output", str(output),
    ]
    if ds["firma"]:
        cmd += ["--firma", ds["firma"]]

    if phase != "1":
        # Fases 2.x reutilizan los baselines de Fase 1: no los reentrenamos.
        cmd += ["--skip-baseline"]
        cmd += ["--observed-stations", *ds["observed_partial"]]

    print(f"\n{'='*78}\n=== ({ds['key']}, fase {phase})\n{'='*78}")
    rc = subprocess.run(cmd).returncode
    if rc != 0:
        print(f"[!!] tune.py para ({ds['key']}, fase {phase}) salió con código {rc}.")
    return output


def _gather_predictions(ds, args):
    """Reúne los predictions.csv de las tres fases con etiquetas distintas."""
    base = args.output_base
    pred_args = []

    # Baselines: viven sólo en Fase 1.
    fase1_dir = base / f"tune-fase1-{ds['key']}"
    for d in args.baseline_desbalance:
        tag = f"d{d:g}".replace(".", "_")
        sub = fase1_dir / f"baseline_{tag}"
        if (sub / "predictions.csv").exists():
            pred_args.append(f"baseline:{tag}={sub}")

    # GNN per fase.
    for phase in args.phases:
        ph_dir = base / f"tune-fase{phase}-{ds['key']}"
        familia = "gnn-fase1" if phase == "1" else f"gnn-fase{phase}"
        for k in args.gnn_kappa:
            tag = f"k{k:g}".replace(".", "_")
            sub = ph_dir / f"gnn_{tag}"
            if (sub / "predictions.csv").exists():
                pred_args.append(f"{familia}:{tag}={sub}")

    return pred_args


def _compare_cross_phase(ds, args, pred_args):
    """Llama a compare_models.py con la unión de variantes de las tres fases."""
    if not pred_args:
        print(f"[compare] sin predictions para {ds['key']}; saltado.")
        return
    output = args.output_base / f"compare-phases-{ds['key']}"
    q_min = _resolve_q_min(ds, args)
    cmd = [
        sys.executable, str(_COMPARE),
        "--predictions", *pred_args,
        "--q-min", str(q_min),
        "--coste-falsa-alarma", str(args.coste_falsa_alarma),
        "--coste-omision", str(args.coste_omision),
        "--max-fn", str(args.max_fn),
        "--output", str(output),
    ]
    print(f"\n{'='*78}\n=== COMPARACIÓN CROSS-PHASE: {ds['key']}\n{'='*78}")
    subprocess.run(cmd)


def main():
    args = parse_args()
    selected = DATASETS if args.datasets is None else \
               [d for d in DATASETS if d["key"] in args.datasets]
    if not selected:
        sys.exit(f"Ningún dataset coincide con --datasets {args.datasets}.")

    # 1. Sweep por (dataset, phase)
    for ds in selected:
        for phase in args.phases:
            _run_phase(ds, phase, args)

    # 2. Comparación cross-phase por dataset
    if not args.skip_compare:
        for ds in selected:
            preds = _gather_predictions(ds, args)
            _compare_cross_phase(ds, args, preds)

    # 3. Resumen global
    print(f"\n{'='*78}\n=== TODO HECHO\n{'='*78}")
    print(f"  Datasets procesados:  {[d['key'] for d in selected]}")
    print(f"  Fases:                {args.phases}")
    if not args.skip_compare:
        print(f"  Comparativos cross-phase en:")
        for ds in selected:
            print(f"    {args.output_base}/compare-phases-{ds['key']}/")


if __name__ == "__main__":
    main()
