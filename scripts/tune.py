"""Orquestador: barrido de hiperparámetros + comparación apples-to-apples.

Para cada familia (Seq2Seq, GNN), entrena varias variantes con
hiperparámetros diferentes y compara cada familia *al óptimo de su propio
barrido* bajo la restricción dura `FN ≤ max_fn`. El ganador es el método
que minimiza coste para el operario de la central, sin que el experimentador
haya tenido que adivinar el valor "correcto" de κ o `desbalance` para cada
modelo.

Las predicciones se cachean en disco; si una variante ya tiene
`predictions.csv` no se reentrena.

Llamada típica:

    python scripts/tune.py \\
        --directorio-datos ../datos-synth/full \\
        --dia-prediccion 2024-12-15 \\
        --epochs 300 --epochs-gnn 200 \\
        --baseline-desbalance 1 2 5 10 20 \\
        --gnn-kappa 5 30 100 \\
        --max-fn 0 --coste-falsa-alarma 1 --coste-omision 100 \\
        --output ../tune-synth-full
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# Permite ejecutar el script directamente sin instalar el paquete.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


_REPO_ROOT = Path(__file__).resolve().parent.parent
_RUN_BASELINE = _REPO_ROOT / "scripts" / "run_baseline.py"
_RUN_GNN = _REPO_ROOT / "scripts" / "run_gnn.py"
_COMPARE = _REPO_ROOT / "scripts" / "compare_models.py"


def parse_args():
    p = argparse.ArgumentParser(description="Barrido y comparación apples-to-apples.")
    p.add_argument("--directorio-datos", required=True, type=Path)
    p.add_argument("--firma", default=None, type=str)
    p.add_argument("--dia-prediccion", required=True, type=str)
    p.add_argument("--output", required=True, type=Path,
                   help="Directorio donde se cachean las predicciones de cada variante.")
    # Barrido del baseline
    p.add_argument("--baseline-desbalance", nargs="+", type=float,
                   default=[1.0, 2.0, 5.0, 10.0, 20.0],
                   help="Valores de --desbalance para Seq2Seq.")
    p.add_argument("--epochs", type=int, default=300,
                   help="Épocas de entrenamiento del baseline.")
    p.add_argument("--skip-baseline", action="store_true")
    # Barrido del GNN
    p.add_argument("--gnn-kappa", nargs="+", type=float,
                   default=[5.0, 30.0, 100.0],
                   help="Valores de --kappa-low-flow para el GNN.")
    p.add_argument("--gnn-fase", default="1", choices=["1", "2.1", "2.2"])
    p.add_argument("--observed-stations", nargs="*", default=None,
                   help="Estaciones observables para Fases 2.1 y 2.2 (ignorado en Fase 1).")
    p.add_argument("--gnn-s-low-flow", type=float, default=5.0)
    p.add_argument("--epochs-gnn", type=int, default=200)
    p.add_argument("--skip-gnn", action="store_true")
    # Análisis
    p.add_argument("--q-min", type=float, default=None,
                   help="Q_min para la comparación. Si se omite, se infiere del basin.")
    p.add_argument("--coste-falsa-alarma", type=float, default=1.0)
    p.add_argument("--coste-omision", type=float, default=100.0)
    p.add_argument("--max-fn", type=int, default=0)
    p.add_argument("--escenario", default="worst", choices=["observed", "worst", "both"])
    p.add_argument("--device", default="cpu")
    p.add_argument("--abort-on-error", action="store_true",
                   help="Si una variante falla, abortar todo el barrido. Por "
                        "defecto se continúa con las siguientes y se loguea el "
                        "fallo en `<output>/logs/<variante>.log`.")
    return p.parse_args()


def _run(cmd, fatal: bool = True, log_file: Path = None) -> int:
    """Ejecuta un sub-proceso. Si `fatal=False`, devuelve el código de salida
    en lugar de abortar."""
    print(f"\n$ {' '.join(str(c) for c in cmd)}")
    if log_file is not None:
        with open(log_file, "ab") as f:
            f.write(f"\n$ {' '.join(str(c) for c in cmd)}\n".encode())
            res = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    else:
        res = subprocess.run(cmd)
    if res.returncode != 0 and fatal:
        raise SystemExit(f"Comando falló (exit {res.returncode}).")
    return res.returncode


def _train_if_missing(plot_dir: Path, build_cmd_fn, *, continue_on_error: bool, log_dir: Path) -> bool:
    """Entrena si no hay `predictions.csv`. Devuelve True si al final
    existe `predictions.csv` (cache hit o entreno OK), False si falló."""
    pred_csv = plot_dir / "predictions.csv"
    if pred_csv.exists():
        print(f"[cache] {pred_csv} ya existe — se reusa (borra el dir para reentrenar).")
        return True
    plot_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{plot_dir.name}.log" if continue_on_error else None
    rc = _run(build_cmd_fn(plot_dir), fatal=not continue_on_error, log_file=log_file)
    if rc != 0:
        print(f"[!!] {plot_dir.name} falló (exit {rc}); log en {log_file}.  "
              f"Continúo con la siguiente variante.")
        return False
    return pred_csv.exists()


def _resolve_q_min(args):
    if args.q_min is not None:
        return args.q_min
    # Auto-detect del manifest (sintética) o del basin del Ebro.
    manifest = args.directorio_datos / "manifest.yaml"
    if manifest.exists():
        import yaml
        m = yaml.safe_load(manifest.read_text())
        return float(m["basin"].get("caudal_minimo_m3s", 30.0))
    # Ebro por defecto
    from seq2seq_runoff.basins import ebro_basin
    return ebro_basin().caudal_minimo_m3s


def main():
    args = parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    log_dir = out / "logs"
    log_dir.mkdir(exist_ok=True)
    continue_on_error = not args.abort_on_error

    common_args = [
        "--directorio-datos", str(args.directorio_datos),
        "--dia-prediccion", args.dia_prediccion,
        "--escenario", args.escenario,
        "--coste-falsa-alarma", str(args.coste_falsa_alarma),
        "--coste-omision", str(args.coste_omision),
        "--max-fn", str(args.max_fn),
    ]
    if args.firma:
        common_args += ["--firma", args.firma]

    # Fase 2.x exige observed-stations; warning temprano si falta.
    if args.gnn_fase != "1" and not args.observed_stations and not args.skip_gnn:
        print(f"[!!] Fase {args.gnn_fase} requiere --observed-stations; "
              f"el sweep del GNN producirá errores.")

    gnn_extras = []
    if args.observed_stations:
        gnn_extras += ["--observed-stations", *args.observed_stations]

    # Etiqueta de familia GNN distinta por fase: las tres convivirán en el
    # comparativo cross-phase sin colisionar.
    gnn_family = "gnn" if args.gnn_fase == "1" else f"gnn-fase{args.gnn_fase}"

    predictions_args = []
    fallos = []  # lista de etiquetas que no produjeron predictions.csv

    # ----- Sweep baseline -----------------------------------------------
    if not args.skip_baseline:
        print(f"\n=== SWEEP BASELINE: desbalance ∈ {args.baseline_desbalance} ===")
        for d in args.baseline_desbalance:
            tag = f"d{d:g}".replace(".", "_")
            sub = out / f"baseline_{tag}"
            def cmd(plot_dir, d=d):
                return [
                    sys.executable, str(_RUN_BASELINE),
                    *common_args,
                    "--epochs", str(args.epochs),
                    "--desbalance", str(d),
                    "--directorio-modelo", str(plot_dir / "modelo"),
                    "--plot", str(plot_dir),
                ]
            ok = _train_if_missing(sub, cmd, continue_on_error=continue_on_error, log_dir=log_dir)
            if ok:
                predictions_args.append(f"baseline:{tag}={sub}")
            else:
                fallos.append(f"baseline:{tag}")

    # ----- Sweep GNN ----------------------------------------------------
    if not args.skip_gnn:
        print(f"\n=== SWEEP GNN: kappa ∈ {args.gnn_kappa} ===")
        for k in args.gnn_kappa:
            tag = f"k{k:g}".replace(".", "_")
            sub = out / f"gnn_{tag}"
            def cmd(plot_dir, k=k):
                return [
                    sys.executable, str(_RUN_GNN),
                    "--fase", args.gnn_fase,
                    *common_args,
                    "--epochs", str(args.epochs_gnn),
                    "--kappa-low-flow", str(k),
                    "--s-low-flow", str(args.gnn_s_low_flow),
                    "--device", args.device,
                    "--directorio-modelo", str(plot_dir / "modelo"),
                    "--plot", str(plot_dir),
                    *gnn_extras,
                ]
            ok = _train_if_missing(sub, cmd, continue_on_error=continue_on_error, log_dir=log_dir)
            if ok:
                predictions_args.append(f"{gnn_family}:{tag}={sub}")
            else:
                fallos.append(f"{gnn_family}:{tag}")

    # ----- Resumen del barrido ------------------------------------------
    print(f"\n=== RESUMEN DEL BARRIDO ===")
    print(f"  Variantes con predictions.csv : {len(predictions_args)}")
    print(f"  Variantes que fallaron         : {len(fallos)}")
    if fallos:
        print(f"  Fallidas: {fallos}")
        print(f"  Logs en: {log_dir}/")

    if not predictions_args:
        raise SystemExit("No hay predicciones (todas las variantes fallaron o se saltaron).")

    # ----- Comparación ---------------------------------------------------
    q_min = _resolve_q_min(args)
    compare_cmd = [
        sys.executable, str(_COMPARE),
        "--predictions", *predictions_args,
        "--q-min", str(q_min),
        "--coste-falsa-alarma", str(args.coste_falsa_alarma),
        "--coste-omision", str(args.coste_omision),
        "--max-fn", str(args.max_fn),
        "--output", str(out),
    ]
    print("\n=== COMPARACIÓN ENTRE GANADORES POR FAMILIA ===")
    _run(compare_cmd)


if __name__ == "__main__":
    main()
