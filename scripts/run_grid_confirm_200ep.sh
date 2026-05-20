#!/usr/bin/env bash
# ============================================================================
# Confirmación de la config C (lam11_init=1.0, sin remediación extra) con
# epochs=200, igualando el régimen de los headlines del paper.
#
# El grid screening (epochs=100) reveló que:
#   * El bypass head es perjudicial en TODOS los basins.
#   * lam11_init=1.0 (sin bypass) gana el grid: N=16 FN=0, N=64 FN=0, Ebro FN=99.
#
# Aquí confirmamos a epochs=200 que ese hallazgo se mantiene (no era un
# artefacto de subentreno). Tres corridas:
#   * synth-N16, config C
#   * synth-N64, config C (con max_windows=2000 igual que el grid)
#   * Ebro,      config C
#
# Tiempo estimado: ~2-3 h en CPU single-process.
#
# Salida: outputs/grid_confirm/C/uagnn-{N16,N64,ebro}/
# NO sobreescribe outputs/grid ni outputs/uagnn-*.
# ============================================================================

set -euo pipefail

PYTHON="${PYTHON:-python}"
export PYTHONUNBUFFERED=1

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

OUT_BASE="outputs/grid_confirm/C"
LOG_DIR="outputs/grid_confirm/_logs"
mkdir -p "$OUT_BASE" "$LOG_DIR"

EPOCHS=200
K_TRAIN=10
K_INFER=50
KAPPA=30
BETA_UA=1e-3
N_RAIN=20
LAM11=1.0     # ← la única remediación que sobrevive el grid

echo "=========================================================================="
echo "[confirm] PYTHON  = $PYTHON"
echo "[confirm] EPOCHS  = $EPOCHS"
echo "[confirm] LAM11   = $LAM11 (config C: sin rain_bypass, sin warmup, sin free_bits)"
echo "=========================================================================="

# --- synth-N16 ---
OUT="$OUT_BASE/uagnn-synth-N16"
if [[ -f "$OUT/headline_metrics.csv" ]]; then
  echo "[confirm] synth-N16 ya existe — skip"
else
  echo ""
  echo "[confirm] synth-N16 ($(date '+%H:%M:%S'))"
  $PYTHON scripts/run_ua_gnn_experiment.py \
    --directorio-datos datos-synth/full --firma SYNTH001 \
    --dia-prediccion 2024-12-15 \
    --rolling-inicio 2022-01-01 --rolling-fin 2024-12-01 \
    --epochs $EPOCHS --K-train $K_TRAIN --K-inference $K_INFER \
    --beta-ua $BETA_UA --kappa $KAPPA \
    --lam11-init $LAM11 \
    --n-rain-samples $N_RAIN \
    --output "$OUT" \
    > "$LOG_DIR/synth-N16.log" 2>&1
  echo "[confirm] synth-N16: OK ($(date '+%H:%M:%S'))"
fi

# --- synth-N64 ---
OUT="$OUT_BASE/uagnn-synth-N64"
if [[ -f "$OUT/headline_metrics.csv" ]]; then
  echo "[confirm] synth-N64 ya existe — skip"
else
  echo ""
  echo "[confirm] synth-N64 ($(date '+%H:%M:%S')) — con max_windows=2000 igual que el grid"
  $PYTHON scripts/run_ua_gnn_experiment.py \
    --directorio-datos datos-synth-N64/full --firma SYNTH-N64 \
    --dia-prediccion 2024-12-15 \
    --rolling-inicio 2022-01-01 --rolling-fin 2024-12-01 \
    --epochs $EPOCHS --K-train $K_TRAIN --K-inference $K_INFER \
    --beta-ua $BETA_UA --kappa $KAPPA \
    --lam11-init $LAM11 \
    --max-windows 2000 --batch-size 64 \
    --n-rain-samples $N_RAIN \
    --output "$OUT" \
    > "$LOG_DIR/synth-N64.log" 2>&1
  echo "[confirm] synth-N64: OK ($(date '+%H:%M:%S'))"
fi

# --- Ebro ---
OUT="$OUT_BASE/uagnn-ebro"
if [[ -f "$OUT/headline_metrics.csv" ]]; then
  echo "[confirm] ebro ya existe — skip"
else
  echo ""
  echo "[confirm] ebro ($(date '+%H:%M:%S'))"
  $PYTHON scripts/run_ua_gnn_experiment.py \
    --directorio-datos datos-06-07-2023 --firma 580734 \
    --dia-prediccion 2023-06-01 \
    --rolling-inicio 2020-01-01 \
    --epochs $EPOCHS --K-train $K_TRAIN --K-inference $K_INFER \
    --beta-ua $BETA_UA --kappa $KAPPA \
    --lam11-init $LAM11 \
    --n-rain-samples $N_RAIN \
    --output "$OUT" \
    > "$LOG_DIR/ebro.log" 2>&1
  echo "[confirm] ebro: OK ($(date '+%H:%M:%S'))"
fi

echo ""
echo "=========================================================================="
echo "[confirm] COMPLETADO."
echo ""
echo "Para verificar los pass-criteria:"
echo "  $PYTHON scripts/summarize_remediation_grid.py \\"
echo "      --grid-root outputs/grid_confirm \\"
echo "      --output outputs/grid_confirm/grid_summary.csv"
echo ""
echo "Si la config C-200ep pasa los tres datasets, podemos reescribir el §4 del"
echo "paper eliminando 'Remediation as a per-basin hyperparameter' y promoviendo"
echo "lam11_init=1.0 como nuevo default universal."
echo "=========================================================================="
