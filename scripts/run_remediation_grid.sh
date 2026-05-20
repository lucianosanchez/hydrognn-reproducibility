#!/usr/bin/env bash
# ============================================================================
# Grid search de configuraciones intermedias de remediación.
#
# Objetivo: encontrar (si existe) una configuración homogénea de los cuatro
# flags de remediación que no degrade Ebro/N=16 y que recupere
# diferenciación de criterios en N=64. Si se encuentra, el paper queda
# más fuerte (no hace falta describir "remediación como hiperparámetro
# per-basin"). Si no se encuentra, los resultados actuales del paper
# (§4.10 "Remediation as a per-basin hyperparameter") quedan como están.
#
# Diseño:
#   * 8 configuraciones cuidadosamente elegidas (A..H) que cubren el
#     espacio entre defaults (A) y full-remediation (H).
#   * 3 datasets por configuración: synth-N16, synth-N64, Ebro.
#   * Screening rápido: 100 épocas, K_train=10, max_windows=2000 en N=64.
#   * NO sobreescribe ningún output previo: todo va a outputs/grid/<id>/.
#
# Tiempo estimado: ~7-9 h en CPU single-process M-series.
#
# Lanzamiento:
#   cd seq2seq_runoff
#   bash scripts/run_remediation_grid.sh
#
# Si quieres lanzar sólo una config (e.g. para tantear):
#   ONLY_CONFIG=D bash scripts/run_remediation_grid.sh
#
# Tras la corrida:
#   python scripts/summarize_remediation_grid.py \
#       --grid-root outputs/grid \
#       --output outputs/grid/grid_summary.csv
# ============================================================================

set -euo pipefail

PYTHON="${PYTHON:-python}"
export PYTHONUNBUFFERED=1

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

GRID_DIR="outputs/grid"
LOG_DIR="$GRID_DIR/_logs"
mkdir -p "$GRID_DIR" "$LOG_DIR"

ONLY_CONFIG="${ONLY_CONFIG:-}"

# ---------------------------------------------------------------------------
# Catálogo de configuraciones.
#
# Formato (separado por |):
#   id|rain_bypass|lam11_init|warmup_epochs|ramp_epochs|free_bits|comentario
# ---------------------------------------------------------------------------
declare -a CONFIGS=(
  "A|off|0.0|0|1|0.0|control: defaults (paper headline)"
  "B|on |0.0|0|1|0.0|sólo bypass, sin lam11 init ni warmup"
  "C|off|1.0|0|1|0.0|sólo lam11=1 (sin bypass)"
  "D|on |0.5|0|1|0.0|bypass + lam11 moderado"
  "E|on |0.0|40|20|0.005|bypass + warmup + free_bits suaves"
  "F|on |1.0|40|20|0.01|bypass + lam moderado + warmup + free_bits"
  "G|on |2.0|80|40|0.005|bypass + lam alto + warmup largo + free_bits suaves"
  "H|on |2.0|80|40|0.02|control: full remediation (N=64 winner del paper actual)"
)

# ---------------------------------------------------------------------------
# Datasets (id, directorio, firma, basin-type, max_windows, K_train)
# ---------------------------------------------------------------------------
declare -a DATASETS=(
  "synth-N16|datos-synth/full|SYNTH001|synth|0|10"
  "synth-N64|datos-synth-N64/full|SYNTH-N64|synth|2000|10"
  "ebro|datos-06-07-2023|580734|ebro|0|10"
)

EPOCHS=100      # screening corto; los headlines del paper usan 200
K_INFER=50
KAPPA=30
BETA_UA=1e-3
N_RAIN=20

# ---------------------------------------------------------------------------
# Loop principal
# ---------------------------------------------------------------------------
echo "=========================================================================="
echo "[grid] PYTHON     = $PYTHON"
echo "[grid] ROOT       = $ROOT"
echo "[grid] CONFIGS    = ${#CONFIGS[@]}"
echo "[grid] DATASETS   = ${#DATASETS[@]}"
echo "[grid] EPOCHS     = $EPOCHS"
echo "[grid] GRID_DIR   = $GRID_DIR"
[[ -n "$ONLY_CONFIG" ]] && echo "[grid] ONLY_CONFIG = $ONLY_CONFIG"
echo "=========================================================================="

for cfg in "${CONFIGS[@]}"; do
  IFS="|" read -r cid bypass lam warmup ramp freebits comment <<< "$cfg"
  cid="$(echo $cid | tr -d ' ')"
  bypass="$(echo $bypass | tr -d ' ')"

  if [[ -n "$ONLY_CONFIG" && "$ONLY_CONFIG" != "$cid" ]]; then
    continue
  fi

  echo ""
  echo "[grid] ==== Config $cid : $comment ===="
  echo "[grid]      bypass=$bypass  lam11=$lam  warmup=$warmup  ramp=$ramp  free_bits=$freebits"

  BYPASS_FLAG=""
  [[ "$bypass" == "on" ]] && BYPASS_FLAG="--rain-bypass"

  for ds in "${DATASETS[@]}"; do
    IFS="|" read -r did dir firma btype max_win k_train <<< "$ds"

    OUT="$GRID_DIR/$cid/uagnn-$did"
    LOGFILE="$LOG_DIR/grid-$cid-$did.log"

    if [[ -f "$OUT/headline_metrics.csv" ]]; then
      echo "[grid]   $did: ya existe $OUT/headline_metrics.csv — skip"
      continue
    fi

    mkdir -p "$OUT"

    MAX_WIN_FLAG=""
    [[ "$max_win" != "0" ]] && MAX_WIN_FLAG="--max-windows $max_win --batch-size 64"

    # Para Ebro y N=16 usamos epochs=100 (screening); para N=64 si hay
    # remediation también 100 épocas (que es la mitad de la corrida del
    # paper headline; suficiente para ver si el modelo escapa al colapso).
    echo "[grid]   $did: lanzando ($(date '+%H:%M:%S'))"
    $PYTHON scripts/run_ua_gnn_experiment.py \
      --directorio-datos "$dir" --firma "$firma" \
      --dia-prediccion 2024-12-15 \
      --rolling-inicio 2022-01-01 --rolling-fin 2024-12-01 \
      --epochs $EPOCHS --K-train $k_train --K-inference $K_INFER \
      --beta-ua $BETA_UA --kappa $KAPPA \
      --warmup-epochs $warmup --ramp-epochs $ramp --free-bits $freebits \
      $BYPASS_FLAG --lam11-init $lam \
      $MAX_WIN_FLAG \
      --n-rain-samples $N_RAIN \
      --output "$OUT" \
      > "$LOGFILE" 2>&1 \
      || { echo "[grid]   $did FAILED — ver $LOGFILE"; continue; }

    echo "[grid]   $did: OK ($(date '+%H:%M:%S'))"
  done
done

echo ""
echo "=========================================================================="
echo "[grid] COMPLETADO. Para resumir:"
echo "  $PYTHON scripts/summarize_remediation_grid.py \\"
echo "      --grid-root $GRID_DIR --output $GRID_DIR/grid_summary.csv"
echo "=========================================================================="
