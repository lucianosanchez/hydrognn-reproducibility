#!/usr/bin/env bash
# ============================================================================
# Operating-cost evaluation for the 8 Phase 2.2 checkpoints in
# outputs/phase22_grid/. Produces outputs/phase22_grid/operating_cost.csv
# with FN/FP/cost per configuration. Feeds tab:phase22_acyclic_grid_cost
# in §5.5 of the paper, closing the gap left by grid_summary.csv
# (which reports only training and physicalisation metrics).
#
# Wall time: ~10-15 min on a single-process CPU (8 checkpoints × ~1 min
# each for the offset sweep over the synthetic evaluation window).
#
# Usage
# -----
#     cd hydrognn
#     PYTHON=python bash scripts/evaluate_phase22_operating_cost.sh
#
# The script is idempotent on the CSV: it appends one row per checkpoint;
# delete the CSV first if you want a clean re-run.
#
# The resulting CSV `outputs/phase22_grid/operating_cost.csv` is the
# numerical source for tab:phase22_acyclic_operating in §5.5 of the
# paper (dense-vs-acyclic candidate-graph operating costs).
# ============================================================================

set -euo pipefail

PYTHON="${PYTHON:-python}"
export PYTHONUNBUFFERED=1

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

GRID_DIR="outputs/phase22_grid"
OUT_CSV="$GRID_DIR/operating_cost.csv"
LOG_DIR="outputs/_logs"
mkdir -p "$LOG_DIR"

# Wipe any partial CSV so the run is reproducible.
rm -f "$OUT_CSV"

echo "=========================================================================="
echo "[phase22-cost] evaluating operating cost for 8 Phase 2.2 checkpoints"
echo "[phase22-cost] output: $OUT_CSV"
echo "=========================================================================="

for ds in synth-N16 synth-N64; do
  for M in 3 6; do
    for mode in dense acyclic; do
      CKPT="$GRID_DIR/${ds}-M${M}-${mode}"
      if [[ ! -f "$CKPT/core.pt" ]]; then
        echo "[phase22-cost] skip $ds M=$M $mode (no checkpoint)"
        continue
      fi
      echo ""
      echo "[phase22-cost] $ds M=$M $mode"
      $PYTHON scripts/evaluate_phase22_operating_cost.py \
        --ckpt "$CKPT" \
        --dataset "$ds" \
        --m-latent "$M" \
        --mode "$mode" \
        --output-csv "$OUT_CSV" \
        > "$LOG_DIR/phase22-cost-${ds}-M${M}-${mode}.log" 2>&1 \
        || echo "[phase22-cost]   FAILED — see $LOG_DIR/phase22-cost-${ds}-M${M}-${mode}.log"
    done
  done
done

echo ""
echo "=========================================================================="
echo "[phase22-cost] DONE. CSV: $OUT_CSV"
echo "[phase22-cost] This file feeds tab:phase22_acyclic_operating in §5.5 of the paper."
echo "=========================================================================="
