#!/usr/bin/env bash
# ============================================================================
# Multi-seed audit of the UA-HydroGNN headline run on Ebro.
#
# WHY THIS SCRIPT EXISTS
# ----------------------
# `scripts/run_paper_experiments.sh robustness` re-trains Ebro with the
# rainfall-bypass remediation profile (--warmup-epochs 60 --ramp-epochs 30
# --free-bits 0.02). That profile collapses the Ebro posterior, so the
# resulting seed-aggregate CSV (outputs/ebro_remediated_seed_robustness.csv)
# confirms only the *regression* documented in
# tab:uagnn_remediation_ablation_results of the paper, not the robustness
# of the headline 719 -> 44 reduction.
#
# This script instead re-trains Ebro with the *canonical headline*
# configuration (no remediation flags, plain low-flow-weighted likelihood)
# for three additional seeds and aggregates the result. It is what the
# §5.3 'Seed and reproducibility' paragraph promises as future work.
#
# WALL TIME
# ---------
# ~30 min per seed on a single-process Apple M-series CPU, ~1.5 h total
# for the three seeds. Add ~1 min for the aggregation step.
#
# USAGE
# -----
#     cd hydrognn
#     PYTHON=/path/to/python bash scripts/run_headline_seed_robustness.sh
#
# Or, if your active interpreter is correct:
#     bash scripts/run_headline_seed_robustness.sh
#
# The script is idempotent: per-seed directories are skipped if the
# headline_metrics.csv already exists.
#
# OUTPUTS
# -------
#   outputs/uagnn-ebro-headline-seed{0,7,123}/
#       headline_metrics.csv, modelo_uagnn/, ua_meta.pkl, ...
#   outputs/ebro_headline_seed_robustness.csv
#       Mean ± std per metric across the three seeds.
#
# RELATIONSHIP TO THE PAPER
# -------------------------
# The aggregate CSV `outputs/ebro_headline_seed_robustness.csv` is the
# numerical source for the seed-robustness numbers reported in §5.3
# "Seed and reproducibility" of the paper. Each row of the CSV (one
# per decision criterion) reports mean ± std across seeds {0, 7, 123};
# the Savage row corresponds to the headline 719 -> 44 audit.
# ============================================================================

set -euo pipefail

PYTHON="${PYTHON:-python}"
export PYTHONUNBUFFERED=1

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

LOG_DIR="outputs/_logs"
mkdir -p "$LOG_DIR"

EPOCHS=200
K_TRAIN=10
K_INFER=50
KAPPA=30
BETA_UA=1e-3
N_RAIN=20

SEEDS=(0 7 123)

echo "=========================================================================="
echo "[headline-seed] PYTHON  = $PYTHON"
echo "[headline-seed] EPOCHS  = $EPOCHS"
echo "[headline-seed] SEEDS   = ${SEEDS[*]}"
echo "[headline-seed] config  = canonical headline (no remediation flags)"
echo "=========================================================================="

for seed in "${SEEDS[@]}"; do
  OUT="outputs/uagnn-ebro-headline-seed${seed}"
  LOG="$LOG_DIR/uagnn-ebro-headline-seed${seed}.log"

  if [[ -f "$OUT/headline_metrics.csv" ]]; then
    echo "[headline-seed] seed=$seed already exists at $OUT — skip"
    continue
  fi

  echo ""
  echo "[headline-seed] seed=$seed launching ($(date '+%H:%M:%S'))"
  $PYTHON scripts/run_ua_gnn_experiment.py \
    --directorio-datos datos-06-07-2023 --firma 580734 \
    --dia-prediccion 2023-06-01 \
    --rolling-inicio 2020-01-01 \
    --epochs $EPOCHS --K-train $K_TRAIN --K-inference $K_INFER \
    --beta-ua $BETA_UA --kappa $KAPPA \
    --n-rain-samples $N_RAIN \
    --seed $seed \
    --output "$OUT" \
    > "$LOG" 2>&1
  echo "[headline-seed] seed=$seed: OK ($(date '+%H:%M:%S'))"
done

echo ""
echo "[headline-seed] aggregating ..."
$PYTHON scripts/summarize_seed_robustness.py \
  --inputs outputs/uagnn-ebro-headline-seed0 \
           outputs/uagnn-ebro-headline-seed7 \
           outputs/uagnn-ebro-headline-seed123 \
  --output outputs/ebro_headline_seed_robustness.csv \
  2>&1 | tee "$LOG_DIR/ebro_headline_seed_robustness.log"

echo ""
echo "=========================================================================="
echo "[headline-seed] DONE."
echo "  Per-seed outputs:  outputs/uagnn-ebro-headline-seed{0,7,123}/"
echo "  Aggregate CSV:     outputs/ebro_headline_seed_robustness.csv"
echo "=========================================================================="
echo ""
echo "Aggregate CSV: outputs/ebro_headline_seed_robustness.csv"
echo "The Savage row of this CSV is the source of the multi-seed"
echo "numbers reported in §5.3 'Seed and reproducibility' of the paper."
