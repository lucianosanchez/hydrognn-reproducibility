#!/usr/bin/env bash
# ============================================================================
# Master wrapper for the three remote audits that close the must-fix
# requirements of the EAAI submission:
#
#   (A) Persistence baseline on all three basins
#       (§5.1 tab:winners_phases persistence rows;
#        §5.3 attribution per-scenario floor).
#
#   (B) Operating cost of the Phase 2.2 latent-storages grid
#       (§5.5 tab:phase22_acyclic_operating; closes the operational
#        reading of the acyclic-vs-dense comparison).
#
#   (C) Cost-ratio re-optimisation of δ for UA-HydroGNN Ebro
#       (§5.3 tab:uagnn_cost_ratio_sensitivity — proper re-optimisation
#        instead of the fixed-δ sensitivity).
#
# Each audit is run by an atomic sub-script and writes one CSV. Wall time
# estimates are individual; the wrapper is sequential to keep memory and
# disk pressure low.
#
# USAGE
# -----
#     cd hydrognn
#     PYTHON=/path/to/python bash scripts/run_remote_paper_audits.sh
#
# Or run a single audit:
#     PYTHON=python bash scripts/run_remote_paper_audits.sh persistence
#     PYTHON=python bash scripts/run_remote_paper_audits.sh phase22
#     PYTHON=python bash scripts/run_remote_paper_audits.sh cost_ratio
#
# OUTPUTS
# -------
#     outputs/persistence_baseline.csv
#     outputs/phase22_grid/operating_cost.csv
#     outputs/uagnn_cost_ratio_sensitivity_reopt.csv
#
# The three CSVs are the numerical sources for the §5 tables of the
# paper (persistence baseline rows of tab:winners_phases; the cost-
# ratio sensitivity tab:uagnn_cost_ratio_sensitivity; and the dense-
# vs-acyclic operating cost tab:phase22_acyclic_operating).
#
# WALL TIME (cumulative, single-process M-series CPU)
# ---------------------------------------------------
#   persistence    ~1 min   (pure Python, no PyTorch)
#   phase22        ~10–15 min
#   cost_ratio     ~10–15 min
#   total all      ~25–30 min
# ============================================================================

set -euo pipefail

PYTHON="${PYTHON:-python}"
export PYTHONUNBUFFERED=1

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

LOG_DIR="outputs/_logs"
mkdir -p "$LOG_DIR"

phase="${1:-all}"

echo "=========================================================================="
echo "[remote-audits] phase = $phase   PYTHON = $PYTHON"
echo "=========================================================================="

# ----------------------------------------------------------------------------
# (A) Persistence baseline
# ----------------------------------------------------------------------------
if [[ "$phase" == "persistence" || "$phase" == "all" ]]; then
  echo ""
  echo "[A] Persistence baseline on Ebro, synth-N16, synth-N64"
  $PYTHON scripts/run_persistence_baseline.py \
    > "$LOG_DIR/persistence_baseline.log" 2>&1 \
    || echo "[A] FAILED — see $LOG_DIR/persistence_baseline.log"
  echo "[A] Result: outputs/persistence_baseline.csv"
fi

# ----------------------------------------------------------------------------
# (B) Phase 2.2 operating cost
# ----------------------------------------------------------------------------
if [[ "$phase" == "phase22" || "$phase" == "all" ]]; then
  echo ""
  echo "[B] Phase 2.2 operating cost — 8 checkpoints under scenario library"
  PYTHON="$PYTHON" bash scripts/evaluate_phase22_operating_cost.sh \
    > "$LOG_DIR/phase22_operating_cost.log" 2>&1 \
    || echo "[B] FAILED — see $LOG_DIR/phase22_operating_cost.log"
  echo "[B] Result: outputs/phase22_grid/operating_cost.csv"
fi

# ----------------------------------------------------------------------------
# (C) Cost-ratio re-optimisation
# ----------------------------------------------------------------------------
if [[ "$phase" == "cost_ratio" || "$phase" == "all" ]]; then
  echo ""
  echo "[C] Cost-ratio re-optimisation of δ for UA-HydroGNN Ebro"
  $PYTHON scripts/analyze_cost_ratio_resweep.py \
    --ckpt outputs/uagnn-ebro-headline/modelo_uagnn \
    --directorio-datos datos-06-07-2023 --firma 580734 \
    --rolling-inicio 2020-01-01 \
    --ratios 10 50 100 200 500 \
    --output outputs/uagnn_cost_ratio_sensitivity_reopt.csv \
    > "$LOG_DIR/cost_ratio_reopt.log" 2>&1 \
    || echo "[C] FAILED — see $LOG_DIR/cost_ratio_reopt.log"
  echo "[C] Result: outputs/uagnn_cost_ratio_sensitivity_reopt.csv"
fi

echo ""
echo "=========================================================================="
echo "[remote-audits] DONE. Produced CSVs (numerical sources of §5 tables):"
echo "  outputs/persistence_baseline.csv               (persistence rows of tab:winners_phases)"
echo "  outputs/phase22_grid/operating_cost.csv        (tab:phase22_acyclic_operating)"
echo "  outputs/uagnn_cost_ratio_sensitivity_reopt.csv (tab:uagnn_cost_ratio_sensitivity)"
echo "=========================================================================="
