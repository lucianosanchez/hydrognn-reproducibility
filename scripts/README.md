# `scripts/`

Entry points (Python and bash) that drive every experiment of the
paper. The canonical pattern is:

```
bash scripts/run_paper_experiments.sh <phase>
```

where `<phase>` is one of `data | train | robustness | l_alpha | w4 |
ensembles | phase22 | ebro_informed | physicalize | phase22_acyclic |
all`. See `REPRODUCING.md` and `docs/EXPERIMENT_MAP.md` for the
mapping to paper sections.

## Bash orchestrators

| File                              | Wall time | Purpose                                                       |
|-----------------------------------|-----------|---------------------------------------------------------------|
| `run_paper_experiments.sh`        | ~4–5 h    | Canonical paper reproduction (10 phases, all sequential)      |
| `run_remediation_grid.sh`         | ~7–9 h    | Optional 8-config × 3-dataset grid (§5.5)                      |
| `run_grid_confirm_200ep.sh`       | ~2–3 h    | Optional 200-epoch confirmation of the grid winner            |

All three respect a `PYTHON=…` environment variable so they can be
pointed at a specific interpreter.

## Python entry points (training / evaluation)

| File                              | Purpose                                                                |
|-----------------------------------|------------------------------------------------------------------------|
| `run_baseline.py`                 | Deterministic Seq2Seq + HydroGNN baselines (Phase 1/2.1/2.2)           |
| `run_gnn.py`                      | HydroGNN single-phase trainer (`--fase 1|2.1|2.2`)                     |
| `run_vae_experiment.py`           | V-Seq2Seq variational model + decision selectors                       |
| `run_ua_gnn_experiment.py`        | UA-HydroGNN main experiment                                            |
| `run_all_phases.py`               | Side-by-side Phase 1 / 2.1 / 2.2 on a single basin                     |
| `run_synth_experiments.py`        | Reproduce the synthetic-basin headline                                 |
| `run_synth_sweep.py`              | Hyperparameter sweep on synthetic basins                               |
| `make_synth_basin.py`             | Regenerate `datos-synth-N64/`                                          |

## Python entry points (analysis)

| File                              | Purpose                                                                |
|-----------------------------------|------------------------------------------------------------------------|
| `analyze_l_alpha.py`              | α-quantile sweep (predictor vs cost mode)                              |
| `analyze_w4_variance.py`          | Aleatoric vs epistemic variance decomposition                          |
| `analyze_scenario_ensembles.py`   | Maximin–Savage agreement audit                                         |
| `analyze_blocks.py`               | High-cost decision block analysis                                      |
| `compare_models.py`               | Pairwise model comparison report                                       |
| `physicalize_topology.py`         | Post-hoc graph cleanup (`--mode strict|soft`)                          |
| `plot_learned_vs_truth.py`        | 2-panel topology recovery figure                                       |
| `make_paper_figures.py`           | Shared figure-rendering pipeline                                       |
| `extract_formal_positions.py`     | Reservoir position extraction from checkpoints                         |
| `summarize_phase22_grid.py`       | Aggregate Phase 2.2 grid results                                       |
| `summarize_remediation_grid.py`   | Aggregate 8-config remediation grid                                    |
| `summarize_seed_robustness.py`    | Mean ± std table from multi-seed runs                                  |

## Python entry points (diagnostics)

| File                              | Purpose                                                                |
|-----------------------------------|------------------------------------------------------------------------|
| `diag_e21_directions.py`          | Audit of $E_{21}$ edge directions in a learned graph                   |
| `diag_n64_collapse.py`            | Replication of the N=64 collapse symptom and step-by-step trace        |
| `tune.py`                         | Generic hyperparameter tuner used during early development             |

## How to add a new entry point

1. Place the new script in `scripts/`.
2. Make it call `python -c "import seq2seq_runoff"` cleanly when
   invoked from the repo root.
3. If it produces a paper-relevant artifact, add an entry in
   `docs/EXPERIMENT_MAP.md`.
4. If it should run as part of the canonical reproduction, add a phase
   to `run_paper_experiments.sh` (copy an existing phase block as a
   template).
