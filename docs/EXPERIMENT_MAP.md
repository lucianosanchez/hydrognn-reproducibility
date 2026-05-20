# Experiment ↔ paper map

This file links every numerical claim of the EAAI manuscript to the
experiment that produced it. If you change an experiment, update the
column "Output file" before modifying any prose elsewhere.

Legend
:   The "Phase" column refers to a phase of
    `scripts/run_paper_experiments.sh` (run `bash
    scripts/run_paper_experiments.sh <phase>` to regenerate). When no
    phase is given, the script is a standalone Python call.

## Headline tables (EAAI §5)

| Table / fig label                  | Paper §                                | Phase           | Output file(s)                                                   |
|------------------------------------|----------------------------------------|-----------------|------------------------------------------------------------------|
| `tab:baseline_seq2seq`             | 5.1 deterministic baselines            | (run baselines) | `outputs/uagnn-{*}/headline_metrics.csv` (Seq2Seq rows)          |
| `tab:vae_decisions`                | 5.2 V-Seq2Seq                          | (separate)      | `outputs/vae-*/headline_metrics.csv` (legacy in `_archive_runs/`) |
| `tab:uagnn_headline`               | 5.3.1 UA-HydroGNN headline             | `train`         | `outputs/uagnn-{synth-N16,synth-N64,ebro-headline}/headline_metrics.csv`  |
| `tab:uagnn_per_scenario`           | 5.3.2 per-scenario costs               | `train`         | idem                                                              |
| `tab:uagnn_fn`                     | 5.3.3 FN reductions                    | `train`         | idem                                                              |
| `tab:uagnn_seed_robustness`        | 5.3.4 multi-seed (headline audit)      | `run_headline_seed_robustness.sh` | `outputs/ebro_headline_seed_robustness.csv`             |
| `tab:uagnn_l_alpha`                | 5.3.5 L_α sweep                        | `l_alpha`       | `outputs/l_alpha_*.csv` (6 files)                                |
| `tab:uagnn_maximin_savage_tie`     | 5.3.6 Maximin–Savage agreement         | `ensembles`     | `outputs/ensemble_decomposition*.csv`                            |
| `tab:uagnn_variance`               | 5.3.7 variance decomposition           | `w4`            | `outputs/w4_decomp.csv`, `outputs/w4_summary.txt`                |
| (§5.4 text only)                   | 5.4 informed-routing anti-hypothesis   | `ebro_informed` | `outputs/uagnn-ebro-informed/headline_metrics.csv`               |
| `tab:remediation_grid`             | 5.5.1 remediation grid                 | (`run_remediation_grid.sh`) | `outputs/grid/grid_summary.csv`                       |
| (§5.5 text only)                   | 5.5.2 grid winner confirmation         | (`run_grid_confirm_200ep.sh`) | `outputs/grid_confirm/`                              |
| `tab:physicalization_metrics`      | 5.6.1 post-hoc physicalization         | `physicalize`   | `outputs/physicalized/*/physicalization_metrics.json`             |
| `tab:phase22_acyclic_grid`         | 5.6.2 acyclic candidate graph          | `phase22_acyclic` | `outputs/phase22_grid/grid_summary.csv`                         |
| `fig:topology_recovery`            | 5.6 topology figures                   | `phase22`       | `figs/topology_synth-{N16,N64}.pdf`                              |
| `fig:topology_physicalised`        | 5.6.1 physicalisation panels           | `physicalize`   | `figs/topology_synth-{N16,N64}_physicalized_{strict,soft}.pdf`   |
| `fig:phase22_acyclic_panels`       | 5.6.2 acyclic vs dense panels          | `phase22_acyclic` | `figs/phase22_grid_synth-{N16,N64}-M{3,6}-{dense,acyclic}.pdf` |

## Decision-theory subtables (technical report `paper_methods.tex` §4.9)

These tables provide additional decompositions used by the technical
report and reproduced in the EAAI appendix.

| Label                              | Source                                                                |
|------------------------------------|------------------------------------------------------------------------|
| `tab:uagnn_hurwicz_sweep`          | `outputs/uagnn-ebro-headline/headline_metrics.csv` (per λ row)         |
| `tab:uagnn_savage_quantile_audit`  | `outputs/ensemble_decomposition_outputs.csv`                           |
| `tab:uagnn_aleatoric_share`        | `outputs/w4_decomp.csv`, `outputs/w4_decomp_outputs.csv`               |
| `tab:uagnn_high_cost_blocks`       | `python scripts/analyze_blocks.py --ckpt outputs/uagnn-ebro-headline/modelo_uagnn` |

## Acyclic candidate graph: hypothesis verification

The §5.6.2 of the paper claims that the acyclic candidate graph
recovers structural interpretability (zero back-flow by construction)
at essentially zero training-quality cost. The check is automatic in
`outputs/phase22_grid/grid_summary.csv`:

| Pattern observed                                              | Implication for paper §5.6.2 |
|---------------------------------------------------------------|-------------------------------|
| `acyclic` NSE-rel ≥ 0.95 × `dense` NSE-rel                    | hypothesis (I) confirmed     |
| `acyclic` NSE-rel ∈ (0.5, 0.95) × `dense`                     | hypothesis (M) — trade-off    |
| `acyclic` NSE-rel < 0.5 × `dense`                             | hypothesis (D) — loops needed |

The current data (May 2026 grid) shows hypothesis (I) firmly: NSE-rel
≥ 0.976 on all four acyclic configurations vs. catastrophic
back-flow shares (60–73 %) for the dense ones.

## Per-paragraph drill-down

If a specific paragraph of the EAAI manuscript needs to be updated to
match a re-run, the table below indicates which paragraph each
output CSV feeds. The line ranges refer to the private manuscript
sources and are reproduced here only as orientation; the public
repository does not contain the LaTeX itself.

| §                                     | Lines      | What to update                                                                |
|---------------------------------------|------------|--------------------------------------------------------------------------------|
| §5.3.1 paragraph 2 (FN drops 96%)     | ~1080–1110 | After `train` re-run, swap the `FN: 99 → 4` numbers against the new headline. |
| §5.3.4 seed and reproducibility       | ~1115–1140 | After `run_headline_seed_robustness.sh`, swap mean/std from `outputs/ebro_headline_seed_robustness.csv`. The `robustness` phase of the orchestrator targets the *remediated* version (for tab:uagnn_remediation_ablation_results) and writes to `outputs/ebro_remediated_seed_robustness.csv`. |
| §5.3.5 L_α                            | ~1240–1290 | After `l_alpha`, swap the α=0.10 vs α=0.50 entries (predictor and cost modes).|
| §5.3.6 Maximin–Savage tie             | ~1300–1330 | After `ensembles`, swap the tie percentages.                                  |
| §5.5.1 remediation grid               | ~1340–1370 | After `run_remediation_grid.sh`, swap the eight-row config table.             |
| §5.6.1 post-hoc physicalization       | ~1380–1420 | After `physicalize`, swap `backflow_share_original` and `verify_nse_relative` |
|                                       |            | numbers. Update both `strict` and `soft` mode rows.                          |
| §5.6.2 acyclic candidate              | ~1425–1500 | After `phase22_acyclic`, swap the grid table and confirm the hypothesis tag. |

## When in doubt

* The **paper is downstream** of the experiments. If a number in the
  paper does not match `outputs/`, the paper is wrong, not the
  experiments. Re-derive from the CSV.
* If the EAAI numbers cannot be traced to a CSV alone, contact the
  corresponding author; the supplementary derivations (sub-table
  breakdowns and chronological run logs) live in a separate private
  repository.
