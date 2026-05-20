# `outputs/`

Curated experimental results that directly feed tables and figures of
the paper. Re-running `bash scripts/run_paper_experiments.sh <phase>`
overwrites the contents idempotently.

Heavyweight intermediate runs (hyperparameter sweeps, Phase-1/2.1/2.2
tuning, V-Seq2Seq β/dz grid) live **outside the repo** in
`../Escorrentía/_archive_runs/`. See `docs/EXPERIMENT_MAP.md` for the
full provenance table.

## Layout

```
outputs/
├── README.md                                    ← this file
│
├── uagnn-ebro-headline/                         ← Ebro canonical headline (719→44 under Savage)
├── uagnn-ebro-headline-seed{0,7,123}/           ← multi-seed audit of the headline
├── uagnn-ebro-remediated/                       ← Ebro with rainfall-bypass remediation
│                                                  (the regressive run; feeds §5.6
│                                                   tab:uagnn_remediation_ablation_results)
├── uagnn-ebro-remediated-seed{0,7,123}/         ← multi-seed audit of the remediated run
├── uagnn-ebro-informed/                         ← Ebro with river-length prior (§5.6.1 ablation)
├── uagnn-ebro-informed-remediated-seed{0,7}/    ← informed × multi-seed (with remediation)
├── uagnn-synth-N16/                             ← synth N=16 headline
├── uagnn-synth-N64/                             ← synth N=64 headline
├── uagnn-synth-N64-fix/                         ← N=64 with remediation (post-collapse fix)
│
│ Each uagnn-*/ contains:
│   ├── headline_metrics.csv                     ← Wald / Hurwicz / Savage rows
│   ├── modelo_uagnn/                            ← PyTorch checkpoint
│   └── ua_meta.pkl                              ← config snapshot
│
├── grid/                                        ← 8-config remediation grid (A..H)
│   └── <id>/uagnn-{synth-N16,synth-N64,ebro}/
│       └── headline_metrics.csv
├── grid_confirm/                                ← 200-epoch confirmation of C
│   └── C/uagnn-{synth-N16,synth-N64,ebro}/
│
├── phase22_grid/                                ← {M=3,6} × {dense, acyclic}
│   ├── {N16,N64}-M{3,6}-{dense,acyclic}/       ← checkpoint
│   ├── {N16,N64}-M{3,6}-{dense,acyclic}-phys/  ← post-hoc physicalization
│   └── grid_summary.csv
│
├── physicalized/                                ← post-hoc graphs (strict + soft)
│   └── {N16,N64}-{strict,soft}/
│       └── physicalization_metrics.json
│
├── ebro_headline_seed_robustness.csv            ← multi-seed audit of the headline
├── ebro_remediated_seed_robustness.csv          ← multi-seed audit of the remediated run
├── ensemble_decomposition.csv                   ← Maximin-Savage tie
├── ensemble_decomposition_outputs.csv
├── l_alpha_*.csv                                ← 6 quantile sweeps
├── l_alpha_sensitivity.csv                      ← summary
└── w4_decomp.csv / w4_summary.txt               ← variance decomposition
    w4_decomp_outputs.csv / w4_summary_outputs.txt
```

## Which file feeds which paper §?

See `docs/EXPERIMENT_MAP.md` for the full mapping. The most consulted
files:

| Paper §                              | File                                                          |
|--------------------------------------|----------------------------------------------------------------|
| §5.3.1 UA-HydroGNN headline          | `uagnn-{synth-N16,synth-N64,ebro-headline}/headline_metrics.csv` |
| §5.3.4 seed and reproducibility      | `ebro_headline_seed_robustness.csv`                            |
| §5.3.5 L_α                           | `l_alpha_ebro_{predictor,cost}.csv`                            |
| §5.3.6 Maximin–Savage tie            | `ensemble_decomposition*.csv`                                  |
| §5.3.7 variance decomposition        | `w4_decomp.csv` + `w4_summary.txt`                             |
| §5.4 informed-routing anti-hypothesis| `uagnn-ebro-informed/headline_metrics.csv`                     |
| §5.5 remediation grid                | `grid/grid_summary.csv`                                        |
| §5.6.1 post-hoc physicalization      | `physicalized/*/physicalization_metrics.json`                  |
| §5.6.2 acyclic candidate grid        | `phase22_grid/grid_summary.csv`                                |
| §5.6 remediation regression on Ebro  | `uagnn-ebro-remediated/headline_metrics.csv` (single seed) + `ebro_remediated_seed_robustness.csv` (3-seed mean) |

## Conventions

* All `headline_metrics.csv` files share the column schema documented
  in `seq2seq_runoff/evaluation.py:write_headline_metrics`.
* `modelo_uagnn/` is a folder, not a file. Load it via
  `UAHydroGNNModel.load("outputs/uagnn-ebro-headline/modelo_uagnn")`.
  The factory inspects `ua_meta.pkl` to reconstruct the right model
  variant (with/without `rain_bypass`, `lam11_init`, etc.).
* `physicalization_metrics.json` is documented in
  `seq2seq_runoff/gnn/physicalize.py`.
* CSVs are UTF-8, comma-separated, with a header row. NaN is encoded
  as the empty string.

## Re-running a single experiment

```bash
# Re-train only Ebro headline UA-HydroGNN
PYTHON=python bash scripts/run_paper_experiments.sh train

# Re-sweep L_α (no retrain needed; uses the existing checkpoints)
PYTHON=python bash scripts/run_paper_experiments.sh l_alpha

# Re-summarise the acyclic candidate grid (no retrain)
python scripts/summarize_phase22_grid.py \
    --grid-root outputs/phase22_grid \
    --output outputs/phase22_grid/grid_summary.csv
```
