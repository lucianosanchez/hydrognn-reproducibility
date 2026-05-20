# Reproducing the paper results

This document is the *operational* companion to the EAAI manuscript. It
tells you **exactly what to run** to regenerate every numerical result,
table and figure of the paper, in what order, and how long each block
takes on a single-process M-series Apple Silicon CPU.

All commands assume the repository root as the working directory:

```bash
cd hydrognn
```

---

## 0. Environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

or

```bash
conda env create -f environment.yml
conda activate hydrognn
pip install -e .
```

Smoke test:

```bash
python -c "import seq2seq_runoff, synth_simulator; print('imports OK')"
bash scripts/run_paper_experiments.sh --help 2>&1 | head  # prints the phase menu
```

> The V-Seq2Seq variational model needs `tensorflow + keras`. Without
> them, every other experiment still works; only
> `seq2seq_runoff.vae.VAESeq2SeqRunoffModel` will raise `ImportError`.

If your interpreter is not on the PATH default:

```bash
PYTHON=/path/to/python bash scripts/run_paper_experiments.sh <phase>
```

---

## 1. One-command reproduction

The full paper experiment suite is orchestrated by a single bash
script:

```bash
bash scripts/run_paper_experiments.sh all
```

Wall time: **≈ 4–5 h CPU**. The phases run sequentially; output goes to
`outputs/<sub-dir>/` and logs to `outputs/_logs/`. Each phase is
idempotent (re-running with the same arguments overwrites the CSVs).

For interactive use, lift one phase at a time:

| Phase            | Wall time | Produces                                                                |
|------------------|-----------|--------------------------------------------------------------------------|
| `data`           | ~5 min    | regenerates `datos-synth-N64/` if not present                            |
| `train`          | ~1.5–2 h  | UA-HydroGNN on synth-N16, synth-N64, Ebro (seed 42)                      |
| `robustness`     | ~1.5 h    | Ebro with seeds {0, 7, 123} for multi-seed robustness                    |
| `l_alpha`        | ~30 min   | α ∈ {0.05…0.95} sweep on 3 datasets × 2 quantile modes                  |
| `w4`             | ~5 min    | aleatoric vs epistemic decomposition (Ebro, top-60 high-cost days)       |
| `ensembles`      | ~10 min   | Maximin–Savage agreement audit (300 Ebro days)                           |
| `phase22`        | ~30–40 min | HydroGNN Phase 2.2 on synth-N16 and synth-N64 (learned reservoir loc.)  |
| `ebro_informed`  | ~30 min   | Ebro UA-HydroGNN with river-length prior (informed routing init)         |
| `physicalize`    | ~2 min    | post-hoc topology cleanup on Phase 2.2 checkpoints                       |
| `phase22_acyclic`| ~50 min   | grid {M=3,6} × {dense, acyclic} × {N16, N64} for identifiability claim   |
| `all`            | ~4–5 h    | all of the above, in order                                               |

Each phase calls atomic Python scripts in `scripts/`. The bash file is
the canonical orchestrator; if you want to inspect a specific Python
call, open `scripts/run_paper_experiments.sh` and grep for the phase
name.

---

## 2. Mapping experiments ↔ paper sections

| Paper §             | Table / figure                       | Experiment phase           | Outputs                                                                |
|---------------------|--------------------------------------|----------------------------|------------------------------------------------------------------------|
| §5.1 baselines      | `tab:baseline_seq2seq`               | `run_baseline.py`          | `outputs/uagnn-*` headline rows                                        |
| §5.2 V-Seq2Seq      | `tab:vae_decisions`                  | `run_vae_experiment.py`    | `outputs/uagnn-*/` headline rows (V-Seq2Seq columns)                    |
| §5.3.1 UA-HydroGNN headline   | `tab:uagnn_headline`         | `train`                    | `outputs/uagnn-{synth-N16,synth-N64,ebro-headline}/headline_metrics.csv` |
| §5.3.2 per-scenario | `tab:uagnn_per_scenario`             | `train`                    | idem                                                                   |
| §5.3.3 FN reduction | `tab:uagnn_fn`                       | `train`                    | idem                                                                   |
| §5.3.4 seed audit   | `tab:uagnn_seed_robustness`          | `run_headline_seed_robustness.sh` | `outputs/ebro_headline_seed_robustness.csv`                     |
| §5.3.5 L_α decision | `tab:uagnn_l_alpha`                  | `l_alpha`                  | `outputs/l_alpha_*.csv`                                                |
| §5.3.6 Maximin-Savage tie     | `tab:uagnn_maximin_savage_tie` | `ensembles`             | `outputs/ensemble_decomposition*.csv`                                  |
| §5.3.7 variance decomp        | `tab:uagnn_variance`         | `w4`                       | `outputs/w4_decomp.csv`, `outputs/w4_summary.txt`                      |
| §5.4 anti-hypothesis | (text only)                        | `ebro_informed`            | `outputs/uagnn-ebro-informed/`                                          |
| §5.5.1 remediation grid       | `tab:remediation_grid`     | `run_remediation_grid.sh`  | `outputs/grid/grid_summary.csv`                                        |
| §5.5.2 grid confirm 200 epochs | (text only)                | `run_grid_confirm_200ep.sh`| `outputs/grid_confirm/`                                                |
| §5.6.1 post-hoc physicalization | `tab:physicalization_metrics`  | `physicalize`        | `outputs/physicalized/{N16,N64}-{strict,soft}/physicalization_metrics.json` + `figs/topology_*_physicalized*.pdf` |
| §5.6.2 acyclic candidate grid | `tab:phase22_acyclic_grid` | `phase22_acyclic`          | `outputs/phase22_grid/grid_summary.csv` + `figs/phase22_grid_*.pdf`     |

The `docs/EXPERIMENT_MAP.md` file expands this with file paths and
expected metric values.

---

## 3. Detail by phase

### 3.1 `data` — regenerate synthetic datasets

```bash
bash scripts/run_paper_experiments.sh data
```

If `datos-synth-N64/full/manifest.yaml` already exists the phase is
a no-op. Otherwise it calls

```bash
python scripts/make_synth_basin.py --n-type1 64 --branching 1.5 --seed 0 --output datos-synth-N64
```

The N=16 dataset (`datos-synth/full/`) ships in the repository.

### 3.2 `train` — UA-HydroGNN canonical training (seed 42)

Three sequential runs over `synth-N16`, `synth-N64` and `Ebro`. The
N=64 run uses the remediation flags (`--warmup-epochs 80
--ramp-epochs 40 --free-bits 0.02 --rain-bypass --lam11-init 2.0`),
documented in §5.5. The other two use defaults.

| Dataset    | epochs | K_train | batch | max_windows | extras                                       |
|------------|--------|---------|-------|-------------|----------------------------------------------|
| synth-N16  | 200    | 10      | 32    | —           | defaults                                     |
| synth-N64  | 200    | 15      | 64    | 3000        | warmup 80, ramp 40, free_bits 0.02, bypass    |
| Ebro       | 200    | 10      | 32    | —           | defaults                                     |

Outputs (two parallel runs are produced; the headline is what the paper reports as canonical, the remediated is what feeds the §5.6 regression table):

```
outputs/uagnn-synth-N16/            headline_metrics.csv, modelo_uagnn/, ua_meta.pkl
outputs/uagnn-synth-N64/            (idem)
outputs/uagnn-ebro-headline/        Ebro canonical, no remediation flags
outputs/uagnn-ebro-remediated/      Ebro with rainfall-bypass remediation
```

`headline_metrics.csv` columns: `model, criterion, scenario_mode,
fn_count, fp_count, total_cost, max_regret, …` (Wald, Hurwicz,
Savage).

### 3.3 Multi-seed audits

There are two separate multi-seed audits, with two separate scripts:

```bash
# Headline audit (no remediation): confirms the 719→44 reduction.
PYTHON=python bash scripts/run_headline_seed_robustness.sh
# Writes outputs/uagnn-ebro-headline-seed{0,7,123}/
# and    outputs/ebro_headline_seed_robustness.csv

# Remediated audit (with rainfall-bypass): confirms the regression
# documented in §5.6 tab:uagnn_remediation_ablation_results.
PYTHON=python bash scripts/run_paper_experiments.sh robustness
# Writes outputs/uagnn-ebro-remediated-seed{0,7,123}/
# and    outputs/ebro_remediated_seed_robustness.csv
```

### 3.4 `l_alpha` — quantile sensitivity sweep

For each of {synth-N16, synth-N64, Ebro} × each `quantile-mode` ∈
{`predictor`, `cost`}, sweep
α ∈ {0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95}.

`predictor` mode is the operational version (decision based on the
α-quantile of the predicted flow). `cost` mode is the strict version
of eq.(11) of the paper (α-quantile of the per-sample cost).

Outputs:

```
outputs/l_alpha_synth-N16_predictor.csv
outputs/l_alpha_synth-N16_cost.csv
outputs/l_alpha_synth-N64_predictor.csv
outputs/l_alpha_synth-N64_cost.csv
outputs/l_alpha_ebro_predictor.csv
outputs/l_alpha_ebro_cost.csv
```

### 3.5 `w4` — variance decomposition (epistemic vs aleatoric)

Identifies the top-60 high-cost days under Savage on Ebro and
decomposes the predictive variance into

  aleatoric + epistemic(initial-state) + epistemic(scenario)

Outputs: `outputs/w4_decomp.csv`, `outputs/w4_summary.txt`.

### 3.6 `ensembles` — Maximin–Savage agreement audit

On 300 random Ebro days, generates 5×8 stochastic sub-scenarios and
measures the fraction of days where Maximin and Savage select an
identical δ*. Includes the robust q_{0.95} per-scenario aggregation.

Output: `outputs/ensemble_decomposition.csv`,
`outputs/ensemble_decomposition_outputs.csv`.

### 3.7 `phase22` — HydroGNN Phase 2.2

Trains HydroGNN with `M_latent` formal reservoir slots and learned
positions on the two synthetic basins:

```
outputs/hydrognn-phase22-synth-N16/   core.pt + meta.pkl  (M_latent=3, obs={SM,ST1})
outputs/hydrognn-phase22-synth-N64/   (M_latent=4, ~15 obs stations)
figs/topology_synth-N16.pdf
figs/topology_synth-N64.pdf
```

The figures are 2-panel "ground-truth vs learned" maps produced by
`scripts/plot_learned_vs_truth.py`.

### 3.8 `ebro_informed` — anti-hypothesis: informed routing init

Re-trains Ebro UA-HydroGNN passing river-segment lengths as a prior
(`--river-velocity 50.0`). Compares against `outputs/uagnn-ebro-headline`
(no length prior).

Output: `outputs/uagnn-ebro-informed/`. The §5.4 of the paper reports
that the river-length prior *does not help* on this basin — an
anti-hypothesis result kept on purpose.

### 3.9 `physicalize` — post-hoc topology cleanup

Applies `seq2seq_runoff.gnn.physicalize` to the Phase 2.2 checkpoints
in both `strict` and `soft` modes, generating 3-panel figures
`figs/topology_<dataset>_physicalized_<mode>.pdf` and metrics in
`outputs/physicalized/<dataset>-<mode>/physicalization_metrics.json`.

Key columns of the JSON:
* `backflow_share_original` — fraction of $E_{21}$ mass that pointed
  upstream in the learned graph (large = chaotic).
* `verify_nse_relative` — 1.0 ⇒ predicted q*(t) unchanged after the
  rewrite; <0.5 ⇒ the loops were operationally needed.

### 3.10 `phase22_acyclic` — acyclic candidate graph (8-run grid)

Trains the grid {dense, acyclic} × {M_latent ∈ {3, 6}} × {synth-N16,
synth-N64} (8 runs) and applies post-hoc physicalization in strict
mode to each. Summary table:

```
outputs/phase22_grid/grid_summary.csv
figs/phase22_grid_*.pdf       # 8 figures
```

In the acyclic variant, candidate edges are restricted by deterministic
BFS-uniform anchors, guaranteeing `backflow_share_original = 0` by
construction. The grid confirms hypothesis (I) of §5.6.2: acyclic
matches dense on NSE-rel while removing 60–73 % of the back-flow mass
of the dense variant.

### 3.11 `run_remediation_grid.sh` (optional)

Independent 8-configuration grid search (`A`…`H`) over
{`rain_bypass`, `lam11_init`, `warmup_epochs`, `ramp_epochs`,
`free_bits`} on the three datasets at 100 epochs.

```bash
bash scripts/run_remediation_grid.sh                  # ~7–9 h CPU
python scripts/summarize_remediation_grid.py \
    --grid-root outputs/grid \
    --output outputs/grid/grid_summary.csv
```

Output: `outputs/grid/<config>/uagnn-<dataset>/headline_metrics.csv`,
plus `outputs/grid/grid_summary.csv`.

### 3.12 `run_grid_confirm_200ep.sh` (optional)

Long-form 200-epoch confirmation of the config-`C` winner of the
remediation grid (`lam11_init=1.0`, no other remediation).

```bash
bash scripts/run_grid_confirm_200ep.sh                # ~2–3 h CPU
```

Output: `outputs/grid_confirm/C/uagnn-{synth-N16,synth-N64,ebro}/`.

---

## 4. Provenance of result tables

The CSVs under `outputs/` are the authoritative numerical source for the
manuscript tables. The map between each `outputs/*.csv` and the
specific manuscript table or paragraph it produces is given in
`docs/EXPERIMENT_MAP.md`.

This public repository does **not** include the manuscript LaTeX or
PDF sources, nor the internal technical reports and session
transcripts. Those artefacts live in a separate private repository
accessible to the authors only.

---

## 6. Troubleshooting

* **`AttributeError: 'UAHydroGNNCore' object has no attribute 'bypass_head'`** —
  you are loading a pre-remediation checkpoint with new code, or
  vice-versa. `UAHydroGNNModel.load()` consults `rain_bypass` in
  `ua_meta.pkl`; old checkpoints default it to `False`.
* **`ImportError: cannot import name 'keras' from 'tensorflow'`** —
  the V-Seq2Seq model requires `tf-keras` or the matching
  `tensorflow + keras` versions pinned in `requirements.txt`. Other
  experiments do not need it.
* **`pdflatex` fails with "undefined reference"** — run it twice. The
  bibliography is in-document (no `bibtex` step needed).
* **Live stdout not appearing in `tail -F`** — Python buffers it. The
  orchestrator script exports `PYTHONUNBUFFERED=1`; if you launch a
  Python script directly, set it yourself.
* **Out-of-memory on Ebro** — `--max-windows 2000 --batch-size 32`
  on Ebro typically uses ~3 GB RAM. Lower `--K-train` to reduce.
* **Non-deterministic results across machines** — the seeds reseed
  Python, NumPy and PyTorch but not CUDA RNG. Stick to CPU for exact
  reproduction.

---

## 7. Wall-time budget (measured on Apple M-series, single process)

| Phase            | Wall time         |
|------------------|-------------------|
| synth-N16 train  | ~15 min           |
| synth-N64 train  | ~45 min           |
| Ebro train       | ~30 min           |
| L_α sweep        | ~12 min / dataset |
| W4 decomposition | ~3 min            |
| Ensembles audit  | ~15 min           |
| Phase 2.2 N16    | ~10 min           |
| Phase 2.2 N64    | ~25 min           |
| Physicalization  | ~30 s × 4         |
| Phase 2.2 grid (8 runs) | ~50 min     |
| **Total (`all`)**| **~4.5 h**        |

Parallelisation would bring it to ~2 h, but `run_paper_experiments.sh`
is intentionally sequential to keep memory and disk pressure under
control.
