# hydrognn — reproducibility package

This repository is the **reproducibility companion** to the article

> **Cost-aware graph hydrology for low-flow abstraction scheduling in
> regulated basins.**
> Sánchez, L., Ranilla-Cortina, S., Izaguirre, C., Couso, I.
> *Engineering Applications of Artificial Intelligence*, 2026 (under review).

It contains the **code, data, scripts and pre-computed result tables**
needed to reproduce every numerical result, table and figure of the
manuscript. The manuscript sources (LaTeX, PDF and earlier drafts) are
*not* included; they live in a separate private repository.

## What is in here

| Directory | Purpose |
|---|---|
| `seq2seq_runoff/` | Python package: Seq2Seq baseline, V-Seq2Seq, HydroGNN (three phases), UA-HydroGNN, decision criteria, evaluation utilities |
| `synth_simulator/` | Synthetic basin generator (directed-tree topology + rainfall + linear-reservoir storage) |
| `scripts/` | Entry points (bash + Python) to train, evaluate and audit |
| `datos-06-07-2023/` | Real Ebro basin upstream of Tudela, daily, 2014–2023 |
| `datos-synth/` | Synthetic basin N₁=16 (full and partial visibility) |
| `datos-synth-N64/` | Synthetic basin N₁=64 |
| `outputs/` | Pre-computed result tables (CSVs) and trained checkpoints |
| `figs/` | Figures referenced from the manuscript |
| `docs/` | Architecture overview, data layout, experiment → paper mapping |

## Quick start

```bash
git clone https://github.com/lucianosanchez/hydrognn-reproducibility.git
cd hydrognn-reproducibility

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

# Smoke test
python -c "import seq2seq_runoff, synth_simulator; print('imports OK')"

# Inspect a key result CSV directly
head outputs/persistence_baseline.csv
head outputs/ebro_headline_seed_robustness.csv

# Reproduce all paper experiments (~4–5 h CPU on Apple M-series)
bash scripts/run_paper_experiments.sh all

# Reproduce only the five post-hoc audits documented in the paper
PYTHON=python bash scripts/run_remote_paper_audits.sh
```

For a detailed walkthrough of each experimental phase, expected wall
times and exact output paths, see **[REPRODUCING.md](REPRODUCING.md)**.
For the mapping between each numerical claim of the manuscript and the
CSV that produces it, see **[docs/EXPERIMENT_MAP.md](docs/EXPERIMENT_MAP.md)**.

## What the package implements

* **Seq2Seq baseline** — LSTM encoder–decoder over basin-aggregated
  rainfall, storage and discharge. Attributed to the LSTM-for-hydrology
  line opened by Kratzert et al.; not a contribution of the paper, used
  here only to isolate the value of scenario-aware decisions in the
  absence of river-network structure.
* **V-Seq2Seq** — variational, scenario-conditioned extension of the
  Seq2Seq with a finite rainfall-scenario library and posterior over a
  latent basin state.
* **HydroGNN** — graph-temporal model with a directed-heterograph
  basin representation: routing nodes (memoryless) and storage nodes
  (with explicit state), three information regimes (supervised storages,
  known storage locations, latent storages) and acyclic candidate
  graphs for structural identifiability.
* **UA-HydroGNN** — uncertainty-aware extension of HydroGNN with a
  posterior over the initial hydrological state.
* **Decision layer** — scenario-conditioned cost surface over operating
  offsets, with four robust criteria (nominal-scenario, Wald maximin,
  Hurwicz maximax, Savage min–max regret).

## Reproducing the headline numbers without retraining

Most of the result tables of the paper can be reproduced **without
retraining** by reusing the pre-computed checkpoints in
`outputs/uagnn-ebro-headline/`, `outputs/uagnn-ebro-headline-seed{0,7,123}/`,
`outputs/uagnn-ebro-remediated*/` and the eight Phase 2.2 grid
checkpoints under `outputs/phase22_grid/`. Each `outputs/*.csv` is the
authoritative numerical source for the corresponding manuscript table;
see `docs/EXPERIMENT_MAP.md`.

## Software requirements

* Python ≥ 3.10
* `torch ≥ 2.1` (UA-HydroGNN, HydroGNN, synth simulator)
* `numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib`, `pyyaml`
* *Optional* — `tensorflow ≥ 2.13` + `keras ≥ 2.13` for the V-Seq2Seq
  model (the rest of the package works without them).

## Data redistribution

The Ebro CSVs in `datos-06-07-2023/` are derived from the SAIH-Ebro
public service. We redistribute them here under the conditions of that
public service; they are not modified beyond what is required to
align timestamps and column types. See `datos-06-07-2023/README.md`.
Synthetic data are released under CC-BY-4.0.

## Citation

If you use this code, the synthetic generator or the pre-computed
result tables, please cite the manuscript:

```bibtex
@article{sanchez2026hydrognn,
  title   = {Cost-aware graph hydrology for low-flow abstraction
             scheduling in regulated basins},
  author  = {S\'anchez, Luciano and Ranilla-Cortina, Sandra and
             Izaguirre, Celia and Couso, In\'es},
  journal = {Engineering Applications of Artificial Intelligence},
  year    = {2026},
  note    = {Manuscript under review}
}
```

A `CITATION.cff` file is provided so GitHub renders a "Cite this
repository" widget.

## License

* Code: **MIT** (see `LICENSE`).
* Synthetic data (`datos-synth/`, `datos-synth-N64/`): **CC-BY-4.0**.
* Real Ebro data (`datos-06-07-2023/`): redistributed under SAIH-Ebro
  public-service terms.

## Authors

Computer Science Department and Statistics Department,
Universidad de Oviedo, Gijón, Spain.

* Luciano Sánchez (`luciano@uniovi.es`) — corresponding author
* Sandra Ranilla-Cortina
* Celia Izaguirre
* Inés Couso

Funding: Cátedra TE — CHIST-ERA Escorrentía project.

## Related private repository

The manuscript sources (LaTeX, PDF, internal reports, session
transcripts) are kept in a separate private repository accessible to
the authors only. The public release in this repository is intentionally
limited to the artefacts needed to reproduce the published results.
