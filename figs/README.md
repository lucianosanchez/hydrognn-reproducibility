# `figs/`

Figures that accompany the EAAI manuscript. Each PDF here is generated
by one of the scripts in `scripts/`. The mapping is:

| File | Producer | Phase |
|---|---|---|
| `topology_synth-N16.pdf` | `scripts/plot_learned_vs_truth.py` | `phase22` |
| `topology_synth-N64.pdf` | `scripts/plot_learned_vs_truth.py` | `phase22` |
| `topology_synth-N16_physicalized.pdf` | `scripts/physicalize_topology.py` | `physicalize` |
| `topology_synth-N16_physicalized_strict.pdf` | idem (`--mode strict`) | `physicalize` |
| `topology_synth-N16_physicalized_soft.pdf` | idem (`--mode soft`) | `physicalize` |
| `topology_synth-N64_physicalized*.pdf` | idem | `physicalize` |
| `phase22_grid_synth-{N16,N64}-M{3,6}-{dense,acyclic}.pdf` | `scripts/physicalize_topology.py` | `phase22_acyclic` |

If a figure is missing, regenerate it by running its producer:

```bash
# Phase 2.2 topology figures (fig:topology_recovery in §5.5)
bash scripts/run_paper_experiments.sh phase22

# Physicalised topology panels (fig:topology_physicalised in §5.5.1)
bash scripts/run_paper_experiments.sh physicalize

# Acyclic-candidate-graph 8-panel grid (fig:phase22_acyclic_panels in §5.5.2)
bash scripts/run_paper_experiments.sh phase22_acyclic
```
