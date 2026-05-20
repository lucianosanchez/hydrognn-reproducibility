# Model overview

This document gives a per-module tour of `seq2seq_runoff/` and
`synth_simulator/`. For a higher-level description of the science see
the EAAI manuscript that accompanies this repository.

## The Seq2Seq baseline

```
                              ┌─────────────┐
                              │  encoder    │
   history (PACUM, EACUM,     │  LSTM(d)    │── state h, c
   A284) ────────────────────►│             │
                              └─────────────┘
                                              │
                          ┌───────────────────┴───────────────────┐
                          │                                       │
              ┌──────────▼──────────┐               ┌────────────▼─────────┐
   PACUM      │ decoder_embalse     │               │ decoder_caudal       │
   future  ──►│ LSTM(d) → Dense(1)  │── EACUM       │ LSTM(d) → Dense(1)   │── A284
              └─────────────────────┘   future      └──────────────────────┘   future
                                                        ▲
                                  PACUM, EACUM ─────────┘
                                  future
```

* The **encoder** compresses `HISTORIA = 20` days of aggregate
  rainfall (PACUM), aggregate reservoir storage (EACUM) and discharge
  (A284) into a state.
* The **reservoir decoder** predicts `HORIZONTE = 10` days of future
  EACUM from future rainfall (real or "worst case = 0").
* The **discharge decoder** receives that reservoir prediction and
  produces the 10-day discharge.
* In production two scenarios are evaluated: observed rainfall
  (validation) and zero rainfall (conservative operational decision).

This is `seq2seq_runoff.model.Seq2SeqRunoffModel`. It deliberately
ignores the graph structure of the basin — it is the *no-prior*
baseline against which the GNN variants are measured.

## The Python package

| File                  | Lines | Role                                                                    |
|-----------------------|-------|--------------------------------------------------------------------------|
| `basin.py`            | 121   | `BasinSpec` / `StationSpec` — generic basin description                  |
| `basins/ebro.py`      | 222   | Ebro factory + graph                                                     |
| `basins/synth.py`     | 167   | Synthetic-basin factory                                                  |
| `config.py`           | 72    | `Config` holds hyper-parameters; every `Config` carries a `BasinSpec`    |
| `data.py`             | 106   | Reads station CSVs through the BasinSpec                                 |
| `transforms.py`       | 63    | Invertible discharge transforms (`logit_inv`, log-scale, etc.)           |
| `windows.py`          | 83    | Sliding-window builder for Seq2Seq                                       |
| `calibration.py`      | 71    | PCHIP monotone spline (logit → observed flow)                            |
| `losses.py`           | 63    | `embalse_loss` + `caudal_loss` of the Seq2Seq baseline                   |
| `model.py`            | 315   | `RunoffModel` abstract base + `Seq2SeqRunoffModel`                       |
| `vae.py`              | 439   | V-Seq2Seq variational baseline (scenario library)                        |
| `decision.py`         | 294   | Decision selectors (naive, Wald, Hurwicz, Savage)                        |
| `scenarios.py`        | 252   | Climatic and initial-state scenario generators                           |
| `evaluation.py`       | 491   | Rolling-window evaluation, FN/FP, total cost, max-regret                 |
| `plotting.py`         | 392   | Shared plotting utilities (confusion, cost curves, …)                    |
| `ua_gnn.py`           | 668   | UA-HydroGNN — Monte-Carlo posterior over initial state + $L_\alpha$      |
| `gnn/core.py`         | 416   | Heterogeneous message-passing on directed bipartite Type-1/Type-2 graph  |
| `gnn/graph.py`        | 346   | `BasinGraph`, dense and `acyclic_candidate_graph(M, anchor_strategy)`    |
| `gnn/model.py`        | 450   | `HydroGNNPhase{1,2_1,2_2}` — three levels of geographic information     |
| `gnn/dataset.py`      | 164   | Window builder honouring the heterograph                                 |
| `gnn/gates.py`        | 58    | Gated combinations for storages                                          |
| `gnn/losses.py`       | 87    | Cost-weighted asymmetric loss                                            |
| `gnn/physicalize.py`  | 588   | Post-hoc graph cleanup (`strict` and `soft` modes)                       |
| `gnn/viz.py`          | 508   | 2-panel and 3-panel topology visualisations                              |

## The synthetic basin generator

`python -m synth_simulator <config.yaml>` runs the simulator. See
`synth_simulator/example_basin.yaml` for a minimal config. The pieces:

| File                   | Role                                                                |
|------------------------|----------------------------------------------------------------------|
| `topology_generator.py`| Branching directed graphs with configurable N, branching ratio, seed |
| `climate.py`           | Synthetic rainfall (Poisson + Gamma intensity)                       |
| `hydro.py`             | Linear-reservoir routing + Type-2 storage dynamics                   |
| `config.py`            | Pydantic config schema                                               |
| `output.py`            | CSV writer matching the operator-side conventions                    |
| `viz.py`               | Network plot                                                         |

The output of one simulator run is a directory with the same layout as
`datos-06-07-2023/` (one CSV per station, plus a `manifest.yaml` that
describes the graph). Both ML pipelines (`seq2seq_runoff` and
`seq2seq_runoff.gnn`) read this layout interchangeably.

## Conventions

* All flows are stored in `m³/s` with timestamps in UTC.
* The asymmetric cost is encoded once in `evaluation.py`:
  `c_FN = 100, c_FP = 1`.
* The hydropower abstraction threshold is `Q_min = 30 m³/s` for Ebro,
  configurable per basin via `BasinSpec.q_min`.
* The decision horizon is `HORIZONTE = 10` days; the historical window
  is `HISTORIA = 20` days.
* Reservoirs are named with the SAIH convention (Ebro): E001 = Ebro,
  E029 = Yesa, E075 = Itoiz. Synthetic reservoirs are R0, R1, R2.
* Posterior-flow decision quantile: `α` in
  `seq2seq_runoff.ua_gnn.decide_l_alpha(...)`. Default `α = 0.10`.

## Reading the code

A productive starting point is:

1. `seq2seq_runoff/model.py:Seq2SeqRunoffModel.fit_and_evaluate()` —
   the simplest end-to-end pipeline.
2. `seq2seq_runoff/decision.py` — the four decision selectors.
3. `seq2seq_runoff/gnn/core.py:HydroGNNCore.forward()` — the
   heterogeneous message-passing step.
4. `seq2seq_runoff/ua_gnn.py:UAHydroGNNModel.fit_and_evaluate()` — the
   posterior-over-initial-state extension.
5. `seq2seq_runoff/gnn/physicalize.py:physicalize_topology()` — the
   post-hoc graph rewriter that recovers an interpretable map without
   retraining.
