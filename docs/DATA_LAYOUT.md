# Data layout

This document describes the on-disk format expected by every loader in
`seq2seq_runoff.data` and `seq2seq_runoff.basins.*`. Both the real
Ebro dataset and the synthetic generator produce the same layout so
that downstream code is agnostic to provenance.

## Per-basin directory layout

```
datos-<basin-id>/
├── manifest.yaml                            # only synthetic basins (see below)
├── DatosHistoricos_<firma>_<station>.csv    # one CSV per station
├── DatosHistoricos_<firma>_<station>.csv
└── …
```

The basin is identified by a `firma` token (`580734` for Ebro,
`SYNTH001` for N=16, `SYNTH-N64` for N=64). Within a basin every
station has its own CSV; the station code embeds:

* The first 1–3 characters identify the sub-catchment / location.
* The next character is the station type (`L`, `Y`, `T`, `E` for
  Ebro; `M`, `Q`, `R0`, `S` for synthetic).
* The suffix encodes the magnitude: `PACUM` (accumulated rainfall,
  mm), `EMBA` (reservoir storage, hm³), `QRIO` (river discharge,
  m³/s), `EMB` (synthetic reservoir storage), `CAUDAL` (synthetic
  discharge).

Examples:

```
DatosHistoricos_580734_A284Z65QRIO1.csv          ← Ebro discharge at Tudela
DatosHistoricos_580734_E001L65VEMBA.csv          ← Ebro reservoir
DatosHistoricos_580734_EM01L84PACUM.csv          ← rain sub-catchment EM01
DatosHistoricos_SYNTH001_SQ-CAUDAL.csv           ← synthetic discharge
DatosHistoricos_SYNTH001_SR0-EMB.csv             ← synthetic reservoir 0
```

## CSV format

Each file is comma-separated with two columns:

```csv
fecha,valor
2014-06-01,0.4
2014-06-02,0.0
…
```

* `fecha` — `YYYY-MM-DD` date (UTC).
* `valor` — magnitude in the unit implied by the station code suffix.
  Missing values are encoded as blank or `NaN`.

The reader (`seq2seq_runoff.data.load_basin_csv`) performs:

1. Parsing of `fecha` to pandas datetime.
2. Coercion of `valor` to float.
3. Reindexing onto the basin's daily grid (forward-fill ≤ 2 days,
   then NaN).
4. Validation against `BasinSpec.station_specs`.

## `manifest.yaml` (synthetic only)

The synthetic generator dumps the full graph for downstream loading:

```yaml
basin_id: SYNTH001
n_type1: 16
n_type2: 3
seed: 0
branching_ratio: 1.5
q_min: 30.0
stations:
  - id: SM-PACUM
    type: rain
    node: SM
  - id: ST1-PACUM
    type: rain
    node: ST1
  …
  - id: SQ-CAUDAL
    type: discharge
    node: outlet
  - id: SR0-EMB
    type: storage
    node: R0
graph:
  type1_nodes: [SM, SM2, ST1, ST1A, ST2, …]
  type2_nodes: [R0, R1, R2]
  edges_11:                          # river arcs
    - {from: SM, to: ST1}
    - {from: SM, to: ST2}
    …
  edges_12:                          # catchment → reservoir
    - {from: ST1, to: R0}
    …
  edges_21:                          # reservoir → downstream
    - {from: R0, to: ST1A}
```

The Ebro basin's graph lives in `seq2seq_runoff.basins.ebro.EBRO_GRAPH`
hard-coded, because the topology was provided by the plant operator
not by a generic GIS layer.

## Datasets shipped with the repository

| Path                                  | Basin id    | N₁ | N₂ | Q_min        | Period            | Source                                  |
|---------------------------------------|-------------|----|----|--------------|--------------------|------------------------------------------|
| `datos-06-07-2023/`                   | 580734      | 16 | 3  | 30 m³/s     | 2014-06 – 2023-07 | SAIH-Ebro public service                 |
| `datos-synth/full/`                   | SYNTH001    | 16 | 3  | 30 m³/s (set) | 2022-01 – 2024-12 | `synth_simulator` seed 0 (in-tree)       |
| `datos-synth/partial/`                | SYNTH001    | 16 | 3  | 30 m³/s     | 2022-01 – 2024-12 | same; only `{SM, ST1}` stations exposed  |
| `datos-synth-N64/full/`               | SYNTH-N64   | 64 | 4  | 30 m³/s     | 2022-01 – 2024-12 | `synth_simulator` seed 0 (regenerable)   |

The `partial/` variant of the N=16 basin hides every station except
SM-PACUM and ST1-PACUM. It is used to simulate the operational case
in which only the rain gauges in the upper sub-basin are available.

## Regenerating the N=64 dataset

```bash
python scripts/make_synth_basin.py \
    --n-type1 64 \
    --branching 1.5 \
    --seed 0 \
    --output datos-synth-N64
```

The script is deterministic given the seed.

## Adding a new basin

1. Drop the CSVs into `datos-<your-basin>/` with the naming scheme
   above.
2. Add a `manifest.yaml` (synthetic) or extend
   `seq2seq_runoff.basins.<your_basin>.py` with a `BasinSpec` and a
   `BasinGraph` factory.
3. Pass `--directorio-datos datos-<your-basin> --firma <your-firma>`
   to any of the `scripts/run_*.py` entry points.

## Operator-side conventions

* All flows are in `m³/s`; storages in `hm³`; rainfall in `mm/day`.
* The decision day is the day **after** the historical window; the
  decision horizon is 10 days.
* The asymmetric cost `(c_FN, c_FP) = (100, 1)` is hard-coded in
  `seq2seq_runoff.evaluation.OPERATING_COST` and *not* a CLI flag —
  changing it makes the comparisons in the paper meaningless.
