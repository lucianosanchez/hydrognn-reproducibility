# `datos-synth/` — synthetic basin N=16

Deterministic synthetic basin produced by `synth_simulator` with seed
0, N₁ = 16 Type-1 nodes and 3 Type-2 reservoirs. Used as the
"informed-prior" sandbox where the graph topology is known exactly,
allowing the GNN to be evaluated against ground truth (Phase 2.2
topology recovery in §5.6).

**Firma**: `SYNTH001`. Period 2022-01-01 to 2024-12-15.

## Layout

```
datos-synth/
├── full/            ← every station of the basin exposed
│   ├── manifest.yaml
│   └── DatosHistoricos_SYNTH001_*.csv     ← 8 stations
└── partial/         ← only the two visible rain gauges (operational case)
    ├── manifest.yaml
    └── DatosHistoricos_SYNTH001_{SM,ST1}-PACUM.csv
```

The `partial/` variant simulates the operational case in which only
the rain gauges of the upper sub-basin are available; the rest of the
basin must be inferred by the model.

## Regenerating

```bash
python -m synth_simulator synth_simulator/example_basin.yaml \
       --output datos-synth --seed 0
```

The output is deterministic given the seed.

## Stations

| Code            | Magnitude       | Unit   | Description                                       |
|-----------------|-----------------|--------|---------------------------------------------------|
| `SM-PACUM`      | rainfall        | mm/d   | Rain gauge at upper meadow                        |
| `ST1-PACUM`     | rainfall        | mm/d   | Rain gauge at tributary 1                         |
| `ST1A-PACUM`    | rainfall        | mm/d   | Rain gauge at tributary 1A                        |
| `ST2-PACUM`     | rainfall        | mm/d   | Rain gauge at tributary 2                         |
| `SQ-CAUDAL`     | discharge       | m³/s   | Outlet discharge                                  |
| `SR0-EMB`       | reservoir storage | hm³  | Reservoir 0                                       |
| `SR1-EMB`       | reservoir storage | hm³  | Reservoir 1                                       |
| `SR2-EMB`       | reservoir storage | hm³  | Reservoir 2                                       |

The full topology (16 Type-1 nodes / 3 Type-2 reservoirs, with the 8
listed stations attached to specific nodes) lives in
`full/manifest.yaml`.
