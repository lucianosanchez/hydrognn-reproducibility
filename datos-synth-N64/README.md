# `datos-synth-N64/` — synthetic basin N=64

Larger synthetic basin (N₁ = 64 Type-1 nodes, 4 Type-2 reservoirs)
used as a stress test for HydroGNN scalability (§5.3). The headline
result is that the N=64 case collapses under defaults and requires
the remediation flags documented in §5.5 of the paper.

**Firma**: `SYNTH-N64`. Period 2022-01-01 to 2024-12-15.

## Regenerating

```bash
python scripts/make_synth_basin.py \
       --n-type1 64 \
       --branching 1.5 \
       --seed 0 \
       --output datos-synth-N64
```

The output is deterministic given the seed. The script wraps
`synth_simulator.topology_generator.random_basin` and re-runs the
hydrological simulator (`synth_simulator.hydro`) over the resulting
graph.

## Layout

```
datos-synth-N64/
└── full/
    ├── manifest.yaml
    └── DatosHistoricos_SYNTH-N64_*.csv     ← ~30 stations
```

The station codes follow the convention

```
DatosHistoricos_SYNTH-N64_S<n>-PACUM.csv     ← rain gauge at node S<n>
DatosHistoricos_SYNTH-N64_R<k>-EMB.csv       ← reservoir k storage
DatosHistoricos_SYNTH-N64_SQ-CAUDAL.csv      ← outlet discharge
```

The exact list (which `<n>` rain stations are observable) is fixed by
the seed; inspect `full/manifest.yaml` to see the topology.
