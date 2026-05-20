# `datos-06-07-2023/` — Ebro upstream of Tudela

Daily CSV time-series from the SAIH-Ebro public service, covering the
Ebro basin upstream of the discharge gauge at Tudela (Spain). Period
2014-06-01 to 2023-07-04 (~9 years).

**Firma**: `580734`.

## Stations

| Code                      | Magnitude       | Unit   | Description                          |
|---------------------------|-----------------|--------|--------------------------------------|
| `A284Z65QRIO1`            | discharge       | m³/s   | Ebro at Tudela (outlet)              |
| `E001L65VEMBA`            | reservoir storage | hm³  | E001 (Ebro)                          |
| `E029Y65VEMBA`            | reservoir storage | hm³  | E029 (Yesa)                          |
| `E075E65VEMBA`            | reservoir storage | hm³  | E075 (Itoiz)                         |
| `EM01L84PACUM`            | rainfall        | mm/d   | Rain sub-catchment EM01              |
| `EM06L84PACUM`            | rainfall        | mm/d   | Rain sub-catchment EM06              |
| `EM09L84PACUM`            | rainfall        | mm/d   | Rain sub-catchment EM09              |
| `EM11L84PACUM`            | rainfall        | mm/d   | Rain sub-catchment EM11              |
| `EM25T84PACUM`            | rainfall        | mm/d   | Rain sub-catchment EM25              |
| `EM29Y84PACUM`            | rainfall        | mm/d   | Rain sub-catchment EM29 (Yesa)       |
| `EM30T84PACUM`            | rainfall        | mm/d   | Rain sub-catchment EM30              |
| `EM71T84PACUM`            | rainfall        | mm/d   | Rain sub-catchment EM71              |
| `EM75E84PACUM`            | rainfall        | mm/d   | Rain sub-catchment EM75 (Itoiz)      |

The graph (operator-side, 16 Type-1 nodes / 3 Type-2 reservoirs) is
hard-coded in `seq2seq_runoff/basins/ebro.py:EBRO_GRAPH` and was
provided by the plant operator, not derived from a generic GIS source.

## Operational parameters

| Parameter          | Value     |
|--------------------|-----------|
| `Q_min`            | 30 m³/s   |
| `c_FN` (false neg) | 100       |
| `c_FP` (false pos) | 1         |
| Decision horizon   | 10 days   |
| Historical window  | 20 days   |

## Provenance

Downloaded from the SAIH-Ebro public service
(<https://saihebro.com>). Each file is the result of a single API
query at the indicated firma. The CSVs were not modified after
download: missing values are encoded as in the source.

## Licence

The data are derived from a public service. Redistribution conditions
follow SAIH-Ebro's terms. If you redistribute a derived dataset,
preserve the firma and station codes so that the original API query
can be reconstructed.
