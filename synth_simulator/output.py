"""Serialización de los resultados de la simulación a CSVs y manifest YAML.

Los CSVs siguen un formato compatible con el lector de
`seq2seq_runoff.data._read_station_csv`: columnas `fecha` y `valor`,
separador `;`, decimal `,`, codificación ISO-8859-1, fecha en formato
`YYYY-MM-DD HH:MM:SS`.

Para cada `VisibilityConfig` se escribe un subdirectorio que contiene:

    DatosHistoricos_<firma>_<code>.csv  (uno por estación/embalse/aforo visible)
    manifest.yaml                       (descripción de lo visible y la topología)

El `manifest.yaml` lo lee `seq2seq_runoff.basins.synth` para construir
`BasinSpec` y `BasinGraph` automáticamente.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Union

import pandas as pd
import yaml

from .config import BasinSimConfig, VisibilityConfig
from .hydro import SimulationResult


_CSV_KW = dict(sep=";", decimal=",", index=False)
_ENCODING = "ISO-8859-1"


def _resolve_visible(items: Union[str, List[str]], all_ids: Iterable[str]) -> List[str]:
    if items == "ALL":
        return list(all_ids)
    return list(items)


def _write_one_series(path: Path, series: pd.Series) -> None:
    df = pd.DataFrame({"fecha": series.index.strftime("%Y-%m-%d %H:%M:%S"),
                       "valor": series.to_numpy()})
    csv = df.to_csv(**_CSV_KW)
    path.write_bytes(csv.encode(_ENCODING))


def _manifest_dict(cfg: BasinSimConfig, vis: VisibilityConfig,
                   visible_stations: List[str], visible_reservoirs: List[str]) -> dict:
    """Construye el diccionario que se serializa como `manifest.yaml`.

    Incluye **toda** la topología (nodos, aristas, embalses) más la lista
    explícita de qué es visible bajo esta configuración. `basins/synth.py`
    elige qué consumir según el modelo (completo/simplificado/Phase 2.2).
    """
    return {
        "basin": {
            "name": cfg.name,
            "firma": cfg.firma,
            "file_pattern": cfg.file_pattern,
            "caudal_minimo_m3s": cfg.caudal_minimo_m3s,
            "river_velocity_km_per_day": cfg.river_velocity_km_per_day,
        },
        "visibility": vis.name,
        "visible_stations": [
            {"node_id": n.id, "code": n.rain_station, "name": n.rain_station}
            for n in cfg.nodes if n.rain_station and n.id in visible_stations
        ],
        "visible_reservoirs": [
            {"id": r.id, "code": r.name, "name": r.name}
            for r in cfg.reservoirs if r.id in visible_reservoirs
        ],
        "flow": {
            "node_id": cfg.outlet().id,
            "code": cfg.outlet().flow_station,
            "name": cfg.outlet().flow_station,
        },
        "topology": {
            "nodes": [
                {"id": n.id,
                 "rain_station": n.rain_station,
                 "is_major_river": n.is_major_river,
                 "catchment_km2": n.catchment_km2}
                for n in cfg.nodes
            ],
            "edges_11": [{"src": e.src, "dst": e.dst, "length_km": e.length_km}
                         for e in cfg.edges_11],
            "reservoirs": [
                {"id": r.id, "name": r.name,
                 "capacity_hm3": r.capacity_hm3,
                 "inflow_from": r.inflow_from, "release_to": r.release_to,
                 "is_biggest": r.is_biggest}
                for r in cfg.reservoirs
            ],
            "outlet": cfg.outlet().id,
            "major_river_node_ids": cfg.major_river_node_ids(),
            "biggest_reservoir_ids": cfg.biggest_reservoir_ids(),
        },
    }


def write_visibility_outputs(
    cfg: BasinSimConfig,
    rainfall: pd.DataFrame,
    sim: SimulationResult,
) -> None:
    """Escribe un subdirectorio por cada `VisibilityConfig` declarada."""
    base = Path(cfg.output_directory)
    outlet = cfg.outlet()
    outlet_q = sim.flow[outlet.id]

    for vis in cfg.output_configurations:
        out_dir = base / vis.name
        out_dir.mkdir(parents=True, exist_ok=True)

        all_station_ids = [n.id for n in cfg.nodes if n.rain_station]
        all_res_ids = [r.id for r in cfg.reservoirs]
        visible_stations = _resolve_visible(vis.visible_rain_stations, all_station_ids)
        visible_reservoirs = _resolve_visible(vis.visible_reservoirs, all_res_ids)

        # Estaciones de pluviosidad
        for n in cfg.nodes:
            if n.rain_station and n.id in visible_stations:
                p = out_dir / cfg.file_pattern.format(firma=cfg.firma, code=n.rain_station)
                _write_one_series(p, rainfall[n.rain_station])

        # Embalses
        for r in cfg.reservoirs:
            if r.id in visible_reservoirs:
                p = out_dir / cfg.file_pattern.format(firma=cfg.firma, code=r.name)
                _write_one_series(p, sim.storage[r.name])

        # Aforo (siempre visible — es el target del modelo)
        p = out_dir / cfg.file_pattern.format(firma=cfg.firma, code=outlet.flow_station)
        _write_one_series(p, outlet_q.rename(outlet.flow_station))

        # Manifest con topología y visibilidad
        manifest = _manifest_dict(cfg, vis, visible_stations, visible_reservoirs)
        (out_dir / "manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True))
