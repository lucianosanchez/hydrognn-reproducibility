"""Cuenca sintética generada por `synth_simulator`.

Lee el `manifest.yaml` que escribe el simulador en cada subdirectorio de
visibilidad y construye:

    `synth_basin(manifest_path)`        → `BasinSpec` con sólo los CSV visibles.
    `synth_graph_full(manifest_path)`   → `BasinGraph` con la topología COMPLETA
                                          (incluye los 3 embalses).
    `synth_graph_simplified(manifest_path)` → `BasinGraph` REDUCIDO al cauce
                                              principal y al embalse más grande.

Las tres funciones se basan exclusivamente en el manifest, así que la misma
ruta funciona tanto para la configuración A (todos los datos) como para B
(estaciones + embalse mayor).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import yaml

from ..basin import BasinSpec, StationSpec
from ..gnn.graph import BasinGraph


def _load_manifest(path: Union[str, Path]) -> dict:
    p = Path(path)
    if p.is_dir():
        p = p / "manifest.yaml"
    return yaml.safe_load(p.read_text())


def synth_basin(manifest_path: Union[str, Path]) -> BasinSpec:
    """`BasinSpec` que lista sólo los CSV efectivamente visibles."""
    m = _load_manifest(manifest_path)

    rain = [
        StationSpec(
            name=s["name"], file_code=s["code"],
            csv_value_column="valor", csv_date_column="fecha",
            kind="rain",
        )
        for s in m["visible_stations"]
    ]
    reservoirs = [
        StationSpec(
            name=r["name"], file_code=r["code"],
            csv_value_column="valor", csv_date_column="fecha",
            kind="reservoir",
        )
        for r in m["visible_reservoirs"]
    ]
    flow = StationSpec(
        name=m["flow"]["name"], file_code=m["flow"]["code"],
        csv_value_column="valor", csv_date_column="fecha",
        kind="flow",
    )

    b = m["basin"]
    return BasinSpec(
        name=f"{b['name']}-{m['visibility']}",
        rainfall_stations=rain,
        reservoirs=reservoirs,
        flow_station=flow,
        file_pattern=b["file_pattern"],
        caudal_minimo_m3s=float(b["caudal_minimo_m3s"]),
    )


# ---------------------------------------------------------------- helpers
# Construcción del BasinGraph desde un subconjunto de nodos/embalses.


def _build_graph(
    nodes_keep: List[str],
    edges_11_all: List[dict],
    reservoirs_all: List[dict],
    res_keep: List[str],
    outlet: str,
    rain_to_node_id: Dict[str, str],
    reservoir_id_to_name: Dict[str, str],
    visible_reservoir_names: set,
) -> BasinGraph:
    """Construye un `BasinGraph` filtrando nodos/embalses por inclusión."""
    keep_set = set(nodes_keep)
    name2idx = {n: i for i, n in enumerate(nodes_keep)}

    edges_11 = [(e["src"], e["dst"]) for e in edges_11_all
                if e["src"] in keep_set and e["dst"] in keep_set]
    s11 = np.array([name2idx[a] for a, _ in edges_11], dtype=np.int64)
    d11 = np.array([name2idx[b] for _, b in edges_11], dtype=np.int64)

    res_data = [r for r in reservoirs_all if r["id"] in res_keep
                and r["inflow_from"] in keep_set and r["release_to"] in keep_set]
    res_names = [r["name"] for r in res_data]
    res2idx = {n: i for i, n in enumerate(res_names)}

    src12 = np.array([name2idx[r["inflow_from"]] for r in res_data], dtype=np.int64)
    dst12 = np.array([res2idx[r["name"]] for r in res_data], dtype=np.int64)
    src21 = np.array([res2idx[r["name"]] for r in res_data], dtype=np.int64)
    dst21 = np.array([name2idx[r["release_to"]] for r in res_data], dtype=np.int64)

    rain_to_type1 = {
        col: name2idx[node_id]
        for col, node_id in rain_to_node_id.items()
        if node_id in keep_set
    }

    res_to_observed = {
        rname: rname for rname in res_names if rname in visible_reservoir_names
    }

    return BasinGraph(
        type1_names=list(nodes_keep),
        edge_index_11=np.stack([s11, d11], axis=0) if edges_11 else np.zeros((2, 0), dtype=np.int64),
        res_names=res_names,
        src12=src12, dst12=dst12, src21=src21, dst21=dst21,
        target_node_idx=name2idx[outlet],
        rain_to_type1=rain_to_type1,
        res_to_observed=res_to_observed,
    )


def _common_args(m: dict) -> Tuple[List[dict], List[dict], str, Dict[str, str], Dict[str, str], set]:
    topo = m["topology"]
    rain_to_node_id = {
        n["rain_station"]: n["id"]
        for n in topo["nodes"]
        if n.get("rain_station")
    }
    reservoir_id_to_name = {r["id"]: r["name"] for r in topo["reservoirs"]}
    visible_res_names = {r["name"] for r in m["visible_reservoirs"]}
    return (topo["edges_11"], topo["reservoirs"], topo["outlet"],
            rain_to_node_id, reservoir_id_to_name, visible_res_names)


# ----------------------------------------------------------------- API


def synth_graph_full(manifest_path: Union[str, Path]) -> BasinGraph:
    """Grafo con todos los nodos Tipo-1 y los 3 embalses (modelo 2)."""
    m = _load_manifest(manifest_path)
    edges_11, res_all, outlet, rain_to_node, _, vis_res_names = _common_args(m)
    nodes_keep = [n["id"] for n in m["topology"]["nodes"]]
    res_keep = [r["id"] for r in res_all]
    return _build_graph(nodes_keep, edges_11, res_all, res_keep, outlet,
                        rain_to_node, _, vis_res_names)


def synth_graph_simplified(manifest_path: Union[str, Path]) -> BasinGraph:
    """Grafo con sólo el cauce principal y el embalse mayor (modelo 3 / 4).

    Para el modelo 4, este grafo se pasa a `HydroGNNPhase2_2` como
    `graph_base`: la fase ignora los embalses Tipo-2 declarados aquí y
    construye internamente un grafo de candidatos densos.
    """
    m = _load_manifest(manifest_path)
    edges_11, res_all, outlet, rain_to_node, _, vis_res_names = _common_args(m)
    nodes_keep = list(m["topology"]["major_river_node_ids"])
    if outlet not in nodes_keep:
        nodes_keep = nodes_keep + [outlet]  # garantía de que el aforo está
    res_keep = list(m["topology"]["biggest_reservoir_ids"])
    return _build_graph(nodes_keep, edges_11, res_all, res_keep, outlet,
                        rain_to_node, _, vis_res_names)
