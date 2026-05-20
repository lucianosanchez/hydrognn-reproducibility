"""Heterograph genérico del modelo Tipo-1 + Tipo-2 (sec. 5.1 de report2.tex).

Contiene la representación ligera `BasinGraph` y el constructor del grafo
de candidatos densos para la Fase 2.2. Las topologías concretas de cada
cuenca viven en `seq2seq_runoff.basins.<cuenca>` (e.g. `ebro_graph`).

Convenciones:
    Type-1 (V_1): nodos de mezcla/encaminamiento, sin memoria explícita.
                  Algunos llevan asociada una estación pluviométrica.
    Type-2 (V_2): embalses con memoria S_k(t).
    E_11        : aristas Tipo-1 → Tipo-1 (topología fluvial GIS, fija).
    E_12        : Tipo-1 → Tipo-2  (entradas al embalse, candidatas).
    E_21        : Tipo-2 → Tipo-1  (sueltas al río, candidatas).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class BasinGraph:
    """Estructura que comparten todas las fases del modelo GNN.

    Las matrices de aristas se guardan como `int64` para que se conviertan a
    `torch.LongTensor` sin coste y se puedan usar directamente como índices
    de scatter.
    """

    type1_names: List[str]
    edge_index_11: np.ndarray            # forma (2, E11): [src; dst]

    res_names: List[str]
    src12: np.ndarray                    # (E12,)
    dst12: np.ndarray                    # (E12,)
    src21: np.ndarray                    # (E21,)
    dst21: np.ndarray                    # (E21,)

    target_node_idx: int                 # índice v* en V_1

    # Mapas de columnas del DataFrame a índices del grafo.
    rain_to_type1: Dict[str, int] = field(default_factory=dict)
    res_to_observed: Dict[str, str] = field(default_factory=dict)

    # ---- Información geográfica opcional -----------------------------------
    # Longitudes fluviales por arista (km). Si están disponibles, permiten
    # inicializar λ_{routing} de forma informada (cf. HydroGNNCore con
    # `river_velocity_km_day`). Si son None, el modelo aprende λ desde su
    # logit_init por defecto.
    edge_len_km_11: Optional[np.ndarray] = None   # shape (E11,)
    len_12: Optional[np.ndarray] = None           # shape (E12,)
    len_21: Optional[np.ndarray] = None           # shape (E21,)
    # Coordenadas (lat, lon) por nodo Tipo-1 y embalse, útiles para el
    # layout geográfico de las figuras (cf. seq2seq_runoff/gnn/viz.py).
    type1_latlon: Optional[Dict[str, Tuple[float, float]]] = None
    res_latlon: Optional[Dict[str, Tuple[float, float]]] = None
    # Datos físicos opcionales de los embalses (volúmenes en hm³, cotas en m
    # s.n.m.). Por ahora consumidos sólo por la documentación y métricas;
    # la dinámica no impone S_k^max todavía.
    reservoir_specs: Optional[Dict[str, Dict[str, float]]] = None

    @property
    def N1(self) -> int:
        return len(self.type1_names)

    @property
    def M(self) -> int:
        return len(self.res_names)

    @property
    def E11(self) -> int:
        return self.edge_index_11.shape[1]

    @property
    def E12(self) -> int:
        return self.src12.shape[0]

    @property
    def E21(self) -> int:
        return self.src21.shape[0]


def dense_candidate_graph(
    graph_base: BasinGraph,
    M: int,
    *,
    excluir_objetivo_e12: bool = True,
) -> BasinGraph:
    """Construye un grafo con `M` embalses latentes "libres" para la Fase 2.2.

    Cada candidato $R_k$ se conecta:
      * con E_12 desde **todos** los nodos Tipo-1 (excepto el aforo si
        `excluir_objetivo_e12=True`),
      * con E_21 hacia **todos** los nodos Tipo-1.

    El reparto de qué fracción del flujo de cada Tipo-1 entra a cada
    embalse — y dónde sale — se aprende mediante los pesos de splitting
    (`logw12`, `logw21`) que ya existen en `HydroGNNCore`. El modelo decide,
    sin penalización por número de embalses activos, dónde colocar la masa
    embalsada formal.

    Esto es lo que justifica la Fase 2.2: si la cobertura de sensores es
    parcial, el modelo posiblemente coloque embalses formales en
    posiciones distintas a las reales, pero con masa total comparable a la
    capacidad real de la cuenca.
    """
    if M < 1:
        raise ValueError("M debe ser >= 1.")
    n1 = graph_base.N1

    fuentes_t1 = list(range(n1))
    if excluir_objetivo_e12:
        fuentes_t1 = [i for i in fuentes_t1 if i != graph_base.target_node_idx]
    destinos_t1 = list(range(n1))

    # E_12 denso: cada Tipo-1 elegible apunta a cada R_k.
    src12 = np.repeat(np.asarray(fuentes_t1, dtype=np.int64), M)
    dst12 = np.tile(np.arange(M, dtype=np.int64), len(fuentes_t1))

    # E_21 denso: cada R_k apunta a cada Tipo-1.
    src21 = np.repeat(np.arange(M, dtype=np.int64), len(destinos_t1))
    dst21 = np.tile(np.asarray(destinos_t1, dtype=np.int64), M)

    return BasinGraph(
        type1_names=graph_base.type1_names,
        edge_index_11=graph_base.edge_index_11,
        res_names=[f"R_lat{k}" for k in range(M)],
        src12=src12,
        dst12=dst12,
        src21=src21,
        dst21=dst21,
        target_node_idx=graph_base.target_node_idx,
        rain_to_type1=graph_base.rain_to_type1,
        res_to_observed={},  # los embalses latentes no se supervisan
    )


# ---------------------------------------------------------------------------
# Grafo de candidatos ACÍCLICO: para explorabilidad / interpretabilidad
# ---------------------------------------------------------------------------


def _bfs_ancestors(graph_base: BasinGraph) -> Dict[int, set]:
    """Ancestros BFS (incluyendo el propio nodo) en E_11."""
    n = graph_base.N1
    in_neighbors: Dict[int, List[int]] = {i: [] for i in range(n)}
    src, dst = graph_base.edge_index_11
    for s, d in zip(src.tolist(), dst.tolist()):
        in_neighbors[int(d)].append(int(s))
    anc: Dict[int, set] = {}
    for j in range(n):
        seen = {j}
        frontier = list(in_neighbors[j])
        while frontier:
            x = frontier.pop()
            if x in seen:
                continue
            seen.add(x)
            frontier.extend(in_neighbors[x])
        anc[j] = seen
    return anc


def _bfs_descendants_base(graph_base: BasinGraph) -> Dict[int, set]:
    """Descendientes BFS (incluyendo el propio nodo) en E_11."""
    n = graph_base.N1
    out_neighbors: Dict[int, List[int]] = {i: [] for i in range(n)}
    src, dst = graph_base.edge_index_11
    for s, d in zip(src.tolist(), dst.tolist()):
        out_neighbors[int(s)].append(int(d))
    desc: Dict[int, set] = {}
    for i in range(n):
        seen = {i}
        frontier = list(out_neighbors[i])
        while frontier:
            x = frontier.pop()
            if x in seen:
                continue
            seen.add(x)
            frontier.extend(out_neighbors[x])
        desc[i] = seen
    return desc


def _bfs_depth_base(graph_base: BasinGraph) -> List[int]:
    """Profundidad BFS desde las cabeceras en E_11."""
    n = graph_base.N1
    in_deg = np.zeros(n, dtype=int)
    out_neighbors: Dict[int, List[int]] = {i: [] for i in range(n)}
    src, dst = graph_base.edge_index_11
    for s, d in zip(src.tolist(), dst.tolist()):
        in_deg[int(d)] += 1
        out_neighbors[int(s)].append(int(d))
    depth = [-1] * n
    frontier = [i for i in range(n) if in_deg[i] == 0]
    cur = 0
    while frontier:
        for i in frontier:
            if depth[i] < 0:
                depth[i] = cur
        nxt = []
        for i in frontier:
            for j in out_neighbors[i]:
                if depth[j] < cur + 1:
                    depth[j] = cur + 1
                nxt.append(j)
        frontier = list(set(nxt))
        cur += 1
        if cur > n + 2:
            break
    for i in range(n):
        if depth[i] < 0:
            depth[i] = cur
    return depth


def _pick_anchor_nodes(
    graph_base: BasinGraph,
    M: int,
    strategy: str = "bfs_uniform",
    exclude_target: bool = True,
) -> List[int]:
    """Elige M nodos Type-1 como anclas tentativas para los embalses formales.

    Estrategias:
      * "bfs_uniform": un ancla por capa BFS, espaciadas uniformemente desde
        las cabeceras hasta el outlet.
      * "headwaters": las M cabeceras con más caminos al outlet (in-degree=0
        en E_11 ordenadas por número de descendientes).
    """
    depth = _bfs_depth_base(graph_base)
    target = graph_base.target_node_idx
    candidates = [i for i in range(graph_base.N1)
                  if not (exclude_target and i == target)]

    if strategy == "headwaters":
        in_deg = np.zeros(graph_base.N1, dtype=int)
        for d in graph_base.edge_index_11[1].tolist():
            in_deg[int(d)] += 1
        desc = _bfs_descendants_base(graph_base)
        heads = [i for i in candidates if in_deg[i] == 0]
        heads.sort(key=lambda i: -len(desc.get(i, set())))
        if not heads:
            heads = candidates
        anchors = heads[:M]
        # padding si no hay suficientes cabeceras
        while len(anchors) < M:
            anchors.append(heads[len(anchors) % len(heads)])
        return anchors

    # bfs_uniform: distribuir M capas equidistantes entre depth=0 y depth=max
    depths_avail = sorted(set(depth[i] for i in candidates))
    if not depths_avail:
        return [candidates[0]] * M
    target_depths = np.linspace(depths_avail[0], depths_avail[-1], M)
    anchors: List[int] = []
    used = set()
    for td in target_depths:
        # el nodo no usado con depth más cercano a td
        ranked = sorted(candidates, key=lambda i: (abs(depth[i] - td), i))
        for i in ranked:
            if i not in used:
                anchors.append(i)
                used.add(i)
                break
        else:
            anchors.append(ranked[0])
    return anchors


def acyclic_candidate_graph(
    graph_base: BasinGraph,
    M: int,
    *,
    anchor_strategy: str = "bfs_uniform",
    exclude_target_e12: bool = True,
) -> BasinGraph:
    """Variante acíclica de `dense_candidate_graph` para Fase 2.2 interpretable.

    Cada embalse formal $R^*_k$ se asocia determinísticamente a un nodo-ancla
    $a_k \\in V_1$ (estrategia configurable). El conjunto de aristas
    candidatas se restringe entonces a:

      E_12 candidate: $(i, k)$ con $i \\in \\mathrm{ancestors}_{E_{11}}(a_k)$
      E_21 candidate: $(k, j)$ con $j \\in \\mathrm{descendants}_{E_{11}}(a_k)$

    Por construcción no hay aristas $E_{21}$ apuntando aguas arriba del
    catchment alimentador → cero back-flow estructural. La identificabilidad
    mejora a costa de capacidad expresiva: el modelo no puede usar embalses
    como buffers temporales con flujo arbitrario.

    Parameters
    ----------
    graph_base : BasinGraph
        Topología física del cauce ($E_{11}$ conocido).
    M : int
        Número de embalses formales latentes.
    anchor_strategy : {"bfs_uniform", "headwaters"}
        Cómo elegir las posiciones-ancla de los embalses.
    exclude_target_e12 : bool, default True
        Excluir el outlet de los orígenes válidos en $E_{12}$ (mismo
        comportamiento que `dense_candidate_graph`).
    """
    if M < 1:
        raise ValueError("M debe ser >= 1.")
    anchors = _pick_anchor_nodes(graph_base, M, strategy=anchor_strategy,
                                  exclude_target=exclude_target_e12)
    ancestors = _bfs_ancestors(graph_base)
    descendants = _bfs_descendants_base(graph_base)

    src12_list, dst12_list = [], []
    src21_list, dst21_list = [], []
    for k, a in enumerate(anchors):
        # E_12: ancestros del ancla (incluido el ancla)
        for i in sorted(ancestors[a]):
            if exclude_target_e12 and i == graph_base.target_node_idx:
                continue
            src12_list.append(int(i)); dst12_list.append(int(k))
        # E_21: descendientes del ancla (incluido el ancla)
        for j in sorted(descendants[a]):
            src21_list.append(int(k)); dst21_list.append(int(j))

    src12 = np.asarray(src12_list, dtype=np.int64)
    dst12 = np.asarray(dst12_list, dtype=np.int64)
    src21 = np.asarray(src21_list, dtype=np.int64)
    dst21 = np.asarray(dst21_list, dtype=np.int64)

    res_names = [f"R_lat{k}_a{a}" for k, a in enumerate(anchors)]

    return BasinGraph(
        type1_names=graph_base.type1_names,
        edge_index_11=graph_base.edge_index_11,
        res_names=res_names,
        src12=src12,
        dst12=dst12,
        src21=src21,
        dst21=dst21,
        target_node_idx=graph_base.target_node_idx,
        rain_to_type1=graph_base.rain_to_type1,
        res_to_observed={},
        edge_len_km_11=getattr(graph_base, "edge_len_km_11", None),
        type1_latlon=getattr(graph_base, "type1_latlon", None),
    )
