"""Generador parametrizado de cuencas sintéticas para experimentación controlada.

Produce un `BasinSimConfig` (compatible con el resto del simulador) a partir
de parámetros macroscópicos:

    n_type1                tamaño del grafo (nº de nodos Tipo-1)
    branching_factor       cuán ramificada es la red fluvial (1=línea, ∞=estrella)
    n_reservoirs           número de embalses reales
    reservoir_strategy     dónde se sitúan ("headwater"/"midstream"/"scattered")
    station_coverage       fracción de cabeceras con pluviómetro
    nonstationarity_amp    amplitud de la deriva temporal del manejo de embalses
    catchment_total_km2    área total de drenaje (calibra los caudales)
    seed                   determinismo de la generación

El árbol se construye con un proceso Galton–Watson truncado: empieza con el
OUTLET y, en cada paso, escoge un nodo hoja (cabecera actual del árbol en
construcción) y le añade `Poisson(branching_factor)` hijos hasta llegar a
`n_type1` nodos.

Topología hidrológica resultante:
  * Las hojas reciben pluviómetro (con prob. `station_coverage`) y un área
    de drenaje proporcional al total.
  * Los nodos interiores son confluencias sin escorrentía local.
  * Los embalses se asignan a `n_reservoirs` nodos elegidos según la
    estrategia y "secuestran" la arista E_11 que iría de su nodo a su padre,
    sustituyéndola por (E_12 al embalse) + (E_21 del embalse al padre).
"""

from __future__ import annotations

from dataclasses import replace
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import (
    BasinSimConfig,
    ClimateConfig,
    EdgeConfig,
    NodeConfig,
    ReservoirConfig,
    VisibilityConfig,
)


# --------------------------------------------------------------------------
# 1. Construcción del árbol (Galton-Watson truncado)
# --------------------------------------------------------------------------


def _grow_tree(n_target: int, branching: float, rng: np.random.Generator):
    """Crea un árbol con `n_target` nodos, raíz en 0 (OUTLET).

    Devuelve:
        parent : list[int|None]   parent[i] = padre de i (None para la raíz)
        children: list[list[int]] children[i] = hijos de i
        depth  : list[int]        depth[i] (0 = OUTLET)
    """
    parent: List[Optional[int]] = [None]
    children: List[List[int]] = [[]]
    depth: List[int] = [0]

    leaves: List[int] = [0]   # candidatos a expansión

    while len(parent) < n_target and leaves:
        i = int(rng.choice(leaves))
        # Cuántos hijos añadir
        k = max(1, int(rng.poisson(max(branching, 0.5))))
        k = min(k, n_target - len(parent))
        for _ in range(k):
            new_id = len(parent)
            parent.append(i)
            children.append([])
            depth.append(depth[i] + 1)
            children[i].append(new_id)
            leaves.append(new_id)
        # i ya no es hoja
        if i in leaves:
            leaves.remove(i)

    return parent, children, depth


def _max_depth(depth: List[int]) -> int:
    return max(depth) if depth else 0


# --------------------------------------------------------------------------
# 2. Asignación de embalses según estrategia
# --------------------------------------------------------------------------


def _select_reservoir_nodes(
    n_reservoirs: int,
    parent: List[Optional[int]],
    children: List[List[int]],
    depth: List[int],
    strategy: str,
    rng: np.random.Generator,
) -> List[int]:
    """Devuelve los IDs de los nodos donde se ubicará cada embalse.

    Restricciones:
      * El OUTLET (0) nunca lleva embalse.
      * Cada embalse intercepta la arista (i → parent[i]); por tanto un nodo
        no puede ser a la vez embalse y outlet de otro embalse en cascada
        sin pensarlo bien — pero como el árbol es DAG y los embalses están
        en nodos distintos, esto es seguro.
    """
    n = len(parent)
    if n_reservoirs <= 0:
        return []

    leaves = [i for i in range(n) if not children[i]]
    interior = [i for i in range(1, n) if children[i]]
    max_d = _max_depth(depth)

    if strategy == "headwater":
        candidates = leaves
    elif strategy == "midstream":
        # nodos interiores con profundidad en el tercio medio del árbol
        lo, hi = max_d / 3.0, 2.0 * max_d / 3.0
        candidates = [i for i in interior if lo <= depth[i] <= hi]
        # fallback si la franja está vacía
        if not candidates:
            candidates = interior
    elif strategy == "scattered":
        candidates = [i for i in range(1, n)]
    elif strategy == "random":
        candidates = [i for i in range(1, n)]
    else:
        raise ValueError(f"Estrategia desconocida: {strategy!r}")

    n_take = min(n_reservoirs, len(candidates))
    return [int(x) for x in rng.choice(candidates, size=n_take, replace=False)]


# --------------------------------------------------------------------------
# 3. Capacidades, áreas, longitudes
# --------------------------------------------------------------------------


def _split_total_among(
    total: float, n: int, skew: float, rng: np.random.Generator
) -> np.ndarray:
    """Reparte `total` entre `n` valores positivos. `skew=0` = uniforme;
    `skew=1` = log-normal con σ=1 (cola pesada)."""
    if skew <= 0:
        # todos iguales con un poco de ruido (5%)
        x = np.ones(n) + 0.05 * rng.standard_normal(n)
        x = np.clip(x, 0.5, None)
    else:
        x = rng.lognormal(mean=0.0, sigma=skew, size=n)
    return total * x / x.sum()


def _calibrate_total_capacity(
    catchment_total_km2: float,
    rainfall_mm_mean: float,
    p_wet_year: float,
    runoff_coef: float,
    days_of_storage: float = 30.0,
) -> float:
    """Capacidad agregada (Hm³) calibrada para almacenar `days_of_storage` días
    de caudal medio.

    Caudal medio (m³/s) = lluvia media diaria × área × runoff / 86400, con
    lluvia media diaria = rainfall_mm_mean × p_wet_year.
    """
    daily_rain_mm = rainfall_mm_mean * p_wet_year
    mean_q_m3s = daily_rain_mm * catchment_total_km2 * runoff_coef * 1000.0 / 86400.0
    # 1 m³/s × 86400 s/day / 1e6 = 0.0864 Hm³/day
    capacity_hm3 = mean_q_m3s * 86400.0 / 1e6 * days_of_storage
    return float(capacity_hm3)


# --------------------------------------------------------------------------
# 4. Layout: posiciones (x, y) por radial layout (para los plots)
# --------------------------------------------------------------------------


def _radial_layout(
    parent: List[Optional[int]], children: List[List[int]], depth: List[int]
) -> Dict[int, Tuple[float, float]]:
    """Layout por niveles: y proporcional a la profundidad, x reparte hijos."""
    n = len(parent)
    pos: Dict[int, Tuple[float, float]] = {}
    max_d = _max_depth(depth) or 1

    # Asignamos x a las hojas en orden de DFS, luego propagamos hacia padres.
    leaves: List[int] = []

    def dfs(node: int):
        if not children[node]:
            leaves.append(node)
        else:
            for c in children[node]:
                dfs(c)

    dfs(0)
    n_leaves = max(len(leaves), 1)
    for i, leaf in enumerate(leaves):
        pos[leaf] = (i, -depth[leaf])

    # Padres se sitúan en la media de los hijos
    def settle(node: int) -> Tuple[float, float]:
        if node in pos:
            return pos[node]
        xs, ys = zip(*(settle(c) for c in children[node]))
        x_mean = sum(xs) / len(xs)
        y_node = -depth[node]
        pos[node] = (x_mean, y_node)
        return pos[node]

    settle(0)
    # Renormaliza x al rango [0, n_leaves-1]
    return pos


# --------------------------------------------------------------------------
# 5. API principal
# --------------------------------------------------------------------------


def random_basin(
    *,
    n_type1: int = 16,
    branching_factor: float = 2.5,
    n_reservoirs: int = 3,
    reservoir_strategy: str = "headwater",
    capacity_skew: float = 0.6,
    edge_length_km_range: Tuple[float, float] = (10.0, 40.0),
    river_velocity_km_per_day: float = 60.0,
    catchment_total_km2: float = 5000.0,
    runoff_coef: float = 0.3,
    station_coverage: float = 1.0,
    rainfall_climate: Optional[ClimateConfig] = None,
    nonstationarity_amp: float = 0.0,
    name: str = "Synth",
    firma: str = "SYNTH",
    output_dir: str = "./datos-synth-sweep",
    caudal_minimo_m3s: Optional[float] = None,
    days_of_storage: float = 30.0,
    visibility: Optional[List[VisibilityConfig]] = None,
    seed: int = 0,
) -> BasinSimConfig:
    """Genera un `BasinSimConfig` aleatorio con parámetros macroscópicos.

    Si `rainfall_climate` es `None` se usa un default razonable
    (estaciones invierno/verano, mm/día medios = 8). El `caudal_minimo_m3s`
    por defecto se calibra al 15 % del caudal medio (régimen de bajo caudal
    raro pero no ínfimo).

    `visibility` permite controlar qué se publica al modelo:
        * None → una sola configuración "full" (todo visible).
        * lista de VisibilityConfig → lo que el usuario decida.
    """
    rng = np.random.default_rng(seed)

    # Clima por defecto.
    if rainfall_climate is None:
        rainfall_climate = ClimateConfig(
            start_date="2010-01-01",
            end_date="2024-12-31",
            seed=seed,
        )

    p_wet_year = (rainfall_climate.p_wet_winter + rainfall_climate.p_wet_summer) / 2

    # 1. Topología.
    parent, children, depth = _grow_tree(n_type1, branching_factor, rng)

    # 2. Roles: el 0 es el OUTLET; las hojas reciben pluviómetro con prob.
    #    `station_coverage` (al menos una estación garantizada).
    leaves = [i for i, ch in enumerate(children) if not ch]
    leaf_mask = rng.random(len(leaves)) < station_coverage
    if not leaf_mask.any():
        leaf_mask[0] = True  # garantía mínima

    # 3. Áreas de drenaje a las hojas con cierta asimetría.
    catchment_areas = _split_total_among(
        catchment_total_km2, len(leaves), skew=capacity_skew, rng=rng
    )

    # 4. Embalses.
    reservoir_nodes = _select_reservoir_nodes(
        n_reservoirs, parent, children, depth, reservoir_strategy, rng
    )
    cap_total_hm3 = _calibrate_total_capacity(
        catchment_total_km2=catchment_total_km2,
        rainfall_mm_mean=rainfall_climate.rainfall_mm_mean,
        p_wet_year=p_wet_year,
        runoff_coef=runoff_coef,
        days_of_storage=days_of_storage,
    )
    capacities_hm3 = (
        _split_total_among(cap_total_hm3, len(reservoir_nodes), capacity_skew, rng)
        if reservoir_nodes
        else np.array([])
    )

    # 5. Layout para los plots.
    layout = _radial_layout(parent, children, depth)

    # 6. Construye los NodeConfig.
    nodes: List[NodeConfig] = []
    leaf_idx_map = {leaf_id: idx for idx, leaf_id in enumerate(leaves)}
    for i in range(len(parent)):
        is_outlet = (i == 0)
        is_leaf = i in leaf_idx_map
        node_id = "OUTLET" if is_outlet else f"N{i}"
        rain = None
        catch = 0.0
        if is_leaf:
            j = leaf_idx_map[i]
            if leaf_mask[j]:
                rain = f"S{i}-PACUM"
            catch = float(catchment_areas[j])
        flow = "SQ-CAUDAL" if is_outlet else None
        # Major river = path desde la hoja con mayor catchment hasta el outlet
        # (lo dejamos para luego; aquí marcamos el outlet y los nodos en la
        # ruta más caudalosa)
        nodes.append(NodeConfig(
            id=node_id,
            rain_station=rain,
            catchment_km2=catch,
            runoff_coef=runoff_coef,
            flow_station=flow,
            is_major_river=False,
            position=list(layout[i]),
        ))

    # Marca el cauce principal: ruta desde la hoja con mayor catchment al OUTLET.
    if leaves:
        biggest_leaf = leaves[int(np.argmax(catchment_areas))]
        i = biggest_leaf
        while i is not None:
            nodes[i].is_major_river = True
            i = parent[i]

    # 7. Aristas E_11: para cada nodo no-outlet sin embalse asociado a su salida,
    #    arista (i → parent[i]) con longitud aleatoria.
    reservoir_set = set(reservoir_nodes)
    edges: List[EdgeConfig] = []
    L_lo, L_hi = edge_length_km_range
    for i in range(1, len(parent)):
        p = parent[i]
        if i in reservoir_set:
            # i sale al embalse, el embalse suelta a parent[i]: no hay E_11
            continue
        length = float(rng.uniform(L_lo, L_hi))
        edges.append(EdgeConfig(src=nodes[i].id, dst=nodes[p].id, length_km=length))

    # 8. Reservorios.
    reservoirs: List[ReservoirConfig] = []
    biggest_idx = int(np.argmax(capacities_hm3)) if len(capacities_hm3) else -1
    for k, r_node in enumerate(reservoir_nodes):
        p = parent[r_node]
        reservoirs.append(ReservoirConfig(
            id=f"R{k}",
            name=f"R{k}-EMB",
            capacity_hm3=float(capacities_hm3[k]),
            inflow_from=nodes[r_node].id,
            release_to=nodes[p].id,
            release_fraction_per_day=0.04,
            initial_storage_hm3=float(capacities_hm3[k] * 0.5),
            is_biggest=(k == biggest_idx),
            position=None,  # auto-layout entre inflow y release
        ))

    # 9. Calibración del caudal mínimo si no lo da el usuario.
    if caudal_minimo_m3s is None:
        mean_q_m3s = (rainfall_climate.rainfall_mm_mean * p_wet_year
                      * catchment_total_km2 * runoff_coef * 1000.0 / 86400.0)
        caudal_minimo_m3s = float(0.15 * mean_q_m3s)  # 15 % del caudal medio

    # 10. Visibilidades por defecto.
    if visibility is None:
        visibility = [VisibilityConfig(name="full")]

    return BasinSimConfig(
        name=f"{name}-N{n_type1}-K{n_reservoirs}-{reservoir_strategy}-seed{seed}",
        river_velocity_km_per_day=river_velocity_km_per_day,
        nodes=nodes,
        edges_11=edges,
        reservoirs=reservoirs,
        climate=rainfall_climate,
        output_configurations=visibility,
        output_directory=output_dir,
        firma=firma,
        caudal_minimo_m3s=caudal_minimo_m3s,
        nonstationarity_amplitude=nonstationarity_amp,
    )
