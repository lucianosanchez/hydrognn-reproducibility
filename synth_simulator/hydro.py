"""Simulación hidrológica determinista (paso diario).

Para cada día t y cada nodo i en orden topológico:

    F_in(i,t) = Σ_{src→i ∈ E_11} q(src, t − Δ(src,i))
              + Σ_{k→i ∈ E_21} O(k, t)
    q(i, t)   = F_in(i,t) + r(i, t)         (escorrentía local)

Si i es fuente de un E_12 hacia el embalse k, la totalidad de q(i, t) entra
al embalse, que se actualiza inmediatamente:

    I(k, t)   = q(i, t)
    A(k, t)   = S(k, t) + I(k, t)
    O(k, t)   = β · A(k, t)                 (suelta proporcional)
    S(k, t+1) = clip(A(k, t) − O(k, t), 0, capacidad)

El retardo Δ(src,i) en E_11 se toma como `round(length / velocity)` en
días, con mínimo de 1 día (suficiente para modelar una cuenca pequeña con
paso diario; un modelo más fino requeriría sub-paso intra-día).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .config import BasinSimConfig


# -- topological sort -----------------------------------------------------


def _topo_sort(cfg: BasinSimConfig) -> List[int]:
    """Orden topológico de los nodos Tipo-1 considerando E_11 y los puentes
    virtuales (inflow_from → release_to) introducidos por cada embalse."""
    node_idx = {n.id: i for i, n in enumerate(cfg.nodes)}
    n = len(cfg.nodes)
    in_deg = np.zeros(n, dtype=int)
    children: List[List[int]] = [[] for _ in range(n)]

    for e in cfg.edges_11:
        children[node_idx[e.src]].append(node_idx[e.dst])
        in_deg[node_idx[e.dst]] += 1
    for r in cfg.reservoirs:
        children[node_idx[r.inflow_from]].append(node_idx[r.release_to])
        in_deg[node_idx[r.release_to]] += 1

    queue = [i for i in range(n) if in_deg[i] == 0]
    order: List[int] = []
    while queue:
        i = queue.pop(0)
        order.append(i)
        for c in children[i]:
            in_deg[c] -= 1
            if in_deg[c] == 0:
                queue.append(c)
    if len(order) != n:
        raise ValueError("La topología contiene un ciclo.")
    return order


# -- core simulation ------------------------------------------------------


@dataclass
class SimulationResult:
    flow: pd.DataFrame      # caudal en m³/s por nodo (incluye el aforo)
    storage: pd.DataFrame   # almacenamiento de cada embalse en Hm³
    release: pd.DataFrame   # suelta de cada embalse en m³/s


def simulate_hydrology(cfg: BasinSimConfig, rainfall: pd.DataFrame) -> SimulationResult:
    """Simula el caudal y los embalses para todo el periodo.

    `rainfall` debe tener una columna por cada `node.rain_station` declarado.
    """
    n_days = len(rainfall)
    fechas = rainfall.index
    nodes = cfg.nodes
    node_idx = {n.id: i for i, n in enumerate(nodes)}
    n_nodes = len(nodes)
    n_res = len(cfg.reservoirs)

    # E_11 incoming: para cada nodo, lista (src_idx, delay).
    incoming_e11: List[List[Tuple[int, int]]] = [[] for _ in range(n_nodes)]
    for e in cfg.edges_11:
        delay = max(1, int(round(e.length_km / cfg.river_velocity_km_per_day)))
        incoming_e11[node_idx[e.dst]].append((node_idx[e.src], delay))

    # E_21 incoming: para cada nodo, lista (res_idx, delay=0). Modelo simple.
    incoming_e21: List[List[Tuple[int, int]]] = [[] for _ in range(n_nodes)]
    for k, r in enumerate(cfg.reservoirs):
        incoming_e21[node_idx[r.release_to]].append((k, 0))

    # E_12: para cada nodo, ¿es fuente de un embalse?
    src_to_res: Dict[int, int] = {node_idx[r.inflow_from]: k for k, r in enumerate(cfg.reservoirs)}

    capacities = np.array([r.capacity_hm3 for r in cfg.reservoirs])
    base_release = np.array([r.release_fraction_per_day for r in cfg.reservoirs])
    # Deriva temporal de la regla de soltada (no-estacionariedad operativa).
    # Cuando `cfg.nonstationarity_amplitude > 0`, la fracción de soltada por
    # embalse oscila como base · (1 + σ_op · sin(2π t / T) + 0.3 σ_op · ε_t)
    # donde ε_t es un AR(1) con autocorrelación 0.9. La amplitud σ_op se
    # interpreta como fracción relativa: σ=0.3 ⇒ ±30 % de variación lenta.
    sigma_op = float(cfg.nonstationarity_amplitude)
    if sigma_op > 0:
        rng_op = np.random.default_rng(cfg.climate.seed + 7919)  # primo arbitrario
        n_res = len(cfg.reservoirs)
        # Componente determinista (sinusoide larga, periodo = ½ del registro).
        period = max(n_days // 2, 1)
        sinus = np.sin(2.0 * np.pi * np.arange(n_days) / period)
        # Componente estocástica AR(1) por embalse para que cada uno derive distinto.
        eps = np.zeros((n_days, n_res), dtype=np.float32)
        rho = 0.9
        for k in range(n_res):
            shocks = rng_op.standard_normal(n_days)
            for t in range(1, n_days):
                eps[t, k] = rho * eps[t - 1, k] + np.sqrt(1 - rho * rho) * shocks[t]
        # Multiplicador (n_days, n_res); mín 0.05 para evitar release nulo.
        deriva = sigma_op * sinus[:, None] + 0.3 * sigma_op * eps
        release_fracs_t = base_release[None, :] * (1.0 + deriva)
        release_fracs_t = np.clip(release_fracs_t, 0.05 * base_release[None, :], None)
    else:
        release_fracs_t = None  # uso el valor base por defecto

    # Escorrentía local (m³/s): mm * km² * coef * 1000 / 86400.
    runoff = np.zeros((n_days, n_nodes), dtype=np.float32)
    for n in nodes:
        if n.rain_station and n.catchment_km2 > 0:
            P = rainfall[n.rain_station].to_numpy(dtype=np.float32)
            runoff[:, node_idx[n.id]] = P * n.catchment_km2 * n.runoff_coef * 1000.0 / 86400.0

    topo = _topo_sort(cfg)

    q = np.zeros((n_days, n_nodes), dtype=np.float32)
    S = np.zeros((n_days + 1, n_res), dtype=np.float32)
    O = np.zeros((n_days, n_res), dtype=np.float32)
    S[0] = np.array([r.initial_storage_hm3 for r in cfg.reservoirs], dtype=np.float32)

    # --- bucle temporal ---------------------------------------------------

    for t in range(n_days):
        for i in topo:
            f_in = 0.0
            for src, d in incoming_e11[i]:
                if t - d >= 0:
                    f_in += q[t - d, src]
            for k, _ in incoming_e21[i]:
                f_in += O[t, k]   # ya estará calculado por orden topológico

            q[t, i] = runoff[t, i] + f_in

            # Si i es fuente de un embalse, todo lo que sale de i va al embalse.
            if i in src_to_res:
                k = src_to_res[i]
                I_hm3 = q[t, i] * 86400.0 / 1e6
                A_hm3 = S[t, k] + I_hm3
                beta_k_t = (release_fracs_t[t, k] if release_fracs_t is not None
                            else base_release[k])
                O_hm3 = beta_k_t * A_hm3
                S_new = A_hm3 - O_hm3
                if S_new > capacities[k]:
                    spill = S_new - capacities[k]
                    S_new = capacities[k]
                    O_hm3 += spill
                S_new = max(0.0, S_new)
                O[t, k] = O_hm3 * 1e6 / 86400.0
                S[t + 1, k] = S_new

    flow_df = pd.DataFrame(q, index=fechas, columns=[n.id for n in nodes])
    storage_df = pd.DataFrame(S[:n_days], index=fechas, columns=[r.name for r in cfg.reservoirs])
    release_df = pd.DataFrame(O, index=fechas, columns=[r.name for r in cfg.reservoirs])
    return SimulationResult(flow=flow_df, storage=storage_df, release=release_df)
