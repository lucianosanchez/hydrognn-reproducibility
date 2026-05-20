"""Visualización de la cuenca y de las series generadas.

Cuatro figuras componibles:

    plot_basin_graph    Topología (nodos Tipo-1, embalses, aristas).
    plot_rainfall       Series de pluviosidad por estación.
    plot_reservoirs     Almacenamiento de cada embalse vs. capacidad.
    plot_outlet_flow    Caudal en el aforo con el umbral mínimo.

`plot_summary` las combina en una única página y `save_all` las exporta en
ficheros separados a un directorio.

Si los nodos del YAML llevan `position: [x, y]`, se respeta esa colocación;
en otro caso el layout se calcula por profundidad topológica (longest path
desde una cabecera). Para topologías reales conviene fijar las posiciones a
mano para que el plot recuerde la geografía.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .config import BasinSimConfig
from .hydro import SimulationResult


# --------------------------------------------------------------------- layout


def _topo_depths(cfg: BasinSimConfig) -> Dict[str, int]:
    """Profundidad topológica (longest path desde una fuente) de cada nodo Tipo-1.

    Considera tanto E_11 como los puentes virtuales `inflow_from → release_to`
    introducidos por cada embalse, igual que en `hydro._topo_sort`.
    """
    nodes = [n.id for n in cfg.nodes]
    node_idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)
    children = [[] for _ in range(n)]
    parents = [[] for _ in range(n)]
    for e in cfg.edges_11:
        children[node_idx[e.src]].append(node_idx[e.dst])
        parents[node_idx[e.dst]].append(node_idx[e.src])
    for r in cfg.reservoirs:
        children[node_idx[r.inflow_from]].append(node_idx[r.release_to])
        parents[node_idx[r.release_to]].append(node_idx[r.inflow_from])

    in_deg = [len(p) for p in parents]
    queue = [i for i in range(n) if in_deg[i] == 0]
    order = []
    while queue:
        i = queue.pop(0)
        order.append(i)
        for c in children[i]:
            in_deg[c] -= 1
            if in_deg[c] == 0:
                queue.append(c)

    depth = [0] * n
    for i in order:
        for p in parents[i]:
            depth[i] = max(depth[i], depth[p] + 1)
    return {nodes[i]: depth[i] for i in range(n)}


def node_positions(cfg: BasinSimConfig) -> Tuple[
    Dict[str, Tuple[float, float]],
    Dict[str, Tuple[float, float]],
]:
    """Devuelve `(pos_t1, pos_res)` con coordenadas (x, y) por id."""
    pos_t1: Dict[str, Tuple[float, float]] = {}
    explicit = {n.id: tuple(n.position) for n in cfg.nodes if n.position}

    if all(n.position for n in cfg.nodes):
        # Todos llevan posición explícita.
        for n in cfg.nodes:
            pos_t1[n.id] = tuple(n.position)
    else:
        depths = _topo_depths(cfg)
        # Agrupamos por profundidad y distribuimos verticalmente.
        by_depth: Dict[int, list] = {}
        for n in cfg.nodes:
            by_depth.setdefault(depths[n.id], []).append(n.id)
        for d, ids in by_depth.items():
            ids = sorted(ids)
            for i, nid in enumerate(ids):
                if nid in explicit:
                    pos_t1[nid] = explicit[nid]
                else:
                    pos_t1[nid] = (float(d), float(i - (len(ids) - 1) / 2))

    pos_res: Dict[str, Tuple[float, float]] = {}
    for r in cfg.reservoirs:
        if r.position:
            pos_res[r.id] = tuple(r.position)
        else:
            x1, y1 = pos_t1[r.inflow_from]
            x2, y2 = pos_t1[r.release_to]
            pos_res[r.id] = ((x1 + x2) / 2.0, (y1 + y2) / 2.0 + 0.35)

    return pos_t1, pos_res


# --------------------------------------------------------------------- plots


def plot_basin_graph(cfg: BasinSimConfig, ax=None):
    """Dibuja la topología de la cuenca."""
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    pos_t1, pos_res = node_positions(cfg)

    # Aristas E_11 (cauce)
    for e in cfg.edges_11:
        x1, y1 = pos_t1[e.src]
        x2, y2 = pos_t1[e.dst]
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color="steelblue", lw=1.5))
        # etiqueta de longitud sobre el centro de la arista
        ax.text((x1 + x2) / 2, (y1 + y2) / 2, f"{e.length_km:.0f} km",
                fontsize=6, color="steelblue", ha="center", va="bottom")

    # Aristas E_12 (entrada al embalse) y E_21 (suelta)
    for r in cfg.reservoirs:
        x_src, y_src = pos_t1[r.inflow_from]
        x_res, y_res = pos_res[r.id]
        x_dst, y_dst = pos_t1[r.release_to]
        ax.annotate("", xy=(x_res, y_res), xytext=(x_src, y_src),
                    arrowprops=dict(arrowstyle="->", color="seagreen", lw=1.0, linestyle="dashed"))
        ax.annotate("", xy=(x_dst, y_dst), xytext=(x_res, y_res),
                    arrowprops=dict(arrowstyle="->", color="darkorange", lw=1.0, linestyle="dashed"))

    # Nodos Tipo-1
    for n in cfg.nodes:
        x, y = pos_t1[n.id]
        if n.flow_station:
            color, marker, size, etiqueta = "tab:red", "*", 260, f"{n.id}\n[{n.flow_station}]"
        elif n.rain_station:
            color, marker, size, etiqueta = "tab:blue", "o", 160, f"{n.id}\n[{n.rain_station}]"
        else:
            color, marker, size, etiqueta = "lightgray", "o", 80, n.id
        ax.scatter([x], [y], s=size, c=color, marker=marker, edgecolor="black", zorder=3)
        ax.text(x, y - 0.2, etiqueta, ha="center", va="top", fontsize=7)

    # Embalses (cuadrado, tamaño ∝ capacidad)
    cap_max = max(r.capacity_hm3 for r in cfg.reservoirs)
    for r in cfg.reservoirs:
        x, y = pos_res[r.id]
        size = 100 + 200 * (r.capacity_hm3 / cap_max)
        ax.scatter([x], [y], s=size, marker="s", c="navy", edgecolor="black", zorder=3)
        ax.text(x, y + 0.2, f"{r.id}\n{r.capacity_hm3:.0f} Hm³",
                ha="center", va="bottom", fontsize=7)

    ax.set_title(f"Topología — {cfg.name}")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="datalim")
    ax.margins(0.15)

    handles = [
        plt.Line2D([0], [0], marker="o", color="tab:blue", linestyle="", label="Estación pluviométrica"),
        plt.Line2D([0], [0], marker="*", color="tab:red", linestyle="", markersize=12, label="Aforo"),
        plt.Line2D([0], [0], marker="s", color="navy", linestyle="", label="Embalse"),
        plt.Line2D([0], [0], color="steelblue", lw=1.5, label="Cauce (E_11)"),
        plt.Line2D([0], [0], color="seagreen", lw=1.0, linestyle="dashed", label="Entrada a embalse (E_12)"),
        plt.Line2D([0], [0], color="darkorange", lw=1.0, linestyle="dashed", label="Suelta de embalse (E_21)"),
    ]
    ax.legend(handles=handles, loc="best", fontsize=7, frameon=True)
    return fig


def _safe_resample_sum(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Resample compatible con pandas ≥2.2 ("ME") y <2.2 ("M")."""
    try:
        return df.resample(freq).sum()
    except ValueError:
        # En pandas <2.2 los aliases nuevos ("ME", "YE", "QE") no existen.
        legacy = {"ME": "M", "YE": "Y", "QE": "Q"}.get(freq)
        if legacy is None:
            raise
        return df.resample(legacy).sum()


def plot_rainfall(rainfall: pd.DataFrame, ax=None, frequency: str = "ME"):
    """Pluviosidad agregada (mensual por defecto) por estación."""
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    else:
        fig = ax.figure

    df = _safe_resample_sum(rainfall, frequency) if frequency else rainfall
    for col in df.columns:
        ax.plot(df.index, df[col].to_numpy(), lw=0.9, alpha=0.75, label=col)
    ax.set_ylabel("mm" + (f" / {frequency}" if frequency else " / día"))
    ax.set_title("Pluviosidad por estación")
    ax.legend(loc="upper right", fontsize=7, ncol=min(3, len(df.columns)))
    ax.grid(True, alpha=0.3)
    return fig


def plot_reservoirs(storage: pd.DataFrame, capacities: Dict[str, float], ax=None):
    """Almacenamiento por embalse y línea horizontal con la capacidad."""
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    else:
        fig = ax.figure

    colors = plt.cm.tab10.colors
    for i, col in enumerate(storage.columns):
        c = colors[i % len(colors)]
        ax.plot(storage.index, storage[col].to_numpy(), lw=1.1, color=c, label=col)
        if col in capacities:
            ax.axhline(capacities[col], color=c, linestyle=":", alpha=0.5, lw=0.8)
    ax.set_ylabel("Almacenamiento (Hm³)")
    ax.set_title("Volumen embalsado (línea punteada = capacidad)")
    ax.legend(loc="best", fontsize=7)
    ax.grid(True, alpha=0.3)
    return fig


def plot_outlet_flow(flow_series: pd.Series, q_min: float, ax=None):
    """Caudal en el aforo con la línea de Q_min."""
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    else:
        fig = ax.figure

    ax.plot(flow_series.index, flow_series.to_numpy(), lw=0.7, color="tab:blue")
    ax.axhline(q_min, color="red", linestyle="--", lw=1.0, label=f"Q_min = {q_min} m³/s")
    bajo = flow_series <= q_min
    if bajo.any():
        # Resaltado de los días por debajo del umbral.
        ax.fill_between(flow_series.index, 0, flow_series.to_numpy(),
                        where=bajo.to_numpy(), color="red", alpha=0.15,
                        label="Por debajo del umbral")
    ax.set_ylabel("Caudal (m³/s)")
    ax.set_title("Caudal en el aforo")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    return fig


# --------------------------------------------------------------------- combo


def plot_summary(cfg: BasinSimConfig, rainfall: pd.DataFrame, sim: SimulationResult):
    """Página única con topología + las tres series."""
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(14, 14))
    gs = GridSpec(4, 1, height_ratios=[2.5, 1, 1, 1], hspace=0.35, figure=fig)

    plot_basin_graph(cfg, ax=fig.add_subplot(gs[0]))
    plot_rainfall(rainfall, ax=fig.add_subplot(gs[1]))
    capacities = {r.name: r.capacity_hm3 for r in cfg.reservoirs}
    plot_reservoirs(sim.storage, capacities, ax=fig.add_subplot(gs[2]))
    outlet = cfg.outlet()
    plot_outlet_flow(sim.flow[outlet.id], cfg.caudal_minimo_m3s, ax=fig.add_subplot(gs[3]))
    fig.suptitle(f"Simulación de la cuenca {cfg.name}", fontsize=14, y=0.995)
    return fig


def save_all(cfg: BasinSimConfig, rainfall: pd.DataFrame, sim: SimulationResult,
             output_dir: Union[str, Path], dpi: int = 150) -> None:
    """Genera y guarda los cuatro plots individuales y el resumen."""
    # En entornos sin display, usar Agg silenciosamente.
    os.environ.setdefault("MPLBACKEND", "Agg")
    import matplotlib.pyplot as plt

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    plot_basin_graph(cfg).savefig(out / "topology.png", dpi=dpi, bbox_inches="tight")
    plt.close("all")

    plot_rainfall(rainfall).savefig(out / "rainfall.png", dpi=dpi, bbox_inches="tight")
    plt.close("all")

    capacities = {r.name: r.capacity_hm3 for r in cfg.reservoirs}
    plot_reservoirs(sim.storage, capacities).savefig(out / "reservoirs.png", dpi=dpi, bbox_inches="tight")
    plt.close("all")

    outlet = cfg.outlet()
    plot_outlet_flow(sim.flow[outlet.id], cfg.caudal_minimo_m3s).savefig(
        out / "outlet_flow.png", dpi=dpi, bbox_inches="tight")
    plt.close("all")

    plot_summary(cfg, rainfall, sim).savefig(out / "summary.png", dpi=dpi, bbox_inches="tight")
    plt.close("all")
