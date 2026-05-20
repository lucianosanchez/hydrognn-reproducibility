"""Visualización de topologías HydroGNN: aprendida vs ground truth.

Útil para Fase 2.2 (grafo de candidatos densos con gates aprendibles)
donde el modelo debe descubrir la ubicación correcta de los embalses
latentes y sus conexiones. En basins sintéticos conocemos la ground
truth, así que podemos cuantificar y visualizar lo bien que el modelo
recupera la topología real.

API principal:
  plot_basin_graph(graph, ax=None, ...)              # dibuja un BasinGraph
  plot_learned_topology(core, ax=None, threshold=..)  # dibuja el aprendido
  plot_comparison(truth_graph, core,                  # side-by-side
                  output_path, threshold=0.10, ...)

Sin dependencias externas (sólo matplotlib).
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

from .graph import BasinGraph


# ============================================================================
# Layout: BFS desde las cabeceras hasta el aforo
# ============================================================================


def _bfs_layers(graph: BasinGraph) -> Dict[int, int]:
    """Profundidad BFS desde las cabeceras (in-degree=0) por E_11.

    Devuelve {node_idx -> layer}, con cabeceras en layer 0 y aforo en
    el layer más profundo. Usado para colocar nodos en columnas en el
    plot.
    """
    n = graph.N1
    in_deg = np.zeros(n, dtype=int)
    out_neighbors: Dict[int, List[int]] = defaultdict(list)
    src, dst = graph.edge_index_11
    for s, d in zip(src.tolist(), dst.tolist()):
        in_deg[d] += 1
        out_neighbors[s].append(d)

    layer: Dict[int, int] = {}
    frontier = [i for i in range(n) if in_deg[i] == 0]
    cur_layer = 0
    while frontier:
        for i in frontier:
            layer.setdefault(i, cur_layer)
        next_layer = []
        for i in frontier:
            for j in out_neighbors[i]:
                # tomar el layer max entre las posibles cabeceras
                layer[j] = max(layer.get(j, 0), cur_layer + 1)
                next_layer.append(j)
        frontier = list(set(next_layer))
        cur_layer += 1
        if cur_layer > n + 2:    # cycle guard
            break
    # los que no se alcanzaron desde cabeceras (raro) van al final
    for i in range(n):
        layer.setdefault(i, cur_layer)
    return layer


def _layout(graph: BasinGraph) -> Tuple[Dict[int, Tuple[float, float]],
                                         Dict[int, Tuple[float, float]]]:
    """Coordenadas (x, y) para nodos Tipo-1 y embalses.

    Si el `BasinGraph` aporta `type1_latlon` y `res_latlon`, se usa
    layout geográfico (lon→x, lat→y). En otro caso, fallback a layout
    por columnas BFS desde las cabeceras.
    """
    # Layout geográfico si hay coordenadas
    if getattr(graph, "type1_latlon", None):
        pos1 = {}
        for i, name in enumerate(graph.type1_names):
            ll = graph.type1_latlon.get(name)
            if ll is None:
                # nodo sin coords → caemos a layout por BFS abajo
                pos1 = {}
                break
            lat, lon = ll
            pos1[i] = (lon, lat)
        if pos1:
            pos2 = {}
            res_latlon = getattr(graph, "res_latlon", None) or {}
            for k, name in enumerate(graph.res_names):
                ll = res_latlon.get(name)
                if ll is not None:
                    lat, lon = ll
                    pos2[k] = (lon, lat)
                else:
                    # embalse sin coords → al lado del primer origen E_12
                    srcs = [int(graph.src12[e]) for e in range(graph.E12)
                            if int(graph.dst12[e]) == k]
                    if srcs:
                        x = float(np.mean([pos1[s][0] for s in srcs]))
                        y = float(np.mean([pos1[s][1] for s in srcs]))
                        pos2[k] = (x + 0.05, y + 0.05)
                    else:
                        pos2[k] = (0.0, 0.0)
            return pos1, pos2

    # Fallback: BFS layers
    layers = _bfs_layers(graph)
    max_layer = max(layers.values()) if layers else 0

    # Type-1: agrupar por layer, repartir verticalmente.
    by_layer: Dict[int, List[int]] = defaultdict(list)
    for i in range(graph.N1):
        by_layer[layers[i]].append(i)
    for L in by_layer:
        by_layer[L].sort()

    pos1: Dict[int, Tuple[float, float]] = {}
    for L, nodes in by_layer.items():
        n = len(nodes)
        for k, i in enumerate(nodes):
            x = L * 1.6
            # centra los nodos verticalmente en su columna
            y = -(k - (n - 1) / 2.0)
            pos1[i] = (x, y)

    # Type-2: ubicar al lado del primer nodo origen E_12.
    pos2: Dict[int, Tuple[float, float]] = {}
    for k in range(graph.M):
        # nodos Tipo-1 que alimentan a este embalse
        srcs = [int(graph.src12[e]) for e in range(graph.E12)
                if int(graph.dst12[e]) == k]
        if not srcs:
            # huérfano: coloca al margen derecho-superior
            pos2[k] = ((max_layer + 1) * 1.6, 3.0 + 0.5 * k)
            continue
        x_mean = float(np.mean([pos1[s][0] for s in srcs]))
        y_mean = float(np.mean([pos1[s][1] for s in srcs]))
        # desplaza ligeramente arriba a la derecha
        pos2[k] = (x_mean + 0.4, y_mean + 0.6)
    return pos1, pos2


# ============================================================================
# Plot primitives
# ============================================================================


def _draw_arrow(ax, x0, y0, x1, y1, **kw):
    """Flecha entre dos coords con FancyArrowPatch (curva ligera)."""
    arrow = FancyArrowPatch(
        (x0, y0), (x1, y1),
        arrowstyle="-|>",
        mutation_scale=10,
        connectionstyle="arc3,rad=0.07",
        shrinkA=8, shrinkB=8,
        zorder=1, **kw,
    )
    ax.add_patch(arrow)


def _plot_nodes(ax, pos1, pos2, type1_names, res_names, gauged_nodes=None,
                target_idx=None, latent_res_idx=None):
    """Dibuja los nodos Tipo-1 (cuadrados) y Tipo-2 (círculos)."""
    gauged_nodes = gauged_nodes or set()
    latent_res_idx = latent_res_idx or set()

    # Type-1
    for i, (x, y) in pos1.items():
        if i == target_idx:
            color = "#d62728"          # rojo: outlet
            shape = "s"; size = 220
        elif i in gauged_nodes:
            color = "#1f77b4"          # azul: con pluviómetro
            shape = "s"; size = 160
        else:
            color = "#aaaaaa"          # gris: junction sin sensor
            shape = "s"; size = 110
        ax.scatter([x], [y], marker=shape, s=size, c=color,
                    edgecolors="black", linewidths=0.8, zorder=3)
        ax.annotate(type1_names[i] if i < len(type1_names) else f"v{i}",
                    (x, y), xytext=(0, -14), textcoords="offset points",
                    ha="center", va="top", fontsize=6, zorder=4)

    # Type-2
    for k, (x, y) in pos2.items():
        if k in latent_res_idx:
            color = "#ff7f0e"          # naranja: latent / learned
            edge = "darkorange"
        else:
            color = "#2ca02c"          # verde: real reservoir
            edge = "darkgreen"
        ax.scatter([x], [y], marker="o", s=220, c=color,
                    edgecolors=edge, linewidths=1.0, zorder=3)
        ax.annotate(res_names[k] if k < len(res_names) else f"R{k}",
                    (x, y), xytext=(0, 14), textcoords="offset points",
                    ha="center", va="bottom", fontsize=6, weight="bold",
                    zorder=4)


def _plot_edges_e11(ax, pos1, edge_index_11, color="black", width=1.0,
                     alpha=0.7):
    """Aristas E_11 (río Tipo-1 → Tipo-1)."""
    src, dst = edge_index_11
    for s, d in zip(src.tolist(), dst.tolist()):
        _draw_arrow(ax, *pos1[int(s)], *pos1[int(d)],
                     color=color, lw=width, alpha=alpha)


def _plot_edges_e12_e21(ax, pos1, pos2, src12, dst12, src21, dst21,
                          weights12=None, weights21=None,
                          color12="#2ca02c", color21="#ff7f0e",
                          base_width=1.5, max_width=3.5, alpha=0.85):
    """Aristas E_12 (Tipo-1 → reservorio) y E_21 (reservorio → Tipo-1)."""
    # E_12: linestyle dashed, color verde
    for e in range(len(src12)):
        s = int(src12[e]); d = int(dst12[e])
        w = float(weights12[e]) if weights12 is not None else 1.0
        if w < 1e-3:
            continue
        lw = base_width + (max_width - base_width) * min(w, 1.0)
        _draw_arrow(ax, *pos1[s], *pos2[d],
                     color=color12, lw=lw, alpha=alpha, linestyle="--")
    # E_21: linestyle dotted, color naranja
    for e in range(len(src21)):
        s = int(src21[e]); d = int(dst21[e])
        w = float(weights21[e]) if weights21 is not None else 1.0
        if w < 1e-3:
            continue
        lw = base_width + (max_width - base_width) * min(w, 1.0)
        _draw_arrow(ax, *pos2[s], *pos1[d],
                     color=color21, lw=lw, alpha=alpha, linestyle=":")


# ============================================================================
# Public API
# ============================================================================


def plot_basin_graph(graph: BasinGraph, ax=None, title: str = "",
                       gauged_nodes: Optional[set] = None,
                       latent_res_idx: Optional[set] = None) -> "plt.Axes":
    """Dibuja un `BasinGraph` con todas sus aristas E_11, E_12, E_21."""
    if not _HAS_MPL:
        raise ImportError("matplotlib es requerido para plot_basin_graph.")
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 6))
    pos1, pos2 = _layout(graph)
    _plot_edges_e11(ax, pos1, graph.edge_index_11)
    _plot_edges_e12_e21(ax, pos1, pos2,
                          graph.src12, graph.dst12,
                          graph.src21, graph.dst21)
    if gauged_nodes is None:
        gauged_nodes = set(graph.rain_to_type1.values()) if graph.rain_to_type1 else set()
    _plot_nodes(ax, pos1, pos2, graph.type1_names, graph.res_names,
                 gauged_nodes=gauged_nodes,
                 target_idx=graph.target_node_idx,
                 latent_res_idx=latent_res_idx)
    ax.set_title(title, fontsize=11)
    ax.axis("off")
    ax.margins(0.10)
    return ax


def plot_learned_topology(core, ax=None, title: str = "",
                            threshold: float = 0.10) -> "plt.Axes":
    """Dibuja la topología aprendida por un `HydroGNNCore` con gates.

    Sólo tiene sentido para Fase 2.2 (`use_gates="nodes_and_edges"`).
    Las aristas con share < `threshold` no se dibujan; el grosor de las
    visibles es proporcional a su share.

    Parameters
    ----------
    core : HydroGNNCore
        Modelo entrenado. Debe haber sido construido con un grafo de
        candidatos densos (`dense_candidate_graph`).
    threshold : float
        Mínimo share para considerar una arista E_12 / E_21 como activa.
    """
    if not _HAS_MPL:
        raise ImportError("matplotlib es requerido para plot_learned_topology.")
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 6))

    info = core.analyze_positions()
    # inflow_share[k, i]: fracción del flujo del nodo Tipo-1 i que va al embalse k
    # outflow_share[k, j]: fracción de salida del embalse k al nodo Tipo-1 j
    inflow = np.asarray(info["inflow_share"])
    outflow = np.asarray(info["outflow_share"])
    z_res = np.asarray(info["z_res"])

    # Criterio de "embalse activo":
    #   * Si los gates están aprendidos (algún z_res < 0.5) → usamos z_res > 0.5.
    #   * Si los gates están fijos en 1 (Phase 1 / Phase 2.2 con use_gates="none")
    #     → un embalse se considera activo si recibe algún flujo significativo
    #       y suelta algo: total_inflow_share[k] > 0.1 AND total_outflow_share[k] > 0.1.
    N1 = inflow.shape[1]
    M = inflow.shape[0]

    gates_learned = bool(np.any(z_res < 0.5))
    if gates_learned:
        active_res = [k for k in range(M) if z_res[k] > 0.5]
    else:
        in_total = inflow.sum(axis=1)         # (M,) — sum over Type-1 sources
        out_total = outflow.sum(axis=1)       # (M,) — sum over Type-1 sinks
        active_res = [k for k in range(M)
                       if in_total[k] > 0.10 and out_total[k] > 0.10]

    src12, dst12, w12 = [], [], []
    for k in range(M):
        if k not in active_res:
            continue
        for i in range(N1):
            if inflow[k, i] >= threshold:
                src12.append(i); dst12.append(k); w12.append(float(inflow[k, i]))
    src21, dst21, w21 = [], [], []
    for k in range(M):
        if k not in active_res:
            continue
        for j in range(N1):
            if outflow[k, j] >= threshold:
                src21.append(k); dst21.append(j); w21.append(float(outflow[k, j]))

    # Construimos un BasinGraph ficticio para reutilizar el layout
    # (mantenemos los nombres del grafo del core).
    type1_names = [n for n in (core.graph.type1_names if hasattr(core, "graph")
                                else [f"v{i}" for i in range(N1)])]
    res_names = [f"R*{k}" for k in active_res]
    # Re-indexamos los embalses al subconjunto activo
    res_map = {k: idx for idx, k in enumerate(active_res)}
    dst12_re = np.array([res_map[k] for k in dst12], dtype=np.int64)
    src21_re = np.array([res_map[k] for k in src21], dtype=np.int64)

    learned = BasinGraph(
        type1_names=type1_names,
        edge_index_11=(core.edge_index_11.cpu().numpy() if hasattr(core, "edge_index_11")
                         else np.zeros((2, 0), dtype=np.int64)),
        res_names=res_names,
        src12=np.array(src12, dtype=np.int64),
        dst12=dst12_re,
        src21=src21_re,
        dst21=np.array(dst21, dtype=np.int64),
        target_node_idx=int(core.target_idx),
        rain_to_type1={},
        res_to_observed={},
    )

    pos1, pos2 = _layout(learned)
    _plot_edges_e11(ax, pos1, learned.edge_index_11, color="#666666")
    _plot_edges_e12_e21(ax, pos1, pos2,
                          learned.src12, learned.dst12,
                          learned.src21, learned.dst21,
                          weights12=np.asarray(w12),
                          weights21=np.asarray(w21))
    _plot_nodes(ax, pos1, pos2, learned.type1_names, learned.res_names,
                 gauged_nodes=None,
                 target_idx=learned.target_node_idx,
                 latent_res_idx=set(range(learned.M)))
    ax.set_title(title or f"Topología aprendida (z_res>0.5, share≥{threshold:g})",
                  fontsize=11)
    ax.axis("off")
    ax.margins(0.10)
    return ax


def topology_recovery_metrics(truth: BasinGraph, core,
                                threshold: float = 0.10) -> dict:
    """Métricas cuantitativas de recuperación de la topología.

    Compara la topología aprendida (de `core.analyze_positions()`) con la
    ground-truth (`truth`):
      * num_active_reservoirs: cuántos embalses (z_res > 0.5) dejó el modelo.
      * e12_recall, e12_precision: aristas Tipo-1 → embalse acertadas vs reales.
      * e21_recall, e21_precision: aristas embalse → Tipo-1.
      * inflow_node_iou: IoU entre conjuntos {nodos Tipo-1 que alimentan
                          algún embalse activo, real y aprendido}.
    """
    info = core.analyze_positions()
    inflow = np.asarray(info["inflow_share"])
    outflow = np.asarray(info["outflow_share"])
    z_res = np.asarray(info["z_res"])

    # Mapeo embalse aprendido ←→ embalse real: lo asignamos por overlap
    # de inflow_share con la matriz binaria real.
    N1 = truth.N1; M_real = truth.M; M_lat = inflow.shape[0]
    in_real = np.zeros((M_real, N1)); out_real = np.zeros((M_real, N1))
    for e in range(truth.E12):
        in_real[int(truth.dst12[e]), int(truth.src12[e])] = 1.0
    for e in range(truth.E21):
        out_real[int(truth.src21[e]), int(truth.dst21[e])] = 1.0

    # Embalses aprendidos activos (mismo criterio que plot_learned_topology).
    gates_learned = bool(np.any(z_res < 0.5))
    if gates_learned:
        active = [k for k in range(M_lat) if z_res[k] > 0.5]
    else:
        in_total = inflow.sum(axis=1)
        out_total = outflow.sum(axis=1)
        active = [k for k in range(M_lat)
                   if in_total[k] > 0.10 and out_total[k] > 0.10]

    # Aristas aprendidas binarizadas
    in_learn = (inflow[active] >= threshold).astype(int) if active else np.zeros((0, N1))
    out_learn = (outflow[active] >= threshold).astype(int) if active else np.zeros((0, N1))

    # Mejor asignación greedy: para cada embalse aprendido, busca el real
    # con mayor IoU sobre inflow_share. Sin emparejamiento global óptimo.
    matched_real_e12 = np.zeros_like(in_real)
    matched_real_e21 = np.zeros_like(out_real)
    for i_lat in range(len(active)):
        best_iou = -1.0; best_k = -1
        for k_real in range(M_real):
            inter = int(np.sum(in_learn[i_lat] & in_real[k_real].astype(int)))
            uni = int(np.sum(in_learn[i_lat] | in_real[k_real].astype(int)))
            iou = inter / max(uni, 1)
            if iou > best_iou:
                best_iou = iou; best_k = k_real
        if best_k >= 0:
            matched_real_e12[best_k] = np.maximum(matched_real_e12[best_k], in_learn[i_lat])
            matched_real_e21[best_k] = np.maximum(matched_real_e21[best_k], out_learn[i_lat])

    def _prec_recall(real, pred):
        real = real.astype(bool); pred = pred.astype(bool)
        tp = int((real & pred).sum())
        fp = int((~real & pred).sum())
        fn = int((real & ~pred).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        return precision, recall, tp, fp, fn

    p12, r12, tp12, fp12, fn12 = _prec_recall(in_real, matched_real_e12)
    p21, r21, tp21, fp21, fn21 = _prec_recall(out_real, matched_real_e21)

    # IoU del conjunto de nodos Tipo-1 que alimentan algún embalse
    in_real_nodes = set(int(s) for s in truth.src12)
    in_learn_nodes = set(i for i_lat in range(len(active))
                            for i in np.where(in_learn[i_lat] > 0)[0].tolist())
    inter_n = len(in_real_nodes & in_learn_nodes)
    union_n = max(len(in_real_nodes | in_learn_nodes), 1)
    iou_nodes = inter_n / union_n

    return {
        "num_active_reservoirs_learned": int(len(active)),
        "num_reservoirs_real": int(M_real),
        "e12_precision": p12, "e12_recall": r12,
        "e12_tp": tp12, "e12_fp": fp12, "e12_fn": fn12,
        "e21_precision": p21, "e21_recall": r21,
        "e21_tp": tp21, "e21_fp": fp21, "e21_fn": fn21,
        "inflow_node_iou": iou_nodes,
    }


def plot_comparison(truth: BasinGraph, core, output_path: Path,
                     threshold: float = 0.10, title_left: str = "Ground truth",
                     title_right: str = "Learned", suptitle: str = "",
                     figsize: Tuple[float, float] = (16, 7)) -> dict:
    """Side-by-side: ground truth (izquierda) vs aprendido (derecha).

    También computa y dibuja en el `suptitle` las métricas de recovery.
    Devuelve las métricas como dict.
    """
    if not _HAS_MPL:
        raise ImportError("matplotlib es requerido para plot_comparison.")
    metrics = topology_recovery_metrics(truth, core, threshold=threshold)

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    plot_basin_graph(truth, ax=axes[0], title=title_left)
    plot_learned_topology(core, ax=axes[1], title=title_right, threshold=threshold)

    summary = (
        f"reservoirs: real={metrics['num_reservoirs_real']}, "
        f"learned={metrics['num_active_reservoirs_learned']}  |  "
        f"E_12 precision={metrics['e12_precision']:.2f} "
        f"recall={metrics['e12_recall']:.2f}  |  "
        f"E_21 precision={metrics['e21_precision']:.2f} "
        f"recall={metrics['e21_recall']:.2f}  |  "
        f"inflow-node IoU={metrics['inflow_node_iou']:.2f}"
    )
    if suptitle:
        fig.suptitle(f"{suptitle}\n{summary}", fontsize=11)
    else:
        fig.suptitle(summary, fontsize=10)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return metrics


__all__ = [
    "plot_basin_graph",
    "plot_learned_topology",
    "plot_comparison",
    "topology_recovery_metrics",
]
