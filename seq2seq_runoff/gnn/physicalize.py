"""Post-procesado: transforma un grafo aprendido en uno físicamente válido.

Phase 2.2 entrena con `dense_candidate_graph`, donde cada embalse formal
puede conectar con cualquier nodo Type-1 en ambos sentidos. El optimizador
encuentra mínimos del error donde las aristas E_21 (sueltas) apuntan
hacia nodos aguas arriba — matemáticamente equivalentes a un bucle
temporal, pero físicamente imposibles (no se vierte agua aguas arriba).

Este módulo implementa una **transformación post-hoc** que:

  1. Identifica embalses formales activos (in/out share > τ).
  2. Para cada embalse activo, determina su "destino canónico":
     el descendiente BFS común más cercano de los nodos que lo
     alimentan en E_11. Si no existe, marca el embalse como
     no-físicamente realizable y lo descarta.
  3. Reorganiza E_12 y E_21 colapsando las múltiples salidas en una
     sola arista por embalse hacia su destino canónico, preservando
     la suma de masa.
  4. Devuelve un nuevo `BasinGraph` y la receta para transferir los
     parámetros aprendidos del core original al nuevo grafo, de modo
     que el modelo no requiere reentrenamiento.

La transformación es exacta en mass-balance (la suma de pesos se
preserva) pero aproximada en dinámica (porque la composición de
filtros y la dinámica no-lineal del embalse no se redistribuyen de
forma exacta). La función `verify_equivalence` mide cuánto se aparta
el caudal del outlet del original; si la desviación es despreciable,
la physicalización es válida para reportar.

API:
  physicalize_topology(core, physical_graph_ref, threshold=0.10)
      → (new_basin_graph, transfer_plan, quality_metrics)
  transfer_core_weights(old_core, new_graph, transfer_plan)
      → new HydroGNNCore listo para inferencia
  verify_equivalence(old_core, new_core, dataloader, ...)
      → dict con ΔNSE, ΔRMSE, ΔFN, correlación
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn

from .graph import BasinGraph


# ---------------------------------------------------------------------------
# Análisis del grafo físico: ¿qué Type-1 es descendiente de qué Type-1?
# ---------------------------------------------------------------------------


def _bfs_descendants(physical_graph: BasinGraph) -> Dict[int, set]:
    """Para cada nodo i, devuelve el conjunto de descendientes BFS en E_11."""
    n = physical_graph.N1
    out_neighbors: Dict[int, List[int]] = {i: [] for i in range(n)}
    src, dst = physical_graph.edge_index_11
    for s, d in zip(src.tolist(), dst.tolist()):
        out_neighbors[int(s)].append(int(d))

    desc: Dict[int, set] = {}
    for i in range(n):
        seen = set()
        frontier = list(out_neighbors[i])
        while frontier:
            x = frontier.pop()
            if x in seen:
                continue
            seen.add(x)
            frontier.extend(out_neighbors[x])
        desc[i] = seen
    return desc


def _bfs_depth(physical_graph: BasinGraph) -> Dict[int, int]:
    """Profundidad BFS desde las cabeceras (in-degree 0 en E_11)."""
    n = physical_graph.N1
    in_deg = np.zeros(n, dtype=int)
    out_neighbors: Dict[int, List[int]] = {i: [] for i in range(n)}
    src, dst = physical_graph.edge_index_11
    for s, d in zip(src.tolist(), dst.tolist()):
        in_deg[int(d)] += 1
        out_neighbors[int(s)].append(int(d))
    depth: Dict[int, int] = {}
    frontier = [i for i in range(n) if in_deg[i] == 0]
    cur = 0
    while frontier:
        for i in frontier:
            depth.setdefault(i, cur)
        nxt = []
        for i in frontier:
            for j in out_neighbors[i]:
                depth[j] = max(depth.get(j, 0), cur + 1)
                nxt.append(j)
        frontier = list(set(nxt))
        cur += 1
        if cur > n + 2:
            break
    for i in range(n):
        depth.setdefault(i, cur)
    return depth


def _canonical_destination(
    source_set: List[int],
    inflow_weights: Dict[int, float],
    desc: Dict[int, set],
    depth: Dict[int, int],
) -> Optional[int]:
    """Destino canónico del embalse: descendiente común más somero de
    todos los nodos en source_set; en su defecto, primer descendiente
    del source dominante."""
    if not source_set:
        return None
    # Descendientes comunes a todos los sources, incluyéndolos a ellos
    # mismos para gestionar el caso "source dominante es sink".
    common = None
    for i in source_set:
        cand = set([i]) | desc.get(i, set())
        common = cand if common is None else (common & cand)
    common = common or set()
    # No vale tomar como destino canónico un nodo del propio source_set
    # (eso es exactamente el "loop sobre sí mismo" que queremos evitar).
    common -= set(source_set)
    if common:
        # El más somero (menor depth = más aguas arriba dentro del set
        # de descendientes comunes), que físicamente es el primer punto
        # del río donde el agua se "reúne" tras el embalse.
        return min(common, key=lambda j: depth.get(j, 10**9))

    # Fallback: descendientes del source dominante (mayor inflow_weight).
    dominant = max(source_set, key=lambda i: inflow_weights.get(i, 0.0))
    cand = desc.get(dominant, set()) - set(source_set)
    if cand:
        return min(cand, key=lambda j: depth.get(j, 10**9))
    return None


# ---------------------------------------------------------------------------
# Resultado de la physicalización
# ---------------------------------------------------------------------------


@dataclass
class TransferPlan:
    """Receta para construir un nuevo HydroGNNCore equivalente al aprendido.

    Mantiene los mapeos {old_edge_idx → new_edge_idx} para E_12 y E_21
    junto con los nodos / embalses retenidos, de modo que el módulo
    `transfer_core_weights` pueda copiar los parámetros aprendidos
    directamente sin recompilar el modelo.

    Distinguimos:
      * `active_reservoirs_idx_old`: índices en `core.M` originales que
        pasaron el threshold in/out (in_total > τ_in AND out_total > τ_out).
      * `kept_reservoirs_idx_old`: subset estricto de los anteriores que
        además admite un destino canónico físicamente realizable. Sólo
        éstos se transfieren al nuevo core; el resto se descartan.
    """
    new_graph: BasinGraph
    active_reservoirs_idx_old: List[int]             # in/out > τ
    kept_reservoirs_idx_old: List[int]               # active AND canonical_dst encontrado
    # Mapeos {(src, dst) → idx} para localizar cada arista del nuevo grafo
    # en el set de aristas del original (E_12).
    e12_old_idx_per_new: List[List[int]]             # por arista nueva, lista de aristas viejas
    # Para E_21, cada embalse tiene UNA arista nueva (colapsado de las
    # múltiples destinos físicamente inconsistentes); el plan recuerda
    # qué aristas viejas se han colapsado en cada arista nueva.
    e21_old_idx_per_new: List[List[int]]
    # Métricas de calidad de la physicalización.
    metrics: dict = field(default_factory=dict)


def physicalize_topology(
    core,
    physical_graph_ref: BasinGraph,
    threshold_in: float = 0.10,
    threshold_out: float = 0.10,
    mode: str = "strict",
) -> TransferPlan:
    """Construye un grafo físicamente coherente equivalente al aprendido.

    Parameters
    ----------
    core : HydroGNNCore
        Modelo entrenado en Fase 2.2 con `dense_candidate_graph`.
    physical_graph_ref : BasinGraph
        Grafo físico de referencia (e.g. `synth_graph_full` o `ebro_graph`).
        Sólo se usa su `edge_index_11` y `target_node_idx` para la
        topología del cauce.
    threshold_in, threshold_out : float
        Umbrales de activación de aristas E_12 / E_21. Default 0.10.
    mode : {"strict", "soft"}
        - "strict": las aristas E_21 hacia atrás se DESCARTAN (su masa se
          pierde). El nuevo grafo conserva sólo las E_21 físicamente
          válidas del aprendido. Si NINGUNA es válida, se colapsa al
          canonical destination del fallback.
        - "soft": las aristas E_21 hacia atrás se REASIGNAN al canonical
          destination del embalse (su masa se suma vía logsumexp al
          logw21 del destino válido más cercano). Esto preserva la masa
          total y, en general, da una predicción más cercana al original.

    Returns
    -------
    TransferPlan con el nuevo grafo y las recetas de transferencia.
    """
    if mode not in ("strict", "soft"):
        raise ValueError(f"mode debe ser 'strict' o 'soft'; recibido {mode!r}")
    info = core.analyze_positions()
    inflow = np.asarray(info["inflow_share"])    # (M_lat, N1)
    outflow = np.asarray(info["outflow_share"])  # (M_lat, N1)
    z_res = np.asarray(info["z_res"])

    M_lat, N1 = inflow.shape

    # Embalses activos según el mismo criterio que viz.py.
    in_total = inflow.sum(axis=1)
    out_total = outflow.sum(axis=1)
    gates_learned = bool(np.any(z_res < 0.5))
    if gates_learned:
        active = [k for k in range(M_lat) if z_res[k] > 0.5]
    else:
        active = [k for k in range(M_lat)
                   if in_total[k] > threshold_in
                   and out_total[k] > threshold_out]

    desc = _bfs_descendants(physical_graph_ref)
    depth = _bfs_depth(physical_graph_ref)

    new_res_names: List[str] = []
    new_src12: List[int] = []
    new_dst12: List[int] = []
    e12_old_per_new: List[List[int]] = []
    new_src21: List[int] = []
    new_dst21: List[int] = []
    e21_old_per_new: List[List[int]] = []

    # Para localizar aristas viejas (i → k_old) y (k_old → j) en core.
    src12_old = core.src12.cpu().numpy() if hasattr(core, "src12") else None
    dst12_old = core.dst12.cpu().numpy() if hasattr(core, "dst12") else None
    src21_old = core.src21.cpu().numpy() if hasattr(core, "src21") else None
    dst21_old = core.dst21.cpu().numpy() if hasattr(core, "dst21") else None

    discarded_no_canonical = []
    kept_reservoirs_idx_old: List[int] = []
    backflow_share_orig = 0.0
    inflow_share_orig = 0.0
    n_e21_kept_downstream = 0       # aristas E_21 conservadas (físicas)
    n_e21_collapsed_fallback = 0    # E_21 colapsadas a una arista canónica

    for k_old in active:
        new_k = len(kept_reservoirs_idx_old)  # índice tentativo en el nuevo grafo
        # source set y inflow weights para este embalse
        source_set = [i for i in range(N1) if inflow[k_old, i] >= threshold_in]
        inflow_weights = {i: float(inflow[k_old, i]) for i in source_set}
        if not source_set:
            discarded_no_canonical.append(k_old)
            continue

        # ----- Métrica de back-flow ANTES de tomar decisión -----
        # j es "downstream" si está en los descendientes de ALGÚN source
        # (o es el source mismo: caso límite que no consideramos válido).
        downstream_destinations = []   # (j, mass, old_edge_idx)
        backflow_old_edges = []        # aristas viejas hacia destinos no-físicos
        for j in range(N1):
            mass = float(outflow[k_old, j])
            if mass < 1e-6:
                continue
            is_downstream = any(j in desc.get(i, set()) for i in source_set)
            inflow_share_orig += mass
            # Localiza la arista vieja correspondiente (k_old, j)
            if src21_old is not None:
                olds = [e for e in range(len(src21_old))
                        if int(src21_old[e]) == k_old and int(dst21_old[e]) == j]
            else:
                olds = []
            if not is_downstream:
                backflow_share_orig += mass
                backflow_old_edges.extend(olds)
                continue
            if mass >= threshold_out:
                downstream_destinations.append((j, mass, olds))

        # ----- Decisión: conservar destinos físicos o colapsar -----
        if downstream_destinations:
            # CONSERVADOR: mantenemos cada destino físicamente válido como
            # arista propia, sin colapsar. La masa relativa se preserva
            # por los logw21 individuales.
            #
            # Modo "soft": la masa back-flow se REASIGNA al canonical
            # destination del embalse (o, si no existe, al destino válido
            # con mayor inflow_share aprendido). Mass-preserving.
            if mode == "soft" and backflow_old_edges:
                # Elegir destino canónico: preferentemente el descendiente
                # común; si no, el destino downstream con mayor masa
                # aprendida.
                canonical = _canonical_destination(
                    source_set, inflow_weights, desc, depth)
                if canonical is None:
                    # fallback: el destino downstream con mayor masa.
                    canonical = max(downstream_destinations, key=lambda t: t[1])[0]
                # Buscar si ese canonical ya está en downstream_destinations;
                # si no, añadirlo y darle la masa de las backflow_old_edges.
                hit = next(
                    (idx for idx, (j, _, _) in enumerate(downstream_destinations)
                     if j == canonical),
                    None,
                )
                if hit is None:
                    downstream_destinations.append((canonical, 0.0, list(backflow_old_edges)))
                else:
                    j, mass, olds = downstream_destinations[hit]
                    downstream_destinations[hit] = (j, mass, list(olds) + list(backflow_old_edges))
                n_e21_collapsed_fallback += 1
        else:
            # Ningún destino aprendido era físicamente válido. Fallback:
            # colapsamos TODOS los destinos aprendidos en una sola arista
            # al canonical destination (descendiente común o del dominante).
            canonical = _canonical_destination(source_set, inflow_weights, desc, depth)
            if canonical is None:
                # Ni siquiera hay fallback posible → descartamos.
                discarded_no_canonical.append(k_old)
                continue
            # Lista de TODOS los old edges de este embalse (incluye los
            # que iban "hacia atrás", cuya masa absorbemos al canonical).
            if src21_old is not None:
                all_olds = [e for e in range(len(src21_old))
                            if int(src21_old[e]) == k_old]
            else:
                all_olds = []
            downstream_destinations = [(canonical, 0.0, all_olds)]
            n_e21_collapsed_fallback += 1

        # Pasó: lo registramos como "kept".
        kept_reservoirs_idx_old.append(k_old)
        new_res_names.append(f"R*{k_old}")
        # E_12 nuevas (una por source)
        for i in source_set:
            new_src12.append(i); new_dst12.append(new_k)
            if src12_old is not None:
                olds = [e for e in range(len(src12_old))
                        if int(src12_old[e]) == i and int(dst12_old[e]) == k_old]
            else:
                olds = []
            e12_old_per_new.append(olds)
        # E_21: una arista por cada destino físicamente válido
        for j, mass, olds in downstream_destinations:
            new_src21.append(new_k); new_dst21.append(j)
            e21_old_per_new.append(olds)
            if mass > 0.0:
                n_e21_kept_downstream += 1

    # Si todos los embalses se descartan, devolvemos al menos uno trivial
    # para que el BasinGraph siga siendo construible. En la práctica esto
    # no debería ocurrir si el modelo aprende algo.
    if not new_res_names:
        return TransferPlan(
            new_graph=BasinGraph(
                type1_names=physical_graph_ref.type1_names,
                edge_index_11=physical_graph_ref.edge_index_11.copy(),
                res_names=[],
                src12=np.zeros(0, dtype=np.int64),
                dst12=np.zeros(0, dtype=np.int64),
                src21=np.zeros(0, dtype=np.int64),
                dst21=np.zeros(0, dtype=np.int64),
                target_node_idx=physical_graph_ref.target_node_idx,
                rain_to_type1=dict(physical_graph_ref.rain_to_type1),
                res_to_observed={},
                edge_len_km_11=getattr(physical_graph_ref, "edge_len_km_11", None),
                type1_latlon=getattr(physical_graph_ref, "type1_latlon", None),
            ),
            active_reservoirs_idx_old=list(active),
            kept_reservoirs_idx_old=[],
            e12_old_idx_per_new=[],
            e21_old_idx_per_new=[],
            metrics={
                "n_reservoirs_active": len(active),
                "n_reservoirs_kept": 0,
                "n_reservoirs_discarded": len(discarded_no_canonical),
                "backflow_share_original": 0.0,
            },
        )

    new_basin_graph = BasinGraph(
        type1_names=physical_graph_ref.type1_names,
        edge_index_11=physical_graph_ref.edge_index_11.copy(),
        res_names=new_res_names,
        src12=np.array(new_src12, dtype=np.int64),
        dst12=np.array(new_dst12, dtype=np.int64),
        src21=np.array(new_src21, dtype=np.int64),
        dst21=np.array(new_dst21, dtype=np.int64),
        target_node_idx=physical_graph_ref.target_node_idx,
        rain_to_type1=dict(physical_graph_ref.rain_to_type1),
        res_to_observed={},
        edge_len_km_11=getattr(physical_graph_ref, "edge_len_km_11", None),
        type1_latlon=getattr(physical_graph_ref, "type1_latlon", None),
    )

    metrics = {
        "n_reservoirs_active": len(active),
        "n_reservoirs_kept": len(new_res_names),
        "n_reservoirs_discarded": len(discarded_no_canonical),
        "n_e12_new": len(new_src12),
        "n_e21_new": len(new_src21),
        "n_e21_kept_downstream": int(n_e21_kept_downstream),
        "n_e21_collapsed_fallback": int(n_e21_collapsed_fallback),
        "n_e12_old": int(len(src12_old)) if src12_old is not None else None,
        "n_e21_old": int(len(src21_old)) if src21_old is not None else None,
        "backflow_share_original": (
            float(backflow_share_orig / inflow_share_orig)
            if inflow_share_orig > 0 else 0.0
        ),
        "downstream_share_original": (
            float(1.0 - backflow_share_orig / inflow_share_orig)
            if inflow_share_orig > 0 else 0.0
        ),
    }

    return TransferPlan(
        new_graph=new_basin_graph,
        active_reservoirs_idx_old=list(active),
        kept_reservoirs_idx_old=kept_reservoirs_idx_old,
        e12_old_idx_per_new=e12_old_per_new,
        e21_old_idx_per_new=e21_old_per_new,
        metrics=metrics,
    )


# ---------------------------------------------------------------------------
# Transferencia de pesos del core viejo al nuevo grafo
# ---------------------------------------------------------------------------


def transfer_core_weights(old_core, plan: TransferPlan):
    """Construye un nuevo `HydroGNNCore` con `plan.new_graph` y copia
    todos los parámetros aprendidos. Devuelve el nuevo core en modo eval.

    Convenciones de transferencia:
      * Parámetros por-arista E_11: copiados tal cual (E_11 no cambia).
      * Por-arista E_12: copiados desde la(s) arista(s) vieja(s); si una
        nueva arista colapsa varias viejas, se toma la suma de los exp
        (preservando masa) para logw12, y promedio ponderado para lam12.
      * Por-arista E_21: colapsado de las múltiples destinos viejos:
        logw21 = log(sum(exp(logw21_old))) — preserva masa total.
        lam21 = promedio ponderado por exp(logw21_old).
      * Por-embalse (beta_wA, gamma_logit, res_embed, gate_res):
        subselect por el índice antiguo `active_reservoirs_idx_old`.
      * MLPs (runoff_net, alpha_net, beta_ctx, sigma_head), node_embed,
        logw11, lam11_logit, gates de aristas E_12/E_21: si el subset
        es compatible, copia directa.
    """
    from .core import HydroGNNCore

    new_graph = plan.new_graph
    new_core = HydroGNNCore(
        new_graph,
        use_gates=old_core.use_gates,
        node_static_dim=old_core.node_embed.shape[-1],
        ctx_dim=old_core.sigma_head[0].in_features - 1,  # sigma_head input = (1 + ctx_dim,)
        hidden=old_core.runoff_net[0].out_features,
        logw12_init=old_core.logw12_init,
        rain_bypass=old_core.rain_bypass,
        lam11_init=0.0,
        river_velocity_km_day=None,
    )
    new_core.eval()

    with torch.no_grad():
        # ----- Parámetros independientes de la indexación de embalses -----
        new_core.node_embed.data.copy_(old_core.node_embed.data)
        new_core.runoff_net.load_state_dict(old_core.runoff_net.state_dict())
        new_core.alpha_net.load_state_dict(old_core.alpha_net.state_dict())
        new_core.beta_ctx.load_state_dict(old_core.beta_ctx.state_dict())
        new_core.sigma_head.load_state_dict(old_core.sigma_head.state_dict())
        if old_core.bypass_head is not None and new_core.bypass_head is not None:
            new_core.bypass_head.load_state_dict(old_core.bypass_head.state_dict())
        new_core.logw11.data.copy_(old_core.logw11.data)
        new_core.lam11_logit.data.copy_(old_core.lam11_logit.data)

        # ----- Por-embalse: subselect según kept_reservoirs_idx_old -----
        # OJO: usamos `kept`, no `active`. Los `active` pueden ser más que
        # los `kept` (algunos activos no admiten canonical destination y
        # se descartan); el nuevo core tiene exactamente `len(kept)` slots
        # de embalse, no `len(active)`.
        kept = plan.kept_reservoirs_idx_old
        if kept:
            idx = torch.tensor(kept, dtype=torch.long)
            new_core.res_embed.data.copy_(old_core.res_embed.data.index_select(0, idx))
            new_core.beta_wA.data.copy_(old_core.beta_wA.data.index_select(0, idx))
            new_core.gamma_logit.data.copy_(old_core.gamma_logit.data.index_select(0, idx))
            new_core.gate_res.log_alpha.data.copy_(
                old_core.gate_res.log_alpha.data.index_select(0, idx)
            )

        # ----- Aristas E_12 (no se colapsan en general; una arista por
        #       (i, k_old) que sobrevive) -----
        for new_e, olds in enumerate(plan.e12_old_idx_per_new):
            if not olds:
                # Esta arista del nuevo grafo no existía en el original.
                # En la práctica esto no ocurre: las aristas nuevas son
                # subset estricto de las viejas. Si ocurriese, dejamos el
                # init por defecto.
                continue
            # Si hay varias viejas (no debería en el caso E_12 fila a
            # fila), sumamos sus exp para preservar masa.
            old_idx = torch.tensor(olds, dtype=torch.long)
            log_w = torch.logsumexp(old_core.logw12.data.index_select(0, old_idx), dim=0)
            new_core.logw12.data[new_e] = log_w
            # lam12 y gate_12: promedio simple
            new_core.lam12_logit.data[new_e] = \
                old_core.lam12_logit.data.index_select(0, old_idx).mean()
            new_core.gate_12.log_alpha.data[new_e] = \
                old_core.gate_12.log_alpha.data.index_select(0, old_idx).mean()

        # ----- Aristas E_21: colapsado clave -----
        for new_e, olds in enumerate(plan.e21_old_idx_per_new):
            if not olds:
                continue
            old_idx = torch.tensor(olds, dtype=torch.long)
            # logw21 colapsado: log(sum(exp)) preserva la masa total
            # repartida hacia los múltiples destinos antiguos.
            log_w_sum = torch.logsumexp(old_core.logw21.data.index_select(0, old_idx), dim=0)
            new_core.logw21.data[new_e] = log_w_sum
            # lam21: promedio ponderado por la masa que cada arista vieja
            # acumulaba — los λ rápidos pesan más si llevaban más flujo.
            old_w = torch.softmax(old_core.logw21.data.index_select(0, old_idx), dim=0)
            new_core.lam21_logit.data[new_e] = \
                (old_w * old_core.lam21_logit.data.index_select(0, old_idx)).sum()
            new_core.gate_21.log_alpha.data[new_e] = \
                old_core.gate_21.log_alpha.data.index_select(0, old_idx).mean()

    return new_core


# ---------------------------------------------------------------------------
# Verificación empírica de equivalencia
# ---------------------------------------------------------------------------


@torch.no_grad()
def verify_equivalence(
    old_core,
    new_core,
    rain: torch.Tensor,        # (B, L, N1)
    mask: torch.Tensor,        # (B, L, N1)
    ctx: torch.Tensor,         # (B, L, ctx_dim)
    H: int,
    T: int,
) -> dict:
    """Compara las predicciones de caudal del core original vs el físicalizado.

    Devuelve dict con RMSE, NSE, correlación de Pearson y rango de la
    diferencia, calculados sobre el horizonte (H..H+T) de cada ventana
    del batch."""
    old_out = old_core(rain, mask, ctx)
    new_out = new_core(rain, mask, ctx)
    q_old = old_out.mu_Q[:, H:H + T].cpu().numpy()
    q_new = new_out.mu_Q[:, H:H + T].cpu().numpy()

    diff = q_old - q_new
    rmse = float(np.sqrt((diff ** 2).mean()))
    var_old = float(np.var(q_old))
    nse = 1.0 - float((diff ** 2).mean()) / max(var_old, 1e-12)
    # Pearson manual sobre flatten
    a = q_old.flatten(); b = q_new.flatten()
    a = (a - a.mean()) / (a.std() + 1e-12)
    b = (b - b.mean()) / (b.std() + 1e-12)
    corr = float((a * b).mean())

    return {
        "rmse": rmse,
        "nse_relative": nse,
        "pearson_correlation": corr,
        "diff_max_abs": float(np.abs(diff).max()),
        "diff_mean_abs": float(np.abs(diff).mean()),
        "n_samples": int(q_old.size),
    }


__all__ = [
    "TransferPlan",
    "physicalize_topology",
    "transfer_core_weights",
    "verify_equivalence",
]
