"""Análisis posicional de los embalses formales aprendidos por Phase 2.2 (H4).

Para cada experimento (config sintética) que entrenó Phase 2.2:

1. Carga el modelo guardado (`gnn_k*/modelo/`).
2. Llama a `analyze_positions()` para obtener `inflow_share[k, i]` —
   fracción del flujo del nodo Tipo-1 i que entra al embalse formal k.
3. Marginaliza para obtener un vector `inflow_attention[i]`
   = Σ_k inflow_share[k, i] (cuán "atractivo" es cada nodo Tipo-1 para
   los embalses formales).
4. Construye, a partir del manifest topológico, el vector de
   "capacidad efectiva aguas arriba" `effective_storage_upstream[i]`
   = suma de capacidades de los embalses reales aguas arriba de i.
5. Compara las dos distribuciones via:
     * Wasserstein-1 (signed: dirección física)
     * Spearman ρ (correlación de orden)
     * top-k overlap: cuántos de los top-k posiciones formales coinciden
       con las top-k reales.

H4 predice: la atención de los embalses formales se concentra en los
nodos cuya capacidad efectiva aguas arriba es alta — es decir, **donde
hay un embalse real aguas arriba**, aunque el modelo no haya recibido
esa información durante el entrenamiento.

Output: una fila por config en `<output>/h4_positions.csv` con todas las
métricas + un plot por config opcional.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import yaml


# ---------------------------------------------------------------------------
# Carga del modelo Phase 2.2 entrenado y extracción de posiciones.
# ---------------------------------------------------------------------------


def _load_phase22_model(model_dir: Path):
    """Carga el HydroGNNPhase2_2 desde `model_dir/modelo/` y devuelve la
    instancia. Falla con error claro si los pesos no están."""
    import torch  # local import: torch no está en el path estándar
    from seq2seq_runoff.gnn import HydroGNNPhase2_2, GNNConfig
    from seq2seq_runoff.basins.synth import (
        synth_basin, synth_graph_simplified,
    )

    pkl = model_dir / "meta.pkl"
    if not pkl.exists():
        raise FileNotFoundError(f"Falta {pkl}.")
    return HydroGNNPhase2_2.load(model_dir, config=None)


def _attention_vector(modelo) -> np.ndarray:
    """`inflow_attention[i] = Σ_k inflow_share[k, i]` (vector len N1)."""
    info = modelo.analyze_positions()
    share = np.asarray(info["inflow_share"])  # (M, N1)
    return share.sum(axis=0), info  # (N1,)


def _topology_from_manifest(manifest_path: Path):
    """Devuelve (id_to_idx, parent, capacity_real_upstream) leídos del manifest."""
    m = yaml.safe_load(manifest_path.read_text())
    nodes = m["topology"]["nodes"]
    edges = m["topology"]["edges_11"]
    reservoirs = m["topology"]["reservoirs"]

    name_to_idx = {n["id"]: i for i, n in enumerate(nodes)}
    n_nodes = len(nodes)

    # parent[i] = nodo aguas abajo de i (a través de E_11), o None si no.
    parent = [None] * n_nodes
    for e in edges:
        parent[name_to_idx[e["src"]]] = name_to_idx[e["dst"]]
    # Si un nodo es fuente de un embalse, su "padre" pasa por el embalse al
    # release_to del embalse.
    for r in reservoirs:
        src_idx = name_to_idx[r["inflow_from"]]
        parent[src_idx] = name_to_idx[r["release_to"]]

    # Capacidad real "aguas arriba o en" cada nodo: suma de capacities de
    # embalses cuyo `inflow_from` está en el subárbol que termina en i.
    capacity_at_node = [0.0] * n_nodes
    for r in reservoirs:
        capacity_at_node[name_to_idx[r["inflow_from"]]] += float(r["capacity_hm3"])

    # Propagamos cumulativamente aguas abajo: for each node i, en post-order,
    # capacity_upstream[i] = capacity_at_node[i] + sum(capacity_upstream[c] for c in children[i])
    children = [[] for _ in range(n_nodes)]
    for i, p in enumerate(parent):
        if p is not None:
            children[p].append(i)

    capacity_upstream = [0.0] * n_nodes

    def dfs(i: int) -> float:
        s = capacity_at_node[i]
        for c in children[i]:
            s += dfs(c)
        capacity_upstream[i] = s
        return s

    # Encontrar la raíz: nodo sin padre.
    roots = [i for i in range(n_nodes) if parent[i] is None]
    for root in roots:
        dfs(root)

    return name_to_idx, capacity_upstream


# ---------------------------------------------------------------------------
# Métricas de comparación distribución formal vs real.
# ---------------------------------------------------------------------------


def _normalize(v: np.ndarray) -> np.ndarray:
    s = v.sum()
    return v / s if s > 0 else v


def _wasserstein_1d(p: np.ndarray, q: np.ndarray) -> float:
    """W₁ entre dos distribuciones discretas sobre el mismo soporte
    (asume el orden del array como eje 1-D)."""
    p, q = _normalize(p), _normalize(q)
    cdf_p = np.cumsum(p)
    cdf_q = np.cumsum(q)
    return float(np.sum(np.abs(cdf_p - cdf_q)))


def _spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    """Correlación de Spearman."""
    n = len(x)
    if n < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    return float(np.sum(rx * ry) / np.sqrt(np.sum(rx * rx) * np.sum(ry * ry)))


def _topk_overlap(x: np.ndarray, y: np.ndarray, k: int) -> float:
    """Fracción de los top-k de x que también están en los top-k de y."""
    k = min(k, len(x))
    if k <= 0:
        return float("nan")
    top_x = set(np.argsort(-x)[:k])
    top_y = set(np.argsort(-y)[:k])
    return len(top_x & top_y) / k


# ---------------------------------------------------------------------------
# Pipeline por config.
# ---------------------------------------------------------------------------


def analyze_one(config_dir: Path, kappas: List[float]) -> List[Dict]:
    """Analiza todos los gnn_k* de Phase 2.2 dentro de `config_dir`."""
    rows = []
    manifest = config_dir / "full" / "manifest.yaml"
    if not manifest.exists():
        return rows
    name_to_idx, capacity_upstream = _topology_from_manifest(manifest)
    cap_vec = np.asarray(capacity_upstream, dtype=float)

    for k in kappas:
        tag = f"k{k:g}".replace(".", "_")
        model_dir = config_dir / "tune-fase2.2" / f"gnn_{tag}" / "modelo"
        if not model_dir.exists():
            continue
        try:
            modelo = _load_phase22_model(model_dir)
            attention, info = _attention_vector(modelo)
        except Exception as e:
            print(f"  [!!] no se pudo cargar {model_dir}: {e}")
            continue

        # Mapear el vector de atención al espacio de nodos del manifest.
        # `info["type1_names"]` es la lista de nombres del grafo del modelo;
        # asumimos que coinciden con name_to_idx.
        type1_names = info["type1_names"]
        att = np.zeros_like(cap_vec)
        for j, nname in enumerate(type1_names):
            if nname in name_to_idx:
                att[name_to_idx[nname]] = float(attention[j])

        n1 = len(cap_vec)
        n_real_res = int((cap_vec > 0).sum())
        rows.append({
            "config_dir": str(config_dir),
            "kappa": k,
            "n_type1": n1,
            "n_real_reservoirs": n_real_res,
            "wasserstein": _wasserstein_1d(att, cap_vec),
            "spearman_rho": _spearman_rho(att, cap_vec),
            "topk_overlap@1": _topk_overlap(att, cap_vec, 1),
            "topk_overlap@3": _topk_overlap(att, cap_vec, 3),
            "topk_overlap@K": _topk_overlap(att, cap_vec, n_real_res),
        })
    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sweep-root", required=True, type=Path,
                   help="Raíz del sweep (e.g. ../sweep-results), buscaremos "
                        "subdirs <bloque>/<config_id>/.")
    p.add_argument("--kappa", nargs="+", type=float, default=[5, 30, 100])
    p.add_argument("--output", type=Path, default=Path("../sweep-analysis"))
    return p.parse_args()


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for block_dir in sorted(args.sweep_root.iterdir()):
        if not block_dir.is_dir():
            continue
        for config_dir in sorted(block_dir.iterdir()):
            if not config_dir.is_dir():
                continue
            print(f"[h4] {config_dir.name}")
            rows.extend(analyze_one(config_dir, args.kappa))

    if not rows:
        print("[h4] no hay modelos Phase 2.2 cargables.")
        return
    out_csv = args.output / "h4_positions.csv"
    with out_csv.open("w") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n[h4] {len(rows)} filas en {out_csv}")
    # Resumen agregado: media de spearman_rho, etc.
    import statistics as st
    rho_vals = [r["spearman_rho"] for r in rows
                if not (isinstance(r["spearman_rho"], float) and np.isnan(r["spearman_rho"]))]
    if rho_vals:
        print(f"[h4] Spearman rho mediana = {st.median(rho_vals):+.3f}, "
              f"IQR = [{np.quantile(rho_vals, 0.25):+.3f}, "
              f"{np.quantile(rho_vals, 0.75):+.3f}]")
    overlap1 = [r["topk_overlap@1"] for r in rows
                if not (isinstance(r["topk_overlap@1"], float) and np.isnan(r["topk_overlap@1"]))]
    if overlap1:
        print(f"[h4] top-1 overlap medio = {sum(overlap1)/len(overlap1):.3f}")


if __name__ == "__main__":
    main()
