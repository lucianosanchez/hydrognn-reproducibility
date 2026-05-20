"""Experimento comparativo sobre la cuenca sintética.

Para cada combinación (4 modelos × 2 configuraciones de visibilidad)
entrena y evalúa, y produce una tabla con las métricas en peor caso.

Modelos:
    1. Seq2Seq                    — sin información geográfica.
    2. GNN grafo completo         — toda la topología y los 3 embalses.
    3. GNN grafo simplificado     — sólo cauce principal + embalse mayor (R0).
    4. GNN posición latente       — cauce simplificado + M_latent embalses libres.

Configuraciones de datos:
    full     — todos los embalses observados.
    partial  — sólo el embalse mayor observado.

Para los modelos 3 y 4, las dos configuraciones producen el mismo
entrenamiento (sólo se observa R0 en cualquier caso); los corremos las
dos veces para reportar la misma métrica y poder leer la tabla en bloque.

Antes de ejecutar este script, genera los datos:

    python -m synth_simulator synth_simulator/example_basin.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Permite ejecutar el script directamente sin instalar el paquete.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from seq2seq_runoff import (
    Config,
    ForecastScenario,
    Seq2SeqRunoffModel,
    load_basin_dataframe,
)
from seq2seq_runoff.basins import synth_basin, synth_graph_full, synth_graph_simplified
from seq2seq_runoff.data import scale_to_unit, split_train_test
from seq2seq_runoff.evaluation import rolling_evaluation, summary
from seq2seq_runoff.gnn import (
    GNNConfig,
    HydroGNNPhase1,
    HydroGNNPhase2_1,
    HydroGNNPhase2_2,
)


# --------------------------------------------------------------------- run


def _train_and_eval(modelo, basin, df_escalado, maximos, base_cfg, label: str) -> dict:
    """Entrena y devuelve las métricas peor-caso del modelo."""
    train, _ = split_train_test(df_escalado, fraccion_test=base_cfg.fraccion_test)
    print(f"\n[{label}] entrenando ({type(modelo).__name__})…")
    modelo.fit(train, maximos)
    fin = df_escalado.index[-base_cfg.horizonte - 1]
    res = rolling_evaluation(
        modelo, basin, df_escalado, maximos,
        fecha_inicio=df_escalado.index[base_cfg.historia],
        fecha_fin=fin,
        horizonte=base_cfg.horizonte,
        caudal_minimo_m3s=base_cfg.caudal_minimo_m3s,
        escenario=ForecastScenario.WORST,
    )
    metricas = summary(res["caudal_obs"], res["caudal_pred"], base_cfg.caudal_minimo_m3s)
    print(f"[{label}] métricas: {metricas}")
    return metricas


def _make_seq2seq(cfg: Config) -> Seq2SeqRunoffModel:
    return Seq2SeqRunoffModel(cfg)


def _make_gnn_full(cfg: Config, manifest_dir: Path, gnn_cfg: GNNConfig):
    """Modelo 2 — Fase 1 si el embalse está observado, Fase 2.1 si no.

    Detectamos automáticamente: si en `manifest` hay 3 embalses visibles
    usamos `HydroGNNPhase1` (supervisión total); si hay 1, usamos
    `HydroGNNPhase2_1` (sólo supervisamos el observado).
    """
    graph = synth_graph_full(manifest_dir)
    if len(graph.res_to_observed) == graph.M:
        return HydroGNNPhase1(gnn_cfg, graph)
    # Phase2_1 espera observed_stations; en el experimento sintético
    # mantenemos todas las estaciones (el escenario B sólo oculta embalses).
    obs = list(cfg.basin.rain_columns)
    gnn_cfg2 = GNNConfig(**{**gnn_cfg.__dict__, "observed_stations": obs})
    return HydroGNNPhase2_1(gnn_cfg2, graph)


def _make_gnn_simplified(cfg: Config, manifest_dir: Path, gnn_cfg: GNNConfig):
    """Modelo 3 — Fase 1 sobre el grafo simplificado (R0 supervisado)."""
    graph = synth_graph_simplified(manifest_dir)
    return HydroGNNPhase1(gnn_cfg, graph)


def _make_gnn_latent(cfg: Config, manifest_dir: Path, gnn_cfg: GNNConfig):
    """Modelo 4 — Fase 2.2 sobre el grafo simplificado, embalses latentes."""
    graph = synth_graph_simplified(manifest_dir)
    obs = list(cfg.basin.rain_columns)
    gnn_cfg2 = GNNConfig(**{**gnn_cfg.__dict__, "observed_stations": obs})
    return HydroGNNPhase2_2(gnn_cfg2, graph)


# --------------------------------------------------------------------- main


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="4 modelos × 2 visibilidades sobre la cuenca sintética.")
    p.add_argument("--datos", required=True, type=Path,
                   help="Directorio raíz que contiene full/ y partial/ con manifest.yaml en cada uno.")
    p.add_argument("--epochs-seq2seq", default=200, type=int)
    p.add_argument("--epochs-gnn", default=30, type=int)
    p.add_argument("--m-latent", default=4, type=int)
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    base = args.datos
    visibilidades = ["full", "partial"]

    resultados: dict[str, dict[str, dict]] = {}

    for vis in visibilidades:
        print(f"\n=== Configuración de datos: {vis.upper()} ===")
        manifest_dir = base / vis
        basin = synth_basin(manifest_dir)
        firma = pd.io.json.dumps if False else None  # noqa
        # Encontramos firma del manifest
        import yaml
        m = yaml.safe_load((manifest_dir / "manifest.yaml").read_text())
        firma = m["basin"]["firma"]

        df = load_basin_dataframe(basin, manifest_dir, firma)
        df_escalado, maximos = scale_to_unit(df)
        print(f"[datos {vis}] {len(df_escalado)} pasos, "
              f"estaciones={len(basin.rain_columns)}, embalses={len(basin.reservoir_columns)}")

        base_cfg = Config(basin=basin, epochs=args.epochs_seq2seq)
        gnn_cfg = GNNConfig(
            basin=basin,
            historia=base_cfg.historia, horizonte=base_cfg.horizonte,
            epochs=args.epochs_gnn, M_latent=args.m_latent, device=args.device,
        )

        modelos = {
            "1-seq2seq":           _make_seq2seq(base_cfg),
            "2-gnn-full":          _make_gnn_full(base_cfg, manifest_dir, gnn_cfg),
            "3-gnn-simplified":    _make_gnn_simplified(base_cfg, manifest_dir, gnn_cfg),
            "4-gnn-latent":        _make_gnn_latent(base_cfg, manifest_dir, gnn_cfg),
        }
        resultados[vis] = {
            label: _train_and_eval(modelo, basin, df_escalado, maximos, base_cfg, f"{vis}/{label}")
            for label, modelo in modelos.items()
        }

    # Tabla resumen
    print("\n=== RESUMEN ===")
    metric_keys = ["f1_alarma", "precision_alarma", "recall_alarma", "nse", "kge"]
    print(f"{'modelo':22s}  {'config':8s}  " + "  ".join(f"{k:>16s}" for k in metric_keys))
    print("-" * (22 + 2 + 8 + 2 + 18 * len(metric_keys)))
    for vis in visibilidades:
        for label, metricas in resultados[vis].items():
            valores = "  ".join(f"{metricas[k]:>16.3f}" for k in metric_keys)
            print(f"{label:22s}  {vis:8s}  {valores}")


if __name__ == "__main__":
    main()
