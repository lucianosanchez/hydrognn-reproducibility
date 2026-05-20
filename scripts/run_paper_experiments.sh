#!/usr/bin/env bash
# ============================================================================
# Experimentos finales para paper_methods.tex §4 (UA-HydroGNN).
#
# Lanzarlo desde el directorio raíz del paquete:
#     cd seq2seq_runoff
#     bash scripts/run_paper_experiments.sh [phase]
#
# Donde [phase] es uno de:
#   data        — regenera datasets sintéticos (~5 min CPU)
#   train       — entrena UA-HydroGNN en {synth-N16, synth-N64, Ebro}
#                  con la remediación (β-warmup, free-bits, rain bypass)
#                  y semilla canónica 42 (~1.5–2 h total CPU)
#   robustness  — re-entrena Ebro con semillas {0, 7, 123} para tablas
#                  de robustez (~1.5 h total CPU)
#   l_alpha     — análisis L_α sensitivity en {N=16, N=64, Ebro} para
#                  ambos modos quantile {predictor, cost} (~30 min CPU)
#   w4          — recomputa W4 (variance attribution) en el nuevo Ebro
#                  (~5 min CPU)
#   ensembles   — recomputa el audit Maximin-Savage en el nuevo Ebro
#                  (~10 min CPU)
#   phase22     — entrena HydroGNN Phase 2.2 (posiciones aprendidas) en
#                  synth-N16 y synth-N64 + genera las figuras
#                  fig:topology_recovery (~30-40 min CPU)
#   ebro_informed — re-entrena UA-HydroGNN en Ebro con las longitudes
#                  fluviales por arco como prior (river-velocity=50 km/d).
#                  Comparar con outputs/uagnn-ebro-headline (sin longitudes).
#                  ~30 min CPU.
#   physicalize  — toma los checkpoints Phase 2.2 (synth-N16 y synth-N64)
#                  y aplica el post-procesado de physicalize_topology
#                  para producir grafos físicos equivalentes. Genera
#                  figuras 3-panel ground/learned/physicalised. ~2 min CPU.
#                  Requiere que phase22 haya terminado antes.
#   phase22_acyclic — grid de Phase 2.2 cruzando {M_latent=3,6} × {dense,
#                  acyclic} en synth-N16 y synth-N64 (8 corridas).
#                  Investiga si restringir el grafo a topologías acíclicas
#                  mejora la identificabilidad estructural sin degradar
#                  la utilidad operacional. ~50 min CPU total.
#                  Aplica physicalización post-hoc a cada checkpoint
#                  para tabla comparativa.
#   all         — todo lo anterior, en orden (~4-5 h total CPU)
#
# Hipótesis sobre el entorno:
#   * Python 3.11 con torch≥2.1, numpy, pandas, scipy, yaml.
#     Si tu intérprete está en otra ruta, edita PYTHON abajo.
#   * Las dependencias del proyecto deben estar accesibles desde
#     `PYTHONPATH=.` (lanza el script desde el directorio del paquete).
#
# Salidas: los CSVs se escriben en `outputs/<sub-dir>/` y los logs en
# `outputs/_logs/`. Cada corrida es idempotente: si re-lanzas con los
# mismos parámetros, se sobreescriben los CSVs.
# ============================================================================

set -euo pipefail

# Intérprete Python: usa `python` del PATH activo. Si tienes un venv
# activado (`source venv/bin/activate` o `pyenv shell ...`), se usará
# ése. Sobreescribible con env-var:
#     PYTHON=/ruta/a/python bash scripts/run_paper_experiments.sh ...
PYTHON="${PYTHON:-python}"
export PYTHONUNBUFFERED=1

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

mkdir -p outputs/_logs

phase="${1:-all}"

echo "=========================================================================="
echo "[paper_experiments] phase = $phase"
echo "[paper_experiments] PYTHON = $PYTHON"
echo "[paper_experiments] ROOT   = $ROOT"
echo "=========================================================================="

# --------------------------------------------------------------------------
# phase: ebro_informed — re-entrena UA-HydroGNN en Ebro con prior de
# longitudes fluviales (inicialización informada de λ vía
# --river-velocity). Comparar contra outputs/uagnn-ebro-headline.
# --------------------------------------------------------------------------
if [[ "$phase" == "ebro_informed" || "$phase" == "all" ]]; then
  echo ""
  echo "[ebro_informed] UA-HydroGNN Ebro con prior de longitudes (v=50 km/d)"
  $PYTHON scripts/run_ua_gnn_experiment.py \
    --directorio-datos datos-06-07-2023 --firma 580734 \
    --dia-prediccion 2023-06-01 \
    --rolling-inicio 2020-01-01 \
    --epochs 200 --K-train 10 --K-inference 50 \
    --beta-ua 1e-3 --kappa 30 \
    --river-velocity 50.0 \
    --n-rain-samples 20 \
    --output outputs/uagnn-ebro-informed \
    > outputs/_logs/uagnn-ebro-informed.log 2>&1 \
    || echo "[ebro_informed] FAILED — revisa outputs/_logs/uagnn-ebro-informed.log"

  echo "[ebro_informed] OK — compara con outputs/uagnn-ebro-headline:"
  echo "  diff <(grep ^savage outputs/uagnn-ebro-headline/headline_metrics.csv) \\"
  echo "       <(grep ^savage outputs/uagnn-ebro-informed/headline_metrics.csv)"
fi


# --------------------------------------------------------------------------
# phase: phase22 — entrena HydroGNN Phase 2.2 (posiciones aprendidas) en
# los dos basins sintéticos para que la figura `fig:topology_recovery`
# tenga contenido informativo. Las posiciones reales se ocultan al modelo;
# se le da un grafo de candidatos densos con `M_latent` embalses formales,
# y aprende dónde colocarlos. Resultado: checkpoints en
# outputs/hydrognn-phase22-{synth-N16,synth-N64}.
# --------------------------------------------------------------------------
if [[ "$phase" == "phase22" || "$phase" == "all" ]]; then
  echo ""
  echo "[phase22] HydroGNN Phase 2.2 — synth-N16 (M_latent=3, 2 of 4 stations visible)"
  $PYTHON scripts/run_gnn.py --fase 2.2 \
    --directorio-datos datos-synth/full --firma SYNTH001 \
    --dia-prediccion 2024-12-15 \
    --rolling-inicio 2022-01-01 --rolling-fin 2024-12-01 \
    --epochs 60 --m-latent 3 \
    --observed-stations SM-PACUM ST1-PACUM \
    --directorio-modelo outputs/hydrognn-phase22-synth-N16 \
    --kappa-low-flow 30 \
    > outputs/_logs/hydrognn-phase22-synth-N16.log 2>&1 \
    || echo "[phase22]   synth-N16 FAILED — revisa outputs/_logs/hydrognn-phase22-synth-N16.log"

  echo "[phase22] HydroGNN Phase 2.2 — synth-N64 (M_latent=4, ~half stations visible)"
  # Lista observada: las primeras estaciones disponibles del manifest. Si
  # tu N=64 tiene otras siglas, ajusta esta lista. Si la dejas vacía cae
  # al default M_latent=4 con N1=64 → 60×4=240 arcos E_12 que el modelo
  # debe esparcir.
  OBS_64="$(ls datos-synth-N64/full/DatosHistoricos_SYNTH-N64_*PACUM.csv 2>/dev/null \
            | sed -E 's|.*DatosHistoricos_SYNTH-N64_([A-Z0-9-]+).csv|\1|' \
            | head -15 | tr '\n' ' ')"
  if [[ -z "$OBS_64" ]]; then
    echo "[phase22]   AVISO: no he encontrado estaciones en datos-synth-N64. Phase 2.2 N=64 saltado."
  else
    echo "[phase22]   observed_stations (N=64) = $OBS_64"
    $PYTHON scripts/run_gnn.py --fase 2.2 \
      --directorio-datos datos-synth-N64/full --firma SYNTH-N64 \
      --dia-prediccion 2024-12-15 \
      --rolling-inicio 2022-01-01 --rolling-fin 2024-12-01 \
      --epochs 60 --m-latent 4 \
      --observed-stations $OBS_64 \
      --directorio-modelo outputs/hydrognn-phase22-synth-N64 \
      --kappa-low-flow 30 \
      > outputs/_logs/hydrognn-phase22-synth-N64.log 2>&1 \
      || echo "[phase22]   synth-N64 FAILED — revisa outputs/_logs/hydrognn-phase22-synth-N64.log"
  fi

  echo "[phase22] Generando figuras 2-panel ground-truth vs learned"
  mkdir -p figs
  for ds in synth-N16 synth-N64; do
    CKPT="outputs/hydrognn-phase22-${ds}"
    case "$ds" in
      synth-N16) DATA_DIR="datos-synth/full"; FIRMA="SYNTH001" ;;
      synth-N64) DATA_DIR="datos-synth-N64/full"; FIRMA="SYNTH-N64" ;;
    esac
    if [[ ! -f "$CKPT/core.pt" ]]; then
      echo "[phase22]   skip $ds (sin checkpoint $CKPT/core.pt)"
      continue
    fi
    $PYTHON scripts/plot_learned_vs_truth.py \
      --ckpt "$CKPT" \
      --basin-dir "$DATA_DIR" --firma "$FIRMA" \
      --output "figs/topology_${ds}.pdf" \
      --title-suffix "Phase 2.2 — M_latent=$([ "$ds" = "synth-N16" ] && echo 3 || echo 4)" \
      > "outputs/_logs/viz-${ds}.log" 2>&1 \
      || echo "[phase22]   plot_learned_vs_truth para $ds FAILED — revisa outputs/_logs/viz-${ds}.log"
    echo "[phase22]   figs/topology_${ds}.pdf"
  done
fi


# --------------------------------------------------------------------------
# phase: data — regenera datasets sintéticos
# --------------------------------------------------------------------------
if [[ "$phase" == "data" || "$phase" == "all" ]]; then
  echo ""
  echo "[data] regenerando datasets sintéticos ..."
  if [[ ! -f datos-synth-N64/full/manifest.yaml ]]; then
    $PYTHON scripts/make_synth_basin.py \
      --n-type1 64 --branching 1.5 --seed 0 \
      --output datos-synth-N64 \
      2>&1 | tee outputs/_logs/data-synth-N64.log
  else
    echo "[data] datos-synth-N64/full/manifest.yaml ya existe — saltando."
  fi
  echo "[data] dataset N=16 esperado en datos-synth/full (ya viene en el repo)"
fi

# --------------------------------------------------------------------------
# phase: train — entrena con remediación canónica (semilla 42)
# --------------------------------------------------------------------------
if [[ "$phase" == "train" || "$phase" == "all" ]]; then
  echo ""
  echo "[train] UA-HydroGNN — synth-N16 (defaults, sin remediación)"
  $PYTHON scripts/run_ua_gnn_experiment.py \
    --directorio-datos datos-synth/full --firma SYNTH001 \
    --dia-prediccion 2024-12-15 \
    --rolling-inicio 2022-01-01 --rolling-fin 2024-12-01 \
    --epochs 200 --K-train 10 --K-inference 50 \
    --beta-ua 1e-3 --kappa 30 \
    --n-rain-samples 20 \
    --output outputs/uagnn-synth-N16 \
    2>&1 | tee outputs/_logs/uagnn-synth-N16.log

  echo ""
  echo "[train] UA-HydroGNN — synth-N64 (CON remediación opt-in)"
  $PYTHON scripts/run_ua_gnn_experiment.py \
    --directorio-datos datos-synth-N64/full --firma SYNTH-N64 \
    --dia-prediccion 2024-12-15 \
    --rolling-inicio 2022-01-01 --rolling-fin 2024-12-01 \
    --epochs 200 --K-train 15 --K-inference 50 \
    --beta-ua 1e-3 --kappa 30 \
    --warmup-epochs 80 --ramp-epochs 40 --free-bits 0.02 \
    --rain-bypass --lam11-init 2.0 \
    --max-windows 3000 --batch-size 64 \
    --n-rain-samples 20 \
    --output outputs/uagnn-synth-N64 \
    2>&1 | tee outputs/_logs/uagnn-synth-N64.log

  echo ""
  echo "[train] UA-HydroGNN — Ebro headline (sin flags de remediación)"
  $PYTHON scripts/run_ua_gnn_experiment.py \
    --directorio-datos datos-06-07-2023 --firma 580734 \
    --dia-prediccion 2023-06-01 \
    --rolling-inicio 2020-01-01 \
    --epochs 200 --K-train 10 --K-inference 50 \
    --beta-ua 1e-3 --kappa 30 \
    --n-rain-samples 20 \
    --output outputs/uagnn-ebro-headline \
    2>&1 | tee outputs/_logs/uagnn-ebro-headline.log

  # Versión con remediación (rain_bypass + free_bits + warmup) — para la
  # tabla §5.6 tab:uagnn_remediation_ablation_results, NO para el headline.
  echo ""
  echo "[train] UA-HydroGNN — Ebro con perfil de remediación (regresivo)"
  $PYTHON scripts/run_ua_gnn_experiment.py \
    --directorio-datos datos-06-07-2023 --firma 580734 \
    --dia-prediccion 2023-06-01 \
    --rolling-inicio 2020-01-01 \
    --epochs 200 --K-train 10 --K-inference 50 \
    --beta-ua 1e-3 --kappa 30 \
    --warmup-epochs 60 --ramp-epochs 30 --free-bits 0.02 \
    --rain-bypass --lam11-init 2.0 \
    --n-rain-samples 20 \
    --output outputs/uagnn-ebro-remediated \
    2>&1 | tee outputs/_logs/uagnn-ebro-remediated.log
fi

# --------------------------------------------------------------------------
# phase: robustness — multi-seed Ebro CON remediación (para tab §5.6).
# Para el multi-seed audit del HEADLINE (sin remediación) usar
#   bash scripts/run_headline_seed_robustness.sh
# --------------------------------------------------------------------------
if [[ "$phase" == "robustness" || "$phase" == "all" ]]; then
  echo ""
  echo "[robustness] Ebro multi-seed {0, 7, 123} CON remediación (perfil regresivo)"
  for seed in 0 7 123; do
    echo ""
    echo "[robustness] Ebro remediated semilla=$seed"
    $PYTHON scripts/run_ua_gnn_experiment.py \
      --directorio-datos datos-06-07-2023 --firma 580734 \
      --dia-prediccion 2023-06-01 \
      --rolling-inicio 2020-01-01 \
      --epochs 200 --K-train 10 --K-inference 50 \
      --beta-ua 1e-3 --kappa 30 \
      --warmup-epochs 60 --ramp-epochs 30 --free-bits 0.02 \
      --n-rain-samples 20 \
      --seed $seed \
      --output "outputs/uagnn-ebro-remediated-seed${seed}" \
      2>&1 | tee "outputs/_logs/uagnn-ebro-remediated-seed${seed}.log"
  done

  # Resumen multi-seed remediated
  $PYTHON scripts/summarize_seed_robustness.py \
    --inputs outputs/uagnn-ebro-remediated-seed0 \
             outputs/uagnn-ebro-remediated-seed7 \
             outputs/uagnn-ebro-remediated-seed123 \
    --output outputs/ebro_remediated_seed_robustness.csv \
    2>&1 | tee outputs/_logs/ebro_remediated_seed_robustness.log
fi

# --------------------------------------------------------------------------
# phase: l_alpha — sensitivity en los 3 datasets, ambos modos
# --------------------------------------------------------------------------
if [[ "$phase" == "l_alpha" || "$phase" == "all" ]]; then
  echo ""
  echo "[l_alpha] sensitivity sweep — 3 datasets × 2 quantile modes"
  for mode in predictor cost; do
    for cfg in "synth-N16:datos-synth/full:SYNTH001:synth:uagnn-synth-N16" \
               "synth-N64:datos-synth-N64/full:SYNTH-N64:synth:uagnn-synth-N64" \
               "ebro:datos-06-07-2023:580734:ebro:uagnn-ebro-headline"; do
      IFS=":" read -r tag datos firma basin ckpt_dir <<< "$cfg"
      ckpt="outputs/${ckpt_dir}/modelo_uagnn"
      if [[ ! -d "$ckpt" ]]; then
        echo "[l_alpha] sin checkpoint $ckpt — saltando $tag-$mode"
        continue
      fi
      echo ""
      echo "[l_alpha] $tag / quantile_mode=$mode"
      $PYTHON scripts/analyze_l_alpha.py \
        --ckpt "$ckpt" --datos "$datos" --firma "$firma" --basin "$basin" \
        --rolling-inicio 2022-01-01 --rolling-fin 2024-12-01 \
        --quantile-mode "$mode" \
        --alphas 0.05 0.10 0.25 0.50 0.75 0.90 0.95 \
        --output "outputs/l_alpha_${tag}_${mode}.csv" \
        2>&1 | tee "outputs/_logs/l_alpha_${tag}_${mode}.log"
    done
  done
fi

# --------------------------------------------------------------------------
# phase: w4 — variance attribution sobre el Ebro recién entrenado
# --------------------------------------------------------------------------
if [[ "$phase" == "w4" || "$phase" == "all" ]]; then
  echo ""
  echo "[w4] descomposición de varianza en Ebro (sobre el headline)"
  CKPT="outputs/uagnn-ebro-headline/modelo_uagnn"
  if [[ ! -d "$CKPT" ]]; then
    echo "[w4] checkpoint $CKPT no existe; lanzar 'train' antes."
    exit 1
  fi
  W4_CKPT="$CKPT" $PYTHON scripts/analyze_w4_variance.py \
    2>&1 | tee outputs/_logs/w4.log
fi

# --------------------------------------------------------------------------
# phase: ensembles — Maximin/Savage agreement
# --------------------------------------------------------------------------
if [[ "$phase" == "ensembles" || "$phase" == "all" ]]; then
  echo ""
  echo "[ensembles] audit Maximin–Savage en Ebro (sobre el headline)"
  CKPT="outputs/uagnn-ebro-headline/modelo_uagnn"
  if [[ ! -d "$CKPT" ]]; then
    echo "[ensembles] checkpoint $CKPT no existe; lanzar 'train' antes."
    exit 1
  fi
  ENS_CKPT="$CKPT" $PYTHON scripts/analyze_scenario_ensembles.py \
    2>&1 | tee outputs/_logs/ensembles.log
fi

# --------------------------------------------------------------------------
# phase: physicalize — post-procesa los checkpoints Phase 2.2 generando
# grafos físicos equivalentes (sin bucles ni vertidos aguas arriba) y
# figuras 3-panel ground/learned/physicalised.
# --------------------------------------------------------------------------
if [[ "$phase" == "physicalize" || "$phase" == "all" ]]; then
  echo ""
  echo "[physicalize] post-procesando checkpoints Phase 2.2 (modos strict y soft)"
  mkdir -p figs outputs/physicalized
  for ds in synth-N16 synth-N64; do
    CKPT="outputs/hydrognn-phase22-${ds}"
    case "$ds" in
      synth-N16) DATA_DIR="datos-synth/full"; FIRMA="SYNTH001" ;;
      synth-N64) DATA_DIR="datos-synth-N64/full"; FIRMA="SYNTH-N64" ;;
    esac
    if [[ ! -f "$CKPT/core.pt" ]]; then
      echo "[physicalize]   skip $ds (sin checkpoint $CKPT/core.pt)"
      continue
    fi
    for mode in strict soft; do
      $PYTHON scripts/physicalize_topology.py \
        --ckpt "$CKPT" \
        --basin-dir "$DATA_DIR" --firma "$FIRMA" --basin-type synth \
        --mode $mode \
        --output-dir "outputs/physicalized/${ds}-${mode}" \
        --fig "figs/topology_${ds}_physicalized_${mode}.pdf" \
        > "outputs/_logs/physicalize-${ds}-${mode}.log" 2>&1 \
        || echo "[physicalize]   $ds/$mode FAILED — revisa outputs/_logs/physicalize-${ds}-${mode}.log"
      echo "[physicalize]   figs/topology_${ds}_physicalized_${mode}.pdf"
    done
  done
  # El paper LaTeX vive en ../Escorrentía/, así que copiamos las figuras
  # a ../figs/ para que \includegraphics las encuentre desde ahí.
  if [[ -d "../figs" ]]; then
    cp figs/topology_*.pdf ../figs/ 2>/dev/null && \
      echo "[physicalize] figuras topology_*.pdf → ../figs/ (sincronizadas con el paper)"
  fi
fi


# --------------------------------------------------------------------------
# phase: phase22_acyclic — grid {M=3,6} × {dense, acyclic} en synth-N16/64
# --------------------------------------------------------------------------
if [[ "$phase" == "phase22_acyclic" || "$phase" == "all" ]]; then
  echo ""
  echo "[phase22_acyclic] grid {M=3,6} × {dense, acyclic} × {N16, N64}"
  mkdir -p figs outputs/_logs outputs/phase22_grid
  for ds in synth-N16 synth-N64; do
    case "$ds" in
      synth-N16) DATA_DIR="datos-synth/full"; FIRMA="SYNTH001";
                 OBS="SM-PACUM ST1-PACUM" ;;
      synth-N64) DATA_DIR="datos-synth-N64/full"; FIRMA="SYNTH-N64";
                 OBS="$(ls datos-synth-N64/full/DatosHistoricos_SYNTH-N64_*PACUM.csv 2>/dev/null \
                       | sed -E 's|.*DatosHistoricos_SYNTH-N64_([A-Z0-9-]+).csv|\1|' \
                       | head -15 | tr '\n' ' ')" ;;
    esac
    for M in 3 6; do
      for mode in dense acyclic; do
        TAG="${ds}-M${M}-${mode}"
        OUT="outputs/phase22_grid/${TAG}"
        if [[ -f "$OUT/core.pt" ]]; then
          echo "[phase22_acyclic]   $TAG ya existe — skip"
          continue
        fi
        ACYCLIC_FLAG=""
        [[ "$mode" == "acyclic" ]] && ACYCLIC_FLAG="--acyclic-candidates"
        echo "[phase22_acyclic]   $TAG ($(date '+%H:%M:%S'))"
        $PYTHON scripts/run_gnn.py --fase 2.2 \
          --directorio-datos "$DATA_DIR" --firma "$FIRMA" \
          --dia-prediccion 2024-12-15 \
          --rolling-inicio 2022-01-01 --rolling-fin 2024-12-01 \
          --epochs 60 --m-latent $M \
          --observed-stations $OBS \
          $ACYCLIC_FLAG \
          --directorio-modelo "$OUT" \
          --kappa-low-flow 30 \
          > "outputs/_logs/phase22_grid-${TAG}.log" 2>&1 \
          || echo "[phase22_acyclic]   $TAG FAILED — outputs/_logs/phase22_grid-${TAG}.log"
      done
    done
  done

  echo ""
  echo "[phase22_acyclic] physicalización post-hoc para los 8 checkpoints"
  for ds in synth-N16 synth-N64; do
    case "$ds" in
      synth-N16) DATA_DIR="datos-synth/full"; FIRMA="SYNTH001" ;;
      synth-N64) DATA_DIR="datos-synth-N64/full"; FIRMA="SYNTH-N64" ;;
    esac
    for M in 3 6; do
      for mode in dense acyclic; do
        TAG="${ds}-M${M}-${mode}"
        CKPT="outputs/phase22_grid/${TAG}"
        if [[ ! -f "$CKPT/core.pt" ]]; then continue; fi
        $PYTHON scripts/physicalize_topology.py \
          --ckpt "$CKPT" \
          --basin-dir "$DATA_DIR" --firma "$FIRMA" --basin-type synth \
          --mode strict \
          --output-dir "outputs/phase22_grid/${TAG}-phys" \
          --fig "figs/phase22_grid_${TAG}.pdf" \
          > "outputs/_logs/phase22_grid-${TAG}-phys.log" 2>&1 \
          || echo "[phase22_acyclic]   physicalize $TAG FAILED"
      done
    done
  done

  echo ""
  echo "[phase22_acyclic] generando tabla resumen"
  $PYTHON scripts/summarize_phase22_grid.py \
    --grid-root outputs/phase22_grid \
    --output outputs/phase22_grid/grid_summary.csv \
    2>&1 | tee outputs/_logs/phase22_grid_summary.log

  # Auto-sincronización con paper
  if [[ -d "../figs" ]]; then
    cp figs/phase22_grid_*.pdf ../figs/ 2>/dev/null && \
      echo "[phase22_acyclic] figuras phase22_grid_*.pdf → ../figs/"
  fi
fi


echo ""
echo "=========================================================================="
echo "[paper_experiments] phase '$phase' COMPLETED"
echo "=========================================================================="
echo "Outputs:    outputs/"
echo "Logs:       outputs/_logs/"
echo ""
echo "The CSVs under outputs/ are the numerical sources of the manuscript"
echo "tables. The full mapping (which CSV feeds which table of §5) is in"
echo "docs/EXPERIMENT_MAP.md."
