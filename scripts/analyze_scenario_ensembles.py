"""Ensembles por escenario: rompe el empate Maximin–Savage.

En la corrida headline del paper, Maximin y Savage coinciden en
`worst_case_cost` porque ambos eligen el mismo δ* en cada día. La razón
es que sólo había **una** realización promedio por escenario, así que el
peor escenario coincidía con el escenario de máximo regret.

Aquí construimos un ensemble de `M_inner` realizaciones por escenario
(no la media); para cada (s, m_inner) computamos el coste L_{s,m_inner};
y aplicamos los criterios de decisión sobre el **cuantil 0.95 interno** de
cada escenario en lugar de su valor medio. Esto da:

  L^{wc}_s   = q_{0.95}_{m_inner} L_{s, m_inner}    (worst-case interno)

y los criterios Maximin / Savage se aplican sobre L^{wc}. Bajo este
régimen, Maximin y Savage divergen cuando la dispersión interna del peor
escenario es diferente del escenario de máximo regret.

Genera `outputs/ensemble_decomposition.csv` con la comparación.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from seq2seq_runoff import load_basin_dataframe
from seq2seq_runoff.basins import ebro_basin, ebro_graph
from seq2seq_runoff.data import scale_to_unit
from seq2seq_runoff.decision import evaluate_all_criteria
from seq2seq_runoff.scenarios import default_library, apply_scenario_to_historical
from seq2seq_runoff.ua_gnn import UAHydroGNNModel


import os

ROOT = Path(__file__).resolve().parent.parent
# Permite override vía env-var ENS_CKPT. Por defecto usa el checkpoint
# canónico del headline:
#     ENS_CKPT=outputs/uagnn-ebro-remediated/modelo_uagnn python analyze_scenario_ensembles.py
_CKPT_ENV = os.environ.get("ENS_CKPT")
CKPT = Path(_CKPT_ENV) if _CKPT_ENV else (ROOT / "outputs/uagnn-ebro-headline/modelo_uagnn")
DATA_DIR = ROOT / "datos-06-07-2023"
_TAG = CKPT.parent.parent.name if _CKPT_ENV else "headline"
OUT_CSV = ROOT / f"outputs/ensemble_decomposition_{_TAG}.csv"


def main():
    M_INNER = 8       # número de realizaciones internas por escenario
    K_INFER = 50

    print(f"[ens] cargando checkpoint {CKPT} ...")
    model = UAHydroGNNModel.load(CKPT)
    cfg = model.cfg
    scenarios = default_library()
    scenario_names = [s.name for s in scenarios]
    basin = ebro_basin()
    df = load_basin_dataframe(basin, DATA_DIR, "580734")
    df_scaled, maximos = scale_to_unit(df)
    flow_col = basin.flow_column
    rain_col = basin.rain_aggregate_column

    rolling_days = pd.date_range("2020-01-01",
                                  df_scaled.index[-cfg.horizonte - 1], freq="D")
    rolling_days = [d for d in rolling_days
                    if (d - pd.Timedelta(days=cfg.historia - 1)) in df_scaled.index
                    and (d + pd.Timedelta(days=cfg.horizonte)) in df_scaled.index]
    # Submuestreamos para no tardar horas: 300 días aleatorios reproducibles.
    rng = np.random.default_rng(0)
    if len(rolling_days) > 300:
        idx = rng.choice(len(rolling_days), size=300, replace=False)
        rolling_days = sorted([rolling_days[i] for i in idx])
    print(f"[ens] {len(rolling_days)} días")
    q_min = basin.caudal_minimo_m3s
    deltas = np.linspace(-q_min, 3.0 * q_min, 121)

    rows = []
    for k, hoy in enumerate(rolling_days):
        manana = hoy + pd.Timedelta(days=1)
        fin = hoy + pd.Timedelta(days=cfg.horizonte)
        observed_pacum = df.loc[manana:fin, rain_col].to_numpy(dtype=np.float32)

        # Para cada escenario, generamos M_INNER realizaciones DIFERENTES
        # (no la media). Pasamos las 5*M_INNER trayectorias como sub-escenarios.
        sub_pacums = []
        sub_names = []
        for s in scenarios:
            for m in range(M_INNER):
                rng_in = np.random.default_rng(
                    int(hoy.toordinal()) + 10000 * m + hash(s.name) % 100000)
                pert = apply_scenario_to_historical(observed_pacum, s, rng_in)
                sub_pacums.append(pert)
                sub_names.append(f"{s.name}#{m}")
        pacum_arr = np.stack(sub_pacums, axis=0).astype(np.float32)

        # Pasada por el modelo: (M_total, K, T).
        dist = model.predict_distribution(df_scaled, hoy, maximos,
                                           pacum_future=pacum_arr, K=K_INFER)
        obs = df_scaled.loc[manana:fin, flow_col].to_numpy() * float(maximos[flow_col])

        # Construimos L_{D, M_total} con sub-escenarios; luego agregamos a
        # 5 escenarios "robustos" tomando q_{0.95} sobre los M_INNER de cada uno.
        # Para mantenerlo simple, computamos los criterios DOS veces:
        #   (a) sobre los 5*M_INNER sub-escenarios como si fueran independientes
        #       (esto da maximin y savage "exhaustivos" sobre el ensemble pleno).
        #   (b) sobre los 5 escenarios "robustos" con coste = q_{0.95} interno.

        # (a) baseline plano:
        res_a = evaluate_all_criteria(
            predicted_distribution=dist, observed=obs,
            deltas=deltas, q_min=q_min,
            coste_falsa_alarma=1.0, coste_omision=100.0,
            scenario_names=sub_names,
            baseline_scenario="baseline#0",   # primer sub-escenario baseline
        )

        # (b) robusto: reducimos los costes por escenario raíz usando
        # q_{0.95}. Lo hacemos manualmente, ya que `evaluate_all_criteria`
        # no soporta este "rollup" anidado de forma directa.
        from seq2seq_runoff.decision import cost_grid_per_scenario, maximin_delta, savage_delta, naive_delta, maximax_delta
        L_full, FN_full, FP_full = cost_grid_per_scenario(
            dist, obs, deltas, q_min,
            coste_falsa_alarma=1.0, coste_omision=100.0,
            scenario_names=sub_names,
        )  # L_full shape (D, 5*M_INNER)

        # Reshape a (D, 5, M_INNER) y agregamos por q_{0.95} sobre M_INNER.
        D = L_full.shape[0]
        L_robust = np.quantile(L_full.reshape(D, 5, M_INNER), q=0.95, axis=2)  # (D, 5)
        FN_robust = np.quantile(FN_full.reshape(D, 5, M_INNER), q=0.95, axis=2)
        FP_robust = np.quantile(FP_full.reshape(D, 5, M_INNER), q=0.95, axis=2)

        # Criterios sobre L_robust.
        res_b = {
            "naive":   naive_delta(L_robust, FN_robust, FP_robust, deltas, scenario_names),
            "maximin": maximin_delta(L_robust, FN_robust, FP_robust, deltas, scenario_names),
            "maximax": maximax_delta(L_robust, FN_robust, FP_robust, deltas, scenario_names),
            "savage":  savage_delta(L_robust, FN_robust, FP_robust, deltas, scenario_names),
        }

        for cname in ("naive", "maximin", "maximax", "savage"):
            r_a = res_a[cname]
            r_b = res_b[cname]
            rows.append({
                "fecha": hoy.date().isoformat(),
                "criterio": cname,
                "flat_delta_star": r_a.delta_star,
                "flat_worst_case": r_a.worst_case_cost,
                "flat_max_regret": r_a.max_regret,
                "robust_delta_star": r_b.delta_star,
                "robust_worst_case": r_b.worst_case_cost,
                "robust_max_regret": r_b.max_regret,
                "agreement_delta": float(abs(r_a.delta_star - r_b.delta_star) < 1e-6),
            })
        if (k + 1) % 50 == 0:
            print(f"   ... {k+1}/{len(rolling_days)}")

    df_out = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(OUT_CSV, index=False)
    print(f"\n[ens] {len(rows)} filas → {OUT_CSV}")

    # Resumen: ¿Maximin y Savage divergen bajo la versión robusta?
    print("\n=== Agreement entre Maximin y Savage ===")
    for variant in ("flat", "robust"):
        sub_max = df_out[df_out["criterio"] == "maximin"].set_index("fecha")
        sub_sav = df_out[df_out["criterio"] == "savage"].set_index("fecha")
        agree = np.isclose(sub_max[f"{variant}_delta_star"].values,
                           sub_sav[f"{variant}_delta_star"].values,
                           atol=1e-6).mean()
        print(f"  {variant:6s}: Maximin y Savage coinciden en δ* en {100*agree:.1f}% de los días")
        # Diferencias en worst_case y max_regret.
        worst_diff = (sub_max[f"{variant}_worst_case"].values
                       - sub_sav[f"{variant}_worst_case"].values)
        print(f"  {variant:6s}:    Δ(worst_case)  median = {np.median(worst_diff):.1f}   "
              f"max = {np.max(np.abs(worst_diff)):.1f}")


if __name__ == "__main__":
    main()
