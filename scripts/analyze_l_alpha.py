"""Análisis L_α: sensibilidad de las decisiones al cuantil α del operador.

El criterio cost-aware del paper §4.5 se reduce a la mediana cuando
α=0.5 (la convención por defecto del baseline). Subir α hace al operador
más conservador respecto a FN (declara alarma con poca evidencia), y
bajarlo lo hace más permisivo. La hipótesis es: barrer α saca a la luz
una "curva de operador" — la frontera de coste vs FN sobre la que el
operador puede situarse según su nivel de aversión al riesgo.

Soporta dos planos para tomar el cuantil:
  * `--quantile-mode predictor` (default): cuantil del caudal μ_Q.
    Es la versión operacional (una decisión por (m, t)).
  * `--quantile-mode cost`: cuantil del coste por sample (cf.
    eq:uagnn_Lalpha del paper). Versión estricta.

Llamadas típicas:

    # Ebro headline (canonical, no remediation)
    python analyze_l_alpha.py \\
        --ckpt outputs/uagnn-ebro-headline/modelo_uagnn \\
        --datos datos-06-07-2023 --firma 580734 \\
        --basin ebro --quantile-mode predictor \\
        --rolling-inicio 2020-01-01 \\
        --output outputs/l_alpha_ebro_predictor.csv

    # Synth N=16 (datos-synth/full)
    python analyze_l_alpha.py \\
        --ckpt outputs/uagnn-synth-full/modelo_uagnn \\
        --datos datos-synth/full --firma SYNTH001 \\
        --basin synth --quantile-mode cost \\
        --rolling-inicio 2022-01-01 --rolling-fin 2024-12-01 \\
        --output outputs/l_alpha_synthN16_cost.csv

    # Synth N=64 (datos-synth-N64/full)
    python analyze_l_alpha.py \\
        --ckpt outputs/uagnn-synth-N64-fix/modelo_uagnn \\
        --datos datos-synth-N64/full --firma SYNTH-N64 \\
        --basin synth --quantile-mode predictor \\
        --rolling-inicio 2022-01-01 --rolling-fin 2024-12-01 \\
        --output outputs/l_alpha_synthN64_predictor.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from seq2seq_runoff import Config, load_basin_dataframe
from seq2seq_runoff.basins import ebro_basin, synth_basin
from seq2seq_runoff.data import scale_to_unit
from seq2seq_runoff.decision import evaluate_all_criteria
from seq2seq_runoff.scenarios import default_library, apply_scenario_to_historical
from seq2seq_runoff.ua_gnn import UAHydroGNNModel


def _parse():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", type=Path, required=True,
                   help="Directorio del checkpoint UA-HydroGNN (con ua_core.pt y ua_meta.pkl).")
    p.add_argument("--datos", type=Path, required=True,
                   help="Directorio de datos del basin (donde están los CSV).")
    p.add_argument("--firma", type=str, required=True,
                   help="Firma de la cuenca (e.g. 580734 para Ebro, SYNTH-N64).")
    p.add_argument("--basin", choices=["ebro", "synth"], required=True)
    p.add_argument("--rolling-inicio", default="2020-01-01")
    p.add_argument("--rolling-fin", default=None)
    p.add_argument("--alphas", type=float, nargs="+",
                   default=[0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])
    p.add_argument("--quantile-mode", choices=["predictor", "cost"],
                   default="predictor")
    p.add_argument("--K-inference", type=int, default=50)
    p.add_argument("--n-rain-samples", type=int, default=20)
    p.add_argument("--coste-fp", type=float, default=1.0)
    p.add_argument("--coste-fn", type=float, default=100.0)
    p.add_argument("--n-delta", type=int, default=161,
                   help="Resolución del grid de δ (default 161).")
    p.add_argument("--delta-range-mult", type=float, nargs=2, default=(-2.0, 3.0),
                   help="Multiplicadores de q_min para los extremos del grid de δ "
                        "(default: -2 a +3 × q_min, evita saturación).")
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def _basin_factory(args):
    if args.basin == "ebro":
        return ebro_basin()
    return synth_basin(args.datos)


def main():
    args = _parse()
    print(f"[l_α] cargando checkpoint {args.ckpt} ...")
    model = UAHydroGNNModel.load(args.ckpt)
    cfg = model.cfg

    basin = _basin_factory(args)
    df = load_basin_dataframe(basin, args.datos, args.firma)
    df_scaled, maximos = scale_to_unit(df)
    flow_col = basin.flow_column
    rain_col = basin.rain_aggregate_column

    scenarios = default_library()
    scenario_names = [s.name for s in scenarios]

    rolling_start = pd.Timestamp(args.rolling_inicio)
    rolling_end = (pd.Timestamp(args.rolling_fin) if args.rolling_fin else
                    df_scaled.index[-cfg.horizonte - 1])
    rolling_days = pd.date_range(rolling_start, rolling_end, freq="D")
    rolling_days = [d for d in rolling_days
                    if (d - pd.Timedelta(days=cfg.historia - 1)) in df_scaled.index
                    and (d + pd.Timedelta(days=cfg.horizonte)) in df_scaled.index]
    print(f"[l_α] {len(rolling_days)} días | basin={args.basin} firma={args.firma}")
    print(f"      quantile_mode={args.quantile_mode}  alphas={args.alphas}")

    q_min = basin.caudal_minimo_m3s
    delta_lo, delta_hi = args.delta_range_mult
    deltas = np.linspace(delta_lo * q_min, delta_hi * q_min, args.n_delta)

    # Pre-cómputo: predicted_distribution por día (M, K, T). Cachear es
    # crítico porque el coste predictivo es ~5s/día con K=50 en Ebro.
    print(f"[l_α] precomputando predictivas para {len(rolling_days)} días...")
    cache = {}
    for k, hoy in enumerate(rolling_days):
        manana = hoy + pd.Timedelta(days=1)
        fin = hoy + pd.Timedelta(days=cfg.horizonte)
        observed_pacum = df.loc[manana:fin, rain_col].to_numpy(dtype=np.float32)
        pacum_per_scenario = []
        for s in scenarios:
            trajs = np.stack([
                apply_scenario_to_historical(
                    observed_pacum, s,
                    rng=np.random.default_rng(int(hoy.toordinal()) + 1000 * m + hash(s.name) % 10000),
                ) for m in range(args.n_rain_samples)
            ], axis=0)
            pacum_per_scenario.append(trajs.mean(axis=0))
        pacum_arr = np.stack(pacum_per_scenario, axis=0)
        dist = model.predict_distribution(df_scaled, hoy, maximos,
                                           pacum_future=pacum_arr.astype(np.float32),
                                           K=args.K_inference)
        obs = df_scaled.loc[manana:fin, flow_col].to_numpy() * float(maximos[flow_col])
        cache[hoy] = (dist, obs)
        if (k + 1) % 100 == 0:
            print(f"   ... {k+1}/{len(rolling_days)}")

    # Evaluación por α.
    rows = []
    for alpha in args.alphas:
        totals = {s: {c: 0.0 for c in ("naive", "maximin", "maximax", "savage")}
                  for s in scenario_names}
        fn_totals = {s: {c: 0.0 for c in ("naive", "maximin", "maximax", "savage")}
                     for s in scenario_names}
        delta_stars = {c: [] for c in ("naive", "maximin", "maximax", "savage")}
        for hoy, (dist, obs) in cache.items():
            res = evaluate_all_criteria(
                predicted_distribution=dist, observed=obs,
                deltas=deltas, q_min=q_min,
                coste_falsa_alarma=args.coste_fp, coste_omision=args.coste_fn,
                scenario_names=scenario_names,
                quantile_alpha=alpha, quantile_mode=args.quantile_mode,
            )
            for cname, r in res.items():
                delta_stars[cname].append(r.delta_star)
                for s in scenario_names:
                    totals[s][cname] += r.cost_per_scenario[s]
                    fn_totals[s][cname] += r.fn_per_scenario[s]

        for cname in ("naive", "maximin", "maximax", "savage"):
            row = {
                "alpha": alpha,
                "criterion": cname,
                "quantile_mode": args.quantile_mode,
                "delta_star_median": float(np.median(delta_stars[cname])),
                "delta_star_p25": float(np.quantile(delta_stars[cname], 0.25)),
                "delta_star_p75": float(np.quantile(delta_stars[cname], 0.75)),
                **{f"cost_{s}": totals[s][cname] for s in scenario_names},
                **{f"fn_{s}":   fn_totals[s][cname] for s in scenario_names},
                "fn_total": sum(fn_totals[s][cname] for s in scenario_names),
                "cost_total": sum(totals[s][cname] for s in scenario_names),
            }
            rows.append(row)

    df_out = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(args.output, index=False)
    print(f"\n[l_α] {len(rows)} filas → {args.output}")

    # Resumen Savage.
    sav = df_out[df_out["criterion"] == "savage"].sort_values("alpha")
    print(f"\n=== Savage L_α sensitivity ({args.quantile_mode} mode) ===")
    print(f"{'α':>6}  {'FN_total':>10}  {'cost_total':>12}  "
          f"{'δ*_med':>8}  {'δ*_p25':>8}  {'δ*_p75':>8}")
    for _, r in sav.iterrows():
        print(f"{r['alpha']:>6.2f}  {r['fn_total']:>10.0f}  {r['cost_total']:>12.0f}  "
              f"{r['delta_star_median']:>+8.2f}  {r['delta_star_p25']:>+8.2f}  "
              f"{r['delta_star_p75']:>+8.2f}")


if __name__ == "__main__":
    main()
