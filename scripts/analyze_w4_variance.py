"""Análisis W4: descomposición de varianza epistémica/aleatórica en Ebro.

Hipótesis W4 (paper §4.5): en días "F5" (donde la decisión Savage difiere
materialmente de Maximin), la varianza epistémica (estado inicial × escenario)
es comparable o mayor a la varianza aleatórica residual — i.e., la
incertidumbre dominante es estructural y no ruido. Si esto se cumple, el
gradiente cost-aware sobre los criterios Savage/Maximin tiene sentido
(las decisiones se basan en algo más que en ruido); si no se cumple,
los criterios son una respuesta a ruido y Maximin/Savage debería degradarse
a naive.

Descomposición clásica (Eq. uagnn_variance del paper):

    Var[Q | t] = E_s[E_k[σ²_{k,s,t}]]        (aleatórica)
               + Var_{k,s}[μ_{k,s,t}]         (epistémica total)

la epistémica se separa en:

    Var_{escenario}      = Var_s[ E_k[μ_{k,s,t}] ]
    Var_{estado_inicial} = E_s[ Var_k[μ_{k,s,t}] ]

(la descomposición no es exacta cuando K, M son finitos, pero es la
estimación insesgada estándar).

El script carga el checkpoint Ebro guardado, lee `headline_per_day.csv`
para identificar los días F5, y para cada uno emite las tres varianzas
agregadas sobre el horizonte T=14. Resultado: CSV `w4_decomp.csv` y un
resumen en stdout.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch

from seq2seq_runoff import load_basin_dataframe
from seq2seq_runoff.basins import ebro_basin, ebro_graph
from seq2seq_runoff.data import scale_to_unit, split_train_test
from seq2seq_runoff.scenarios import default_library, apply_scenario_to_historical
from seq2seq_runoff.ua_gnn import UAHydroGNNModel


import os

ROOT = Path(__file__).resolve().parent.parent
# Permite override vía env-var W4_CKPT. Por defecto usa el checkpoint
# canónico del headline. Para usar otro:
#     W4_CKPT=outputs/uagnn-ebro-remediated/modelo_uagnn python analyze_w4_variance.py
_CKPT_ENV = os.environ.get("W4_CKPT")
CKPT = Path(_CKPT_ENV) if _CKPT_ENV else (ROOT / "outputs/uagnn-ebro-headline/modelo_uagnn")
# El headline CSV usado para identificar F5 days; debe corresponder al
# mismo checkpoint. Si W4_CKPT está set, buscamos al lado del checkpoint.
if _CKPT_ENV:
    HEADLINE = CKPT.parent / "headline_per_day.csv"
else:
    HEADLINE = ROOT / "outputs/uagnn-ebro-headline/headline_per_day.csv"
DATA_DIR = ROOT / "datos-06-07-2023"     # firma 580734 = Ebro real
# Output CSV tagged por el nombre del checkpoint para no sobreescribir.
_TAG = CKPT.parent.parent.name if _CKPT_ENV else "archive3"
OUT_CSV = ROOT / f"outputs/w4_decomp_{_TAG}.csv"
OUT_SUMMARY = ROOT / f"outputs/w4_summary_{_TAG}.txt"


def _f5_days(headline: pd.DataFrame, top_k: int = 50) -> list[pd.Timestamp]:
    """Días F5: los `top_k` con mayor coste flashy bajo el criterio Savage."""
    sub = headline[headline["criterio"] == "savage"].copy()
    sub["fecha"] = pd.to_datetime(sub["fecha"])
    sub = sub.sort_values("cost_flashy", ascending=False).head(top_k)
    return [pd.Timestamp(d) for d in sub["fecha"].tolist()]


def _scenario_pacums(observed: np.ndarray, scenarios, hoy: pd.Timestamp,
                     n_samples: int) -> np.ndarray:
    """`(M, T)` lluvia agregada media por escenario sobre `n_samples`
    realizaciones, igual que en `run_ua_gnn_experiment.py`."""
    out = []
    for s in scenarios:
        trajs = np.stack([
            apply_scenario_to_historical(
                observed, s,
                rng=np.random.default_rng(
                    int(hoy.toordinal()) + 1000 * m + hash(s.name) % 10000),
            ) for m in range(n_samples)
        ], axis=0)
        out.append(trajs.mean(axis=0))
    return np.stack(out, axis=0)


def main():
    print(f"[w4] cargando checkpoint {CKPT} ...")
    model = UAHydroGNNModel.load(CKPT)
    cfg = model.cfg
    K = 60   # subir un poco para mejor estimación de varianza epistémica
    scenarios = default_library()
    scenario_names = [s.name for s in scenarios]

    basin = ebro_basin()
    df = load_basin_dataframe(basin, DATA_DIR, "580734")
    df_scaled, maximos = scale_to_unit(df)
    flow_col = basin.flow_column
    rain_col = basin.rain_aggregate_column

    headline = pd.read_csv(HEADLINE)
    f5 = _f5_days(headline, top_k=60)
    f5 = [d for d in f5
          if (d - pd.Timedelta(days=cfg.historia - 1)) in df_scaled.index
          and (d + pd.Timedelta(days=cfg.horizonte)) in df_scaled.index]
    print(f"[w4] {len(f5)} días F5 (flashy más caro bajo Savage)")

    model.core.eval()
    flow_max = float(maximos[flow_col])
    q_min = basin.caudal_minimo_m3s

    rows = []
    for hoy in f5:
        manana = hoy + pd.Timedelta(days=1)
        fin = hoy + pd.Timedelta(days=cfg.horizonte)
        observed = df.loc[manana:fin, rain_col].to_numpy(dtype=np.float32)
        pacums = _scenario_pacums(observed, scenarios, hoy, n_samples=20)  # (M, T)

        # mu_Q (K, M, T) y log_sigma (K, M, T) directos del core, sin
        # promediar — necesitamos la varianza entre K muestras.
        from seq2seq_runoff.gnn.dataset import build_window
        df_local = df_scaled.copy()
        if fin not in df_local.index:
            ultimo = df_local.index[-1]
            if fin > ultimo:
                n_extra = (fin - ultimo).days
                idx = pd.date_range(ultimo + pd.Timedelta(days=1), fin, freq="D")
                pad = pd.DataFrame(np.repeat(df_local.iloc[[-1]].to_numpy(),
                                              n_extra, axis=0),
                                    index=idx, columns=df_local.columns)
                df_local = pd.concat([df_local, pad])
        H, T = cfg.historia, cfg.horizonte
        ventana = build_window(df_local, model.graph, hoy, H, T,
                                flow_column=flow_col,
                                observed_stations=cfg.observed_stations)
        rain = ventana.rain.unsqueeze(0)
        mask = ventana.mask.unsqueeze(0)
        ctx = ventana.ctx.unsqueeze(0)

        # Inyecta cada escenario en la lluvia futura.
        nodos_obs = sorted(set(model.graph.rain_to_type1.values()))
        pacum_norm = pacums / float(maximos[rain_col])
        per_station = pacum_norm[:, :, None] / max(len(nodos_obs), 1)
        M_sc = pacums.shape[0]
        rain_sc = rain.repeat(M_sc, 1, 1)
        for n_idx in nodos_obs:
            rain_sc[:, H:H + T, n_idx] = torch.tensor(per_station[:, :, 0],
                                                       dtype=rain_sc.dtype)
        mask_sc = mask.repeat(M_sc, 1, 1)
        ctx_sc = ctx.repeat(M_sc, 1, 1)

        with torch.no_grad():
            out = model.core.forward_mc(rain_sc, mask_sc, ctx_sc, H=H, K=K)
        mu = out["mu_Q"][:, :, H:H + T].numpy() * flow_max     # (K, M, T)
        sigma = torch.nn.functional.softplus(
            out["log_sigma"][:, :, H:H + T]).numpy() * flow_max  # (K, M, T)

        # Descomposición sobre el horizonte (mediana en T y máximo en T).
        E_k_mu = mu.mean(axis=0)             # (M, T) — media sobre K para cada (s, t)
        var_k_mu = mu.var(axis=0)            # (M, T) — varianza-K para cada (s, t)
        E_s_E_k_sigma2 = (sigma**2).mean(axis=(0, 1))   # (T,)

        var_scenario = E_k_mu.var(axis=0)    # (T,)
        var_state = var_k_mu.mean(axis=0)    # (T,)
        var_epist = var_scenario + var_state # (T,)
        var_aleat = E_s_E_k_sigma2           # (T,)

        # Agregamos en el horizonte: la "peor" descomposición (máx sobre t)
        # y la mediana sobre t.
        for stat_name, fn in (("max_T", np.max), ("median_T", np.median)):
            rows.append({
                "fecha": hoy.date().isoformat(),
                "stat_T": stat_name,
                "var_scenario": float(fn(var_scenario)),
                "var_state": float(fn(var_state)),
                "var_epist": float(fn(var_epist)),
                "var_aleat": float(fn(var_aleat)),
                "ratio_epist_aleat": float(fn(var_epist) / max(float(fn(var_aleat)), 1e-9)),
                "mean_mu":   float(E_k_mu.mean()),
                "mean_sigma": float(sigma.mean()),
            })

    df_out = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(OUT_CSV, index=False)
    print(f"[w4] {len(rows)} filas → {OUT_CSV}")

    # Resumen agregado: H1 del paper.
    print("\n=== W4: ¿es la varianza epistémica comparable o mayor a la aleatórica? ===")
    for stat in ("max_T", "median_T"):
        sub = df_out[df_out["stat_T"] == stat]
        med_ratio = sub["ratio_epist_aleat"].median()
        p25 = sub["ratio_epist_aleat"].quantile(0.25)
        p75 = sub["ratio_epist_aleat"].quantile(0.75)
        share_above_1 = float((sub["ratio_epist_aleat"] >= 1.0).mean())
        print(f"  [{stat}]   ratio epist/aleat:  "
              f"P25={p25:.3f}  median={med_ratio:.3f}  P75={p75:.3f}   "
              f"% días con epist >= aleat: {100*share_above_1:.1f}%")
        var_sc_share = float((sub["var_scenario"] / (sub["var_epist"] + 1e-12)).median())
        var_st_share = 1.0 - var_sc_share
        print(f"  [{stat}]   composición epist:   "
              f"escenario {100*var_sc_share:.1f}%  /  estado_inicial {100*var_st_share:.1f}%")

    with open(OUT_SUMMARY, "w") as f:
        for stat in ("max_T", "median_T"):
            sub = df_out[df_out["stat_T"] == stat]
            f.write(f"[{stat}]\n")
            f.write(f"  ratio epist/aleat: P25={sub['ratio_epist_aleat'].quantile(0.25):.3f}  "
                    f"median={sub['ratio_epist_aleat'].median():.3f}  "
                    f"P75={sub['ratio_epist_aleat'].quantile(0.75):.3f}\n")
            f.write(f"  share epist >= aleat: {100*(sub['ratio_epist_aleat']>=1).mean():.1f}%\n\n")
    print(f"[w4] resumen → {OUT_SUMMARY}")


if __name__ == "__main__":
    main()
