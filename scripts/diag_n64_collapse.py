"""Diagnóstico del colapso UA-HydroGNN en synth-N64.

Carga el checkpoint guardado en `outputs/_archive3/uagnn-synth-N64` y
contesta tres preguntas:

  Q1. ¿Cuánto vale el posterior `(mu, log_var)` aprendido?
      Si `|mu|` ≈ 0 y `log_var` ≈ 0, el posterior es esencialmente el
      prior N(0, I) — colapso puro.
      Si `log_var` ≈ -10 (clamp), el posterior es delta de Dirac en `mu`
      — colapso "rígido" (la NLL ignora la KL).

  Q2. ¿La predicción `mu_Q` es invariante al input de lluvia futura?
      Tomamos una ventana real, sustituimos la lluvia futura por (a) la
      observada, (b) 0, (c) 5× observada, y medimos cuánto cambia el
      output. Si el rango es < 1% → el modelo ignora la entrada.

  Q3. ¿Las K muestras del posterior producen `mu_Q` diversa?
      Calculamos `std(mu_Q, axis=K) / mean(mu_Q, axis=K)` para 50
      muestras MC en una ventana. Si CV ≈ 0 → la posterior es trivial.

El script imprime un informe estructurado y guarda
`outputs/_diag_n64.txt`.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch

from seq2seq_runoff.basins import synth_basin, synth_graph_full
from seq2seq_runoff.data import load_basin_dataframe, scale_to_unit, split_train_test
from seq2seq_runoff.gnn.dataset import build_window
from seq2seq_runoff.ua_gnn import UAHydroGNNModel


ARCHIVE = Path(__file__).resolve().parent.parent / "outputs/_archive3/uagnn-synth-N64/modelo_uagnn"


def main():
    # Regeneramos primero el dataset N=64 sintético (mismo seed que paper).
    out_dir = Path(__file__).resolve().parent.parent / "datos-synth-N64"
    if not (out_dir / "full" / "manifest.yaml").exists():
        print("[diag] regenerando dataset N=64 ...")
        import subprocess
        subprocess.run([
            sys.executable, str(Path(__file__).parent / "make_synth_basin.py"),
            "--n-type1", "64", "--branching", "1.5", "--seed", "0",
            "--output", str(out_dir),
        ], check=True)

    print(f"[diag] cargando checkpoint {ARCHIVE} ...")
    model = UAHydroGNNModel.load(ARCHIVE)
    cfg = model.cfg
    graph = model.graph
    print(f"       graph N1={graph.N1} M={graph.M} E11={graph.E11} "
          f"E12={graph.E12} E21={graph.E21}")
    d_state = graph.M + graph.E11 + graph.E12 + graph.E21
    print(f"       d_state = {d_state}")

    # Carga datos.
    basin = synth_basin(out_dir / "full")
    df = load_basin_dataframe(basin, out_dir / "full", "SYNTH-N64")
    df_scaled, maximos = scale_to_unit(df)
    train, test = split_train_test(df_scaled, fraccion_test=0.2)

    # Toma 100 ventanas de validación para muestrear posterior.
    H, T = cfg.historia, cfg.horizonte
    rolling_days = test.index[H:H + 100]
    print(f"\n[diag] {len(rolling_days)} ventanas test desde {rolling_days[0].date()}")

    model.core.eval()
    device = torch.device("cpu")

    # --- Q1: distribución de posterior params -----------------------------
    mus, log_vars = [], []
    with torch.no_grad():
        for hoy in rolling_days:
            try:
                w = build_window(df_scaled, graph, hoy, H, T,
                                 flow_column=cfg.basin.flow_column,
                                 observed_stations=cfg.observed_stations)
            except Exception:
                continue
            rain = w.rain.unsqueeze(0)
            mu, log_var = model.core.posterior_params(rain, H)
            mus.append(mu.cpu().numpy().flatten())
            log_vars.append(log_var.cpu().numpy().flatten())

    mus = np.stack(mus)              # (N, d_state)
    log_vars = np.stack(log_vars)
    sigmas = np.exp(0.5 * log_vars)
    kl_per_dim = -0.5 * (1.0 + log_vars - mus**2 - np.exp(log_vars))  # (N, d_state)

    print("\n=== Q1: Posterior parameters ===")
    print(f"  |mu|        : mean = {np.abs(mus).mean():.4f}   "
          f"max = {np.abs(mus).max():.4f}   "
          f"std-across-windows = {mus.std(axis=0).mean():.4f}")
    print(f"  log_var     : mean = {log_vars.mean():.3f}   "
          f"min = {log_vars.min():.3f}   max = {log_vars.max():.3f}")
    print(f"  sigma       : mean = {sigmas.mean():.4f}   "
          f"min = {sigmas.min():.4f}   max = {sigmas.max():.4f}")
    print(f"  KL/dim      : mean = {kl_per_dim.mean():.4f}   "
          f"<0.01: {(kl_per_dim.mean(axis=0) < 0.01).sum()}/{d_state} dims colapsadas")
    print(f"  KL total    : mean over windows = {kl_per_dim.sum(axis=1).mean():.3f}")

    # --- Q2: sensibilidad a la lluvia futura ------------------------------
    hoy = rolling_days[len(rolling_days) // 2]
    w = build_window(df_scaled, graph, hoy, H, T,
                     flow_column=cfg.basin.flow_column,
                     observed_stations=cfg.observed_stations)
    rain_base = w.rain.unsqueeze(0)   # (1, L, N1)
    mask_base = w.mask.unsqueeze(0)
    ctx_base = w.ctx.unsqueeze(0)

    # Tres escenarios: observado, 0, 5× observado.
    rain_zero = rain_base.clone()
    rain_zero[:, H:H + T, :] = 0.0
    rain_5x = rain_base.clone()
    rain_5x[:, H:H + T, :] = rain_base[:, H:H + T, :] * 5.0

    print("\n=== Q2: Sensibilidad a la lluvia futura ===")
    with torch.no_grad():
        out_obs = model.core.forward_mc(rain_base, mask_base, ctx_base, H=H, K=20)
        out_zero = model.core.forward_mc(rain_zero, mask_base, ctx_base, H=H, K=20)
        out_5x = model.core.forward_mc(rain_5x, mask_base, ctx_base, H=H, K=20)
    mu_obs = out_obs["mu_Q"][:, 0, H:H + T].mean(dim=0).cpu().numpy()
    mu_zero = out_zero["mu_Q"][:, 0, H:H + T].mean(dim=0).cpu().numpy()
    mu_5x = out_5x["mu_Q"][:, 0, H:H + T].mean(dim=0).cpu().numpy()
    print(f"  mu_Q obs      : {mu_obs[:6].round(4)}")
    print(f"  mu_Q zero     : {mu_zero[:6].round(4)}")
    print(f"  mu_Q 5x       : {mu_5x[:6].round(4)}")
    print(f"  Δ(0 vs obs)   : abs mean = {np.abs(mu_zero - mu_obs).mean():.6f}   "
          f"max = {np.abs(mu_zero - mu_obs).max():.6f}")
    print(f"  Δ(5x vs obs)  : abs mean = {np.abs(mu_5x - mu_obs).mean():.6f}   "
          f"max = {np.abs(mu_5x - mu_obs).max():.6f}")

    # --- Q3: dispersión MC entre muestras ---------------------------------
    print("\n=== Q3: Dispersión de muestras MC ===")
    with torch.no_grad():
        out50 = model.core.forward_mc(rain_base, mask_base, ctx_base, H=H, K=50)
    mu_K = out50["mu_Q"][:, 0, H:H + T].cpu().numpy()   # (K=50, T)
    mu_K_mean = mu_K.mean(axis=0)
    mu_K_std = mu_K.std(axis=0)
    cv = mu_K_std / (np.abs(mu_K_mean) + 1e-8)
    print(f"  mu_Q por sample (T=14): mean = {mu_K_mean.mean():.4f}   "
          f"std-across-K = {mu_K_std.mean():.6f}   "
          f"CV = {cv.mean():.6f}")
    print(f"  rango: min sample = {mu_K.min():.4f}   max sample = {mu_K.max():.4f}")

    # --- Diagnóstico ------------------------------------------------------
    print("\n=== DIAGNÓSTICO ===")
    posterior_collapsed = np.abs(mus).mean() < 0.05 and np.abs(log_vars.mean()) < 0.5
    input_invariant = np.abs(mu_zero - mu_obs).mean() < 1e-3
    mc_collapsed = mu_K_std.mean() < 1e-3
    print(f"  posterior ≈ prior     : {posterior_collapsed}  "
          f"(|mu|={np.abs(mus).mean():.4f}, log_var={log_vars.mean():.3f})")
    print(f"  output invariante     : {input_invariant}  "
          f"(Δ(0→obs)={np.abs(mu_zero - mu_obs).mean():.6f})")
    print(f"  muestras MC colapsan  : {mc_collapsed}  "
          f"(std-K={mu_K_std.mean():.6f})")


if __name__ == "__main__":
    main()
