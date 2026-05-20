"""Variational Seq2Seq (V-Seq2Seq) — sec. 3 de paper_methods.tex.

Extensión probabilística del baseline `Seq2SeqRunoffModel` en la que el
codificador produce los parámetros de una distribución gaussiana sobre el
código latente `z ∈ R^{d_z}` en lugar de un estado determinista. Permite
muestrear `K` trayectorias predictivas y, junto con una librería de
escenarios de lluvia (`scenarios.py`), evaluar criterios de decisión
robustos (`decision.py`).

Diferencias principales con `Seq2SeqRunoffModel`:
    * El codificador produce `(μ, log σ²)` en lugar de `(h, c)`.
    * `z = μ + σ ⊙ ε` con `ε ~ N(0, I)` (reparameterización).
    * Los estados iniciales de los dos decodificadores se proyectan
      linealmente desde `z`.
    * La pérdida añade un término KL ponderado por `β` (β-VAE).
    * `predict_distribution()` devuelve K muestras predictivas.

La interfaz `RunoffModel` se respeta: `predict()` devuelve la MEDIA de
las K muestras, lo que permite reusar la tubería existente (rolling
evaluation, comparativos, plots).
"""

from __future__ import annotations

import pickle
from datetime import timedelta
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras import Model
from keras import ops as K_ops
from keras.callbacks import EarlyStopping
from keras.layers import LSTM, Dense, Input, Lambda, Layer, TimeDistributed

from .calibration import MonotonicCalibrator
from .config import Config
from .losses import make_caudal_loss, make_embalse_loss
from .model import Forecast, ForecastScenario, RunoffModel
from .windows import build_windows


# ===========================================================================
# Helpers internos
# ===========================================================================


class _SamplingLayer(Layer):
    """Reparametrización z = μ + σ · ε con ε ~ N(0, I).

    En Keras 3 la aleatoriedad dentro de una `Layer` se gestiona con un
    `SeedGenerator` que produce semillas distintas en cada paso. Encapsular
    el muestreo en una `Layer` evita la incompatibilidad de
    `tf.random.normal` con los `KerasTensor` de la API funcional.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.seed_generator = keras.random.SeedGenerator(seed=None)

    def call(self, inputs):
        mu, log_var = inputs
        eps = keras.random.normal(shape=K_ops.shape(mu),
                                  seed=self.seed_generator)
        return mu + K_ops.exp(0.5 * log_var) * eps

    def compute_output_shape(self, input_shape):
        return input_shape[0]


class _KLDivLayer(Layer):
    """Layer-side-effect que añade KL(q(z|x) || N(0,I)) como pérdida del modelo.

    El cálculo se hace dentro de `call()` para que (i) Keras 3 acepte los
    `KerasTensor` simbólicos vía `keras.ops`, y (ii) `add_loss` registre el
    término en `model.losses` y participe del backward.
    """

    def __init__(self, beta: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.beta = float(beta)

    def call(self, inputs):
        mu, log_var = inputs
        kl = -0.5 * K_ops.mean(
            K_ops.sum(
                1.0 + log_var - K_ops.square(mu) - K_ops.exp(log_var),
                axis=-1,
            )
        )
        self.add_loss(self.beta * kl)
        # Pass-through: el grafo sigue con μ y log σ² inalterados.
        return [mu, log_var]

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        cfg = super().get_config()
        cfg["beta"] = self.beta
        return cfg


def _build_encoder(config: Config, latent_dim_z: int) -> Model:
    """Codificador con salida (μ, log σ²) ∈ R^{d_z}."""
    n_enc = config.num_variables_codificador
    d = config.latent_dim
    enc_in = Input(shape=(config.historia, n_enc), name="enc_in")
    _, h, _ = LSTM(d, return_state=True, name="enc_lstm")(enc_in)
    mu = Dense(latent_dim_z, name="z_mu")(h)
    log_var = Dense(latent_dim_z, name="z_log_var")(h)
    return Model(enc_in, [mu, log_var], name="encoder")


def _build_decoders(config: Config, latent_dim_z: int) -> Model:
    """Decodificadores compartidos: (z, dec_cau_in, dec_emb_in) → (cau_out, emb_out)."""
    d = config.latent_dim
    n_dec_emb = len(config.variables_decoder_embalse)
    n_dec_cau = len(config.variables_decoder_caudal)

    z_in = Input(shape=(latent_dim_z,), name="z_in")
    dec_cau_in = Input(shape=(config.horizonte, n_dec_cau), name="dec_cau_in")
    dec_emb_in = Input(shape=(config.horizonte, n_dec_emb), name="dec_emb_in")

    # Proyecciones lineales z → estados iniciales LSTM
    h_V = Dense(d, name="z_to_h_V")(z_in)
    c_V = Dense(d, name="z_to_c_V")(z_in)
    h_Q = Dense(d, name="z_to_h_Q")(z_in)
    c_Q = Dense(d, name="z_to_c_Q")(z_in)

    dec_emb_seq, _, _ = LSTM(d, return_sequences=True, return_state=True,
                              name="dec_emb_lstm")(dec_emb_in, initial_state=[h_V, c_V])
    dec_emb_out = TimeDistributed(Dense(1, activation="sigmoid"),
                                   name="dec_emb_out")(dec_emb_seq)

    dec_cau_seq, _, _ = LSTM(d, return_sequences=True, return_state=True,
                              name="dec_cau_lstm")(dec_cau_in, initial_state=[h_Q, c_Q])
    dec_cau_out = TimeDistributed(Dense(1, activation="sigmoid"),
                                   name="dec_cau_out")(dec_cau_seq)

    return Model([z_in, dec_cau_in, dec_emb_in], [dec_cau_out, dec_emb_out],
                 name="decoders")


def _build_joint_with_kl(encoder: Model, decoders: Model,
                          latent_dim_z: int, config: Config, beta: float) -> Model:
    """Modelo conjunto entrenable. Incluye el término KL añadido vía add_loss."""
    n_enc = config.num_variables_codificador
    n_dec_emb = len(config.variables_decoder_embalse)
    n_dec_cau = len(config.variables_decoder_caudal)

    enc_in = Input(shape=(config.historia, n_enc), name="enc_in")
    dec_cau_in = Input(shape=(config.horizonte, n_dec_cau), name="dec_cau_in")
    dec_emb_in = Input(shape=(config.horizonte, n_dec_emb), name="dec_emb_in")

    mu, log_var = encoder(enc_in)
    # Inyecta la divergencia KL como pérdida side-effect; los tensores siguen
    # propagándose sin cambios para la cabeza de muestreo y los decoders.
    mu_kl, log_var_kl = _KLDivLayer(beta=beta, name="kl_loss")([mu, log_var])
    z = _SamplingLayer(name="z_sample")([mu_kl, log_var_kl])
    cau_out, emb_out = decoders([z, dec_cau_in, dec_emb_in])

    joint = Model([enc_in, dec_cau_in, dec_emb_in], [cau_out, emb_out],
                  name="vae_joint")
    return joint


# ===========================================================================
# Modelo principal
# ===========================================================================


class VAESeq2SeqRunoffModel(RunoffModel):
    """Variational Seq2Seq (sec. 3 del paper).

    Parameters
    ----------
    config
        `Config` del experimento; reutiliza historia, horizonte, dimensiones
        de LSTM, época, batch size, etc.
    latent_dim_z
        Dimensión del código latente `z`. Valores típicos: 4–16.
    beta
        Peso del término KL en la ELBO (β-VAE). Valores típicos:
        10⁻³–10⁰.
    n_latent_samples
        Número de muestras de `z` usadas en inferencia para aproximar la
        distribución predictiva.
    calibrador
        Igual que en `Seq2SeqRunoffModel`.
    """

    nombre = "vae-seq2seq"

    def __init__(
        self,
        config: Config,
        latent_dim_z: int = 16,
        beta: float = 1e-2,
        n_latent_samples: int = 100,
        calibrador: Optional[MonotonicCalibrator] = None,
    ):
        self.config = config
        self.latent_dim_z = int(latent_dim_z)
        self.beta = float(beta)
        self.n_latent_samples = int(n_latent_samples)
        self.calibrador = calibrador or MonotonicCalibrator.identity()
        self.encoder = _build_encoder(config, self.latent_dim_z)
        self.decoders = _build_decoders(config, self.latent_dim_z)
        self.joint = _build_joint_with_kl(
            self.encoder, self.decoders, self.latent_dim_z, config, self.beta,
        )

    # ---------- entrenamiento ------------------------------------------------

    def fit(self, df_train: pd.DataFrame, maximos: pd.Series):
        cfg = self.config
        ventanas = build_windows(
            df_train,
            historia=cfg.historia,
            horizonte=cfg.horizonte,
            variables_codificador=cfg.variables_codificador,
            variables_decoder_embalse=cfg.variables_decoder_embalse,
            variables_decoder_caudal=cfg.variables_decoder_caudal,
            variable_objetivo_caudal=cfg.basin.flow_column,
            variable_objetivo_embalse=cfg.basin.reservoir_aggregate_column,
        )
        caudal_minimo_norm = float(cfg.caudal_minimo_m3s / maximos[cfg.basin.flow_column])
        embalse_loss = make_embalse_loss(cfg.peso_suavidad_embalse)
        caudal_loss = make_caudal_loss(caudal_minimo_norm, desbalance=cfg.desbalance)

        self.joint.compile(optimizer="adam", loss=[caudal_loss, embalse_loss])

        early = EarlyStopping(
            monitor="val_loss",
            patience=int(cfg.epochs * cfg.paciencia_factor),
            restore_best_weights=True,
        )

        history = self.joint.fit(
            x=[ventanas.encoder_inputs, ventanas.decoder_inputs_caudal,
               ventanas.decoder_inputs_embalse],
            y=[ventanas.target_caudal, ventanas.target_embalse],
            batch_size=cfg.batch_size,
            epochs=cfg.epochs,
            validation_split=cfg.fraccion_validacion,
            verbose=0,
            callbacks=[early],
        )

        # Calibrador sobre la media predictiva del último paso (idéntico al baseline).
        logits = self._predict_caudal_mean(
            ventanas.encoder_inputs, ventanas.decoder_inputs_caudal,
            ventanas.decoder_inputs_embalse,
        )[:, -1, 0]
        observados = ventanas.target_caudal[:, -1, 0]
        self.calibrador.fit(logits, observados)
        return history

    # ---------- inferencia ---------------------------------------------------

    def _predict_caudal_mean(
        self, enc_in: np.ndarray, dec_cau_in: np.ndarray, dec_emb_in: np.ndarray,
    ) -> np.ndarray:
        """Media de K muestras predictivas (output del decoder caudal)."""
        K = self.n_latent_samples
        mu, log_var = self.encoder.predict(enc_in, verbose=0)
        sigma = np.exp(0.5 * log_var)
        # Vectorizamos muestreo: (K, B, dz)
        eps = np.random.randn(K, *mu.shape).astype(np.float32)
        zs = mu[None, ...] + sigma[None, ...] * eps  # (K, B, dz)
        # Apilamos para llamar al decoder con batch K·B
        B = enc_in.shape[0]
        z_flat = zs.reshape(K * B, -1)
        dec_cau_rep = np.tile(dec_cau_in, (K, 1, 1))
        dec_emb_rep = np.tile(dec_emb_in, (K, 1, 1))
        cau_out, _ = self.decoders.predict([z_flat, dec_cau_rep, dec_emb_rep], verbose=0)
        cau_out = cau_out.reshape(K, B, *cau_out.shape[1:])
        return cau_out.mean(axis=0)  # (B, T, 1)

    @tf.autograph.experimental.do_not_convert
    def predict(
        self,
        df: pd.DataFrame,
        hoy: pd.Timestamp,
        maximos: pd.Series,
        escenario: str = ForecastScenario.OBSERVED,
    ) -> Forecast:
        """Predicción puntual (media sobre K muestras) — compatible con RunoffModel."""
        cfg = self.config
        manana = hoy + timedelta(days=1)
        fin_horizonte = hoy + timedelta(days=cfg.horizonte)
        inicio_historia = hoy - timedelta(days=cfg.historia - 1)

        if df.index.empty or hoy not in df.index or inicio_historia not in df.index:
            raise ValueError(
                f"No hay historia suficiente para hoy={hoy.date()}: necesito "
                f"{inicio_historia.date()} … {hoy.date()} ({cfg.historia} días)."
            )
        if escenario == ForecastScenario.OBSERVED and fin_horizonte not in df.index:
            raise ValueError(
                f"Escenario `observed` necesita futuro hasta {fin_horizonte.date()}; "
                f"el dataset acaba en {df.index[-1].date()}."
            )

        enc_in = df.loc[inicio_historia:hoy, cfg.variables_codificador]
        enc_in = enc_in.to_numpy(dtype=np.float32).reshape(
            1, cfg.historia, len(cfg.variables_codificador))

        # Sin embalse predicho aquí — pasamos PACUM/EACUM observados o ceros (worst).
        rain_col = cfg.basin.rain_aggregate_column
        eacum_col = cfg.basin.reservoir_aggregate_column
        if escenario == ForecastScenario.OBSERVED:
            pacum_fut = df.loc[manana:fin_horizonte, [rain_col]].to_numpy(dtype=np.float32)
            eacum_fut = df.loc[manana:fin_horizonte, [eacum_col]].to_numpy(dtype=np.float32)
        else:
            pacum_fut = np.zeros((cfg.horizonte, 1), dtype=np.float32)
            # eacum desconocido en peor caso → mantenemos último valor observado
            eacum_last = float(df.loc[hoy, eacum_col])
            eacum_fut = np.full((cfg.horizonte, 1), eacum_last, dtype=np.float32)

        dec_emb_in = pacum_fut.reshape(1, cfg.horizonte, 1)
        dec_cau_in = np.concatenate([pacum_fut, eacum_fut], axis=-1).reshape(
            1, cfg.horizonte, 2)

        caudal_logit = self._predict_caudal_mean(enc_in, dec_cau_in, dec_emb_in)  # (1,T,1)
        caudal_norm = self.calibrador(caudal_logit[0, :, 0])

        fechas = pd.date_range(start=manana, end=fin_horizonte)
        return Forecast(
            fechas=fechas,
            caudal=caudal_norm * maximos[cfg.basin.flow_column],
            embalse=np.zeros(cfg.horizonte),  # no relevante en predict() puntual
            caudal_logit=caudal_logit[0, :, 0],
        )

    # ---------- predicción distribucional (lo que abre la puerta a §3.5) ----

    def predict_distribution(
        self,
        df: pd.DataFrame,
        hoy: pd.Timestamp,
        maximos: pd.Series,
        pacum_future: np.ndarray,                # (T,) o (M, T): M escenarios
        eacum_future: Optional[np.ndarray] = None,  # idem; default = último valor
        n_latent_samples: Optional[int] = None,
    ) -> np.ndarray:
        """Devuelve la distribución predictiva de caudal (m³/s) para uno o
        varios escenarios futuros de pluviosidad.

        Returns
        -------
        np.ndarray shape (M, K, T) con caudal predicho en m³/s, donde:
            M = número de escenarios de lluvia futura (incluye 1 si pacum_future es 1D)
            K = self.n_latent_samples (o el override)
            T = horizonte
        """
        cfg = self.config
        K = n_latent_samples or self.n_latent_samples

        # Aceptamos pacum_future como (T,) o (M, T)
        pacum_arr = np.asarray(pacum_future, dtype=np.float32)
        if pacum_arr.ndim == 1:
            pacum_arr = pacum_arr.reshape(1, -1)
        M, T = pacum_arr.shape
        assert T == cfg.horizonte, f"pacum_future debe tener T={cfg.horizonte} pasos"

        if eacum_future is None:
            last_eacum = float(df.loc[hoy, cfg.basin.reservoir_aggregate_column])
            eacum_arr = np.full((M, T), last_eacum, dtype=np.float32)
        else:
            eacum_arr = np.asarray(eacum_future, dtype=np.float32)
            if eacum_arr.ndim == 1:
                eacum_arr = eacum_arr.reshape(1, -1)
            assert eacum_arr.shape == (M, T)

        # Encoder único (no depende del escenario)
        inicio_historia = hoy - timedelta(days=cfg.historia - 1)
        enc_in = df.loc[inicio_historia:hoy, cfg.variables_codificador]
        enc_in = enc_in.to_numpy(dtype=np.float32).reshape(
            1, cfg.historia, len(cfg.variables_codificador))
        mu, log_var = self.encoder.predict(enc_in, verbose=0)
        sigma = np.exp(0.5 * log_var)
        # Muestras: (K, dz)
        eps = np.random.randn(K, mu.shape[1]).astype(np.float32)
        zs = mu[0] + sigma[0] * eps  # (K, dz)

        # Cross-product: (M·K) ejemplos
        z_flat = np.tile(zs[None, :, :], (M, 1, 1)).reshape(M * K, -1)
        dec_cau = np.stack(
            [np.stack([pacum_arr[m], eacum_arr[m]], axis=-1) for m in range(M)],
            axis=0,
        )  # (M, T, 2)
        dec_cau_flat = np.repeat(dec_cau, K, axis=0)  # (M·K, T, 2)
        dec_emb_flat = np.repeat(pacum_arr.reshape(M, T, 1), K, axis=0)  # (M·K, T, 1)

        cau_out, _ = self.decoders.predict(
            [z_flat, dec_cau_flat, dec_emb_flat], verbose=0
        )
        cau_out = cau_out.reshape(M, K, T)  # (M, K, T) en [0, 1] normalizado
        # Calibra y reescala a m³/s
        caudal_m3s = self.calibrador(cau_out) * maximos[cfg.basin.flow_column]
        return caudal_m3s

    # ---------- persistencia -------------------------------------------------

    def save(self, directorio: str | Path) -> None:
        d = Path(directorio)
        d.mkdir(parents=True, exist_ok=True)
        self.joint.save(d / "joint.keras")
        self.encoder.save(d / "encoder.keras")
        self.decoders.save(d / "decoders.keras")
        with open(d / "vae_meta.pkl", "wb") as f:
            pickle.dump({
                "latent_dim_z": self.latent_dim_z,
                "beta": self.beta,
                "n_latent_samples": self.n_latent_samples,
                "calibrador": self.calibrador,
            }, f)

    @classmethod
    def load(cls, directorio: str | Path, config: Config) -> "VAESeq2SeqRunoffModel":
        from keras.models import load_model
        d = Path(directorio)
        with open(d / "vae_meta.pkl", "rb") as f:
            meta = pickle.load(f)
        instancia = cls.__new__(cls)
        instancia.config = config
        instancia.latent_dim_z = meta["latent_dim_z"]
        instancia.beta = meta["beta"]
        instancia.n_latent_samples = meta["n_latent_samples"]
        instancia.calibrador = meta["calibrador"]
        instancia.encoder = load_model(d / "encoder.keras", compile=False)
        instancia.decoders = load_model(d / "decoders.keras", compile=False)
        instancia.joint = load_model(d / "joint.keras", compile=False)
        return instancia
