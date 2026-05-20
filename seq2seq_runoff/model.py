"""Modelo Seq2Seq de escorrentía con dos decodificadores.

Esquema (tres redes que comparten el codificador):

                                ┌─────────────┐
                                │  encoder    │
    histórico (PACUM, EACUM,    │  LSTM(d)    │── estado h, c
    A284) ──────────────────────►             │
                                └─────────────┘
                                                │
                            ┌───────────────────┴───────────────────┐
                            │                                       │
                ┌──────────▼──────────┐               ┌────────────▼─────────┐
   PACUM       │ decoder_embalse     │               │ decoder_caudal       │
   futuro ────►│ LSTM(d) → Dense(1)  │── EACUM       │ LSTM(d) → Dense(1)   │── A284
                └─────────────────────┘   futuro      └──────────────────────┘   futuro
                                                          ▲
                                       PACUM, EACUM ──────┘
                                       futuro

Lo que en producción se hace es:
  1. el modelo del embalse predice EACUM futuro a partir de PACUM (real o cero),
  2. ese EACUM estimado entra como entrada al modelo del caudal,
  3. el modelo del caudal predice A284 futuro.

El "modelo conjunto" es un solo `keras.Model` que une las tres piezas y se
entrena una sola vez con dos pérdidas (una por decoder).

Este módulo también define la clase abstracta `RunoffModel` que sirve como
contrato común para que otros modelos (GNN tipo-1, tipo-1+2, etc.) se
puedan comparar en la misma tubería.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Optional, Sequence, Tuple, TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    # Type-only imports — never executed at module load. Allow the package
    # to be imported when TensorFlow is not installed (e.g. UA-HydroGNN /
    # HydroGNN-only deployments).
    import tensorflow as tf
    from keras import Model

# TensorFlow / Keras are imported lazily inside Seq2SeqRunoffModel methods
# and only at the point a Seq2Seq model is actually instantiated. The
# rest of the package (RunoffModel ABC, Forecast, ForecastScenario) is
# free of TensorFlow dependencies.
try:
    import tensorflow as _tf_probe  # noqa: F401
    _TF_AVAILABLE = True
    _TF_IMPORT_ERROR: Optional[Exception] = None
except Exception as _err:  # pragma: no cover — environment dependent
    _TF_AVAILABLE = False
    _TF_IMPORT_ERROR = _err

from .calibration import MonotonicCalibrator
from .config import Config
from .losses import make_caudal_loss, make_embalse_loss
from .windows import WindowSet, build_windows


# ---------------------------------------------------------------------------
# Tipos de retorno comunes a todos los modelos.
# ---------------------------------------------------------------------------


class ForecastScenario:
    """Cómo rellenar las exógenas futuras (PACUM) que necesita el decodificador.

    En producción no conocemos la lluvia que va a caer, así que evaluamos dos
    escenarios y comparamos:

        OBSERVED  : usamos la pluviosidad real registrada (sólo para validación).
        WORST     : suponemos lluvia cero (peor caso operativo del caudal mínimo).
    """

    OBSERVED = "observed"
    WORST = "worst"


@dataclass
class Forecast:
    """Salida estandarizada de cualquier `RunoffModel`."""

    fechas: pd.DatetimeIndex
    caudal: np.ndarray         # m³/s en escala original
    embalse: np.ndarray        # Hm³ totales en escala original
    caudal_logit: np.ndarray   # confianza bruta en [0, 1] (útil para calibración/UQ)


# ---------------------------------------------------------------------------
# Contrato común para todos los modelos comparados.
# ---------------------------------------------------------------------------


class RunoffModel(abc.ABC):
    """Interfaz común para los modelos del experimento comparativo.

    Cualquier alternativa al baseline (GNN tipo-1, GNN tipo-1+2, etc.) debe
    implementar estos cuatro métodos para enchufarse a la tubería de
    `scripts/run_baseline.py`.
    """

    nombre: str

    @abc.abstractmethod
    def fit(self, df_train: pd.DataFrame, maximos: pd.Series) -> None: ...

    @abc.abstractmethod
    def predict(
        self,
        df: pd.DataFrame,
        hoy: pd.Timestamp,
        maximos: pd.Series,
        escenario: str = ForecastScenario.OBSERVED,
    ) -> Forecast: ...

    @abc.abstractmethod
    def save(self, directorio: str | Path) -> None: ...

    @classmethod
    @abc.abstractmethod
    def load(cls, directorio: str | Path, config: Config) -> "RunoffModel": ...


# ---------------------------------------------------------------------------
# Definición de la red.
# ---------------------------------------------------------------------------


def _require_tensorflow() -> None:
    """Raise a clear ImportError if TF/Keras are needed but unavailable."""
    if not _TF_AVAILABLE:
        raise ImportError(
            "Seq2SeqRunoffModel requires tensorflow + keras to be installed. "
            "Install them with:\n"
            "    pip install 'tensorflow>=2.13' 'keras>=2.13'\n"
            "or skip the Seq2Seq baseline and use the GNN models (HydroGNN, "
            "UA-HydroGNN) which only need PyTorch.\n"
            f"Original import error: {_TF_IMPORT_ERROR}"
        )


def _build_keras_models(config: Config):
    """Construye los tres `keras.Model`: embalse, caudal y conjunto.

    TF/Keras are imported lazily so that this module can be loaded
    without the TF stack; only callers of `Seq2SeqRunoffModel` need
    them installed.
    """
    _require_tensorflow()
    from keras import Model
    from keras.layers import LSTM, Dense, Input, TimeDistributed

    n_enc = config.num_variables_codificador
    n_dec_emb = len(config.variables_decoder_embalse)
    n_dec_cau = len(config.variables_decoder_caudal)
    d = config.latent_dim

    # Codificador compartido.
    encoder_inputs = Input(shape=(config.historia, n_enc), name="enc_in")
    _, h, c = LSTM(d, return_state=True, name="enc_lstm")(encoder_inputs)
    encoder_states = [h, c]

    # Decodificador del embalse.
    dec_emb_in = Input(shape=(config.horizonte, n_dec_emb), name="dec_emb_in")
    dec_emb_seq, _, _ = LSTM(d, return_sequences=True, return_state=True, name="dec_emb_lstm")(
        dec_emb_in, initial_state=encoder_states
    )
    dec_emb_out = TimeDistributed(Dense(1, activation="sigmoid"), name="dec_emb_out")(dec_emb_seq)
    model_embalse = Model([encoder_inputs, dec_emb_in], dec_emb_out, name="embalse")

    # Decodificador del caudal.
    dec_cau_in = Input(shape=(config.horizonte, n_dec_cau), name="dec_cau_in")
    dec_cau_seq, _, _ = LSTM(d, return_sequences=True, return_state=True, name="dec_cau_lstm")(
        dec_cau_in, initial_state=encoder_states
    )
    dec_cau_out = TimeDistributed(Dense(1, activation="sigmoid"), name="dec_cau_out")(dec_cau_seq)
    model_caudal = Model([encoder_inputs, dec_cau_in], dec_cau_out, name="caudal")

    # Modelo conjunto: salidas en el orden [caudal, embalse].
    model_conjunto = Model(
        inputs=[encoder_inputs, dec_cau_in, dec_emb_in],
        outputs=[dec_cau_out, dec_emb_out],
        name="conjunto",
    )

    return model_embalse, model_caudal, model_conjunto


# ---------------------------------------------------------------------------
# Modelo concreto: Seq2Seq baseline.
# ---------------------------------------------------------------------------


class Seq2SeqRunoffModel(RunoffModel):
    """Baseline Seq2Seq con dos decodificadores (sec. 3 de report2.tex)."""

    nombre = "seq2seq"

    def __init__(self, config: Config, calibrador: Optional[MonotonicCalibrator] = None):
        self.config = config
        self.calibrador = calibrador or MonotonicCalibrator.identity()
        self.model_embalse, self.model_caudal, self.model_conjunto = _build_keras_models(config)

    # ---------- entrenamiento ------------------------------------------------

    def fit(self, df_train: pd.DataFrame, maximos: pd.Series):
        _require_tensorflow()
        from keras.callbacks import EarlyStopping
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

        self.model_conjunto.compile(optimizer="adam", loss=[caudal_loss, embalse_loss])
        self.model_embalse.compile(loss=embalse_loss)
        self.model_caudal.compile(loss=caudal_loss)

        early = EarlyStopping(
            monitor="val_loss",
            patience=int(cfg.epochs * cfg.paciencia_factor),
            restore_best_weights=True,
        )

        history = self.model_conjunto.fit(
            x=[ventanas.encoder_inputs, ventanas.decoder_inputs_caudal, ventanas.decoder_inputs_embalse],
            y=[ventanas.target_caudal, ventanas.target_embalse],
            batch_size=cfg.batch_size,
            epochs=cfg.epochs,
            validation_split=cfg.fraccion_validacion,
            verbose=0,
            callbacks=[early],
        )

        # Calibra logit→caudal sobre training, último paso del horizonte.
        logits = self.model_caudal.predict(
            [ventanas.encoder_inputs, ventanas.decoder_inputs_caudal], verbose=0
        )[:, -1, 0]
        observados = ventanas.target_caudal[:, -1, 0]
        self.calibrador.fit(logits, observados)
        return history

    # ---------- predicción ---------------------------------------------------

    def predict(
        self,
        df: pd.DataFrame,
        hoy: pd.Timestamp,
        maximos: pd.Series,
        escenario: str = ForecastScenario.OBSERVED,
    ) -> Forecast:
        cfg = self.config
        manana = hoy + timedelta(days=1)
        fin_horizonte = hoy + timedelta(days=cfg.horizonte)
        inicio_historia = hoy - timedelta(days=cfg.historia - 1)

        # Validaciones explícitas: el error que da numpy al reshape vacío es opaco.
        if df.index.empty:
            raise ValueError("El DataFrame está vacío.")
        if hoy not in df.index or inicio_historia not in df.index:
            raise ValueError(
                f"No hay historia suficiente para hoy={hoy.date()}: necesito "
                f"{inicio_historia.date()} … {hoy.date()} ({cfg.historia} días) "
                f"pero el dataset va de {df.index[0].date()} a {df.index[-1].date()}."
            )
        if escenario == ForecastScenario.OBSERVED and fin_horizonte not in df.index:
            raise ValueError(
                f"Para `escenario=observed` necesito datos futuros hasta "
                f"{fin_horizonte.date()}, pero el dataset acaba en "
                f"{df.index[-1].date()}. Usa `escenario='worst'` para no requerir "
                f"el futuro, o elige una fecha hoy ≤ "
                f"{(df.index[-1] - timedelta(days=cfg.horizonte)).date()}."
            )

        # Codificador: últimos `historia` días observados.
        enc_in = df.loc[inicio_historia:hoy, cfg.variables_codificador]
        enc_in = enc_in.to_numpy(dtype=np.float32).reshape(1, cfg.historia, len(cfg.variables_codificador))

        # Decodificador del embalse: PACUM futuro real o cero según escenario.
        if escenario == ForecastScenario.OBSERVED:
            dec_emb_in = df.loc[manana:fin_horizonte, cfg.variables_decoder_embalse]
            dec_emb_in = dec_emb_in.to_numpy(dtype=np.float32).reshape(1, cfg.horizonte, -1)
        elif escenario == ForecastScenario.WORST:
            dec_emb_in = np.zeros((1, cfg.horizonte, len(cfg.variables_decoder_embalse)), dtype=np.float32)
        else:
            raise ValueError(f"Escenario desconocido: {escenario!r}")

        eacum_pred = self.model_embalse.predict([enc_in, dec_emb_in], verbose=0)  # (1, T, 1)

        # Decodificador del caudal: lluvia agregada (escenario) y embalse estimado.
        rain_col = cfg.basin.rain_aggregate_column
        if escenario == ForecastScenario.OBSERVED:
            pacum_fut = df.loc[manana:fin_horizonte, [rain_col]].to_numpy(dtype=np.float32)
        else:
            pacum_fut = np.zeros((cfg.horizonte, 1), dtype=np.float32)
        dec_cau_in = np.concatenate([pacum_fut, eacum_pred[0]], axis=-1).reshape(1, cfg.horizonte, 2)
        caudal_logit = self.model_caudal.predict([enc_in, dec_cau_in], verbose=0)  # (1, T, 1)

        caudal_norm = self.calibrador(caudal_logit[0, :, 0])
        fechas = pd.date_range(start=manana, end=fin_horizonte)
        return Forecast(
            fechas=fechas,
            caudal=caudal_norm * maximos[cfg.basin.flow_column],
            embalse=eacum_pred[0, :, 0] * maximos[cfg.basin.reservoir_aggregate_column],
            caudal_logit=caudal_logit[0, :, 0],
        )

    # ---------- persistencia -------------------------------------------------

    def save(self, directorio: str | Path) -> None:
        d = Path(directorio)
        d.mkdir(parents=True, exist_ok=True)
        # Formato `.keras` (nativo, único soportado por Keras 3 además de `.h5`).
        self.model_conjunto.save(d / "conjunto.keras")
        self.model_embalse.save(d / "embalse.keras")
        self.model_caudal.save(d / "caudal.keras")
        # El calibrador es ligero; lo persistimos como pickle.
        import pickle
        with open(d / "calibrador.pkl", "wb") as f:
            pickle.dump(self.calibrador, f)

    @classmethod
    def load(cls, directorio: str | Path, config: Config) -> "Seq2SeqRunoffModel":
        from keras.models import load_model
        d = Path(directorio)
        instancia = cls.__new__(cls)
        instancia.config = config
        instancia.model_conjunto = load_model(d / "conjunto.keras", compile=False)
        instancia.model_embalse = load_model(d / "embalse.keras", compile=False)
        instancia.model_caudal = load_model(d / "caudal.keras", compile=False)
        import pickle
        with open(d / "calibrador.pkl", "rb") as f:
            instancia.calibrador = pickle.load(f)
        return instancia
