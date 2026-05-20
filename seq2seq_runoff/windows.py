"""Construcción de ventanas deslizantes para el entrenamiento.

El notebook tenía tres funciones casi idénticas (`create_dataset_caudal`,
`create_dataset_embalse`, `create_dataset_conjunto`). Aquí hay una sola:
`build_windows` produce todas las matrices que necesita el modelo conjunto, y
los modelos individuales se obtienen seleccionando un subconjunto.

Cada muestra ocupa una posición `i` en la serie y produce:

    encoder_input[i]    : pasos i .. i+H-1   sobre las variables del codificador.
    decoder_input[i]    : pasos i+H .. i+H+T-1 sobre las exógenas del decoder.
    decoder_target[i]   : el mismo intervalo, pero la variable objetivo.
    fecha[i]            : la última fecha del histórico (i+H-1).

Donde H = `historia` y T = `horizonte`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
import pandas as pd


@dataclass
class WindowSet:
    """Conjunto de ventanas listas para alimentar a un modelo Seq2Seq.

    Las dimensiones siguen la convención de Keras: (batch, tiempo, variables).
    """

    encoder_inputs: np.ndarray            # (N, H, V_enc)
    decoder_inputs_embalse: np.ndarray    # (N, T, V_dec_emb)  (típicamente PACUM futuro)
    decoder_inputs_caudal: np.ndarray     # (N, T, V_dec_cau)  (PACUM y EACUM futuros)
    target_embalse: np.ndarray            # (N, T, 1)          (EACUM futuro)
    target_caudal: np.ndarray             # (N, T, 1)          (A284 futuro)
    fechas: List[pd.Timestamp]            # última fecha del histórico de cada muestra


def _stack(df: pd.DataFrame, columnas: Sequence[str], desde: int, hasta: int) -> np.ndarray:
    """Devuelve `df[columnas].iloc[desde:hasta]` como un array (T, len(columnas))."""
    return df.iloc[desde:hasta][list(columnas)].to_numpy(dtype=np.float32)


def build_windows(
    df: pd.DataFrame,
    *,
    historia: int,
    horizonte: int,
    variables_codificador: Sequence[str],
    variables_decoder_embalse: Sequence[str],
    variables_decoder_caudal: Sequence[str],
    variable_objetivo_caudal: str,
    variable_objetivo_embalse: str = "EACUM",
) -> WindowSet:
    """Construye todas las ventanas deslizantes de la serie."""
    n = len(df)
    n_muestras = n - historia - horizonte
    if n_muestras <= 0:
        raise ValueError(
            f"La serie tiene {n} pasos; insuficiente para historia={historia} y "
            f"horizonte={horizonte} (se necesitan al menos {historia + horizonte + 1})."
        )

    enc, dec_emb, dec_cau, tgt_emb, tgt_cau, fechas = [], [], [], [], [], []
    for i in range(n_muestras):
        fechas.append(df.index[i + historia - 1])
        enc.append(_stack(df, variables_codificador, i, i + historia))
        dec_emb.append(_stack(df, variables_decoder_embalse, i + historia, i + historia + horizonte))
        dec_cau.append(_stack(df, variables_decoder_caudal, i + historia, i + historia + horizonte))
        tgt_emb.append(_stack(df, [variable_objetivo_embalse], i + historia, i + historia + horizonte))
        tgt_cau.append(_stack(df, [variable_objetivo_caudal], i + historia, i + historia + horizonte))

    return WindowSet(
        encoder_inputs=np.stack(enc),
        decoder_inputs_embalse=np.stack(dec_emb),
        decoder_inputs_caudal=np.stack(dec_cau),
        target_embalse=np.stack(tgt_emb),
        target_caudal=np.stack(tgt_cau),
        fechas=fechas,
    )
