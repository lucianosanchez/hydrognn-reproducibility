"""Funciones de pérdida personalizadas (compatibles con Keras 2 y Keras 3).

Dos pérdidas, cada una con una motivación clara:

* `make_embalse_loss`: el embalse evoluciona suavemente, así que penalizamos
  saltos día a día además del MSE puntual.
* `make_caudal_loss`: lo planteamos como clasificación (¿supera el umbral?)
  con pesos asimétricos, porque las anomalías de bajo caudal son raras pero
  es lo que importa decidir.

Implementadas con `tf.*` (no con `keras.backend`) para evitar que el
refactor de Keras 3 — que movió las operaciones matemáticas a `keras.ops`
y las quitó de `keras.backend` — rompa el código.
"""

from __future__ import annotations

from typing import Callable, TYPE_CHECKING

if TYPE_CHECKING:
    import tensorflow as tf  # type-only: avoids hard import at module load


_EPS = 1e-7


def make_embalse_loss(peso_suavidad: float) -> "Callable[[tf.Tensor, tf.Tensor], tf.Tensor]":
    """MSE más penalización por la norma de la primera diferencia.

    TensorFlow is imported lazily so the rest of `seq2seq_runoff` can be
    used without the TF/Keras stack when only the GNN models are needed.
    """
    import tensorflow as tf  # noqa: F401 — lazy import

    def embalse_loss(y_true, y_pred):
        mse = tf.reduce_mean(tf.square(y_true - y_pred), axis=-1)
        if peso_suavidad <= 0:
            return mse
        y_pred = tf.clip_by_value(y_pred, _EPS, 1.0 - _EPS)
        diff = y_pred[:, 1:] - y_pred[:, :-1]
        return mse + peso_suavidad * tf.reduce_mean(tf.square(diff))

    return embalse_loss


def make_caudal_loss(
    caudal_minimo_normalizado: float,
    desbalance: float = 10.0,
) -> "Callable[[tf.Tensor, tf.Tensor], tf.Tensor]":
    """Cross-entropy ponderada para el problema "caudal por encima del umbral".

    El modelo emite la probabilidad (sigmoide) de superar `caudal_minimo`. Al
    haber muchos más días normales que anómalos, los días normales reciben
    peso 1 y los anómalos peso `desbalance`.
    """
    import tensorflow as tf  # noqa: F401 — lazy import

    # LSR Cuidado con el desbalance, bajar para que aumente la precisión a costa de recall

    def caudal_loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, _EPS, 1.0 - _EPS)
        # Etiqueta binaria: 1 si supera el umbral mínimo.
        etiqueta = tf.sign(tf.nn.relu(y_true - caudal_minimo_normalizado))
        loss = tf.reduce_mean(
            -tf.reduce_sum(etiqueta * tf.math.log(y_pred), axis=-1)
            - desbalance * tf.reduce_sum((1 - etiqueta) * tf.math.log(1 - y_pred), axis=-1)
        )
        return loss / 1000.0

    return caudal_loss
