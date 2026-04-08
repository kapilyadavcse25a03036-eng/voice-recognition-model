"""
src/model.py
────────────
Dense Neural Network (DNN) for phoneme classification.

Architecture
------------
Input  →  Dense(256, ReLU) + Dropout  →  Dense(128, ReLU) + Dropout
       →  Dense(64, ReLU)  + Dropout  →  Dense(NUM_CLASSES, Softmax)

Regularisation is applied via:
* Dropout layers (rate = config.DROPOUT_RATE).
* L2 weight decay on Dense layers (lambda = config.L2_LAMBDA).

The model is compiled with the Adam optimiser and categorical cross-entropy
loss for multi-class classification.
"""

import os
import sys
import logging

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_model(
    input_size: int = config.FEATURE_SIZE,
    num_classes: int = config.NUM_CLASSES,
    hidden_units: list = None,
    dropout_rate: float = config.DROPOUT_RATE,
    l2_lambda: float = config.L2_LAMBDA,
    learning_rate: float = config.LEARNING_RATE,
) -> "tensorflow.keras.Model":
    """
    Build and compile the Dense Neural Network for phoneme classification.

    Args:
        input_size:    Length of the flattened MFCC feature vector.
        num_classes:   Number of output classes (phoneme categories).
        hidden_units:  List of units for each hidden Dense layer.
                       Defaults to ``config.HIDDEN_UNITS``.
        dropout_rate:  Dropout probability after each hidden layer.
        l2_lambda:     L2 regularisation coefficient.
        learning_rate: Adam learning rate.

    Returns:
        Compiled ``tf.keras.Sequential`` model.
    """
    import tensorflow as tf
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
    from tensorflow.keras.regularizers import l2
    from tensorflow.keras.optimizers import Adam

    if hidden_units is None:
        hidden_units = config.HIDDEN_UNITS

    model = Sequential(name="phoneme_dnn")

    # Input layer
    model.add(Input(shape=(input_size,), name="mfcc_input"))

    # Hidden layers
    for i, units in enumerate(hidden_units):
        model.add(
            Dense(
                units,
                activation="relu",
                kernel_regularizer=l2(l2_lambda),
                name=f"dense_{i + 1}",
            )
        )
        model.add(BatchNormalization(name=f"bn_{i + 1}"))
        model.add(Dropout(dropout_rate, name=f"dropout_{i + 1}"))

    # Output layer
    model.add(Dense(num_classes, activation="softmax", name="output"))

    # Compile
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


# ---------------------------------------------------------------------------
# Model persistence
# ---------------------------------------------------------------------------

def save_model(model, path: str = config.MODEL_PATH) -> None:
    """
    Save a Keras model to disk (native Keras format or H5).

    Args:
        model: Trained ``tf.keras.Model`` instance.
        path:  Destination file path.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model.save(path)
    logger.info("Model saved → %s", path)


def load_model(path: str = config.MODEL_PATH) -> "tensorflow.keras.Model":
    """
    Load a Keras model from disk.

    Args:
        path: Path to the saved model file.

    Returns:
        Loaded ``tf.keras.Model``.

    Raises:
        FileNotFoundError: If *path* does not exist.
    """
    import tensorflow as tf

    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    model = tf.keras.models.load_model(path)
    logger.info("Model loaded ← %s", path)
    return model


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def get_model_summary(model) -> str:
    """
    Return the model summary as a string.

    Args:
        model: ``tf.keras.Model`` instance.

    Returns:
        Multi-line string with layer names, output shapes, and parameter counts.
    """
    lines = []
    model.summary(print_fn=lambda x: lines.append(x))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    model = build_model()
    summary = get_model_summary(model)
    logger.info("\n%s", summary)

    # Forward pass with random data
    rng = np.random.default_rng(0)
    dummy = rng.standard_normal((4, config.FEATURE_SIZE)).astype(np.float32)
    probs = model.predict(dummy, verbose=0)
    logger.info("Output shape: %s  (expected: (4, %d))", probs.shape, config.NUM_CLASSES)
    logger.info("Row sums (should all be ~1.0): %s", probs.sum(axis=1))
    logger.info("Smoke-test passed.")
