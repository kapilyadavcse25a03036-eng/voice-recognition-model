"""
src/train.py
────────────
Training pipeline for the phoneme classification DNN.

Steps
-----
1. Load preprocessed waveforms from pickle files (train / val splits).
2. Extract MFCC feature vectors and globally standardise them.
3. Build the Dense Neural Network via ``src/model.py``.
4. Train with early stopping and model-checkpoint callbacks.
5. Save the best model weights and training history.
6. Generate and save loss / accuracy training curves.
"""

import os
import sys
import logging
import csv
import json

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402
from src.preprocess import load_split_from_file
from src.features import extract_features_batch, fit_scaler, apply_scaler
from src.model import build_model, save_model

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    epochs: int = config.EPOCHS,
    batch_size: int = config.BATCH_SIZE,
    learning_rate: float = config.LEARNING_RATE,
    patience: int = config.PATIENCE,
) -> dict:
    """
    Full training pipeline.

    Loads preprocessed data, extracts features, trains the DNN, saves the
    model and logs metrics.

    Args:
        epochs:        Maximum number of training epochs.
        batch_size:    Mini-batch size.
        learning_rate: Adam optimiser learning rate.
        patience:      Early-stopping patience.

    Returns:
        Dictionary containing final training and validation accuracy/loss.
    """
    import tensorflow as tf
    from tensorflow.keras.callbacks import (
        EarlyStopping,
        ModelCheckpoint,
        CSVLogger,
    )

    # ── 1. Load data ─────────────────────────────────────────────────────────
    logger.info("Loading preprocessed data …")
    X_train_raw, y_train = load_split_from_file(config.TRAIN_DATA_PATH)
    X_val_raw, y_val     = load_split_from_file(config.VAL_DATA_PATH)
    logger.info("Train: %d samples | Val: %d samples", len(X_train_raw), len(X_val_raw))

    # ── 2. Feature extraction ─────────────────────────────────────────────────
    logger.info("Extracting MFCC features …")
    X_train = extract_features_batch(X_train_raw)
    X_val   = extract_features_batch(X_val_raw)

    # Global standardisation (fit on train only)
    scaler = fit_scaler(X_train)
    X_train = apply_scaler(X_train, scaler)
    X_val   = apply_scaler(X_val, scaler)

    # Persist scaler alongside the model
    import pickle
    scaler_path = os.path.join(config.MODELS_DIR, "scaler.pkl")
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    with open(scaler_path, "wb") as fh:
        pickle.dump(scaler, fh)
    logger.info("Scaler saved → %s", scaler_path)

    # ── 3. Build model ────────────────────────────────────────────────────────
    logger.info("Building model …")
    model = build_model(
        input_size=X_train.shape[1],
        learning_rate=learning_rate,
    )
    model.summary(print_fn=logger.info)

    # ── 4. Callbacks ──────────────────────────────────────────────────────────
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=patience,
            min_delta=config.MIN_DELTA,
            restore_best_weights=True,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=config.MODEL_PATH,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        CSVLogger(config.TRAINING_HISTORY, append=False),
    ]

    # ── 5. Training ───────────────────────────────────────────────────────────
    logger.info("Starting training (max %d epochs, batch_size=%d) …", epochs, batch_size)
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    # ── 6. Save model & plots ─────────────────────────────────────────────────
    save_model(model, config.MODEL_PATH)
    _plot_training_curves(history)

    # Collect final metrics
    final_metrics = {
        "final_train_loss":     float(history.history["loss"][-1]),
        "final_train_accuracy": float(history.history["accuracy"][-1]),
        "final_val_loss":       float(history.history["val_loss"][-1]),
        "final_val_accuracy":   float(history.history["val_accuracy"][-1]),
        "epochs_trained":       len(history.history["loss"]),
    }
    _save_training_metrics(final_metrics)
    logger.info("Training complete.  Final metrics: %s", final_metrics)
    return final_metrics


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_training_curves(history) -> None:
    """
    Save a two-panel figure showing loss and accuracy over epochs.

    Args:
        history: ``tf.keras.callbacks.History`` object returned by ``model.fit``.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = range(1, len(history.history["loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    ax1.plot(epochs, history.history["loss"],     label="Train loss")
    ax1.plot(epochs, history.history["val_loss"], label="Val loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training and Validation Loss")
    ax1.legend()
    ax1.grid(True)

    # Accuracy
    ax2.plot(epochs, history.history["accuracy"],     label="Train accuracy")
    ax2.plot(epochs, history.history["val_accuracy"], label="Val accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Training and Validation Accuracy")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    plt.savefig(config.TRAINING_PLOT, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Training curves saved → %s", config.TRAINING_PLOT)


def _save_training_metrics(metrics: dict) -> None:
    """
    Append or create a JSON file with training summary metrics.

    Args:
        metrics: Dictionary of metric names to float values.
    """
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    existing = {}
    if os.path.exists(config.METRICS_PATH):
        with open(config.METRICS_PATH) as fh:
            try:
                existing = json.load(fh)
            except json.JSONDecodeError:
                pass
    existing.update({"training": metrics})
    with open(config.METRICS_PATH, "w") as fh:
        json.dump(existing, fh, indent=2)
    logger.info("Training metrics saved → %s", config.METRICS_PATH)


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    train()
