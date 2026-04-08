"""
src/train.py
────────────
Training pipeline for the phoneme classification Random Forest.

Steps
-----
1. Load preprocessed waveforms from pickle files (train / val splits).
2. Extract MFCC feature vectors and globally standardise them.
3. Build the Random Forest classifier via ``src/model.py``.
4. Fit the model on training data.
5. Save the trained model and evaluate on the validation set.
6. Generate and save a feature-importance bar chart.
"""

import os
import sys
import logging
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

def train() -> dict:
    """
    Full training pipeline.

    Loads preprocessed data, extracts features, fits the Random Forest
    classifier, saves the model and logs metrics.

    Returns:
        Dictionary containing training and validation accuracy.
    """
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
    logger.info("Building Random Forest model …")
    model = build_model()

    # ── 4. Training ───────────────────────────────────────────────────────────
    logger.info("Fitting model on %d training samples …", len(X_train))
    model.fit(X_train, y_train)
    logger.info("Training complete.")

    # ── 5. Save model & compute metrics ──────────────────────────────────────
    save_model(model, config.MODEL_PATH)

    train_accuracy = float(model.score(X_train, y_train))
    val_accuracy   = float(model.score(X_val, y_val))
    logger.info("Train accuracy: %.4f | Val accuracy: %.4f", train_accuracy, val_accuracy)

    # ── 6. Feature-importance plot ────────────────────────────────────────────
    _plot_feature_importances(model)

    final_metrics = {
        "train_accuracy": round(train_accuracy, 4),
        "val_accuracy":   round(val_accuracy, 4),
    }
    _save_training_metrics(final_metrics)
    logger.info("Training complete.  Final metrics: %s", final_metrics)
    return final_metrics


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_feature_importances(model) -> None:
    """
    Save a bar chart of the top-40 most important MFCC feature dimensions.

    Args:
        model: Fitted ``sklearn.ensemble.RandomForestClassifier``.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    importances = model.feature_importances_
    top_n = min(40, len(importances))
    indices = np.argsort(importances)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(range(top_n), importances[indices])
    ax.set_title(f"Top-{top_n} Feature Importances (Random Forest)")
    ax.set_xlabel("Feature index rank")
    ax.set_ylabel("Importance")
    ax.grid(True, axis="y")

    plt.tight_layout()
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    plt.savefig(config.TRAINING_PLOT, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Feature importances saved → %s", config.TRAINING_PLOT)


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
