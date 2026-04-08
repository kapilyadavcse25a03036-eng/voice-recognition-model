"""
src/evaluate.py
───────────────
Evaluation pipeline for the trained phoneme classification model.

Produces
--------
* Overall accuracy, precision, recall, and weighted F1-score.
* Per-class metrics (printed to console).
* Confusion matrix visualisation saved to ``results/confusion_matrix.png``.
* Metrics written / merged into ``results/metrics.json``.
"""

import os
import sys
import json
import logging

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402
from src.preprocess import load_split_from_file
from src.features import extract_features_batch, apply_scaler
from src.model import load_model

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def evaluate(
    model_path: str = config.MODEL_PATH,
    test_data_path: str = config.TEST_DATA_PATH,
) -> dict:
    """
    Evaluate the trained model on the held-out test set.

    Args:
        model_path:     Path to the saved model pickle file.
        test_data_path: Path to the test-set pickle file.

    Returns:
        Dictionary containing ``accuracy``, ``precision``, ``recall``,
        ``f1_score`` (all weighted averages), and ``per_class`` metrics.
    """
    import pickle

    # ── 1. Load model ─────────────────────────────────────────────────────────
    model = load_model(model_path)

    # ── 2. Load scaler ────────────────────────────────────────────────────────
    scaler_path = os.path.join(config.MODELS_DIR, "scaler.pkl")
    if os.path.exists(scaler_path):
        with open(scaler_path, "rb") as fh:
            scaler = pickle.load(fh)
        logger.info("Scaler loaded ← %s", scaler_path)
    else:
        scaler = None
        logger.warning("Scaler not found at %s – using unscaled features.", scaler_path)

    # ── 3. Load test data ─────────────────────────────────────────────────────
    X_test_raw, y_test = load_split_from_file(test_data_path)
    logger.info("Test samples: %d", len(X_test_raw))

    # ── 4. Feature extraction ─────────────────────────────────────────────────
    X_test = extract_features_batch(X_test_raw)
    if scaler is not None:
        X_test = apply_scaler(X_test, scaler)

    # ── 5. Predictions ────────────────────────────────────────────────────────
    y_pred = model.predict(X_test)

    # ── 6. Metrics ────────────────────────────────────────────────────────────
    metrics = compute_metrics(y_test, y_pred)

    # ── 7. Persist ────────────────────────────────────────────────────────────
    _save_metrics(metrics)
    _plot_confusion_matrix(y_test, y_pred)

    return metrics


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute classification metrics from ground-truth and predicted labels.

    Args:
        y_true: 1-D array of integer ground-truth labels.
        y_pred: 1-D array of integer predicted labels.

    Returns:
        Dictionary with keys: ``accuracy``, ``precision``, ``recall``,
        ``f1_score``, ``per_class``.
    """
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        classification_report,
    )

    accuracy  = float(accuracy_score(y_true, y_pred))
    precision = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    recall    = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    f1        = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    report = classification_report(
        y_true,
        y_pred,
        target_names=config.PHONEME_CLASSES,
        output_dict=True,
        zero_division=0,
    )

    logger.info("\n%s", classification_report(
        y_true, y_pred, target_names=config.PHONEME_CLASSES, zero_division=0
    ))

    per_class = {}
    for cls in config.PHONEME_CLASSES:
        if cls in report:
            per_class[cls] = {
                "precision": round(report[cls]["precision"], 4),
                "recall":    round(report[cls]["recall"], 4),
                "f1-score":  round(report[cls]["f1-score"], 4),
                "support":   int(report[cls]["support"]),
            }

    return {
        "accuracy":  round(accuracy, 4),
        "precision": round(precision, 4),
        "recall":    round(recall, 4),
        "f1_score":  round(f1, 4),
        "per_class": per_class,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """
    Save a colour-coded confusion matrix to disk.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
    """
    from sklearn.metrics import confusion_matrix
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = confusion_matrix(y_true, y_pred, labels=list(range(config.NUM_CLASSES)))

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)

    tick_marks = np.arange(config.NUM_CLASSES)
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(config.PHONEME_CLASSES, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(config.PHONEME_CLASSES, fontsize=9)

    # Annotate cells
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=8,
            )

    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix – Phoneme Classification")
    plt.tight_layout()

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    plt.savefig(config.CONFUSION_MATRIX, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Confusion matrix saved → %s", config.CONFUSION_MATRIX)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def _save_metrics(metrics: dict) -> None:
    """
    Merge evaluation metrics into ``results/metrics.json``.

    Args:
        metrics: Evaluation metrics dictionary.
    """
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    existing = {}
    if os.path.exists(config.METRICS_PATH):
        with open(config.METRICS_PATH) as fh:
            try:
                existing = json.load(fh)
            except json.JSONDecodeError:
                pass
    existing["evaluation"] = metrics
    with open(config.METRICS_PATH, "w") as fh:
        json.dump(existing, fh, indent=2)
    logger.info("Evaluation metrics saved → %s", config.METRICS_PATH)

    # Print summary
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"  Accuracy : {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall   : {metrics['recall']:.4f}")
    print(f"  F1-Score : {metrics['f1_score']:.4f}")
    print("=" * 50)


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    metrics = evaluate()
    print(json.dumps(metrics, indent=2))
