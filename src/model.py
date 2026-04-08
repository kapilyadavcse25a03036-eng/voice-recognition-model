"""
src/model.py
────────────
Random Forest classifier for phoneme classification.

The model uses scikit-learn's RandomForestClassifier – a classic ensemble
machine learning algorithm that requires no deep learning framework.

Persistence
-----------
Trained models are saved / loaded as pickle files via Python's standard
``pickle`` module (compatible with ``joblib``).
"""

import os
import sys
import logging
import pickle
from typing import Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_model(
    n_estimators: int = config.N_ESTIMATORS,
    max_depth: Optional[int] = config.MAX_DEPTH,
    min_samples_leaf: int = config.MIN_SAMPLES_LEAF,
    random_state: int = config.RANDOM_SEED,
    n_jobs: int = -1,
):
    """
    Build a Random Forest classifier for phoneme classification.

    Args:
        n_estimators:      Number of trees in the forest.
        max_depth:         Maximum depth of each tree (``None`` = unlimited).
        min_samples_leaf:  Minimum samples required at each leaf node.
        random_state:      Random seed for reproducibility.
        n_jobs:            Number of parallel jobs (``-1`` = all CPUs).

    Returns:
        Untrained ``sklearn.ensemble.RandomForestClassifier``.
    """
    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        class_weight="balanced",
        n_jobs=n_jobs,
    )
    logger.info(
        "Random Forest: n_estimators=%d  max_depth=%s  min_samples_leaf=%d",
        n_estimators,
        max_depth,
        min_samples_leaf,
    )
    return model


# ---------------------------------------------------------------------------
# Model persistence
# ---------------------------------------------------------------------------

def save_model(model, path: str = config.MODEL_PATH) -> None:
    """
    Save a scikit-learn model to disk using pickle.

    Args:
        model: Trained scikit-learn estimator.
        path:  Destination file path.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(model, fh, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("Model saved → %s", path)


def load_model(path: str = config.MODEL_PATH):
    """
    Load a scikit-learn model from disk.

    Args:
        path: Path to the saved pickle file.

    Returns:
        Loaded scikit-learn estimator.

    Raises:
        FileNotFoundError: If *path* does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    with open(path, "rb") as fh:
        model = pickle.load(fh)
    logger.info("Model loaded ← %s", path)
    return model


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    model = build_model()
    logger.info("Model: %s", model)

    # Quick fit with random data
    rng = np.random.default_rng(0)
    X_dummy = rng.standard_normal((40, config.FEATURE_SIZE)).astype(np.float32)
    y_dummy = rng.integers(0, config.NUM_CLASSES, size=40)
    model.fit(X_dummy, y_dummy)

    proba = model.predict_proba(X_dummy[:4])
    logger.info("Output shape: %s  (expected: (4, %d))", proba.shape, config.NUM_CLASSES)
    logger.info("Row sums (should all be ~1.0): %s", proba.sum(axis=1))
    logger.info("Smoke-test passed.")
