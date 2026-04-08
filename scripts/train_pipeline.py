"""
scripts/train_pipeline.py
──────────────────────────
End-to-end training pipeline script.

Steps
-----
1. Generate / download raw audio data (``data/download_data.py``).
2. Preprocess audio and save train / val / test pickle files.
3. Train the Random Forest classifier.
4. Evaluate on the test set.
5. Print a final summary.

Usage (from the project root):
    python scripts/train_pipeline.py
"""

import os
import sys
import logging
import argparse
import json

# Make the project root importable
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import config  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def step_data() -> None:
    """Download / generate synthetic data and check the raw directory."""
    from data.download_data import main as download_main  # type: ignore

    logger.info("=" * 60)
    logger.info("STEP 1 – Data download / generation")
    logger.info("=" * 60)
    download_main()


def step_preprocess() -> None:
    """Preprocess audio files and save pickled train/val/test splits."""
    from src.preprocess import prepare_dataset

    logger.info("=" * 60)
    logger.info("STEP 2 – Preprocessing")
    logger.info("=" * 60)
    prepare_dataset()


def step_train() -> dict:
    """Train the Random Forest classifier and return training summary metrics."""
    from src.train import train

    logger.info("=" * 60)
    logger.info("STEP 3 – Training")
    logger.info("=" * 60)
    return train()


def step_evaluate() -> dict:
    """Evaluate the trained model on the test set."""
    from src.evaluate import evaluate

    logger.info("=" * 60)
    logger.info("STEP 4 – Evaluation")
    logger.info("=" * 60)
    return evaluate()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Full training pipeline for the phoneme classification Random Forest."
    )
    parser.add_argument(
        "--skip-data",
        action="store_true",
        help="Skip data download / generation (use existing raw files).",
    )
    parser.add_argument(
        "--skip-preprocess",
        action="store_true",
        help="Skip preprocessing (use existing pickle files).",
    )
    args = parser.parse_args()

    logger.info("Voice Recognition – Full Training Pipeline")

    if not args.skip_data:
        step_data()

    if not args.skip_preprocess:
        step_preprocess()

    train_metrics = step_train()

    eval_metrics = step_evaluate()

    # ── Final summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Train accuracy : {train_metrics['train_accuracy']:.4f}")
    print(f"  Val   accuracy : {train_metrics['val_accuracy']:.4f}")
    print(f"  Test  accuracy : {eval_metrics['accuracy']:.4f}")
    print(f"  Test  F1-score : {eval_metrics['f1_score']:.4f}")
    print("=" * 60)
    print(f"\nArtifacts:")
    print(f"  Model          : {config.MODEL_PATH}")
    print(f"  Feature imports: {config.TRAINING_PLOT}")
    print(f"  Confusion matrix: {config.CONFUSION_MATRIX}")
    print(f"  Metrics JSON   : {config.METRICS_PATH}")


if __name__ == "__main__":
    main()
