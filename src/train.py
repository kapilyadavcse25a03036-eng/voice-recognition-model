"""
src/train.py
────────────
STT pipeline orchestration: prepare the dataset manifest and run evaluation.

For the Whisper-based STT system, "training" means:

1. Loading the pre-trained Whisper model (weights downloaded automatically).
2. Running it on the prepared dataset manifest.
3. Computing WER / CER metrics and saving results.

Fine-tuning Whisper on a custom dataset is an advanced optional step
documented in the README.  This script focuses on the evaluation workflow
that verifies the system is working correctly on your data.

Usage
-----
    python src/train.py
    python src/train.py --model-size small --language en
"""

import os
import sys
import logging
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402
from src.model import save_model_card
from src.evaluate import evaluate

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    model_size: str = config.WHISPER_MODEL_SIZE,
    language: str = config.STT_LANGUAGE,
    manifest_path: str = config.MANIFEST_PATH,
) -> dict:
    """
    Full STT pipeline: load model → evaluate on manifest → save metrics.

    Args:
        model_size:    Whisper model size (``"tiny"``, ``"base"``, etc.).
        language:      Target language code (e.g. ``"en"``).
        manifest_path: Path to the dataset manifest JSON file.

    Returns:
        Evaluation metrics dictionary.
    """
    logger.info("=== Speech-to-Text Pipeline ===")
    logger.info("Model size : %s", model_size)
    logger.info("Language   : %s", language)
    logger.info("Manifest   : %s", manifest_path)

    # ── 1. Save model card ────────────────────────────────────────────────────
    save_model_card(size=model_size, device=config.STT_DEVICE)

    # ── 2. Evaluate ───────────────────────────────────────────────────────────
    logger.info("Running evaluation …")
    metrics = evaluate(
        manifest_path=manifest_path,
        model_size=model_size,
        language=language,
    )

    # ── 3. Persist summary metrics ────────────────────────────────────────────
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    summary_path = os.path.join(config.RESULTS_DIR, "metrics.json")
    existing = {}
    if os.path.exists(summary_path):
        with open(summary_path) as fh:
            try:
                existing = json.load(fh)
            except json.JSONDecodeError:
                pass
    existing["pipeline"] = {
        "model_size": model_size,
        "language":   language,
        "wer":        metrics.get("wer"),
        "cer":        metrics.get("cer"),
        "avg_confidence": metrics.get("avg_confidence"),
        "num_samples": metrics.get("num_samples"),
    }
    with open(summary_path, "w") as fh:
        json.dump(existing, fh, indent=2)
    logger.info("Summary saved → %s", summary_path)

    return metrics


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Run the STT evaluation pipeline.")
    parser.add_argument(
        "--model-size",
        default=config.WHISPER_MODEL_SIZE,
        help="Whisper model size (tiny/base/small/medium/large).",
    )
    parser.add_argument(
        "--language",
        default=config.STT_LANGUAGE,
        help="Language code (e.g. 'en').",
    )
    parser.add_argument(
        "--manifest",
        default=config.MANIFEST_PATH,
        help="Path to the dataset manifest JSON file.",
    )
    args = parser.parse_args()

    run_pipeline(
        model_size=args.model_size,
        language=args.language,
        manifest_path=args.manifest,
    )

