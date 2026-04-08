"""
scripts/train_pipeline.py
──────────────────────────
End-to-end Speech-to-Text pipeline script.

Steps
-----
1. Prepare the dataset manifest (download + synthetic TTS generation).
2. Run Whisper-based STT evaluation on the manifest.
3. Print a final WER / CER summary.

Usage (from the project root):
    python scripts/train_pipeline.py
    python scripts/train_pipeline.py --skip-data
    python scripts/train_pipeline.py --model-size small --language en
"""

import os
import sys
import logging
import argparse

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
    """Download real samples and/or generate synthetic TTS clips."""
    from data.download_data import main as download_main  # type: ignore

    logger.info("=" * 60)
    logger.info("STEP 1 – Dataset preparation")
    logger.info("=" * 60)
    download_main()


def step_evaluate(model_size: str, language: str) -> dict:
    """Run STT evaluation on the prepared manifest and return metrics."""
    from src.train import run_pipeline

    logger.info("=" * 60)
    logger.info("STEP 2 – STT Evaluation  (model: %s, lang: %s)", model_size, language)
    logger.info("=" * 60)
    return run_pipeline(model_size=model_size, language=language)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Full Speech-to-Text pipeline."
    )
    parser.add_argument(
        "--skip-data",
        action="store_true",
        help="Skip dataset preparation (use existing manifest).",
    )
    parser.add_argument(
        "--model-size",
        default=config.WHISPER_MODEL_SIZE,
        help="Whisper model size: tiny / base / small / medium / large.",
    )
    parser.add_argument(
        "--language",
        default=config.STT_LANGUAGE,
        help="Target language code (e.g. 'en').  Pass 'auto' for detection.",
    )
    args = parser.parse_args()

    language = None if args.language == "auto" else args.language

    logger.info("Voice Recognition – Speech-to-Text Pipeline")

    if not args.skip_data:
        step_data()

    metrics = step_evaluate(model_size=args.model_size, language=language)

    # ── Final summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    wer = metrics.get("wer")
    cer = metrics.get("cer")
    conf = metrics.get("avg_confidence")
    n = metrics.get("num_successful", 0)
    if wer is not None:
        print(f"  Samples evaluated : {n}")
        print(f"  WER               : {wer:.4f}")
        print(f"  CER               : {cer:.4f}")
        print(f"  Avg confidence    : {conf:.4f}")
    print("=" * 60)
    print(f"\nArtifacts:")
    print(f"  Model card : {os.path.join(config.MODELS_DIR, 'model_card.json')}")
    print(f"  Metrics    : {config.METRICS_PATH}")


if __name__ == "__main__":
    main()

