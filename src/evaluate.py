"""
src/evaluate.py
───────────────
Evaluation pipeline for the Speech-to-Text system.

Computes standard ASR metrics:

* **WER** – Word Error Rate  (lower is better; 0 = perfect)
* **CER** – Character Error Rate
* **Average confidence** – mean Whisper confidence score across samples.

Metrics are written to ``results/metrics.json``.

Usage
-----
    python src/evaluate.py                          # use default manifest
    python src/evaluate.py --manifest my_data.json
"""

import os
import sys
import json
import logging
from typing import List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402
from src.preprocess import load_manifest, filter_manifest
from src.transcribe import load_whisper_model, transcribe_file

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# WER / CER helpers
# ---------------------------------------------------------------------------

def _edit_distance(a: List, b: List) -> int:
    """
    Compute the Levenshtein edit distance between two sequences.

    Args:
        a: Reference sequence (list of tokens / characters).
        b: Hypothesis sequence.

    Returns:
        Minimum edit distance (int).
    """
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


def compute_wer(reference: str, hypothesis: str) -> float:
    """
    Compute Word Error Rate between a reference and hypothesis transcript.

    WER = (S + D + I) / N  where N = number of words in the reference.

    Args:
        reference:  Ground-truth transcript (case-insensitive).
        hypothesis: Predicted transcript.

    Returns:
        WER as a float (can exceed 1.0 if hypothesis is much longer).
        Returns 0.0 if both strings are empty.
    """
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    return _edit_distance(ref_words, hyp_words) / len(ref_words)


def compute_cer(reference: str, hypothesis: str) -> float:
    """
    Compute Character Error Rate between a reference and hypothesis transcript.

    CER = edit_distance(ref_chars, hyp_chars) / len(ref_chars).

    Args:
        reference:  Ground-truth transcript (case-insensitive, spaces included).
        hypothesis: Predicted transcript.

    Returns:
        CER as a float.  Returns 0.0 if both strings are empty.
    """
    ref_chars = list(reference.lower())
    hyp_chars = list(hypothesis.lower())
    if not ref_chars:
        return 0.0 if not hyp_chars else 1.0
    return _edit_distance(ref_chars, hyp_chars) / len(ref_chars)


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def evaluate(
    manifest_path: str = config.MANIFEST_PATH,
    model_size: str = config.WHISPER_MODEL_SIZE,
    language: str = config.STT_LANGUAGE,
) -> dict:
    """
    Evaluate the STT system on a labelled manifest.

    Each manifest entry must have both ``"audio"`` and ``"text"`` keys.
    Entries missing ground-truth text are skipped.

    Args:
        manifest_path: Path to the JSON manifest file.
        model_size:    Whisper model size to use.
        language:      Language hint for transcription.

    Returns:
        Dictionary with keys ``wer``, ``cer``, ``avg_confidence``,
        ``num_samples``, and ``per_sample`` (list of per-file metrics).
    """
    manifest = load_manifest(manifest_path)
    manifest = filter_manifest(manifest, require_text=True)

    if not manifest:
        raise RuntimeError(
            "No labelled samples found in the manifest.  "
            "Add 'text' fields or run `python data/download_data.py`."
        )

    model = load_whisper_model(size=model_size)

    per_sample = []
    total_wer = 0.0
    total_cer = 0.0
    total_confidence = 0.0

    for i, entry in enumerate(manifest, 1):
        audio_path = entry["audio"]
        reference  = entry["text"]

        logger.info("[%d/%d] %s", i, len(manifest), os.path.basename(audio_path))
        try:
            result     = transcribe_file(audio_path, model=model, language=language)
            hypothesis = result["text"]
            confidence = result["confidence"]
        except Exception as exc:
            logger.error("  Error: %s", exc)
            per_sample.append({"audio": audio_path, "error": str(exc)})
            continue

        wer = compute_wer(reference, hypothesis)
        cer = compute_cer(reference, hypothesis)

        total_wer        += wer
        total_cer        += cer
        total_confidence += confidence

        logger.info(
            "  REF : %s\n  HYP : %s\n  WER : %.3f  CER : %.3f  conf: %.3f",
            reference, hypothesis, wer, cer, confidence,
        )

        per_sample.append({
            "audio":      audio_path,
            "reference":  reference,
            "hypothesis": hypothesis,
            "wer":        round(wer, 4),
            "cer":        round(cer, 4),
            "confidence": round(confidence, 4),
        })

    n = len(per_sample)
    successful = [s for s in per_sample if "error" not in s]
    ns = len(successful)

    metrics = {
        "wer":            round(total_wer / ns, 4) if ns else None,
        "cer":            round(total_cer / ns, 4) if ns else None,
        "avg_confidence": round(total_confidence / ns, 4) if ns else None,
        "num_samples":    n,
        "num_successful": ns,
        "per_sample":     per_sample,
    }

    _save_metrics(metrics)
    _print_summary(metrics)
    return metrics


# ---------------------------------------------------------------------------
# Persistence / display helpers
# ---------------------------------------------------------------------------

def _save_metrics(metrics: dict) -> None:
    """Merge evaluation metrics into ``results/metrics.json``."""
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    existing = {}
    if os.path.exists(config.METRICS_PATH):
        with open(config.METRICS_PATH) as fh:
            try:
                existing = json.load(fh)
            except json.JSONDecodeError:
                pass

    # Store a summary (without the verbose per_sample list) at top level
    summary = {k: v for k, v in metrics.items() if k != "per_sample"}
    existing["evaluation"] = summary

    with open(config.METRICS_PATH, "w") as fh:
        json.dump(existing, fh, indent=2)
    logger.info("Evaluation metrics saved → %s", config.METRICS_PATH)


def _print_summary(metrics: dict) -> None:
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"  Samples      : {metrics['num_successful']} / {metrics['num_samples']}")
    if metrics["wer"] is not None:
        print(f"  WER          : {metrics['wer']:.4f}")
        print(f"  CER          : {metrics['cer']:.4f}")
        print(f"  Avg Confidence: {metrics['avg_confidence']:.4f}")
    print("=" * 50)


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Evaluate the STT system on a manifest.")
    parser.add_argument(
        "--manifest",
        default=config.MANIFEST_PATH,
        help="Path to the JSON manifest file.",
    )
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
    args = parser.parse_args()

    metrics = evaluate(
        manifest_path=args.manifest,
        model_size=args.model_size,
        language=args.language,
    )
    print(json.dumps({k: v for k, v in metrics.items() if k != "per_sample"}, indent=2))

