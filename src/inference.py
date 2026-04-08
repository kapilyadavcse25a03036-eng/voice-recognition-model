"""
src/inference.py
────────────────
Inference engine: predict the phoneme class of a single audio file.

Public API
----------
``predict(audio_path)`` → ``dict`` with keys:
    * ``predicted_class`` – human-readable phoneme label.
    * ``predicted_index`` – integer class index.
    * ``confidence``      – probability assigned to the top class (0-1).
    * ``all_probabilities`` – dict mapping each class name to its probability.

``predict_from_array(audio)`` → same dict, but accepts a raw numpy waveform.
"""
import os
import sys
import logging
import pickle
from typing import Optional, Union

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402
from src.preprocess import preprocess_audio
from src.features import mfcc_to_vector
from src.model import load_model

logger = logging.getLogger(__name__)

# Module-level cache so the model is loaded only once per process lifetime.
_MODEL_CACHE = None
_SCALER_CACHE = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_model(model_path: str = config.MODEL_PATH):
    """Return the cached model, loading from disk on first call."""
    global _MODEL_CACHE
    if _MODEL_CACHE is None:
        _MODEL_CACHE = load_model(model_path)
    return _MODEL_CACHE


def _get_scaler(scaler_path: Optional[str] = None):
    """Return the cached scaler, loading from disk on first call."""
    global _SCALER_CACHE
    if _SCALER_CACHE is not None:
        return _SCALER_CACHE
    if scaler_path is None:
        scaler_path = os.path.join(config.MODELS_DIR, "scaler.pkl")
    if os.path.exists(scaler_path):
        with open(scaler_path, "rb") as fh:
            _SCALER_CACHE = pickle.load(fh)
        logger.info("Scaler loaded ← %s", scaler_path)
    else:
        logger.warning("Scaler not found at %s – skipping normalisation.", scaler_path)
    return _SCALER_CACHE


def _audio_to_features(audio: np.ndarray) -> np.ndarray:
    """
    Convert a preprocessed waveform to a scaled feature vector.

    Args:
        audio: Fixed-length 1-D float32 waveform (``config.N_SAMPLES`` samples).

    Returns:
        2-D float32 array of shape ``(1, feature_size)`` ready for the model.
    """
    vec = mfcc_to_vector(audio)          # shape: (feature_size,)
    vec = vec.reshape(1, -1)             # shape: (1, feature_size)
    scaler = _get_scaler()
    if scaler is not None:
        vec = scaler.transform(vec)
    return vec.astype(np.float32)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def predict_from_array(
    audio: np.ndarray,
    model_path: str = config.MODEL_PATH,
) -> dict:
    """
    Predict the phoneme class from a raw numpy waveform.

    The waveform is preprocessed (normalised, padded/truncated) before
    feature extraction.

    Args:
        audio:      1-D numpy array of audio samples (any length, any dtype).
        model_path: Path to the saved model pickle file (defaults to config value).

    Returns:
        Dictionary with keys ``predicted_class``, ``predicted_index``,
        ``confidence``, and ``all_probabilities``.
    """
    from src.preprocess import normalise_audio, pad_or_truncate

    # Ensure consistent preprocessing
    audio = np.asarray(audio, dtype=np.float32)
    audio = normalise_audio(audio)
    audio = pad_or_truncate(audio)

    features = _audio_to_features(audio)
    model    = _get_model(model_path)
    proba    = model.predict_proba(features)[0]  # shape: (num_classes,)

    pred_idx    = int(np.argmax(proba))
    pred_class  = config.PHONEME_CLASSES[pred_idx]
    confidence  = float(proba[pred_idx])

    all_probs = {
        cls: round(float(proba[i]), 6)
        for i, cls in enumerate(config.PHONEME_CLASSES)
    }

    return {
        "predicted_class":    pred_class,
        "predicted_index":    pred_idx,
        "confidence":         round(confidence, 6),
        "all_probabilities":  all_probs,
    }


def predict(
    audio_path: str,
    model_path: str = config.MODEL_PATH,
) -> dict:
    """
    Predict the phoneme class for an audio file.

    Args:
        audio_path: Path to a WAV or MP3 audio file.
        model_path: Path to the saved model pickle file (defaults to config value).

    Returns:
        Dictionary with keys ``predicted_class``, ``predicted_index``,
        ``confidence``, and ``all_probabilities``.

    Raises:
        FileNotFoundError: If *audio_path* does not exist.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    audio = preprocess_audio(audio_path)
    return predict_from_array(audio, model_path=model_path)


def batch_predict(
    audio_paths: list,
    model_path: str = config.MODEL_PATH,
) -> list:
    """
    Predict phoneme classes for a list of audio files.

    Args:
        audio_paths: List of paths to audio files.
        model_path:  Path to the saved model pickle file.

    Returns:
        List of prediction dictionaries (one per file), in the same order as
        *audio_paths*.  If a file cannot be loaded, its entry will contain
        ``{"error": <message>}``.
    """
    results = []
    for path in audio_paths:
        try:
            results.append(predict(path, model_path=model_path))
        except Exception as exc:
            logger.error("Failed to predict for %s: %s", path, exc)
            results.append({"error": str(exc)})
    return results


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Phoneme inference on an audio file.")
    parser.add_argument("audio_file", help="Path to a WAV or MP3 audio file.")
    parser.add_argument(
        "--model", default=config.MODEL_PATH, help="Path to the trained model pickle file."
    )
    args = parser.parse_args()

    result = predict(args.audio_file, model_path=args.model)
    print(json.dumps(result, indent=2))
