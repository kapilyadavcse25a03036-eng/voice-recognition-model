"""
src/inference.py
────────────────
Inference engine: transcribe speech from a single audio file to text.

Public API
----------
``predict(audio_path)`` → dict with keys:
    * ``text``        – full transcription string.
    * ``segments``    – list of timed segment dicts (start, end, text).
    * ``language``    – detected or configured language code.
    * ``confidence``  – mean segment confidence score (0-1).

``predict_from_array(audio)`` → same dict, but accepts a raw numpy waveform.

``batch_predict(audio_paths)`` → list of result dicts (one per file).
"""
import os
import sys
import logging
from typing import Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402
from src.transcribe import load_whisper_model, transcribe_file, transcribe_array

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def predict_from_array(
    audio: np.ndarray,
    sr: int = config.SAMPLE_RATE,
    model_size: str = config.WHISPER_MODEL_SIZE,
    language: Optional[str] = config.STT_LANGUAGE,
) -> dict:
    """
    Transcribe speech from a raw numpy waveform.

    Args:
        audio:      1-D numpy array of audio samples (any length, any dtype).
        sr:         Sample rate of *audio* in Hz.
        model_size: Whisper model size to use.
        language:   Language hint (e.g. ``"en"``).  Pass ``None`` for
                    automatic language detection.

    Returns:
        Dictionary with keys ``text``, ``segments``, ``language``,
        ``confidence``.
    """
    audio = np.asarray(audio, dtype=np.float32)
    model = load_whisper_model(size=model_size)
    return transcribe_array(audio, sr=sr, model=model, language=language)


def predict(
    audio_path: str,
    model_size: str = config.WHISPER_MODEL_SIZE,
    language: Optional[str] = config.STT_LANGUAGE,
) -> dict:
    """
    Transcribe speech from an audio file.

    Args:
        audio_path: Path to a WAV or MP3 audio file.
        model_size: Whisper model size to use.
        language:   Language hint (e.g. ``"en"``).  Pass ``None`` for
                    automatic language detection.

    Returns:
        Dictionary with keys ``text``, ``segments``, ``language``,
        ``confidence``.

    Raises:
        FileNotFoundError: If *audio_path* does not exist.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    model = load_whisper_model(size=model_size)
    return transcribe_file(audio_path, model=model, language=language)


def batch_predict(
    audio_paths: list,
    model_size: str = config.WHISPER_MODEL_SIZE,
    language: Optional[str] = config.STT_LANGUAGE,
) -> list:
    """
    Transcribe a list of audio files.

    The Whisper model is loaded once and reused for all files.

    Args:
        audio_paths: List of paths to audio files.
        model_size:  Whisper model size to use.
        language:    Language hint applied to every file.

    Returns:
        List of result dicts (one per file, same order as *audio_paths*).
        If a file cannot be transcribed, its entry will contain
        ``{"error": "<message>"}``.
    """
    from src.transcribe import batch_transcribe

    model = load_whisper_model(size=model_size)
    return batch_transcribe(audio_paths, model=model, language=language)


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Transcribe an audio file to text.")
    parser.add_argument("audio_file", help="Path to a WAV or MP3 audio file.")
    parser.add_argument(
        "--model-size",
        default=config.WHISPER_MODEL_SIZE,
        help="Whisper model size (tiny/base/small/medium/large).",
    )
    parser.add_argument(
        "--language",
        default=config.STT_LANGUAGE,
        help="Language code (e.g. 'en').  Omit for auto-detection.",
    )
    args = parser.parse_args()

    result = predict(
        args.audio_file,
        model_size=args.model_size,
        language=args.language,
    )
    print(json.dumps(result, indent=2))

