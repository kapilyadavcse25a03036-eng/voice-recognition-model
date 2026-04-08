"""
src/transcribe.py
─────────────────
Core Speech-to-Text transcription engine powered by OpenAI Whisper.

Whisper is a general-purpose automatic speech recognition (ASR) model trained
on 680 000 hours of multilingual audio.  It runs fully offline once the model
weights have been downloaded on first use.

Public API
----------
``load_whisper_model(size, device)``
    Load (and cache) a Whisper model of the requested size.

``transcribe_file(audio_path, model)`` → dict
    Transcribe a WAV/MP3 file and return a result dict.

``transcribe_array(audio, sr, model)`` → dict
    Transcribe a raw numpy waveform (resampled to 16 kHz internally).

``batch_transcribe(paths, model)`` → list[dict]
    Transcribe multiple files and return a list of result dicts.

Result dict keys
----------------
* ``text``       – full transcription string (stripped).
* ``segments``   – list of timed segment dicts from Whisper.
* ``language``   – detected language code (e.g. "en").
* ``confidence`` – mean log-probability converted to a 0-1 score.
"""

import os
import sys
import logging
from typing import Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402

logger = logging.getLogger(__name__)

# Module-level model cache: maps (size, device) → whisper model object
_MODEL_CACHE: dict = {}


# ---------------------------------------------------------------------------
# Model management
# ---------------------------------------------------------------------------

def load_whisper_model(
    size: str = config.WHISPER_MODEL_SIZE,
    device: str = config.STT_DEVICE,
):
    """
    Load a Whisper model, returning a cached instance if already loaded.

    Models are downloaded automatically to ``~/.cache/whisper`` on first use.

    Args:
        size:   Model size string – one of ``"tiny"``, ``"base"``, ``"small"``,
                ``"medium"``, ``"large"``.  Smaller models are faster but less
                accurate.
        device: Inference device – ``"cpu"``, ``"cuda"``, or ``"mps"``.

    Returns:
        Loaded ``whisper.Whisper`` model instance.
    """
    cache_key = (size, device)
    if cache_key not in _MODEL_CACHE:
        try:
            import whisper
        except ImportError as exc:
            raise ImportError(
                "openai-whisper is required for transcription.  "
                "Install it with: pip install openai-whisper"
            ) from exc

        logger.info("Loading Whisper model '%s' on device '%s' …", size, device)
        model = whisper.load_model(size, device=device)
        _MODEL_CACHE[cache_key] = model
        logger.info("Whisper model loaded.")

    return _MODEL_CACHE[cache_key]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _avg_confidence(result: dict) -> float:
    """
    Compute a 0-1 confidence score from Whisper's avg_logprob values.

    Whisper reports ``avg_logprob`` per segment (always ≤ 0).  We convert the
    mean across segments to a probability-like score via ``exp``.

    Args:
        result: Raw Whisper result dict.

    Returns:
        Float in [0, 1].  Returns 0.0 if no segments are available.
    """
    import math

    segments = result.get("segments", [])
    if not segments:
        return 0.0
    avg_logprob = sum(s.get("avg_logprob", -1.0) for s in segments) / len(segments)
    return round(float(math.exp(avg_logprob)), 4)


def _format_result(raw: dict) -> dict:
    """
    Convert a raw Whisper output dict to the project's standard result format.

    Args:
        raw: Dict returned by ``whisper.model.transcribe()``.

    Returns:
        Cleaned result dict with keys ``text``, ``segments``, ``language``,
        ``confidence``.
    """
    segments = [
        {
            "id":    s.get("id", i),
            "start": round(s.get("start", 0.0), 3),
            "end":   round(s.get("end", 0.0), 3),
            "text":  s.get("text", "").strip(),
        }
        for i, s in enumerate(raw.get("segments", []))
    ]

    return {
        "text":       raw.get("text", "").strip(),
        "segments":   segments,
        "language":   raw.get("language", "unknown"),
        "confidence": _avg_confidence(raw),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def transcribe_file(
    audio_path: str,
    model=None,
    language: Optional[str] = config.STT_LANGUAGE,
) -> dict:
    """
    Transcribe a WAV or MP3 audio file to text using Whisper.

    Args:
        audio_path: Path to the audio file.
        model:      Pre-loaded Whisper model.  If ``None``, the default model
                    from ``config`` is loaded automatically.
        language:   Language hint (e.g. ``"en"``).  Pass ``None`` for
                    automatic detection.

    Returns:
        Result dict with keys ``text``, ``segments``, ``language``,
        ``confidence``.

    Raises:
        FileNotFoundError: If *audio_path* does not exist.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    if model is None:
        model = load_whisper_model()

    decode_options = {}
    if language:
        decode_options["language"] = language

    logger.debug("Transcribing %s …", audio_path)
    raw = model.transcribe(audio_path, **decode_options)
    return _format_result(raw)


def transcribe_array(
    audio: np.ndarray,
    sr: int = config.SAMPLE_RATE,
    model=None,
    language: Optional[str] = config.STT_LANGUAGE,
) -> dict:
    """
    Transcribe a raw numpy waveform to text.

    The waveform is resampled to 16 kHz if needed before being passed to
    Whisper.

    Args:
        audio:    1-D float32 numpy array of audio samples.
        sr:       Sample rate of *audio* in Hz.
        model:    Pre-loaded Whisper model.  Loaded automatically if ``None``.
        language: Language hint.  Pass ``None`` for automatic detection.

    Returns:
        Result dict with keys ``text``, ``segments``, ``language``,
        ``confidence``.
    """
    import whisper

    if model is None:
        model = load_whisper_model()

    # Whisper expects float32 mono at 16 000 Hz
    audio = np.asarray(audio, dtype=np.float32)
    if sr != 16000:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

    # Whisper's pad_or_trim expects exactly 30 s for the encoder; feeding shorter
    # audio directly via transcribe() handles this automatically.
    audio = whisper.pad_or_trim(audio)

    decode_options = {}
    if language:
        decode_options["language"] = language

    raw = model.transcribe(audio, **decode_options)
    return _format_result(raw)


def batch_transcribe(
    audio_paths: list,
    model=None,
    language: Optional[str] = config.STT_LANGUAGE,
) -> list:
    """
    Transcribe a list of audio files, returning one result dict per file.

    Args:
        audio_paths: List of paths to audio files.
        model:       Pre-loaded Whisper model.  Loaded once and reused.
        language:    Language hint applied to every file.

    Returns:
        List of result dicts (same order as *audio_paths*).  If a file cannot
        be transcribed, its entry will be ``{"error": "<message>"}``.
    """
    if model is None:
        model = load_whisper_model()

    results = []
    for i, path in enumerate(audio_paths, 1):
        logger.info("  [%d/%d] Transcribing %s …", i, len(audio_paths), path)
        try:
            results.append(transcribe_file(path, model=model, language=language))
        except Exception as exc:
            logger.error("Failed to transcribe %s: %s", path, exc)
            results.append({"error": str(exc)})

    return results


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Transcribe an audio file with Whisper.")
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

    whisper_model = load_whisper_model(size=args.model_size)
    result = transcribe_file(args.audio_file, model=whisper_model, language=args.language)
    print(json.dumps(result, indent=2))
