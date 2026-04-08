"""
src/preprocess.py
─────────────────
Audio loading and preprocessing utilities for the Speech-to-Text system.

Responsibilities
----------------
* Load WAV / MP3 files with librosa (resampled to ``config.SAMPLE_RATE``).
* Normalise waveforms to a consistent amplitude.
* Load and save JSON dataset manifests: lists of
  ``{"audio": path, "text": transcript}`` entries.
"""

import os
import sys
import json
import logging
from typing import List, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Low-level audio helpers
# ---------------------------------------------------------------------------

def load_audio(file_path: str, target_sr: int = config.SAMPLE_RATE) -> np.ndarray:
    """
    Load an audio file and return a mono waveform at *target_sr* Hz.

    Args:
        file_path: Path to a WAV or MP3 file.
        target_sr: Desired sample rate in Hz.

    Returns:
        1-D float32 numpy array of audio samples, or a silent array on error.
    """
    try:
        import librosa
        audio, _ = librosa.load(file_path, sr=target_sr, mono=True)
        return audio.astype(np.float32)
    except Exception as exc:
        logger.error("Failed to load %s: %s", file_path, exc)
        return np.zeros(target_sr, dtype=np.float32)


def normalise_audio(audio: np.ndarray) -> np.ndarray:
    """
    Peak-normalise an audio waveform to the range [-1, 1].

    Args:
        audio: Raw audio waveform (1-D numpy array).

    Returns:
        Peak-normalised waveform.  If the waveform is silent (all zeros),
        the original array is returned unchanged.
    """
    peak = np.max(np.abs(audio))
    if peak > 0.0:
        return audio / peak
    return audio


def preprocess_audio(file_path: str) -> np.ndarray:
    """
    Full preprocessing pipeline for a single audio file.

    Steps: load → normalise.

    Args:
        file_path: Path to the audio file.

    Returns:
        Peak-normalised float32 waveform at ``config.SAMPLE_RATE``.
    """
    audio = load_audio(file_path)
    return normalise_audio(audio)


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------

def load_manifest(path: str = config.MANIFEST_PATH) -> List[dict]:
    """
    Load a JSON dataset manifest.

    Each entry must have at least an ``"audio"`` key (path to audio file).
    An optional ``"text"`` key holds the ground-truth transcript.

    Args:
        path: Path to the manifest JSON file.

    Returns:
        List of entry dicts.

    Raises:
        FileNotFoundError: If *path* does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Manifest not found: {path}.  "
            "Run `python data/download_data.py` first."
        )
    with open(path) as fh:
        manifest = json.load(fh)
    logger.info("Loaded %d entries from %s", len(manifest), path)
    return manifest


def save_manifest(entries: List[dict], path: str = config.MANIFEST_PATH) -> None:
    """
    Save a list of entry dicts as a JSON manifest file.

    Args:
        entries: List of ``{"audio": path, "text": transcript}`` dicts.
        path:    Destination file path.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(entries, fh, indent=2)
    logger.info("Saved manifest with %d entries → %s", len(entries), path)


def filter_manifest(
    manifest: List[dict],
    require_text: bool = False,
) -> List[dict]:
    """
    Filter manifest entries to those whose audio files exist on disk.

    Args:
        manifest:     List of manifest entry dicts.
        require_text: If True, also drop entries that lack a ``"text"`` key.

    Returns:
        Filtered list.
    """
    filtered = []
    for entry in manifest:
        audio_path = entry.get("audio", "")
        if not os.path.exists(audio_path):
            logger.warning("Audio file missing, skipping: %s", audio_path)
            continue
        if require_text and not entry.get("text"):
            logger.warning("No transcript for %s, skipping.", audio_path)
            continue
        filtered.append(entry)
    logger.info(
        "Manifest filtered: %d / %d entries kept.", len(filtered), len(manifest)
    )
    return filtered


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    try:
        manifest = load_manifest()
        manifest = filter_manifest(manifest)
        logger.info("Ready to transcribe %d files.", len(manifest))
    except FileNotFoundError as exc:
        logger.error(str(exc))

