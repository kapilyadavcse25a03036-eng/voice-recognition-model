"""
data/download_data.py
─────────────────────
Prepare a labelled speech dataset for the Speech-to-Text system.

Two data sources are supported (tried in order):

1. **LibriSpeech test-clean** – small subset of publicly available read-speech
   audio clips with verified transcripts (downloaded automatically).
2. **Synthetic TTS** – short clips generated from a fixed sentence list using
   the system's ``espeak`` TTS engine (or ``pyttsx3`` if available).
   Always works without internet access.

Output
------
All audio files are stored under ``data/raw/`` and a JSON manifest is written
to ``data/processed/manifest.json`` with entries::

    {"audio": "/abs/path/to/file.wav", "text": "ground truth transcript"}

Usage (from the project root):
    python data/download_data.py
"""

import os
import sys
import json
import logging
import urllib.request
import zipfile
import io

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fixed sentence list used for synthetic TTS generation
# ---------------------------------------------------------------------------
SENTENCES = [
    "the quick brown fox jumps over the lazy dog",
    "speech recognition converts spoken language into text",
    "hello world this is a test of the system",
    "artificial intelligence is transforming every industry",
    "please speak clearly into the microphone",
    "the weather today is sunny with a chance of clouds",
    "machine learning models require large amounts of data",
    "open source software enables collaboration worldwide",
    "she sells seashells by the seashore",
    "how much wood would a woodchuck chuck",
    "to be or not to be that is the question",
    "all that glitters is not gold",
    "a journey of a thousand miles begins with a single step",
    "the early bird catches the worm",
    "actions speak louder than words",
    "every cloud has a silver lining",
    "time flies when you are having fun",
    "knowledge is power",
    "practice makes perfect",
    "where there is a will there is a way",
]

# ---------------------------------------------------------------------------
# LibriSpeech sample download
# ---------------------------------------------------------------------------

# A tiny publicly accessible subset of LibriSpeech test-clean
# (individual flac files from OpenSLR, no auth required).
LIBRISPEECH_SAMPLES = [
    {
        "url": "https://www.openslr.org/resources/12/test-clean/1089/134686/1089-134686-0000.flac",
        "filename": "libri_1089_134686_0000.flac",
        "text": "he hoped there would be stew for dinner turnips and carrots and bruised potatoes and fat mutton pieces to be ladled out in thick peppered flour fattened sauce",
    },
    {
        "url": "https://www.openslr.org/resources/12/test-clean/1089/134686/1089-134686-0001.flac",
        "filename": "libri_1089_134686_0001.flac",
        "text": "stuff it into you his belly counselled him",
    },
    {
        "url": "https://www.openslr.org/resources/12/test-clean/1089/134686/1089-134686-0002.flac",
        "filename": "libri_1089_134686_0002.flac",
        "text": "after early nightfall the yellow lamps would light up here and there the squalid quarter of the brothels",
    },
]


def _download_librispeech_samples(out_dir: str) -> list:
    """
    Attempt to download a handful of LibriSpeech test-clean samples.

    Args:
        out_dir: Directory to save downloaded files.

    Returns:
        List of ``{"audio": path, "text": transcript}`` dicts for
        successfully downloaded files.
    """
    os.makedirs(out_dir, exist_ok=True)
    entries = []
    for sample in LIBRISPEECH_SAMPLES:
        dest = os.path.join(out_dir, sample["filename"])
        if os.path.exists(dest):
            logger.info("Already have %s – skipping download.", sample["filename"])
            entries.append({"audio": dest, "text": sample["text"]})
            continue
        try:
            logger.info("Downloading %s …", sample["filename"])
            urllib.request.urlretrieve(sample["url"], dest)
            entries.append({"audio": dest, "text": sample["text"]})
            logger.info("  OK → %s", dest)
        except Exception as exc:
            logger.warning("Could not download %s: %s", sample["url"], exc)
    return entries


# ---------------------------------------------------------------------------
# Synthetic TTS generation
# ---------------------------------------------------------------------------

def _generate_tts_clip(text: str, out_path: str) -> bool:
    """
    Generate a WAV clip for *text* using an available TTS engine.

    Tries ``pyttsx3`` first (cross-platform, offline), then falls back to
    ``espeak`` (Linux).  If neither is available, generates a simple sine-wave
    placeholder and logs a warning.

    Args:
        text:     The sentence to synthesise.
        out_path: Destination WAV file path.

    Returns:
        True if a real TTS clip was generated, False for a placeholder.
    """
    # ── pyttsx3 (cross-platform) ─────────────────────────────────────────────
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty("rate", 150)
        engine.save_to_file(text, out_path)
        engine.runAndWait()
        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            return True
    except Exception:
        pass

    # ── espeak (Linux) ────────────────────────────────────────────────────────
    try:
        ret = os.system(
            f'espeak -s 150 -w "{out_path}" "{text}" 2>/dev/null'
        )
        if ret == 0 and os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            return True
    except Exception:
        pass

    # ── Placeholder sine wave ─────────────────────────────────────────────────
    logger.warning(
        "No TTS engine found – generating placeholder audio for: %s", text
    )
    sr = config.SAMPLE_RATE
    duration = max(1.0, len(text.split()) * 0.4)
    t = np.linspace(0.0, duration, int(sr * duration), endpoint=False)
    audio = (0.5 * np.sin(2.0 * np.pi * 220.0 * t)).astype(np.float32)
    try:
        import soundfile as sf
        sf.write(out_path, audio, sr)
    except Exception:
        import scipy.io.wavfile as wav
        wav.write(out_path, sr, (audio * 32767).astype(np.int16))
    return False


def generate_synthetic_dataset(out_dir: str = None) -> list:
    """
    Generate TTS audio clips for every sentence in ``SENTENCES``.

    Args:
        out_dir: Directory to store generated WAV files.
                 Defaults to ``data/raw/synthetic/``.

    Returns:
        List of ``{"audio": path, "text": transcript}`` dicts.
    """
    if out_dir is None:
        out_dir = os.path.join(config.RAW_DATA_DIR, "synthetic")
    os.makedirs(out_dir, exist_ok=True)

    entries = []
    for i, sentence in enumerate(SENTENCES):
        fname = f"synthetic_{i:04d}.wav"
        out_path = os.path.join(out_dir, fname)
        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            logger.info("  Already have %s – skipping.", fname)
        else:
            _generate_tts_clip(sentence, out_path)
            logger.info("  Generated %s", fname)
        entries.append({"audio": out_path, "text": sentence})

    logger.info("Synthetic dataset: %d clips in %s", len(entries), out_dir)
    return entries


# ---------------------------------------------------------------------------
# Main entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Build the dataset manifest.

    1. Try to download LibriSpeech samples.
    2. Generate synthetic TTS clips to fill any gaps.
    3. Write the combined manifest to ``data/processed/manifest.json``.
    """
    os.makedirs(config.RAW_DATA_DIR, exist_ok=True)
    os.makedirs(config.PROC_DATA_DIR, exist_ok=True)

    entries = []

    # ── Real speech (LibriSpeech) ─────────────────────────────────────────────
    libri_dir = os.path.join(config.RAW_DATA_DIR, "librispeech")
    logger.info("Attempting LibriSpeech sample download …")
    real_entries = _download_librispeech_samples(libri_dir)
    if real_entries:
        logger.info("Downloaded %d real speech samples.", len(real_entries))
        entries.extend(real_entries)
    else:
        logger.info("LibriSpeech download skipped (no internet or already cached).")

    # ── Synthetic TTS ─────────────────────────────────────────────────────────
    logger.info("Generating synthetic TTS dataset …")
    synthetic_entries = generate_synthetic_dataset()
    entries.extend(synthetic_entries)

    # ── Write manifest ────────────────────────────────────────────────────────
    manifest_path = config.MANIFEST_PATH
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    with open(manifest_path, "w") as fh:
        json.dump(entries, fh, indent=2)

    logger.info(
        "Manifest written: %d entries → %s", len(entries), manifest_path
    )
    logger.info("Run `python src/evaluate.py` to evaluate the STT system.")


if __name__ == "__main__":
    main()

