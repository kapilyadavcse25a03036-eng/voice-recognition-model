"""
src/preprocess.py
─────────────────
Audio loading, normalisation, and dataset splitting utilities.

Responsibilities
----------------
* Load WAV / MP3 files with librosa (resampled to ``config.SAMPLE_RATE``).
* Pad or truncate waveforms to a fixed length (``config.N_SAMPLES``).
* Discover labelled audio in ``data/raw/<class_name>/*.wav``.
* Split the dataset into train / validation / test sets.
* Persist the splits to pickle files for reproducible training.
"""

import os
import sys
import glob
import logging
import pickle
from typing import Tuple, List

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
        1-D float32 numpy array of audio samples, or an empty array on error.
    """
    try:
        import librosa
        audio, _ = librosa.load(file_path, sr=target_sr, mono=True)
        return audio.astype(np.float32)
    except Exception as exc:
        logger.error("Failed to load %s: %s", file_path, exc)
        return np.zeros(config.N_SAMPLES, dtype=np.float32)


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


def pad_or_truncate(audio: np.ndarray, n_samples: int = config.N_SAMPLES) -> np.ndarray:
    """
    Ensure the waveform has exactly *n_samples* samples.

    * If the audio is shorter  → zero-pad at the end.
    * If the audio is longer   → truncate (keep the beginning).

    Args:
        audio:    Input waveform.
        n_samples: Target length in samples.

    Returns:
        Fixed-length float32 numpy array.
    """
    if len(audio) >= n_samples:
        return audio[:n_samples]
    pad_width = n_samples - len(audio)
    return np.pad(audio, (0, pad_width), mode="constant")


def preprocess_audio(file_path: str) -> np.ndarray:
    """
    Full preprocessing pipeline for a single audio file.

    Steps: load → normalise → pad/truncate.

    Args:
        file_path: Path to the audio file.

    Returns:
        Fixed-length, peak-normalised float32 waveform.
    """
    audio = load_audio(file_path)
    audio = normalise_audio(audio)
    audio = pad_or_truncate(audio)
    return audio


# ---------------------------------------------------------------------------
# Dataset discovery
# ---------------------------------------------------------------------------

def discover_dataset(raw_dir: str = config.RAW_DATA_DIR) -> Tuple[List[str], List[int]]:
    """
    Walk ``raw_dir`` and build lists of file paths and integer labels.

    Expected directory layout::

        raw_dir/
        ├── silence/
        │   ├── sample_001.wav
        │   └── …
        ├── vowel_open/
        │   └── …
        └── …

    Only directories whose name appears in ``config.PHONEME_CLASSES`` are
    included.  Supported extensions: ``.wav``, ``.mp3``.

    Args:
        raw_dir: Path to the root directory that contains per-class folders.

    Returns:
        Tuple of (file_paths, labels) where labels are integer indices into
        ``config.PHONEME_CLASSES``.
    """
    file_paths: List[str] = []
    labels: List[int] = []

    for idx, class_name in enumerate(config.PHONEME_CLASSES):
        class_dir = os.path.join(raw_dir, class_name)
        if not os.path.isdir(class_dir):
            logger.warning("Class directory not found: %s", class_dir)
            continue
        patterns = [
            os.path.join(class_dir, "*.wav"),
            os.path.join(class_dir, "*.mp3"),
        ]
        found = []
        for pat in patterns:
            found.extend(glob.glob(pat))
        if not found:
            logger.warning("No audio files in %s", class_dir)
        file_paths.extend(found)
        labels.extend([idx] * len(found))
        logger.info("  Class %-12s (label %d): %d files", class_name, idx, len(found))

    logger.info("Total samples discovered: %d", len(file_paths))
    return file_paths, labels


# ---------------------------------------------------------------------------
# Train / validation / test split
# ---------------------------------------------------------------------------

def split_dataset(
    file_paths: List[str],
    labels: List[int],
    train_ratio: float = config.TRAIN_RATIO,
    val_ratio: float = config.VAL_RATIO,
    random_seed: int = config.RANDOM_SEED,
) -> Tuple:
    """
    Split file paths and labels into stratified train / val / test subsets.

    Args:
        file_paths:  List of audio file paths.
        labels:      Corresponding integer labels.
        train_ratio: Fraction of data for training.
        val_ratio:   Fraction of data for validation.
        random_seed: Random seed for reproducibility.

    Returns:
        Tuple of six lists:
        ``(train_paths, train_labels, val_paths, val_labels, test_paths, test_labels)``
    """
    from sklearn.model_selection import train_test_split

    test_ratio = 1.0 - train_ratio - val_ratio

    # First split: train vs (val + test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        file_paths,
        labels,
        test_size=(val_ratio + test_ratio),
        stratify=labels,
        random_state=random_seed,
    )

    # Second split: val vs test (from the temp pool)
    relative_test = test_ratio / (val_ratio + test_ratio)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=relative_test,
        stratify=y_temp,
        random_state=random_seed,
    )

    logger.info(
        "Split: train=%d  val=%d  test=%d", len(X_train), len(X_val), len(X_test)
    )
    return X_train, y_train, X_val, y_val, X_test, y_test


# ---------------------------------------------------------------------------
# Batch loading
# ---------------------------------------------------------------------------

def load_split(
    file_paths: List[str], labels: List[int]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load and preprocess every file in a split, returning feature-ready arrays.

    Args:
        file_paths: List of audio file paths.
        labels:     Corresponding integer labels.

    Returns:
        ``(X, y)`` where *X* has shape ``(N, config.N_SAMPLES)`` and
        *y* has shape ``(N,)`` (integer labels).
    """
    X = np.zeros((len(file_paths), config.N_SAMPLES), dtype=np.float32)
    y = np.array(labels, dtype=np.int32)
    for i, fp in enumerate(file_paths):
        X[i] = preprocess_audio(fp)
        if (i + 1) % 100 == 0:
            logger.info("  Loaded %d / %d files …", i + 1, len(file_paths))
    return X, y


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def save_split(data: Tuple, path: str) -> None:
    """
    Save a data split tuple to a pickle file.

    Args:
        data: Any pickle-serialisable object (typically ``(X, y)``).
        path: Destination file path.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(data, fh, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("Saved split → %s", path)


def load_split_from_file(path: str) -> Tuple:
    """
    Load a data split tuple from a pickle file.

    Args:
        path: Path to the pickle file.

    Returns:
        The deserialised object stored in the file.
    """
    with open(path, "rb") as fh:
        data = pickle.load(fh)
    logger.info("Loaded split ← %s", path)
    return data


# ---------------------------------------------------------------------------
# Convenience entry-point
# ---------------------------------------------------------------------------

def prepare_dataset(raw_dir: str = config.RAW_DATA_DIR) -> None:
    """
    Full dataset preparation pipeline:

    1. Discover labelled audio files.
    2. Split into train / val / test.
    3. Load & preprocess waveforms.
    4. Save to pickle files.

    Args:
        raw_dir: Directory containing per-class sub-directories.
    """
    logger.info("=== Dataset Preparation ===")
    file_paths, labels = discover_dataset(raw_dir)

    if not file_paths:
        raise RuntimeError(
            f"No audio files found in {raw_dir}.  "
            "Run `python data/download_data.py` first."
        )

    X_train_p, y_train, X_val_p, y_val, X_test_p, y_test = split_dataset(
        file_paths, labels
    )

    logger.info("Loading training data …")
    X_train, y_train = load_split(X_train_p, y_train)

    logger.info("Loading validation data …")
    X_val, y_val = load_split(X_val_p, y_val)

    logger.info("Loading test data …")
    X_test, y_test = load_split(X_test_p, y_test)

    save_split((X_train, y_train), config.TRAIN_DATA_PATH)
    save_split((X_val, y_val), config.VAL_DATA_PATH)
    save_split((X_test, y_test), config.TEST_DATA_PATH)

    logger.info("Dataset preparation complete.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    prepare_dataset()
