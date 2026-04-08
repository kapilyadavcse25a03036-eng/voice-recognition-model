"""
src/features.py
───────────────
Feature extraction utilities for audio-based phoneme classification.

Supported features
------------------
* **MFCC** (Mel-Frequency Cepstral Coefficients) – primary feature used by
  the Dense Neural Network.
* **Log-Mel spectrogram** – useful for visual exploration and CNN-based
  models.
* **Chroma** – pitch-class profile (supplementary).

All feature arrays are normalised (zero-mean, unit-variance per feature
dimension) and returned as fixed-length 1-D vectors ready for the DNN input.
"""

import os
import sys
import logging
from typing import Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Raw feature extraction
# ---------------------------------------------------------------------------

def extract_mfcc(
    audio: np.ndarray,
    sr: int = config.SAMPLE_RATE,
    n_mfcc: int = config.N_MFCC,
    n_fft: int = config.N_FFT,
    hop_length: int = config.HOP_LENGTH,
) -> np.ndarray:
    """
    Compute MFCC features for a waveform.

    Args:
        audio:      1-D float32 waveform (fixed length ``config.N_SAMPLES``).
        sr:         Sample rate in Hz.
        n_mfcc:     Number of MFCC coefficients.
        n_fft:      FFT window size.
        hop_length: Hop length between STFT frames.

    Returns:
        2-D array of shape ``(n_mfcc, n_frames)``.
    """
    import librosa

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
    )
    return mfcc.astype(np.float32)


def extract_log_mel_spectrogram(
    audio: np.ndarray,
    sr: int = config.SAMPLE_RATE,
    n_mels: int = config.N_MELS,
    n_fft: int = config.N_FFT,
    hop_length: int = config.HOP_LENGTH,
    fmin: float = config.FMIN,
    fmax: float = config.FMAX,
) -> np.ndarray:
    """
    Compute a log-power Mel spectrogram.

    Args:
        audio:      1-D float32 waveform.
        sr:         Sample rate in Hz.
        n_mels:     Number of Mel frequency bins.
        n_fft:      FFT window size.
        hop_length: Hop length between STFT frames.
        fmin:       Minimum frequency (Hz).
        fmax:       Maximum frequency (Hz).

    Returns:
        2-D array of shape ``(n_mels, n_frames)`` in dB scale.
    """
    import librosa

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        fmin=fmin,
        fmax=fmax,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return log_mel.astype(np.float32)


def extract_chroma(
    audio: np.ndarray,
    sr: int = config.SAMPLE_RATE,
    n_fft: int = config.N_FFT,
    hop_length: int = config.HOP_LENGTH,
) -> np.ndarray:
    """
    Compute chroma (pitch-class profile) features.

    Args:
        audio:      1-D float32 waveform.
        sr:         Sample rate in Hz.
        n_fft:      FFT window size.
        hop_length: Hop length between STFT frames.

    Returns:
        2-D array of shape ``(12, n_frames)``.
    """
    import librosa

    chroma = librosa.feature.chroma_stft(
        y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length
    )
    return chroma.astype(np.float32)


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def normalise_features(features: np.ndarray) -> np.ndarray:
    """
    Standardise features to zero mean and unit variance along axis 0.

    Args:
        features: 2-D array ``(feature_dim, time_frames)`` or 1-D vector.

    Returns:
        Normalised array with the same shape as *features*.
    """
    mean = features.mean()
    std  = features.std()
    if std > 0:
        return (features - mean) / std
    return features - mean


# ---------------------------------------------------------------------------
# Feature → flat vector
# ---------------------------------------------------------------------------

def mfcc_to_vector(
    audio: np.ndarray,
    n_mfcc: int = config.N_MFCC,
    n_frames: int = config.N_FRAMES,
    normalise: bool = True,
) -> np.ndarray:
    """
    Extract MFCC features and return a fixed-length 1-D vector.

    The 2-D MFCC array ``(n_mfcc, n_frames)`` is padded/truncated along the
    time axis to ensure a consistent shape before flattening.

    Args:
        audio:     Fixed-length 1-D waveform (``config.N_SAMPLES`` samples).
        n_mfcc:    Number of MFCC coefficients.
        n_frames:  Target number of time frames.
        normalise: Whether to standardise the features.

    Returns:
        1-D float32 vector of length ``n_mfcc * n_frames``.
    """
    mfcc = extract_mfcc(audio, n_mfcc=n_mfcc)

    # Pad or truncate along the time axis
    if mfcc.shape[1] < n_frames:
        pad_width = n_frames - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode="constant")
    else:
        mfcc = mfcc[:, :n_frames]

    if normalise:
        mfcc = normalise_features(mfcc)

    return mfcc.flatten().astype(np.float32)


# ---------------------------------------------------------------------------
# Batch feature extraction
# ---------------------------------------------------------------------------

def extract_features_batch(
    X_raw: np.ndarray,
    n_mfcc: int = config.N_MFCC,
    n_frames: int = config.N_FRAMES,
) -> np.ndarray:
    """
    Extract MFCC feature vectors for a batch of waveforms.

    Args:
        X_raw:   2-D array of shape ``(N, config.N_SAMPLES)`` (raw waveforms).
        n_mfcc:  Number of MFCC coefficients.
        n_frames: Target number of time frames.

    Returns:
        2-D float32 array of shape ``(N, n_mfcc * n_frames)``.
    """
    feature_size = n_mfcc * n_frames
    X_feat = np.zeros((len(X_raw), feature_size), dtype=np.float32)

    for i, audio in enumerate(X_raw):
        X_feat[i] = mfcc_to_vector(audio, n_mfcc=n_mfcc, n_frames=n_frames)
        if (i + 1) % 100 == 0:
            logger.info("  Features extracted: %d / %d", i + 1, len(X_raw))

    logger.info("Feature extraction complete: shape %s", X_feat.shape)
    return X_feat


# ---------------------------------------------------------------------------
# Global scaler (fit on train, apply to val / test)
# ---------------------------------------------------------------------------

def fit_scaler(X_train: np.ndarray):
    """
    Fit a StandardScaler on the training feature matrix.

    Args:
        X_train: 2-D training feature array ``(N_train, feature_size)``.

    Returns:
        Fitted ``sklearn.preprocessing.StandardScaler``.
    """
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler


def apply_scaler(X: np.ndarray, scaler) -> np.ndarray:
    """
    Transform a feature matrix using a pre-fitted scaler.

    Args:
        X:      Feature matrix to transform.
        scaler: Fitted ``sklearn.preprocessing.StandardScaler``.

    Returns:
        Scaled feature matrix (same shape as *X*).
    """
    return scaler.transform(X).astype(np.float32)


# ---------------------------------------------------------------------------
# Standalone entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Quick smoke-test with a random signal
    rng = np.random.default_rng(0)
    dummy_audio = rng.uniform(-1, 1, config.N_SAMPLES).astype(np.float32)

    vec = mfcc_to_vector(dummy_audio)
    logger.info("MFCC vector shape: %s", vec.shape)
    logger.info("Expected:          (%d,)", config.FEATURE_SIZE)
    assert vec.shape == (config.FEATURE_SIZE,), "Feature size mismatch!"
    logger.info("Smoke-test passed.")
