"""
src/features.py
───────────────
Audio feature extraction utilities for visualisation and analysis.

These features are no longer used as classifier inputs (the STT model
operates directly on raw audio), but they remain useful for inspecting
audio quality and exploring the dataset.

Supported features
------------------
* **MFCC** (Mel-Frequency Cepstral Coefficients)
* **Log-Mel spectrogram**
* **Chroma** – pitch-class profile
"""

import os
import sys
import logging

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
        audio:      1-D float32 waveform.
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
# Standalone entry-point (visualisation smoke-test)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    rng = np.random.default_rng(0)
    dummy_audio = rng.uniform(-1, 1, config.SAMPLE_RATE * 3).astype(np.float32)

    mfcc = extract_mfcc(dummy_audio)
    logger.info("MFCC shape: %s  (expected: (%d, ~94))", mfcc.shape, config.N_MFCC)

    log_mel = extract_log_mel_spectrogram(dummy_audio)
    logger.info("Log-Mel shape: %s  (expected: (%d, ~94))", log_mel.shape, config.N_MELS)

    logger.info("Smoke-test passed.")

