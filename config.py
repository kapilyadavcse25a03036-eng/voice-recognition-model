"""
config.py - Centralized configuration for the Speech-to-Text system.

All parameters are defined here for easy tuning without touching the source code.
"""

import os

# ─── Base Paths ───────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR       = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR   = os.path.join(DATA_DIR, "raw")
PROC_DATA_DIR  = os.path.join(DATA_DIR, "processed")
MODELS_DIR     = os.path.join(BASE_DIR, "models")
RESULTS_DIR    = os.path.join(BASE_DIR, "results")
NOTEBOOKS_DIR  = os.path.join(BASE_DIR, "notebooks")

# Dataset manifest (JSON list of {"audio": path, "text": transcript} objects)
MANIFEST_PATH = os.path.join(PROC_DATA_DIR, "manifest.json")

# Evaluation output paths
METRICS_PATH       = os.path.join(RESULTS_DIR, "metrics.json")
SPECTROGRAM_PLOT   = os.path.join(RESULTS_DIR, "spectrogram.png")

# ─── Audio Parameters ─────────────────────────────────────────────────────────
SAMPLE_RATE = 16000   # Hz – Whisper expects 16 kHz mono audio

# ─── MFCC / Spectrogram Parameters (used for visualisation) ──────────────────
N_MFCC     = 40    # number of MFCC coefficients
N_FFT      = 2048  # FFT window size
HOP_LENGTH = 512   # frames between consecutive STFT columns
N_MELS     = 128   # number of Mel filterbank bins
FMIN       = 0     # lowest frequency for Mel filterbank (Hz)
FMAX       = 8000  # highest frequency for Mel filterbank (Hz)

# ─── Whisper STT Model ────────────────────────────────────────────────────────
# Available sizes (smallest → largest): "tiny", "base", "small", "medium", "large"
# "base" gives a good speed/accuracy tradeoff for most use-cases.
WHISPER_MODEL_SIZE = "base"

# Target language for transcription.  Set to None for automatic language detection.
STT_LANGUAGE = "en"

# Device for Whisper inference: "cpu", "cuda", or "mps" (Apple Silicon).
STT_DEVICE = "cpu"

# ─── Dataset / Split Parameters ───────────────────────────────────────────────
RANDOM_SEED = 42

# Number of synthetic TTS samples to generate per sentence when no real data
# is available.
N_SYNTHETIC_SAMPLES = 20
