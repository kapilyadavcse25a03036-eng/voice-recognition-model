"""
config.py - Centralized configuration for the Speech-to-Text voice recognition project.

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

# Processed data pickle paths
TRAIN_DATA_PATH = os.path.join(PROC_DATA_DIR, "train_data.pkl")
VAL_DATA_PATH   = os.path.join(PROC_DATA_DIR, "val_data.pkl")
TEST_DATA_PATH  = os.path.join(PROC_DATA_DIR, "test_data.pkl")

# Model output paths
MODEL_PATH         = os.path.join(MODELS_DIR, "voice_model.pkl")
TRAINING_PLOT      = os.path.join(RESULTS_DIR, "feature_importances.png")
CONFUSION_MATRIX   = os.path.join(RESULTS_DIR, "confusion_matrix.png")
METRICS_PATH       = os.path.join(RESULTS_DIR, "metrics.json")

# ─── Audio Parameters ─────────────────────────────────────────────────────────
SAMPLE_RATE    = 16000   # Hz – target sample rate for all audio files
DURATION       = 3.0     # seconds – clips shorter than this are padded, longer are truncated
N_SAMPLES      = int(SAMPLE_RATE * DURATION)  # total number of audio samples per clip

# ─── MFCC / Feature Parameters ────────────────────────────────────────────────
N_MFCC    = 40    # number of MFCC coefficients
N_FFT     = 2048  # FFT window size
HOP_LENGTH = 512  # frames between consecutive STFT columns
N_MELS    = 128   # number of Mel filterbank bins (used for spectrogram)
FMIN      = 0     # lowest frequency for Mel filterbank (Hz)
FMAX      = 8000  # highest frequency for Mel filterbank (Hz)

# Derived: number of time frames after STFT
N_FRAMES  = 1 + N_SAMPLES // HOP_LENGTH  # ~94 for 3 s at 16 kHz / 512 hop

# Flattened MFCC feature vector length (N_MFCC × N_FRAMES)
FEATURE_SIZE = N_MFCC * N_FRAMES

# ─── Dataset / Split Parameters ───────────────────────────────────────────────
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15
RANDOM_SEED = 42

# Phoneme categories (10 broad classes used for this demo)
PHONEME_CLASSES = [
    "silence",     # 0 – background / silence
    "vowel_open",  # 1 – open vowels  (a, æ)
    "vowel_mid",   # 2 – mid vowels   (e, o)
    "vowel_close", # 3 – close vowels (i, u)
    "fricative",   # 4 – fricatives   (s, f, sh)
    "plosive",     # 5 – plosives     (p, b, t, d, k, g)
    "nasal",       # 6 – nasals       (m, n, ng)
    "affricate",   # 7 – affricates   (ch, j)
    "approximant", # 8 – approximants (r, l, w, y)
    "sibilant",    # 9 – sibilants    (z, zh)
]
NUM_CLASSES = len(PHONEME_CLASSES)

# ─── Model Architecture ───────────────────────────────────────────────────────
N_ESTIMATORS   = 200    # number of trees in the Random Forest
MAX_DEPTH      = None   # maximum depth of each tree (None = unlimited)
MIN_SAMPLES_LEAF = 2    # minimum samples required to be at a leaf node

# ─── Training Parameters ──────────────────────────────────────────────────────
# (no epoch-based training for Random Forest)

# ─── Data Generation (synthetic demo) ────────────────────────────────────────
N_SYNTHETIC_SAMPLES = 500   # synthetic samples per class when no real data available
