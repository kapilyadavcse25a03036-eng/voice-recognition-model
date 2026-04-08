# Voice Recognition Model

A complete **Speech-to-Text** voice recognition system that classifies audio into phoneme categories using a **Dense Neural Network (DNN)** trained on Common Voice dataset samples.

---

## Project Overview

This project demonstrates a full machine-learning pipeline for audio classification:

1. **Data preparation** – generate or download labelled audio samples.
2. **Feature extraction** – convert raw waveforms to MFCC feature vectors.
3. **Model training** – train a Dense Neural Network with dropout regularisation.
4. **Evaluation** – measure accuracy, precision, recall, F1-score, and confusion matrix.
5. **Inference** – classify new audio files in real time.
6. **Demo** – interactive script for end-to-end demonstration.

---

## Project Structure

```
voice-recognition-model/
├── config.py                  # Centralised configuration
├── requirements.txt           # Python dependencies
├── README.md
│
├── data/
│   ├── download_data.py       # Download / generate audio data
│   ├── raw/                   # Raw audio files (per-class sub-directories)
│   └── processed/             # Preprocessed pickle files
│
├── src/
│   ├── __init__.py
│   ├── preprocess.py          # Audio loading, normalisation, dataset splitting
│   ├── features.py            # MFCC & spectrogram feature extraction
│   ├── model.py               # Dense Neural Network architecture
│   ├── train.py               # Training pipeline
│   ├── evaluate.py            # Evaluation metrics & plots
│   └── inference.py           # Inference engine
│
├── scripts/
│   ├── train_pipeline.py      # End-to-end training script
│   └── demo.py                # Interactive demonstration
│
├── notebooks/
│   └── exploration.ipynb      # Jupyter notebook for exploration
│
├── models/                    # Saved model files (generated at runtime)
└── results/                   # Plots, metrics, training history (generated)
```

---

## Installation

### Prerequisites

- Python 3.9 or later
- pip

### Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** If you are on a machine without a GPU, TensorFlow will run on CPU.
> Training on the synthetic dataset takes only a few minutes on a modern CPU.

---

## Dataset

The project uses two data sources:

| Source | Description |
|--------|-------------|
| **Synthetic** | Programmatically generated audio clips with characteristic frequency profiles for each phoneme class. Always available, no download required. |
| **Common Voice** | Mozilla's open speech dataset. Real samples are downloaded automatically when available. |

### Phoneme Classes (10 categories)

| Index | Class | Description |
|-------|-------|-------------|
| 0 | silence | Background / silence |
| 1 | vowel_open | Open vowels (a, æ) |
| 2 | vowel_mid | Mid vowels (e, o) |
| 3 | vowel_close | Close vowels (i, u) |
| 4 | fricative | Fricatives (s, f, sh) |
| 5 | plosive | Plosives (p, b, t, d, k, g) |
| 6 | nasal | Nasals (m, n, ng) |
| 7 | affricate | Affricates (ch, j) |
| 8 | approximant | Approximants (r, l, w, y) |
| 9 | sibilant | Sibilants (z, zh) |

---

## How to Run

### 1. Full Training Pipeline (recommended)

Run the entire pipeline – data generation → preprocessing → training → evaluation:

```bash
python scripts/train_pipeline.py
```

Optional arguments:

```bash
python scripts/train_pipeline.py --epochs 100 --batch-size 32 --lr 0.001
python scripts/train_pipeline.py --skip-data        # skip data generation
python scripts/train_pipeline.py --skip-preprocess  # skip preprocessing
```

### 2. Step-by-Step

```bash
# Step 1 – Generate data
python data/download_data.py

# Step 2 – Preprocess audio
python src/preprocess.py

# Step 3 – Train model
python src/train.py

# Step 4 – Evaluate
python src/evaluate.py
```

### 3. Inference on a New Audio File

```bash
python src/inference.py path/to/audio.wav
```

### 4. Interactive Demo

```bash
python scripts/demo.py                         # auto demo with synthetic audio
python scripts/demo.py --audio my_audio.wav   # demo with your own file
python scripts/demo.py --interactive          # interactive mode
python scripts/demo.py --train                # force re-training before demo
```

### 5. Jupyter Notebook

```bash
jupyter notebook notebooks/exploration.ipynb
```

---

## Model Architecture

```
Input  (FEATURE_SIZE = N_MFCC x N_FRAMES = 40 x 94 = 3760)
  |
  v
Dense(256, ReLU) -> BatchNorm -> Dropout(0.3)
  |
  v
Dense(128, ReLU) -> BatchNorm -> Dropout(0.3)
  |
  v
Dense(64,  ReLU) -> BatchNorm -> Dropout(0.3)
  |
  v
Dense(10, Softmax)   <- output: probability over 10 phoneme classes
```

**Regularisation:** L2 weight decay (lambda = 1e-4) + Dropout (p = 0.3) + Batch Normalisation.

**Optimiser:** Adam (lr = 1e-3).

**Loss:** Sparse Categorical Cross-Entropy.

---

## Feature Extraction

1. Load audio at 16 kHz, mono.
2. Peak-normalise waveform to [-1, 1].
3. Pad / truncate to exactly 3 seconds (48 000 samples).
4. Compute **MFCC** with 40 coefficients, FFT size 2048, hop 512.
5. Pad / truncate MFCC matrix to 94 time frames.
6. Standardise (zero mean, unit variance) using a scaler fitted on training data.
7. Flatten to a 1-D vector of length **3 760**.

---

## Results & Performance Metrics

After training, the following files are generated in `results/`:

| File | Description |
|------|-------------|
| `training_curves.png` | Loss and accuracy over epochs |
| `confusion_matrix.png` | Per-class confusion matrix |
| `training_history.csv` | Epoch-by-epoch metrics |
| `metrics.json` | Final accuracy, precision, recall, F1-score |

Example `metrics.json`:

```json
{
  "training": {
    "final_train_accuracy": 0.92,
    "final_val_accuracy": 0.87,
    "epochs_trained": 45
  },
  "evaluation": {
    "accuracy": 0.85,
    "precision": 0.86,
    "recall": 0.85,
    "f1_score": 0.85
  }
}
```

> Actual results will vary depending on the dataset and training run.

---

## Configuration

All parameters are centralised in `config.py`:

```python
SAMPLE_RATE   = 16000       # Hz
DURATION      = 3.0         # seconds
N_MFCC        = 40          # MFCC coefficients
HIDDEN_UNITS  = [256, 128, 64]
DROPOUT_RATE  = 0.3
EPOCHS        = 100
BATCH_SIZE    = 32
LEARNING_RATE = 1e-3
```

Edit `config.py` to tune the model without touching source code.

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `librosa` | Audio loading and feature extraction |
| `numpy` | Numerical operations |
| `scipy` | Signal processing helpers |
| `scikit-learn` | Preprocessing, metrics, train/test split |
| `matplotlib` | Visualisations |
| `tensorflow` | Dense Neural Network |
| `pandas` | Data handling |
| `soundfile` | WAV file I/O |
| `tqdm` | Progress bars |

Install all with:

```bash
pip install -r requirements.txt
```

---

## Acknowledgements

- [Mozilla Common Voice](https://commonvoice.mozilla.org/) for the open speech dataset.
- [librosa](https://librosa.org/) for excellent audio processing utilities.
- [TensorFlow / Keras](https://www.tensorflow.org/) for the deep learning framework.
