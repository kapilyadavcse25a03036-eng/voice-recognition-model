# Voice Recognition – Speech-to-Text System

A complete **Speech-to-Text (STT)** system that transcribes spoken audio into
text using [OpenAI Whisper](https://github.com/openai/whisper) – a
state-of-the-art transformer model trained on 680 000 hours of multilingual
speech.

---

## Project Overview

This project delivers a full end-to-end speech-to-text pipeline:

1. **Data preparation** – download LibriSpeech samples and/or generate
   synthetic TTS clips; write a JSON manifest.
2. **Transcription** – run Whisper inference on any WAV/MP3 file.
3. **Evaluation** – compute **WER** (Word Error Rate) and **CER**
   (Character Error Rate) on a labelled dataset.
4. **Inference** – transcribe new audio files from the command line or via
   the Python API.
5. **Demo** – interactive script for end-to-end demonstration.

---

## Project Structure

```
voice-recognition-model/
├── config.py                  # Centralised configuration
├── requirements.txt           # Python dependencies
├── README.md
│
├── data/
│   ├── download_data.py       # Download / generate labelled audio dataset
│   ├── raw/                   # Raw audio files
│   └── processed/
│       └── manifest.json      # {"audio": path, "text": transcript} entries
│
├── src/
│   ├── __init__.py
│   ├── transcribe.py          # Core Whisper transcription engine
│   ├── preprocess.py          # Audio loading, normalisation, manifest helpers
│   ├── features.py            # MFCC & spectrogram extraction (visualisation)
│   ├── model.py               # Whisper model management & model card
│   ├── train.py               # STT evaluation / pipeline orchestration
│   ├── evaluate.py            # WER / CER evaluation pipeline
│   └── inference.py           # Public inference API
│
├── scripts/
│   ├── train_pipeline.py      # End-to-end STT pipeline script
│   └── demo.py                # Interactive transcription demo
│
├── notebooks/
│   └── exploration.ipynb      # Jupyter notebook for exploration
│
├── models/                    # Model card JSON (generated at runtime)
└── results/                   # Metrics JSON, spectrograms (generated)
```

---

## Installation

### Prerequisites

- Python 3.9 or later
- pip
- *(Optional)* `espeak` or `pyttsx3` for synthetic TTS data generation
- *(Optional)* CUDA GPU for faster Whisper inference

### Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `torch` is a large dependency (~2 GB).  Whisper weights are
> downloaded automatically (~140 MB for the `base` model) on first use to
> `~/.cache/whisper`.

---

## Dataset

Two data sources are supported:

| Source | Description |
|--------|-------------|
| **LibriSpeech test-clean** | Freely available read-speech clips with verified transcripts (downloaded automatically). |
| **Synthetic TTS** | Short clips generated from a fixed sentence list via `espeak` or `pyttsx3`.  Always works offline. |

The manifest format (`data/processed/manifest.json`):

```json
[
  {"audio": "/abs/path/to/clip.wav", "text": "the ground truth transcript"},
  ...
]
```

---

## How to Run

### 1. Full Pipeline (recommended)

Data preparation → STT evaluation → WER/CER report:

```bash
python scripts/train_pipeline.py
```

Optional arguments:

```bash
python scripts/train_pipeline.py --skip-data          # skip dataset preparation
python scripts/train_pipeline.py --model-size small   # use a larger model
python scripts/train_pipeline.py --language auto      # auto-detect language
```

### 2. Step-by-Step

```bash
# Step 1 – Prepare dataset (downloads + synthetic TTS)
python data/download_data.py

# Step 2 – Evaluate STT on the manifest
python src/evaluate.py

# Or run the full pipeline from src/train.py
python src/train.py
```

### 3. Transcribe a Single Audio File

```bash
python src/inference.py path/to/audio.wav
python src/inference.py path/to/audio.wav --model-size small --language en
```

### 4. Interactive Demo

```bash
python scripts/demo.py                          # auto demo with synthetic audio
python scripts/demo.py --audio my_audio.wav    # transcribe your own file
python scripts/demo.py --interactive            # interactive mode
python scripts/demo.py --model-size small      # use a more accurate model
```

### 5. Jupyter Notebook

```bash
jupyter notebook notebooks/exploration.ipynb
```

---

## Model

The transcription engine is **OpenAI Whisper**, a transformer encoder-decoder
trained with weak supervision on 680 000 hours of diverse audio.

| Model Size | Parameters | Relative Speed | Recommended Use |
|------------|-----------|----------------|-----------------|
| `tiny`     | 39 M      | ~10×           | Quick tests, low-resource devices |
| `base`     | 74 M      | ~7×            | **Default** – good accuracy/speed tradeoff |
| `small`    | 244 M     | ~4×            | Better accuracy |
| `medium`   | 769 M     | ~2×            | High accuracy |
| `large`    | 1 550 M   | 1×             | Best accuracy |

Configure the model size in `config.py`:

```python
WHISPER_MODEL_SIZE = "base"   # change to "small", "medium", etc.
STT_LANGUAGE       = "en"     # set to None for automatic language detection
STT_DEVICE         = "cpu"    # change to "cuda" for GPU inference
```

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **WER** | Word Error Rate – `(substitutions + deletions + insertions) / reference_words` |
| **CER** | Character Error Rate – same formula at character level |
| **Avg Confidence** | Mean Whisper segment log-probability converted to [0, 1] |

Results are saved to `results/metrics.json`:

```json
{
  "evaluation": {
    "wer": 0.08,
    "cer": 0.04,
    "avg_confidence": 0.72,
    "num_samples": 23,
    "num_successful": 23
  }
}
```

> Actual results depend on audio quality, background noise, and model size.
> The `base` model typically achieves WER < 10 % on clean English speech.

---

## Configuration

All parameters are centralised in `config.py`:

```python
WHISPER_MODEL_SIZE = "base"   # Whisper model size
STT_LANGUAGE       = "en"     # transcription language (None = auto-detect)
STT_DEVICE         = "cpu"    # inference device
SAMPLE_RATE        = 16000    # Hz – Whisper expects 16 kHz audio
```

---

## Python API

```python
from src.inference import predict, predict_from_array, batch_predict

# Transcribe a file
result = predict("audio.wav")
print(result["text"])        # "hello world"
print(result["confidence"])  # 0.85

# Transcribe a numpy array
import numpy as np
audio = np.zeros(16000, dtype=np.float32)   # 1 s of silence
result = predict_from_array(audio, sr=16000)

# Batch transcription
results = batch_predict(["clip1.wav", "clip2.wav"])
```

Result dict keys:

| Key | Type | Description |
|-----|------|-------------|
| `text` | str | Full transcription |
| `segments` | list | Timed segments `[{start, end, text}]` |
| `language` | str | Detected language code (e.g. `"en"`) |
| `confidence` | float | Mean segment confidence (0–1) |

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `openai-whisper` | Pre-trained STT model (Whisper) |
| `torch` | PyTorch – required by Whisper |
| `librosa` | Audio loading and feature extraction |
| `numpy` | Numerical operations |
| `scipy` | Signal processing helpers |
| `scikit-learn` | Evaluation utilities |
| `matplotlib` | Visualisations |
| `soundfile` | WAV file I/O |

Install all with:

```bash
pip install -r requirements.txt
```

---

## Acknowledgements

- [OpenAI Whisper](https://github.com/openai/whisper) for the pre-trained STT model.
- [LibriSpeech](https://www.openslr.org/12/) for the open speech corpus.
- [librosa](https://librosa.org/) for excellent audio processing utilities.

