"""
src/model.py
────────────
Whisper model management for the Speech-to-Text system.

Provides convenience wrappers around ``src/transcribe.py`` for loading,
caching, and saving Whisper model metadata to disk.

The Whisper weights themselves are managed by the ``openai-whisper`` library
(stored in ``~/.cache/whisper``).  This module saves a lightweight JSON
"model card" to ``models/model_card.json`` so the chosen configuration is
recorded alongside evaluation results.
"""

import os
import sys
import json
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402

logger = logging.getLogger(__name__)

MODEL_CARD_PATH = os.path.join(config.MODELS_DIR, "model_card.json")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(
    size: str = config.WHISPER_MODEL_SIZE,
    device: str = config.STT_DEVICE,
):
    """
    Load and return a Whisper model, using the module-level cache.

    Args:
        size:   Model size – ``"tiny"``, ``"base"``, ``"small"``,
                ``"medium"``, or ``"large"``.
        device: Inference device – ``"cpu"``, ``"cuda"``, or ``"mps"``.

    Returns:
        Loaded ``whisper.Whisper`` instance.
    """
    from src.transcribe import load_whisper_model

    return load_whisper_model(size=size, device=device)


# ---------------------------------------------------------------------------
# Model card persistence
# ---------------------------------------------------------------------------

def save_model_card(
    size: str = config.WHISPER_MODEL_SIZE,
    device: str = config.STT_DEVICE,
    extra: dict = None,
) -> None:
    """
    Write a JSON model card describing the current configuration.

    Args:
        size:   Whisper model size string.
        device: Inference device string.
        extra:  Optional dictionary of additional metadata to include.
    """
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    card = {
        "model_type":  "openai/whisper",
        "model_size":  size,
        "device":      device,
        "language":    config.STT_LANGUAGE,
        "sample_rate": config.SAMPLE_RATE,
    }
    if extra:
        card.update(extra)
    with open(MODEL_CARD_PATH, "w") as fh:
        json.dump(card, fh, indent=2)
    logger.info("Model card saved → %s", MODEL_CARD_PATH)


def load_model_card() -> dict:
    """
    Load the saved model card JSON.

    Returns:
        Model card dict, or an empty dict if the file does not exist.
    """
    if not os.path.exists(MODEL_CARD_PATH):
        return {}
    with open(MODEL_CARD_PATH) as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    save_model_card()
    card = load_model_card()
    logger.info("Model card: %s", json.dumps(card, indent=2))
    logger.info("Smoke-test passed.")

