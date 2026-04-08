"""
scripts/demo.py
───────────────
Interactive demonstration of the voice recognition system.

The demo covers three scenarios:
  1. **Trained model** – if a saved model exists, load it and run inference on
     a supplied audio file (or a freshly generated synthetic clip).
  2. **Quick local training** – if no model is found, train a small model in
     ~30 seconds on synthetic data, then demonstrate inference.
  3. **Feature visualisation** – display MFCC and Mel-spectrogram plots for
     the sample audio.

Usage (from the project root):
    python scripts/demo.py                        # auto mode
    python scripts/demo.py --audio path/to/file.wav
    python scripts/demo.py --train               # force re-training
"""

import os
import sys
import logging
import argparse

import numpy as np

# Make the project root importable
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import config  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature visualisation
# ---------------------------------------------------------------------------

def visualise_features(audio: np.ndarray, title: str = "Sample Audio") -> None:
    """
    Display MFCC and log-Mel spectrogram for an audio waveform.

    Args:
        audio: 1-D float32 waveform (``config.N_SAMPLES`` samples).
        title: Figure title prefix.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.features import extract_mfcc, extract_log_mel_spectrogram

    mfcc     = extract_mfcc(audio)
    log_mel  = extract_log_mel_spectrogram(audio)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6))

    # MFCC
    import librosa.display
    img1 = librosa.display.specshow(
        mfcc, sr=config.SAMPLE_RATE, hop_length=config.HOP_LENGTH,
        x_axis="time", ax=axes[0],
    )
    axes[0].set_title(f"{title} – MFCC ({config.N_MFCC} coefficients)")
    axes[0].set_ylabel("MFCC coefficient")
    plt.colorbar(img1, ax=axes[0], format="%+2.0f")

    # Log-Mel spectrogram
    img2 = librosa.display.specshow(
        log_mel, sr=config.SAMPLE_RATE, hop_length=config.HOP_LENGTH,
        x_axis="time", y_axis="mel", fmin=config.FMIN, fmax=config.FMAX,
        ax=axes[1],
    )
    axes[1].set_title(f"{title} – Log-Mel Spectrogram")
    axes[1].set_ylabel("Frequency (Hz)")
    plt.colorbar(img2, ax=axes[1], format="%+2.0f dB")

    plt.tight_layout()
    out_path = os.path.join(config.RESULTS_DIR, "demo_features.png")
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Feature visualisation saved → {out_path}")


# ---------------------------------------------------------------------------
# Quick training on synthetic data (used when no model exists)
# ---------------------------------------------------------------------------

def quick_train() -> None:
    """
    Train a small DNN on synthetic data so the demo can run without a
    pre-existing model.  Takes ~30–90 s on a typical laptop CPU.
    """
    print("\n[Quick Training] Generating synthetic data and training a small model …")
    from data.download_data import generate_synthetic_dataset  # type: ignore
    from src.preprocess import prepare_dataset
    from src.train import train

    generate_synthetic_dataset()
    prepare_dataset()
    train(epochs=20, batch_size=32)
    print("[Quick Training] Done.\n")


# ---------------------------------------------------------------------------
# Core demo
# ---------------------------------------------------------------------------

def run_demo(audio_path: str = None, force_train: bool = False) -> None:
    """
    Run the end-to-end demo.

    Args:
        audio_path:  Optional path to a WAV/MP3 file to classify.
                     If None, a synthetic clip is generated automatically.
        force_train: If True, re-train the model even if one already exists.
    """
    from src.preprocess import preprocess_audio, normalise_audio, pad_or_truncate
    from src.features import mfcc_to_vector
    from src.inference import predict_from_array

    # ── 1. Ensure model exists ─────────────────────────────────────────────
    if force_train or not os.path.exists(config.MODEL_PATH):
        quick_train()

    if not os.path.exists(config.MODEL_PATH):
        print("ERROR: Model not found after training.  Aborting demo.")
        return

    # ── 2. Prepare sample audio ────────────────────────────────────────────
    if audio_path and os.path.exists(audio_path):
        print(f"\n[Demo] Using provided audio file: {audio_path}")
        audio = preprocess_audio(audio_path)
    else:
        if audio_path:
            print(f"  WARNING: {audio_path} not found – using synthetic audio.")
        print("\n[Demo] Generating synthetic audio sample (class: 'plosive') …")
        rng = np.random.default_rng(42)
        t   = np.linspace(0, config.DURATION, config.N_SAMPLES, endpoint=False)
        # Plosive-like signal: short burst of broadband noise + decay
        burst = np.zeros(config.N_SAMPLES, dtype=np.float32)
        burst[:int(0.05 * config.SAMPLE_RATE)] = rng.standard_normal(
            int(0.05 * config.SAMPLE_RATE)
        ).astype(np.float32)
        audio = burst

    audio = normalise_audio(audio)
    audio = pad_or_truncate(audio)

    # ── 3. Visualise features ──────────────────────────────────────────────
    print("\n[Demo] Extracting and visualising features …")
    visualise_features(audio, title="Demo Sample")

    # ── 4. Run inference ───────────────────────────────────────────────────
    print("\n[Demo] Running phoneme prediction …")
    result = predict_from_array(audio)

    print("\n" + "=" * 50)
    print("PREDICTION RESULT")
    print("=" * 50)
    print(f"  Predicted class : {result['predicted_class']}")
    print(f"  Confidence      : {result['confidence']:.2%}")
    print("\n  Class probabilities:")
    for cls, prob in sorted(result["all_probabilities"].items(),
                            key=lambda kv: -kv[1]):
        bar = "█" * int(prob * 30)
        print(f"    {cls:<15} {prob:.4f}  {bar}")
    print("=" * 50)


# ---------------------------------------------------------------------------
# Interactive mode
# ---------------------------------------------------------------------------

def interactive_mode() -> None:
    """
    Allow the user to provide audio files interactively via the terminal.
    """
    print("\n" + "=" * 60)
    print("VOICE RECOGNITION – INTERACTIVE DEMO")
    print("=" * 60)
    print("Type the path to an audio file (WAV or MP3) to classify it,")
    print("or press Enter to use a synthetic sample.  Type 'quit' to exit.\n")

    while True:
        try:
            user_input = input("Audio file path (or Enter for synthetic): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        run_demo(audio_path=user_input if user_input else None)
        print()


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Voice recognition demo script."
    )
    parser.add_argument(
        "--audio", default=None, help="Path to a WAV or MP3 audio file to classify."
    )
    parser.add_argument(
        "--train", action="store_true", help="Force re-training before demo."
    )
    parser.add_argument(
        "--interactive", "-i", action="store_true",
        help="Enter interactive mode to classify multiple files.",
    )
    args = parser.parse_args()

    if args.interactive:
        if not os.path.exists(config.MODEL_PATH):
            quick_train()
        interactive_mode()
    else:
        run_demo(audio_path=args.audio, force_train=args.train)


if __name__ == "__main__":
    main()
