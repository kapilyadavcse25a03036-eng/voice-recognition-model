"""
scripts/demo.py
───────────────
Interactive demonstration of the Speech-to-Text system.

Scenarios
---------
1. **Transcribe a provided audio file** – supply ``--audio path/to/file.wav``.
2. **Auto demo** – if no file is given, a synthetic TTS clip is generated and
   transcribed (requires ``espeak`` or ``pyttsx3``).
3. **Interactive mode** – repeatedly prompts for an audio file path and
   prints the transcript.
4. **Feature visualisation** – saves MFCC and Log-Mel spectrogram plots to
   ``results/demo_features.png``.

Usage (from the project root):
    python scripts/demo.py                        # auto demo
    python scripts/demo.py --audio path/to/file.wav
    python scripts/demo.py --interactive
    python scripts/demo.py --model-size small
"""

import os
import sys
import logging
import argparse

import numpy as np

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
    Save MFCC and log-Mel spectrogram plots for an audio waveform.

    Args:
        audio: 1-D float32 waveform at ``config.SAMPLE_RATE`` Hz.
        title: Figure title prefix.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import librosa.display
    from src.features import extract_mfcc, extract_log_mel_spectrogram

    mfcc    = extract_mfcc(audio)
    log_mel = extract_log_mel_spectrogram(audio)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6))

    img1 = librosa.display.specshow(
        mfcc, sr=config.SAMPLE_RATE, hop_length=config.HOP_LENGTH,
        x_axis="time", ax=axes[0],
    )
    axes[0].set_title(f"{title} – MFCC ({config.N_MFCC} coefficients)")
    axes[0].set_ylabel("MFCC coefficient")
    plt.colorbar(img1, ax=axes[0], format="%+2.0f")

    img2 = librosa.display.specshow(
        log_mel, sr=config.SAMPLE_RATE, hop_length=config.HOP_LENGTH,
        x_axis="time", y_axis="mel", fmin=config.FMIN, fmax=config.FMAX,
        ax=axes[1],
    )
    axes[1].set_title(f"{title} – Log-Mel Spectrogram")
    axes[1].set_ylabel("Frequency (Hz)")
    plt.colorbar(img2, ax=axes[1], format="%+2.0f dB")

    plt.tight_layout()
    out_path = config.SPECTROGRAM_PLOT
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Feature visualisation saved → {out_path}")


# ---------------------------------------------------------------------------
# Core demo
# ---------------------------------------------------------------------------

def run_demo(
    audio_path: str = None,
    model_size: str = config.WHISPER_MODEL_SIZE,
) -> None:
    """
    Run the end-to-end STT demo.

    Args:
        audio_path:  Path to a WAV/MP3 file.  If ``None``, a synthetic clip
                     is generated automatically.
        model_size:  Whisper model size to use.
    """
    from src.preprocess import load_audio, normalise_audio
    from src.inference import predict

    # ── 1. Prepare audio ───────────────────────────────────────────────────
    if audio_path and os.path.exists(audio_path):
        print(f"\n[Demo] Transcribing: {audio_path}")
        audio = normalise_audio(load_audio(audio_path))
        target_path = audio_path
    else:
        if audio_path:
            print(f"  WARNING: {audio_path!r} not found – using synthetic audio.")

        target_path = os.path.join(config.RAW_DATA_DIR, "synthetic", "synthetic_0000.wav")
        os.makedirs(os.path.dirname(target_path), exist_ok=True)

        if not os.path.exists(target_path):
            print("\n[Demo] Generating synthetic TTS clip …")
            from data.download_data import _generate_tts_clip  # type: ignore
            _generate_tts_clip("the quick brown fox jumps over the lazy dog", target_path)

        print(f"\n[Demo] Transcribing synthetic clip: {target_path}")
        audio = normalise_audio(load_audio(target_path))

    # ── 2. Visualise features ──────────────────────────────────────────────
    print("\n[Demo] Extracting and visualising features …")
    visualise_features(audio, title="Demo Sample")

    # ── 3. Transcribe ─────────────────────────────────────────────────────
    print(f"\n[Demo] Running Whisper ({model_size}) transcription …")
    result = predict(target_path, model_size=model_size)

    print("\n" + "=" * 55)
    print("TRANSCRIPTION RESULT")
    print("=" * 55)
    print(f"  Text       : {result['text']}")
    print(f"  Language   : {result['language']}")
    print(f"  Confidence : {result['confidence']:.4f}")
    if result.get("segments"):
        print("\n  Timed segments:")
        for seg in result["segments"]:
            print(f"    [{seg['start']:6.2f}s – {seg['end']:6.2f}s]  {seg['text']}")
    print("=" * 55)


# ---------------------------------------------------------------------------
# Interactive mode
# ---------------------------------------------------------------------------

def interactive_mode(model_size: str = config.WHISPER_MODEL_SIZE) -> None:
    """Repeatedly prompt for an audio file path and print the transcript."""
    print("\n" + "=" * 60)
    print("SPEECH-TO-TEXT – INTERACTIVE DEMO")
    print("=" * 60)
    print("Enter the path to an audio file (WAV or MP3) to transcribe it.")
    print("Press Enter with no input to use a synthetic clip.")
    print("Type 'quit' to exit.\n")

    while True:
        try:
            user_input = input("Audio file path (or Enter for synthetic): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        run_demo(audio_path=user_input if user_input else None, model_size=model_size)
        print()


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Speech-to-Text demo script.")
    parser.add_argument(
        "--audio", default=None, help="Path to a WAV or MP3 audio file to transcribe."
    )
    parser.add_argument(
        "--model-size",
        default=config.WHISPER_MODEL_SIZE,
        help="Whisper model size (tiny/base/small/medium/large).",
    )
    parser.add_argument(
        "--interactive", "-i", action="store_true",
        help="Enter interactive mode to transcribe multiple files.",
    )
    args = parser.parse_args()

    if args.interactive:
        interactive_mode(model_size=args.model_size)
    else:
        run_demo(audio_path=args.audio, model_size=args.model_size)


if __name__ == "__main__":
    main()

