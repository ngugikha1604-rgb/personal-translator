"""
main.py — Conversation Copilot entry point.

Run:
    cd backend
    python main.py

Controls:
    SPACE (hold)  →  mute while you speak
    Q             →  quit
    Ctrl+C        →  quit
"""

import sys
import threading
import time

from config import GROQ_API_KEY
from services.audio import record_chunk, CHUNK_SECONDS
from services.conversation import conversation
from services.copilot import copilot_service
from services.display import display
from services.speech import speech_service


# ─── Startup validation ───────────────────────────────────────────────────────

def _validate() -> None:
    if not GROQ_API_KEY:
        print("Error: GROQ_API_KEY is not set. Check your .env file.")
        sys.exit(1)


# ─── Keyboard (push-to-mute) ──────────────────────────────────────────────────

def _setup_keyboard(muted: threading.Event) -> None:
    """
    Attach push-to-mute listener via pynput.
    Silently degraded if pynput is not installed — Ctrl+C to quit instead.

    Device target: hardware mute button or proximity sensor on glasses frame.
    """
    try:
        from pynput import keyboard

        def on_press(key):
            if key == keyboard.Key.space:
                muted.set()
                display.status("muted — you're speaking")
            elif hasattr(key, "char") and key.char == "q":
                display.status("Quit.")
                sys.exit(0)

        def on_release(key):
            if key == keyboard.Key.space:
                muted.clear()
                display.status("listening...")

        keyboard.Listener(
            on_press=on_press,
            on_release=on_release,
            daemon=True,
        ).start()

    except ImportError:
        display.status("pynput not installed — push-to-mute disabled. Ctrl+C to quit.")


# ─── Main loop ────────────────────────────────────────────────────────────────

def main() -> None:
    _validate()
    display.header()

    muted = threading.Event()
    _setup_keyboard(muted)
    display.status("listening...")

    try:
        while True:
            # Push-to-mute: skip chunk if user is speaking
            if muted.is_set():
                time.sleep(0.05)
                continue

            # Capture audio chunk from mic (Mic 1 — other person)
            audio_bytes = record_chunk(CHUNK_SECONDS)

            # STT with VAD gate — skips silent chunks automatically
            speech = speech_service.transcribe_other_audio(audio_bytes, "chunk.wav")
            if speech.skipped or not speech.transcript:
                continue

            display.status("processing...")

            # Add transcript to conversation, run LLM analysis
            conversation.add_other(speech.transcript)
            try:
                result = copilot_service.analyze_turns(conversation.get_all())
                display.result(result.intent, result.reply)
            except Exception as e:
                display.error(f"Analysis failed: {e}")

            display.status("listening...")

    except KeyboardInterrupt:
        display.status("Session ended.")


if __name__ == "__main__":
    main()
