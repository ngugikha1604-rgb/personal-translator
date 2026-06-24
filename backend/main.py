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
from services.copilot import copilot_service, CopilotResult
from services.display import display
from services.speech import speech_service, SpeechResult

DIM   = "\033[90m"
RESET = "\033[0m"


# ─── Startup validation ───────────────────────────────────────────────────────

def _validate() -> None:
    if not GROQ_API_KEY:
        print("Error: GROQ_API_KEY is not set. Check your .env file.")
        sys.exit(1)


# ─── Logging ─────────────────────────────────────────────────────────────────

def _log_turn(
    turn_n: int,
    speech: SpeechResult,
    result: CopilotResult,
) -> None:
    """Structured per-turn metrics log."""
    total_ms = speech.stt_ms + result.llm_ms
    tokens   = (
        f"{result.prompt_tokens}p + {result.completion_tokens}c = {result.total_tokens} tokens"
        if result.total_tokens > 0 else "tokens: n/a (upgrade groq sdk)"
    )

    print(
        f"\n{DIM}"
        f"[TURN {turn_n}]\n"
        f"  stt    {speech.stt_ms:>5}ms   \"{speech.transcript}\"\n"
        f"  llm    {result.llm_ms:>5}ms   ttft: {result.ttft_ms}ms  |  {tokens}\n"
        f"  total  {total_ms:>5}ms"
        f"{RESET}"
    )


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
                if not muted.is_set():          # only print on first press, not on repeat
                    display.status("muted — you're speaking")
                muted.set()
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

    muted    = threading.Event()
    turn_n   = 0
    _setup_keyboard(muted)
    display.status("listening...")

    try:
        while True:
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
            turn_n += 1

            # Add transcript to conversation, run LLM analysis
            conversation.add_other(speech.transcript)
            try:
                result = copilot_service.analyze_turns(conversation.get_all())
                _log_turn(turn_n, speech, result)
                display.result(result.intent, result.reply)
            except Exception as e:
                display.error(f"Analysis failed: {e}")

            display.status("listening...")

    except KeyboardInterrupt:
        display.status(f"Session ended. {turn_n} turns.")


if __name__ == "__main__":
    main()
