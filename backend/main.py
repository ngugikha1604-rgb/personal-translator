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

import signal
import sys
import threading
import time

from config import GROQ_API_KEY
from services.audio import record_chunk, CHUNK_SECONDS
from services.conversation import conversation
from services.copilot import copilot_service
from services.display import display
from services.speech import speech_service, SpeechResult, SpeechStatus
from services.utterance_filter import classify_utterance
from services.memory import learn_from_session

_session_start = None


# ─── Startup validation ───────────────────────────────────────────────────────

def _validate() -> None:
    if not GROQ_API_KEY:
        print("Error: GROQ_API_KEY is not set. Check your .env file.")
        sys.exit(1)


# ─── Thread worker ────────────────────────────────────────────────────────────

def _process_other_turn(turn_n: int, speech: SpeechResult, turns: list) -> None:
    """Run LLM off capture loop so audio recording does not pause."""
    try:
        result = copilot_service.analyze_turns(turns)
        display.result(
            result.intent,
            result.reply,
            timing_ms=speech.stt_ms + result.llm_ms,
            tokens={
                "total_tokens": result.total_tokens,
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
            },
        )
        display.status("listening...")
    except Exception as e:
        display.error(f"Analysis failed: {e}")
        display.status("listening...")


# ─── Keyboard (push-to-mute) ──────────────────────────────────────────────────

def _setup_keyboard(muted: threading.Event) -> None:
    """Attach push-to-mute listener via pynput.

    Device target: hardware mute button or proximity sensor on glasses frame.
    """
    try:
        from pynput import keyboard

        def on_press(key):
            if key == keyboard.Key.space:
                if not muted.is_set():
                    display.status("muted — you're speaking")
                muted.set()
            elif hasattr(key, "char") and key.char == "q":
                display.session_summary(0, time.time() - (_session_start or time.time()))
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
    global _session_start
    _session_start = time.time()

    _validate()
    display.header()

    muted    = threading.Event()
    turn_n   = 0
    user_audio_chunks = []
    _setup_keyboard(muted)
    display.status("listening...")

    # Clean shutdown on SIGTERM
    def _handle_signal(signum, frame):
        elapsed = time.time() - _session_start
        display.session_summary(turn_n, elapsed)
        threading.Thread(
            target=learn_from_session,
            args=(list(conversation.get_all()),),
            daemon=True,
        ).start()
        sys.exit(0)
    signal.signal(signal.SIGTERM, _handle_signal)

    try:
        while True:
            if muted.is_set():
                # Keep chunking for speed, defer STT/verification until release.
                user_audio_chunks.append(record_chunk(CHUNK_SECONDS))
                continue

            if user_audio_chunks:
                speech = speech_service.transcribe_user_audio(
                    b"".join(user_audio_chunks),
                    "user_utterance.wav",
                )
                user_audio_chunks.clear()

                transcript = speech.transcript.strip()
                if transcript:
                    conversation.add_user(transcript)
                    try:
                        result = copilot_service.analyze_user_speech(
                            conversation.get_all())
                        display.verification(
                            result.understanding_correct,
                            result.warning,
                            result.llm_ms,
                        )
                    except Exception as e:
                        display.error(f"Verification failed: {e}")

                display.status("listening...")
                continue

            # Capture audio chunk from mic (Mic 1 — other person)
            audio_bytes = record_chunk(CHUNK_SECONDS)

            # STT with VAD gate
            speech = speech_service.transcribe_other_audio(audio_bytes, "chunk.wav")
            transcript = speech.transcript.strip()

            if speech.status == SpeechStatus.ERROR:
                continue
            if speech.status == SpeechStatus.SKIPPED:
                continue

            # Utterance filter
            classification = classify_utterance(transcript)
            if classification == "noise":
                continue
            if classification == "backchannel":
                conversation.add_other(transcript)
                continue

            display.status("processing...")
            turn_n += 1

            conversation.add_other(transcript)
            turns = conversation.get_all()
            threading.Thread(
                target=_process_other_turn,
                args=(turn_n, speech, turns),
                daemon=True,
            ).start()

    except KeyboardInterrupt:
        elapsed = time.time() - _session_start
        display.session_summary(turn_n, elapsed)
        threading.Thread(
            target=learn_from_session,
            args=(list(conversation.get_all()),),
            daemon=True,
        ).start()


if __name__ == "__main__":
    main()
