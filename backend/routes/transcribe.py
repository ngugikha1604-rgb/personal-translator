import time
from flask import Blueprint, request, jsonify
from services.stt import transcribe_audio
from services.vad import has_speech
from services.conversation import conversation

transcribe_bp = Blueprint("transcribe", __name__)


@transcribe_bp.route("/transcribe", methods=["POST"])
def transcribe():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files["audio"]
    audio_bytes = audio_file.read()

    if not audio_bytes:
        return jsonify({"error": "Empty audio file"}), 400

    # VAD gate: skip chunks with no speech (save latency & cost)
    if not has_speech(audio_bytes):
        print("[VAD] Skipped — no speech detected")
        return jsonify({"transcript": "", "stt_ms": 0})

    try:
        t0 = time.perf_counter()
        transcript = transcribe_audio(audio_bytes, audio_file.filename or "audio.webm")
        stt_ms = (time.perf_counter() - t0) * 1000
        print(f"[LAT] STT: {stt_ms:.0f}ms | transcript: \"{transcript}\"")

        if transcript.strip():
            conversation.add("other", transcript)

        return jsonify({"transcript": transcript, "stt_ms": round(stt_ms)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
