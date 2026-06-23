from flask import Blueprint, request, jsonify

from services.conversation import conversation
from services.speech import speech_service

transcribe_bp = Blueprint("transcribe", __name__)


@transcribe_bp.route("/transcribe", methods=["POST"])
def transcribe():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files["audio"]
    audio_bytes = audio_file.read()

    if not audio_bytes:
        return jsonify({"error": "Empty audio file"}), 400

    try:
        speech = speech_service.transcribe_other_audio(
            audio_bytes,
            audio_file.filename or "audio.webm",
        )
        if speech.transcript:
            conversation.add_other(speech.transcript)

        return jsonify({"transcript": speech.transcript, "stt_ms": speech.stt_ms})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
