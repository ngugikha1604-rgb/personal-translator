from flask import Blueprint, request, jsonify
from services.stt import transcribe_audio
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

    try:
        transcript = transcribe_audio(audio_bytes, audio_file.filename or "audio.webm")
        conversation.add("other", transcript)
        return jsonify({"transcript": transcript})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
