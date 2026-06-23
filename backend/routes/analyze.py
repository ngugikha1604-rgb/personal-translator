import json
from flask import Blueprint, Response, request, stream_with_context
from services.llm import stream_analysis
from services.conversation import conversation

analyze_bp = Blueprint("analyze", __name__)


@analyze_bp.route("/analyze", methods=["GET"])
def analyze():
    turns = conversation.get_all()

    def generate():
        if not turns:
            yield f"data: {json.dumps({'error': 'No conversation yet'})}\n\n"
            return

        full_text = ""
        try:
            for token in stream_analysis(turns):
                full_text += token
                yield f"data: {json.dumps({'token': token})}\n\n"
            yield f"data: {json.dumps({'done': True, 'full': full_text})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        }
    )


@analyze_bp.route("/log_user", methods=["POST"])
def log_user():
    """Log the user's spoken response to maintain conversation context."""
    data = request.get_json(silent=True) or {}
    text = data.get("text", "").strip()
    if text:
        conversation.add("user", text)
    return json.dumps({"status": "ok"}), 200, {"Content-Type": "application/json"}


@analyze_bp.route("/clear", methods=["POST"])
def clear():
    conversation.clear()
    return json.dumps({"status": "cleared"}), 200, {"Content-Type": "application/json"}

