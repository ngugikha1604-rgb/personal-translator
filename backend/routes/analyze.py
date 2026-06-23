import json
import time
from flask import Blueprint, Response, request, stream_with_context
from services.llm import stream_analysis
from services.conversation import conversation
from services.context import context_manager
from services.stt import transcribe_audio

analyze_bp = Blueprint("analyze", __name__)


@analyze_bp.route("/analyze", methods=["GET"])
def analyze():
    turns = conversation.get_all()
    t0 = time.perf_counter()

    def generate():
        if not turns:
            yield f"data: {json.dumps({'error': 'No conversation yet'})}\n\n"
            return

        full_text = ""
        first_token = True
        try:
            for token in stream_analysis(turns):
                if first_token:
                    ttft_ms = (time.perf_counter() - t0) * 1000
                    print(f"[LAT] LLM first token: {ttft_ms:.0f}ms")
                    first_token = False
                full_text += token
                yield f"data: {json.dumps({'token': token})}\n\n"

            total_ms = (time.perf_counter() - t0) * 1000
            print(f"[LAT] LLM total: {total_ms:.0f}ms")

            # Parse detected_context from LLM output
            try:
                parsed = json.loads(full_text)
                ctx = parsed.get("detected_context")
                conf = parsed.get("confidence")
                if ctx and conf:
                    context_manager.update_detected(ctx, conf)
                    print(f"[CTX] Auto-detected: \"{ctx}\" (confidence={conf})")
            except (json.JSONDecodeError, KeyError):
                pass

            yield f"data: {json.dumps({'done': True, 'full': full_text, 'total_ms': round(total_ms)})}\n\n"
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


@analyze_bp.route("/analyze", methods=["POST"])
def analyze_message():
    data = request.get_json(silent=True) or {}
    message = data.get("message", "").strip()
    if not message:
        return json.dumps({"error": "Message is required"}), 400, {"Content-Type": "application/json"}

    try:
        full_text = "".join(stream_analysis([{"speaker": "other", "text": message}]))
        parsed = json.loads(full_text)
        return json.dumps({
            "intent": parsed.get("intent", ""),
            "summary": parsed.get("summary", ""),
            "reply": parsed.get("reply", ""),
        }), 200, {"Content-Type": "application/json"}
    except Exception:
        return json.dumps({"error": "Something went wrong"}), 500, {"Content-Type": "application/json"}


@analyze_bp.route("/log_user", methods=["POST"])
def log_user():
    """Log the user's spoken response to maintain conversation context."""
    data = request.get_json(silent=True) or {}
    text = data.get("text", "").strip()
    if text:
        conversation.add("user", text)
    return json.dumps({"status": "ok"}), 200, {"Content-Type": "application/json"}


@analyze_bp.route("/set_context", methods=["POST"])
def set_context():
    data = request.get_json(silent=True) or {}
    context_manager.set_static(
        meeting_type=data.get("meeting_type"),
        other_name=data.get("other_name"),
        other_role=data.get("other_role"),
        user_goal=data.get("user_goal"),
        language_level=data.get("language_level"),
    )
    return json.dumps({"status": "ok", "context": context_manager.get_static()}), 200, {"Content-Type": "application/json"}


@analyze_bp.route("/clear", methods=["POST"])
def clear():
    conversation.clear()
    context_manager.reset()
    return json.dumps({"status": "cleared"}), 200, {"Content-Type": "application/json"}

@analyze_bp.route("/analyze_audio", methods=["POST"])
def analyze_audio():
    if "audio" not in request.files:
        return json.dumps({"error": "No audio file provided"}), 400, {"Content-Type": "application/json"}

    audio_file = request.files["audio"]
    audio_bytes = audio_file.read()

    if not audio_bytes:
        return json.dumps({"error": "Empty audio"}), 400, {"Content-Type": "application/json"}

    try:
        # STT
        t0 = time.perf_counter()
        transcript = transcribe_audio(audio_bytes, filename=audio_file.filename or "audio.webm")
        stt_ms = (time.perf_counter() - t0) * 1000
        print(f"[LAT] STT: {stt_ms:.0f}ms | transcript: \"{transcript}\"")

        if not transcript.strip():
            return json.dumps({"transcript": "", "intent": "", "summary": "", "reply": ""}), 200, {"Content-Type": "application/json"}

        # Add to buffer so LLM sees full conversation history
        conversation.add("other", transcript)
        turns = conversation.get_all()

        # LLM
        t1 = time.perf_counter()
        full_text = "".join(stream_analysis(turns))
        llm_ms = (time.perf_counter() - t1) * 1000
        total_ms = (time.perf_counter() - t0) * 1000
        print(f"[LAT] LLM: {llm_ms:.0f}ms | total: {total_ms:.0f}ms")

        parsed = json.loads(full_text)

        # Auto-detection: persist if confidence is high/medium
        ctx = parsed.get("detected_context")
        conf = parsed.get("confidence")
        if ctx and conf:
            context_manager.update_detected(ctx, conf)
            print(f"[CTX] Auto-detected: \"{ctx}\" (confidence={conf})")

        return json.dumps({
            "transcript": transcript,
            "intent":     parsed.get("intent", ""),
            "summary":    parsed.get("summary", ""),
            "reply":      parsed.get("reply", ""),
            "stt_ms":     round(stt_ms),
            "llm_ms":     round(llm_ms),
            "total_ms":   round(total_ms),
        }), 200, {"Content-Type": "application/json"}

    except json.JSONDecodeError:
        return json.dumps({"error": "LLM returned invalid JSON"}), 500, {"Content-Type": "application/json"}
    except Exception as e:
        return json.dumps({"error": str(e)}), 500, {"Content-Type": "application/json"}
