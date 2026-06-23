import json
import time
from dataclasses import dataclass
from typing import Generator

from services.conversation import conversation
from services.intent import extract_intent
from services.llm import stream_analysis
from services.speech import speech_service
from services.suggested_reply import extract_suggested_reply
from services.summary import extract_summary


@dataclass(frozen=True)
class CopilotResult:
    intent: str
    summary: str
    reply: str
    raw: str
    llm_ms: int

    def display_payload(self) -> dict:
        return {
            "intent": self.intent,
            "summary": self.summary,
            "reply": self.reply,
        }


class CopilotService:
    """Conversation-aware reasoning service for intent and reply suggestions."""

    def stream_turns(self, turns: list) -> Generator[str, None, None]:
        yield from stream_analysis(turns)

    def analyze_turns(self, turns: list) -> CopilotResult:
        t0 = time.perf_counter()
        full_text = "".join(self.stream_turns(turns))
        llm_ms = round((time.perf_counter() - t0) * 1000)
        print(f"[LAT] LLM: {llm_ms}ms")

        parsed = json.loads(full_text)
        return CopilotResult(
            intent=extract_intent(parsed),
            summary=extract_summary(parsed),
            reply=extract_suggested_reply(parsed),
            raw=full_text,
            llm_ms=llm_ms,
        )

    def analyze_other_text(self, text: str) -> CopilotResult:
        return self.analyze_turns([{"speaker": "other", "text": text}])

    def analyze_other_audio(self, audio_bytes: bytes, filename: str = "audio.webm") -> dict:
        total_start = time.perf_counter()
        speech = speech_service.transcribe_other_audio(audio_bytes, filename)

        if not speech.transcript:
            return {
                "transcript": "",
                "intent": "",
                "summary": "",
                "reply": "",
                "stt_ms": speech.stt_ms,
                "llm_ms": 0,
                "total_ms": round((time.perf_counter() - total_start) * 1000),
            }

        conversation.add_other(speech.transcript)
        result = self.analyze_turns(conversation.get_all())
        total_ms = round((time.perf_counter() - total_start) * 1000)
        print(f"[LAT] Total: {total_ms}ms")

        payload = result.display_payload()
        payload.update(
            {
                "transcript": speech.transcript,
                "stt_ms": speech.stt_ms,
                "llm_ms": result.llm_ms,
                "total_ms": total_ms,
            }
        )
        return payload


copilot_service = CopilotService()
