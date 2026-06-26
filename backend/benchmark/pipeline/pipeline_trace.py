"""pipeline_trace.py — Measure complete end-to-end latency breakdown.

Simulates the production pipeline:
  capture → STT → VAD gate → utterance filter → Analyzer → display

Records timing at each stage.

Usage:
    cd backend
    python benchmark/pipeline/pipeline_trace.py --record              # capture + process live
    python benchmark/pipeline/pipeline_trace.py --wav path.wav        # process existing WAV
    python benchmark/pipeline/pipeline_trace.py --text "hello there"  # skip STT, analysis only

Output:
    benchmark_results/pipeline_trace.jsonl
"""

import json
import os
import sys
import time
from statistics import mean, median

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from services.analyzer import Analyzer
from services.stt_factory import get_stt_provider
from services.vad import has_speech
from services.speech import SpeechStatus

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "benchmark_results"
)


def run_pipeline(text: str = None, audio_bytes: bytes = None, filename: str = "") -> dict:
    """Simulate the production pipeline with timing at each stage.

    Returns timing breakdown in milliseconds.
    """
    timeline = {
        "audio_capture_ms": 0,   # Time from start to capture end
        "vad_ms": 0,
        "stt_ms": 0,
        "utter_filter_ms": 0,
        "analyzer_ms": 0,
        "display_ms": 0,
        "e2e_ms": 0,
        "transcript": "",
        "status": "ok",
    }

    pipeline_start = time.perf_counter()

    # ── Stage: STT (if audio provided) ──
    if audio_bytes is not None:
        stt_start = time.perf_counter()

        # VAD gate
        vad_start = time.perf_counter()
        has_sp = has_speech(audio_bytes)
        timeline["vad_ms"] = round((time.perf_counter() - vad_start) * 1000)

        if not has_sp:
            stt_end = time.perf_counter()
            timeline["stt_ms"] = round((stt_end - stt_start) * 1000)
            timeline["transcript"] = ""
            timeline["status"] = "vad_skipped"
            timeline["e2e_ms"] = round((stt_end - pipeline_start) * 1000)
            return timeline

        # STT
        stt = get_stt_provider()
        try:
            transcript = stt.transcribe(audio_bytes, filename)
        except Exception as exc:
            stt_end = time.perf_counter()
            timeline["stt_ms"] = round((stt_end - stt_start) * 1000)
            timeline["status"] = f"stt_error: {str(exc)[:100]}"
            timeline["e2e_ms"] = round((stt_end - pipeline_start) * 1000)
            return timeline

        stt_end = time.perf_counter()
        timeline["stt_ms"] = round((stt_end - stt_start) * 1000)
        timeline["transcript"] = transcript

    else:
        # Text-based: skip STT, use provided text
        transcript = text or ""
        stt_end = pipeline_start
        timeline["stt_ms"] = 0
        timeline["transcript"] = transcript

    # ── Stage: Utterance filter ──
    utfl_start = time.perf_counter()
    transcript_clean = transcript.strip()
    timeline["utter_filter_ms"] = round((time.perf_counter() - utfl_start) * 1000)

    if not transcript_clean:
        utfl_end = time.perf_counter()
        timeline["status"] = "empty_transcript"
        timeline["e2e_ms"] = round((utfl_end - pipeline_start) * 1000)
        return timeline

    # ── Stage: Analyzer ──
    analyzer_start = time.perf_counter()
    turns = [{"speaker": "other", "text": transcript_clean}]

    try:
        result = Analyzer().analyze(turns)
    except Exception as exc:
        analyzer_end = time.perf_counter()
        timeline["analyzer_ms"] = round((analyzer_end - analyzer_start) * 1000)
        timeline["status"] = f"analyzer_error: {str(exc)[:100]}"
        timeline["e2e_ms"] = round((analyzer_end - pipeline_start) * 1000)
        return timeline

    analyzer_end = time.perf_counter()
    timeline["analyzer_ms"] = round((analyzer_end - analyzer_start) * 1000)

    # ── Stage: Display (simulate) ──
    display_start = time.perf_counter()
    # Simulate: format intent + reply for display
    display_payload = {
        "intent": result.intent,
        "reply": result._parsed.get("reply", "") if result._parsed else "",
    }
    display_end = time.perf_counter()
    timeline["display_ms"] = round((display_end - display_start) * 1000)

    # ── Totals ──
    timeline["e2e_ms"] = round((display_end - pipeline_start) * 1000)
    timeline["intent"] = result.intent
    timeline["reply"] = display_payload["reply"]
    timeline["llm_ms"] = result.llm_ms
    timeline["ttft_ms"] = result.ttft_ms
    timeline["status"] = "ok"

    return timeline


def run_benchmark_from_text(texts: list):
    """Run pipeline with text inputs (no mic needed)."""
    all_results = []
    for text in texts:
        result = run_pipeline(text=text)
        all_results.append(result)
        label = result["transcript"][:40] if result["transcript"] else "(empty)"
        ms = result["e2e_ms"]
        print(f"  \"{label}\" → e2e={ms}ms  stt={result['stt_ms']}ms  analyzer={result['analyzer_ms']}ms")

    return all_results


def run_benchmark_from_audio(wav_path: str):
    """Run pipeline with a WAV file."""
    with open(wav_path, "rb") as f:
        audio = f.read()
    result = run_pipeline(audio_bytes=audio, filename=os.path.basename(wav_path))
    print(f"  Audio: {os.path.basename(wav_path)}")
    print(f"  Transcript: {result['transcript'][:60]}")
    print(f"  E2E: {result['e2e_ms']}ms  STT: {result['stt_ms']}ms  Analyzer: {result['analyzer_ms']}ms")
    return [result]


def summarize(results: list, label: str):
    """Print summary statistics for each timing stage."""
    stages = ["audio_capture_ms", "vad_ms", "stt_ms", "utter_filter_ms",
               "analyzer_ms", "display_ms", "e2e_ms"]
    print(f"\n  {'=' * 50}")
    print(f"  Pipeline trace — {label} ({len(results)} samples)")
    print(f"  {'=' * 50}")
    print(f"  {'Stage':20s} {'mean':>8s} {'median':>8s} {'p95':>8s} {'min':>8s} {'max':>8s}")
    for stage in stages:
        vals = sorted([r[stage] for r in results])
        if not vals:
            continue
        n = len(vals)
        mn = mean(vals)
        md = median(vals)
        p95 = vals[int(n * 0.95)]
        mn_v = min(vals)
        mx_v = max(vals)
        print(f"  {stage:20s} {mn:>8.0f} {md:>8.0f} {p95:>8.0f} {mn_v:>8.0f} {mx_v:>8.0f}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Pipeline trace benchmark")
    parser.add_argument("--wav", type=str, default=None, help="WAV file to process")
    parser.add_argument("--text", type=str, default=None, help="Single text turn (skip STT)")
    parser.add_argument("--batch", type=str, default=None, help="JSON file with list of texts")
    parser.add_argument("--record", action="store_true", help="Record from mic")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "pipeline_trace.jsonl")

    if args.wav:
        results = run_benchmark_from_audio(args.wav)
    elif args.text:
        results = run_benchmark_from_text([args.text])
    elif args.batch:
        with open(args.batch) as f:
            texts = json.load(f)
        results = run_benchmark_from_text(texts)
    elif args.record:
        from services.audio import record_chunk
        print("  Recording 5s from mic...")
        audio = record_chunk(5.0)
        print(f"  Captured {len(audio)} bytes")
        results = run_benchmark_from_audio_path(audio, "recorded.wav")
    else:
        # Default: run with corpus sample texts (no mic needed)
        texts = [
            "Hey, nice to meet you! So what brings you here?",
            "That's cool. What do you work on?",
            "Oh nice! What got you into that?",
            "How long have you been doing competitive programming?",
            "Do you compete in programming contests?",
            "What do you think about the new data privacy regulations?",
            "What are your career goals for the next 5 years?",
            "Tell me about a time you disagreed with a technical decision.",
            "We're debating whether to break our monolith into microservices. What's your take?",
            "I just got back from Vietnam. It was incredible!",
        ]
        results = run_benchmark_from_text(texts)

    # Save
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n  Saved: {out_path}")

    # Summary
    label = "live_audio" if (args.wav or args.record) else "text_mode"
    summarize([r for r in results if r["status"] == "ok"], label)

    # Also show raw analyzer metrics (when available)
    ok_results = [r for r in results if r.get("llm_ms") is not None and r.get("ttft_ms") is not None]
    if ok_results:
        ttfts = sorted([r["ttft_ms"] for r in ok_results])
        llms  = sorted([r["llm_ms"] for r in ok_results])
        n = len(ttfts)
        print(f"\n  Analyzer LLM breakdown ({n} samples):")
        print(f"    ttft_ms: mean={mean(ttfts):.0f}  median={median(ttfts):.0f}  p95={ttfts[int(n*0.95)]}")
        print(f"    llm_ms:  mean={mean(llms):.0f}   median={median(llms):.0f}   p95={llms[int(n*0.95)]}")


if __name__ == "__main__":
    main()
