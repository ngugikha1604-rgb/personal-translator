# AGENTS.md

Authoritative guide for AI coding assistants working on this repository.

Read this file before reviewing, refactoring, or implementing any feature.

This document is the source of truth for engineering decisions.

---

# One-Sentence Project Definition

This project is not a desktop application that may someday become wearable.

It is a future wearable Conversation Copilot that temporarily happens to run on a desktop.

Every engineering decision should be evaluated through that lens.

---

# Product Vision

The goal is to build a real-time Conversation Copilot.

The system helps the user:

* understand what someone actually means
* avoid misunderstandings
* respond faster
* stay engaged in conversation

The AI assists thinking.

The AI does not replace thinking.

The user remains responsible for what they say.

---

# What This Project Is

This project is:

* a Conversation Copilot
* an Intent Understanding System
* a Response Planning Assistant
* a Real-Time Cognitive Support Tool

The system exists to reduce conversational thinking latency.

---

# What This Project Is NOT

This project is NOT:

* a translator
* a chatbot
* a voice assistant
* an AI companion
* an automatic responder
* a grammar tutor
* a language learning platform
* a customer support system

Do not optimize for those use cases.

---

# Primary Success Metric

The primary metric is:

Conversation Latency Reduction

Definition:

Time between:

Other person finishes speaking

and

User begins responding.

A feature that reduces this delay is valuable.

A feature that does not reduce this delay must justify its existence.

---

# Current Development Stage

Current code is a desktop CLI prototype.

The terminal is a testing environment.

The terminal is not the product.

Current hardware:

Desktop microphone
→ Desktop terminal

Future hardware:

Phone
→ Earbuds / AirPods microphone
→ AI processing
→ Glasses / Watch / Phone overlay display

Assume the terminal UI will eventually disappear.

---

# Long-Term Architecture Vision

Target architecture:

Audio Stream
→ Voice Activity Detection
→ Speech Recognition
→ Conversation Buffer
→ Understanding Layer
→ Suggestion Layer
→ Wearable Display

Future features:

* understanding verification
* answer verification
* social warnings
* self-speech analysis
* conversation coaching

Design code so these can be added later.

Do not implement them unless explicitly requested.

---

# Non-Negotiable Principles

## 1. Truthfulness

Most important rule.

The AI must never invent facts about the user.

Bad:

"I am a machine learning engineer."

Good:

"I am interested in AI."

Every suggestion must remain factually consistent with known user information.

Never weaken this rule.

---

## 2. Latency First

Latency is a feature.

Real-time usefulness is more important than theoretical accuracy.

Prefer:

* fewer network calls
* simpler pipelines
* smaller outputs

A response delivered in 500ms is often better than a slightly better response delivered in 3 seconds.

---

## 3. Fixed Output Schema

Output schema is:

```json
{
  "intent": "string",
  "summary": "string",
  "reply": "string"
}
```

Do not change this schema without updating every consumer.

---

## 4. Reply Is Not A Script

The reply is:

* a direction
* a talking point
* a key idea

The reply is NOT a full sentence the user should read verbatim.

The user speaks in their own words.

---

## 5. Summary Is Internal

Summary exists only for reasoning.

Never display summary.

Only display:

* reply
* intent

---

## 6. Streaming Future

Current implementation processes chunks.

Future implementation will process streams.

When refactoring ask:

"Will this survive streaming STT?"

Prefer designs that can evolve toward:

audio stream
→ transcript stream
→ reasoning stream
→ display updates

without major rewrites.

---

## 7. Real-Time Path Has Priority

Always prioritize:

Listening
→ Understanding
→ Suggestion

over:

Verification
→ Coaching
→ Learning
→ Analytics

If a feature risks delaying the next utterance, it probably belongs outside the real-time path.

---

# Engineering Philosophy

This project intentionally remains small.

Avoid complexity unless a real bottleneck has been proven.

Do not add systems simply because they are popular.

---

# Explicitly Avoid

Do NOT introduce:

* RAG
* Vector databases
* Knowledge graphs
* LangGraph
* CrewAI
* Multi-agent systems
* Authentication
* User accounts
* Dashboards
* Analytics platforms
* Long-term memory systems

unless explicitly requested.

These are not current bottlenecks.

---

# Current Runtime Flow

Current runtime:

Mic Input
↓
Audio Service
↓
VAD
↓
STT
↓
Conversation Buffer
↓
LLM
↓
Copilot Parser
↓
Display

Current implementation validates product assumptions.

It is not the final deployment architecture.

---

# Repository Structure

```text
backend/
├── main.py
├── config.py
├── user_profile.json
└── services/
    ├── audio.py
    ├── vad.py
    ├── speech.py
    ├── stt.py
    ├── llm.py
    ├── conversation.py
    ├── context.py
    ├── copilot.py
    └── display.py
```

---

# Service Boundaries

Service boundaries are important.

Prefer replacing implementations over changing interfaces.

The goal is hardware portability.

---

## Audio

Responsible only for audio capture.

Must not contain:

* STT
* LLM logic
* Prompt logic

Future implementations:

* phone mic
* earbuds mic
* AirPods mic
* beamforming arrays

---

## VAD

Responsible only for speech detection.

Must remain:

* fast
* local
* lightweight

Cloud VAD should not sit in the real-time path.

---

## STT

Responsible only for speech transcription.

Current provider:

Groq Whisper

Future providers:

* Faster Whisper
* Whisper.cpp
* FunASR
* OS speech APIs

Code outside the STT boundary should not know which provider is used.

---

## LLM

Responsible for:

* prompt construction
* model invocation
* latency metrics

Must preserve:

* truthfulness rule
* output schema
* concise outputs

---

## Conversation Buffer

Single source of truth.

Conversation history ownership must remain clear.

Avoid hidden state.

Avoid duplicate buffers.

Avoid state scattered across services.

---

## Display

Current display is terminal.

Treat it as a smart-glasses simulator.

Output should remain:

* short
* glanceable
* readable in under one second

Never display summary.

---

# Real-Time Path vs Learning Path

Keep separated.

Real-Time Path:

Mic
→ VAD
→ STT
→ Conversation Buffer
→ LLM
→ Display

Future Learning Path:

Session Logs
→ Local Cache
→ Background Processing
→ Profile Updates

Learning must never block conversation.

---

# Review Priorities

When reviewing code:

Focus on:

1. Latency
2. Streaming readiness
3. State ownership
4. Provider independence
5. Hardware portability

Do NOT focus primarily on:

* naming
* style
* formatting
* architectural purity

unless they create real maintenance problems.

---

# Streaming Migration Checklist

When proposing a change ask:

1. Will this survive streaming STT?
2. Will this survive wearable deployment?
3. Will this survive provider replacement?
4. Will this reduce future rewrites?

If not, reconsider the design.

---

# Disposable Code Analysis

When reviewing architecture, explicitly identify:

1. Code likely to survive future hardware migration.
2. Code likely to be rewritten.
3. Code likely to be deleted.

This information is often more valuable than style feedback.

---

# LLM Usage Philosophy

LLM calls are expensive.

Before adding a new LLM call ask:

1. Can a heuristic solve this?
2. Can existing information solve this?
3. Can this happen asynchronously?
4. Does this improve conversation latency?

If not, avoid the call.

---

# Benchmark Results

All performance decisions are based on empirical measurement.

Do not propose optimizations that contradict the results below.

Do not re-optimize things already benchmarked.

---

## LLM is not the bottleneck

Measured:

```
Groq 70B: ≈390 ms
Groq 8B:  ≈437 ms
```

Smaller model is not faster.

Do not downgrade the model to save latency.

Do not reduce conversation history aggressively for latency.

Do not remove few-shot examples from the prompt for latency.

These changes save only ≈20–30 ms and are not worth the quality cost.

---

## STT is the dominant bottleneck

The pipeline bottleneck is in STT, not LLM.

Optimize STT before anything else.

---

## Faster Whisper: tiny.en is the correct model

`base.en` measured ≈6–10 s for 5 s of audio. Unacceptable for real-time.

`tiny.en` measured ≈1.6–3.6 s for 5 s of audio.

Do not switch back to `base.en`.

Do not propose larger models without a new benchmark.

---

## Faster Whisper: cpu_threads=4 is optimal

Thread scaling results:

| Threads | Mean latency |
|---------|--------------|
| 1       | 1064 ms      |
| 2       | 695 ms       |
| 4       | 617 ms       |
| 8       | 1008 ms      |
| 16      | 867 ms       |

4 threads is the sweet spot on current hardware.

More threads causes scheduling overhead and cache contention.

Do not increase `cpu_threads` without re-running the benchmark.

---

## Faster Whisper: internal time breakdown

```
WAV → float32: ≈0.5 ms   — negligible
transcribe():   ≈10–20 ms
segment iter:   97–100% of total time
```

All time is inside CTranslate2 inference.

Do not optimize Python-side parsing.

Do not optimize WAV conversion.

Do not propose chunked WAV streaming as a latency fix — the bottleneck is inference, not I/O.

---

## Confirmed Faster Whisper configuration

```
model:                      tiny.en
device:                     cpu
compute_type:               int8
cpu_threads:                4
beam_size:                  1
condition_on_previous_text: False
```

This is the benchmarked optimum on current hardware.

Do not change any of these values without a new benchmark.

---

## Current STT latency baseline

```
≈600–700 ms for 5 s audio
RTF ≈ 0.12
```

Real-time is achievable. Architecture is now the remaining constraint.

---

## Model initialization is not repeated

The STT provider is a singleton.

The model loads once at startup.

Do not propose caching or lazy loading as an optimization — it is already solved.

---

## Verification LLM is not a latency concern

Verification runs only at end-of-utterance.

It is not on the hot path.

Do not remove it for latency reasons.

Only remove it if token budget becomes a concern.

---

## Remaining architectural bottleneck

After all model-level optimizations, the bottleneck is the fixed recording window:

```
Current:  record 5 s → STT → LLM
Target:   speech ends → STT immediately → LLM
```

The next meaningful latency gain requires VAD-based dynamic chunking (P1 below).

---

# Optimization Roadmap

## P0 — Completed

* ✅ Benchmark full pipeline
* ✅ Benchmark STT latency
* ✅ Benchmark Faster Whisper internals
* ✅ Benchmark thread scaling
* ✅ Switch to `tiny.en`
* ✅ Set `cpu_threads=4`

## P1 — High ROI (implement next)

Replace `record_chunk(5)` with `record_until_silence()`.

Potential savings: 200 ms – 4000 ms depending on utterance length.

This is the highest-ROI remaining optimization.

Requires: reliable VAD (Silero VAD or equivalent).

## P2 — Medium ROI

Overlap audio recording with STT.

Do not block recording while waiting for STT result.

## P3 — Optional cost reduction

Remove Verification LLM.

Not a latency concern. Only relevant if token budget is constrained.

---

# Utterance Filtering

Not every transcript deserves an LLM call.

Examples:

* okay
* good
* right
* yeah
* uh-huh

may contain little semantic value.

Future architecture may include:

Transcript
→ Utterance Classifier
→ Intent Analysis

Keep this possibility open.

---

# Preferred Refactor Format

When proposing changes provide:

Issue:
...

Why it matters:
...

Impact on latency:
...

Impact on streaming migration:
...

Recommended fix:
...

Priority:
Critical / Important / Future

This is more valuable than generic style advice.

---

# Final Rule

Before implementing any change ask:

"Will this still make sense when the terminal disappears and the product becomes a pair of smart glasses?"

If the answer is no, reconsider the change.
