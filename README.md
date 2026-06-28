# Personal Conversation Copilot

Real-time AI copilot for live conversations.

This project helps the user participate in conversations faster by understanding intent, reducing thinking latency, and suggesting truthful response directions.

The user remains in control.

The AI never speaks for the user.

---

# Vision

Most AI conversation tools focus on:

* translation
* grammar correction
* chatbot interaction
* voice assistants

This project focuses on a different problem:

Human conversation latency.

The user can often understand English.

The bottleneck is:

1. Understanding what the other person actually means.
2. Deciding what to say.
3. Responding quickly.
4. Avoiding misunderstandings.

The goal of this system is to reduce that delay.

---

# What This Project Is

This project is:

* a Conversation Copilot
* an Intent Understanding System
* a Response Planning Assistant
* a real-time conversational support tool

The AI helps the user:

* understand intent
* detect misunderstandings
* think of responses faster
* stay engaged in conversations

---

# What This Project Is NOT

This project is NOT:

* a translator
* a chatbot
* an AI companion
* an AI friend
* a voice assistant
* an automatic responder
* a grammar tutor
* a language learning application

The AI should never replace the user.

The AI assists the user.

---

# Core Success Metric

The primary metric is:

Conversation Latency Reduction

Definition:

Time between:

Other person finishes speaking

and

User begins responding.

The system succeeds if this delay becomes shorter.

---

# Long-Term Vision

The current CLI application is only a prototype.

The final target is:

Microphone
→ Streaming Speech Recognition
→ Conversation Understanding
→ Response Planning
→ Wearable Display

Potential hardware:

* Smart Glasses
* Earbuds
* AirPods-style microphones
* Phone companion
* Smart watch

The architecture should evolve toward wearable deployment.

---

# Product Philosophy

The user should always remain the speaker.

The AI should remain a copilot.

The AI should:

* suggest
* guide
* warn
* verify

The AI should never:

* impersonate
* automatically answer
* fabricate information

---

# Truthfulness Rule

This is the most important rule in the entire project.

The AI must never invent facts about the user.

Bad:

"I am a machine learning engineer."

when the user is not.

Good:

"I am interested in AI and software engineering."

All suggestions must remain truthful.

---

# Current Capabilities

The system currently focuses on:

1. Intent Detection
2. Reply Suggestion

Current output:

```json
{
  "intent": "...",
  "summary": "...",
  "reply": "..."
}
```

Example:

```json
{
  "intent": "understanding educational background",
  "summary": "They want to know what the user studies.",
  "reply": "studying AI and software engineering"
}
```

Display:

```text
studying AI and software engineering
understanding educational background
```

Summary is internal only.

---

# Future Capabilities

These features are planned but not yet implemented.

## Understanding Check

Help detect misunderstanding.

Example:

Speaker:

"What got you interested in AI?"

System:

"They are asking WHY you became interested in AI, not HOW you learned AI."

---

## Answer Check

After the user responds:

Determine whether the user's answer actually addressed the question.

Example:

Speaker:

"Why are you interested in AI?"

User:

"I study software engineering."

System:

"You answered WHAT you study, not WHY you are interested in AI."

---

## Social Warning

Detect possible:

* indirect requests
* polite disagreement
* social ambiguity
* cultural nuances

Warnings should be rare.

Only when confidence is high.

---

# Engineering Philosophy

This project intentionally stays simple.

Do NOT add:

* RAG
* Vector Databases
* Knowledge Graphs
* Multi-Agent Systems
* CrewAI
* LangGraph
* Authentication Systems
* Analytics Platforms
* Dashboard Systems

unless a real bottleneck has been proven.

Complexity should only be added when necessary.

---

# Architecture Philosophy

Current implementation is a desktop prototype.

Future implementation will likely involve:

Streaming Audio
→ Streaming STT
→ Conversation State
→ Understanding Layer
→ Suggestion Layer
→ Wearable Display

All architectural decisions should move the codebase closer to that future.

---

# Latency Philosophy

Latency is a first-class feature.

A slightly worse answer delivered in 500ms is often more useful than a better answer delivered in 3 seconds.

When making engineering decisions:

Prefer:

* lower latency
* fewer network calls
* simpler pipelines

Avoid unnecessary processing.

---

# Streaming Philosophy

Even if the current implementation is request-response based, future systems will process continuous streams.

Design components so they can later support:

partial transcript
→ partial reasoning
→ incremental updates

without major rewrites.

---

# Hardware Independence

Service contracts should remain stable.

Implementations may change.

Examples:

STT:

Today:

* Groq Whisper

Future:

* Faster Whisper
* Whisper.cpp
* FunASR
* OS speech APIs

LLM:

Today:

* Groq LLaMA

Future:

* Local models
* llama.cpp
* MLX
* Core ML
* Cloud fallback

Code outside service boundaries should not care which provider is used.

---

# Current Runtime Flow

Mic Input
↓
Audio Service
↓
Voice Activity Detection
↓
Speech-To-Text
↓
Conversation Buffer
↓
LLM Analysis
↓
Display

Current architecture validates the AI pipeline.

It is not the final deployment architecture.

---

# Performance Benchmarks

All performance decisions in this project are based on empirical measurement, not assumption.

The following results were collected on the current development machine.

---

## Pipeline Bottleneck

STT is the dominant bottleneck in the pipeline.

LLM is not the bottleneck.

Measured Groq LLM latency:

```
Groq 70B: ≈390 ms
Groq 8B:  ≈437 ms
```

Smaller model was not faster.
Downgrading model quality is not justified by latency gains.

---

## STT Model Selection

Faster Whisper `base.en` was too slow for real-time use:

```
base.en: ≈6–10 s for 5 s of audio
```

After switching to `tiny.en`:

```
tiny.en: ≈1.6–3.6 s for 5 s of audio
```

Approximately 2.5–3× faster with no architectural change.

---

## Faster Whisper Internal Breakdown

Time breakdown inside `model.transcribe()`:

```
WAV → float32 conversion: ≈0.5 ms
transcribe() call:         ≈10–20 ms
segment iteration:         97–100% of total time
```

All time is inside CTranslate2 inference.
Python overhead is negligible.
WAV conversion is negligible.

---

## Thread Scaling Results

Measured `cpu_threads` impact on Faster Whisper latency:

| Threads | Mean latency |
|---------|--------------|
| 1       | 1064 ms      |
| 2       | 695 ms       |
| 4       | 617 ms       |
| 8       | 1008 ms      |
| 16      | 867 ms       |

Optimal: `cpu_threads=4`

Higher thread counts increase scheduling overhead and cache contention.

---

## Current Faster Whisper Configuration

Best configuration measured on current hardware:

```
model:          tiny.en
device:         cpu
compute_type:   int8
cpu_threads:    4
beam_size:      1
condition_on_previous_text: False
```

Do not change these values without re-running the thread scaling benchmark.

---

## Current STT Latency Baseline

After all optimizations:

```
≈600–700 ms for 5 s of audio
RTF ≈ 0.12
```

The system processes audio approximately 8× faster than real-time.

---

## Other Findings

JSON parsing overhead: `≈0.5 ms` — negligible.

Reducing `CONVERSATION_MAX_TURNS` from 10 to 4: saves `≈20–30 ms` — not worth the context loss.

Prompt trimming: saves `≈20–30 ms` — not worth sacrificing suggestion quality.

Verification LLM: not on the hot path. Runs only at end-of-utterance. Not a latency bottleneck.

Model singleton: `_provider` is initialized once at startup. Model is not reloaded per STT call.

---

## Remaining Bottleneck

After all current optimizations, the remaining bottleneck is pipeline architecture:

```
Current:   record 5 s → STT → LLM
Target:    speech end → STT immediately → LLM
```

The fixed 5-second recording window adds unnecessary latency at the architecture level.

This is addressed by P1 on the optimization roadmap.

---

# Optimization Roadmap

## P0 — Completed

* ✅ Full pipeline latency benchmark
* ✅ STT latency benchmark
* ✅ Faster Whisper internal probe
* ✅ Thread scaling benchmark
* ✅ Switch to `tiny.en`
* ✅ Set `cpu_threads=4`

## P1 — High ROI (next)

Replace fixed-duration recording with VAD-based dynamic chunking.

Change `record_chunk(5)` to `record_until_silence()`.

Potential savings: `200 ms – 4000 ms` depending on utterance length.

This is the highest-ROI remaining optimization.

## P2 — Medium ROI

Overlap audio recording with STT processing.

Do not block recording while waiting for STT to complete.

## P3 — Optional

Remove Verification LLM to reduce token usage.

Not a latency issue. Purely a cost optimization if needed.

---

# Project Structure

```text
personal_conversation_copilot/
├── README.md
├── AGENTS.md
└── backend/
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

# Current Priorities

Priority #1

Reduce conversation latency.

Next target: VAD-based dynamic chunking (P1 on roadmap).

Priority #2

Prepare architecture for streaming.

Priority #3

Keep provider boundaries clean.

Priority #4

Prepare for wearable deployment.

---

# Current Non-Goals

Not currently working on:

* Mobile apps
* Smart glasses SDK integration
* Authentication
* User accounts
* Cloud memory
* Long-term profiles
* Group conversations
* Automatic responses
* Translation mode

These can be revisited later.

---

# Guidelines For Contributors

Before introducing any feature, ask:

1. Does this reduce conversation latency?
2. Does this improve streaming readiness?
3. Will this still be useful when the UI becomes smart glasses?

If the answer is no, reconsider the change.

The goal is not to build a better chatbot.

The goal is to build a real-time Conversation Copilot.
