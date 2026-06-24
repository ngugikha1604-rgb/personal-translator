# Personal Translator — Conversation Copilot

Real-time AI copilot for live English conversations.

System listens to the other person, detects intent, and suggests a short truthful reply fragment. User still decides what to say.

Not a translator. Not a chatbot. Not a voice assistant. Not an automatic responder.

Goal now: keep desktop prototype simple while shaping architecture so it can later move onto constrained hardware: phone + earbuds/AirPods-style mic input, then glasses / watch / phone overlay display.

---

## Core Problem

Live second-language conversation has latency.

User must:

1. Hear words
2. Decode English
3. Understand intent
4. Decide response
5. Speak naturally

This project reduces steps 3–4 latency. It gives user a glanceable reply direction, not a script.

---

## Current Status

Current code is a **desktop CLI hardware prototype**.

It uses:

| Layer | Current implementation | Future hardware target |
|---|---|---|
| Mic input | `sounddevice` default desktop mic | phone mic, earbuds/AirPods mic, beamformed mic array |
| VAD | cheap byte-variance heuristic | WebRTC VAD, Silero VAD, platform VAD |
| STT | Groq Whisper API | local Whisper, whisper.cpp, phone OS speech API |
| LLM | Groq LLaMA 3.3 70B | quantized local model, phone NPU/Core ML/NNAPI, cloud fallback |
| Display | terminal ANSI overlay | glasses, phone overlay, watch, companion app |
| Memory | in-memory rolling buffer | same real-time buffer + optional background learning later |

No Flask server. No web frontend. No routes.

---

## Runtime Flow

```text
Mic input
  ↓
backend/services/audio.py
record_chunk() returns WAV bytes
  ↓
backend/services/vad.py
has_speech() skips likely silence
  ↓
backend/services/stt.py
transcribe_audio() sends audio to Groq Whisper
  ↓
backend/services/conversation.py
conversation.add_other() stores rolling turns
  ↓
backend/services/llm.py
call_llm() sends turns + user profile to Groq LLaMA
  ↓
backend/services/copilot.py
parses fixed JSON
  ↓
backend/services/display.py
prints reply + intent in terminal
```

Output shape is fixed:

```json
{
  "intent": "asking about field of study",
  "summary": "They want to know what the user is currently studying.",
  "reply": "studying AI, mostly building LLM stuff"
}
```

Display shows only:

```text
studying AI, mostly building LLM stuff
asking about field of study
```

`summary` is internal only.

---

## Project Structure

```text
personal_translator/
├── README.md
├── AGENTS.md
└── backend/
    ├── main.py                  # CLI entry point and real-time loop
    ├── config.py                # Env, model names, constants
    ├── requirements.txt
    ├── .env.example
    ├── user_profile.json        # Local user facts/style injected into prompt
    └── services/
        ├── audio.py             # Mic capture: record_chunk() -> WAV bytes
        ├── vad.py               # Speech gate: has_speech()
        ├── stt.py               # Groq Whisper STT wrapper
        ├── speech.py            # VAD + STT orchestration
        ├── conversation.py      # Rolling in-memory buffer singleton
        ├── context.py           # Optional session context manager
        ├── llm.py               # Prompt + LLM call + latency metrics
        ├── copilot.py           # JSON parse + CopilotResult
        └── display.py           # Terminal now, glasses later
```

---

## Hardware-Ready Boundaries

Main rule: replace internals, keep service contracts stable.

### Audio

```python
record_chunk(duration: float) -> bytes
```

Current: desktop mic via `sounddevice`.

Later:
- phone mic
- Bluetooth earbuds mic
- AirPods/headset input through OS APIs
- beamformed mic array

`main.py` should not know device details.

### VAD

```python
has_speech(audio_bytes: bytes) -> bool
```

Current: fast byte-variance heuristic.

Later:
- WebRTC VAD
- Silero VAD
- native phone VAD

VAD stays local and fast. Cloud VAD should not sit in real-time path.

### STT

```python
transcribe_audio(audio_bytes: bytes, filename: str = "audio.webm") -> str
```

Current: Groq Whisper.

Later:
- local Whisper
- faster-whisper
- whisper.cpp
- OS speech APIs when latency/privacy fit

### LLM

```python
call_llm(turns: list) -> LLMResult
```

Current: Groq LLaMA with streaming, TTFT, total latency, token metrics.

Later:
- small local model
- llama.cpp / MLX / Core ML / NNAPI
- cloud fallback only if latency acceptable

Prompt must keep:
- truthfulness rule
- fixed JSON schema
- short reply rule
- no extra fields

### Display

Current display is terminal. It acts like a glasses simulator:

```text
reply  ← primary, big/glanceable
intent ← secondary, small/context
```

Later display can be:
- glasses optical overlay
- phone overlay
- watch display
- companion app UI

Never display `summary`.

---

## Real-Time Path vs Learning Path

Keep separated.

```text
REAL-TIME PATH
Mic / earbuds
  ↓
VAD
  ↓
STT
  ↓
Conversation buffer
  ↓
LLM
  ↓
Display
```

```text
LEARNING PATH (future)
User speech + session logs
  ↓
Local cache after session
  ↓
Background cloud sync
  ↓
Profile learning
  ↓
Updated user_profile.json / device profile
```

Learning path must never block real-time conversation.

Current code implements only real-time path.

---

## Conversation Model

Current system captures only **other person**.

```python
conversation.add_other(text)
```

Reserved for future self-speech logging:

```python
conversation.add_user(text)
```

Speaker meanings:
- `other` — person user is talking to
- `user` — what user actually said, once Mic 2 / self-speech logging exists

---

## Prompt Contract

System prompt lives in:

```text
backend/services/llm.py
```

Critical rule:

> The reply MUST be truthful. Never invent facts about the user.

Allowed output only:

```json
{ "intent": "string", "summary": "string", "reply": "string" }
```

No markdown. No code fences. No extra keys.

---

## User Profile

Local profile file:

```text
backend/user_profile.json
```

Example:

```json
{
  "interests": ["AI", "Programming", "Competitive Programming"],
  "communication_style": ["logical", "concise", "truthful"]
}
```

This profile is injected into prompt. LLM may use it only for truthful reply suggestions.

---

## Setup

Windows:

```bat
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Edit `.env`:

```env
GROQ_API_KEY=your_groq_api_key_here
```

Run:

```bat
python main.py
```

Controls:

```text
Hold SPACE → mute while user speaks
Q          → quit
Ctrl+C     → quit
```

---

## Config

```text
backend/config.py
```

Current config values:
- `GROQ_API_KEY`
- `WHISPER_MODEL`
- `LLM_MODEL`
- `CONVERSATION_MAX_TURNS`
- `USER_PROFILE_PATH`

Model names stay in config. Services read config. `main.py` should not hardcode models.

---

## Development Priorities

Current priority: prepare for device deployment without overbuilding.

1. Keep interfaces stable.
2. Keep `main.py` thin.
3. Keep real-time path isolated.
4. Keep output glanceable.
5. Measure latency.
6. Replace internals only when needed.

Good next steps:
- add CLI/manual session context setter
- improve VAD with local dependency only if false positives hurt
- test chunk size vs latency and reply usefulness
- add optional local STT backend behind same `transcribe_audio()` contract
- add hardware input adapter later, not now

---

## Out of Scope For Now

- Flask routes
- web frontend
- authentication
- multi-speaker group conversation
- automatic replies
- translation mode
- emotion/psychological analysis
- cloud learning pipeline
- long-term people profiles
- hardware-specific SDK integration
- mobile app scaffolding

---

## Current Command

```bat
cd backend
python main.py
```
