# AGENTS.md — Conversation Copilot

Authoritative guide for AI coding assistants working on this repo.

Read before changing code.

---

## What This Project Is

Real-time AI copilot for live English conversations.

System listens to other person, detects intent, and suggests short truthful reply fragment. User still decides what to say.

Current goal: prepare software architecture so it can later run on constrained hardware such as phone + earbuds/AirPods-style mics, then glasses or other lightweight display.

**This is not:**
- translator
- chatbot
- voice assistant
- automatic responder

AI reduces conversation latency. It does not speak for user.

---

## Current Product Direction

Old repo docs described Flask routes and web frontend. Current code is now a **desktop CLI hardware prototype**.

Current runtime:

```
Mic input
  ↓
audio.py records short WAV chunk
  ↓
vad.py skips likely silence
  ↓
stt.py sends audio to Groq Whisper
  ↓
conversation.py stores rolling turns
  ↓
llm.py sends turns + user profile to Groq LLaMA
  ↓
copilot.py parses fixed JSON
  ↓
display.py prints glasses-like terminal overlay
```

Near-term target: keep interfaces stable while replacing internals for device deployment.

Production target examples:
- Phone app does capture, VAD, STT/LLM orchestration
- AirPods / earbuds provide mic input and optional push-to-mute affordance
- Glasses / phone overlay / watch provide display
- Cloud remains optional background learning path, never real-time dependency

---

## Non-Negotiable Principles

1. **Truthfulness** — LLM must never invent facts about user. Truthfulness rule must stay in system prompt.

2. **Latency first** — Every feature judged by time-to-useful-reply. Do not add real-time steps unless needed.

3. **Fixed output schema** — Always:

```json
{ "intent": "string", "summary": "string", "reply": "string" }
```

Do not add/remove fields without updating every consumer.

4. **Reply is short phrase** — `reply` is key point(s), not full script. User speaks in own words.

5. **Summary is internal only** — `summary` helps reasoning. Never display it.

6. **Services swappable, entrypoint thin** — Model, audio, VAD, display, storage internals belong in `services/`. `main.py` orchestrates only.

7. **Singleton conversation buffer** — `conversation` in `services/conversation.py` is shared session state. Do not create new buffers inside runtime flow.

8. **Real-time path isolated** — Hardware capture, STT, LLM, and display path must not depend on background learning/cloud sync.

---

## Current Repository Layout

```
personal_translator/
├── README.md
├── AGENTS.md
└── backend/
    ├── main.py                  # CLI entry point and real-time loop
    ├── config.py                # Env, model names, file paths, constants
    ├── requirements.txt
    ├── .env.example
    ├── user_profile.json        # Local user facts/style injected into prompt
    └── services/
        ├── audio.py             # Mic capture: record_chunk() -> WAV bytes
        ├── vad.py               # Speech gate: has_speech()
        ├── stt.py               # STT abstraction: transcribe_audio()
        ├── speech.py            # VAD + STT service
        ├── conversation.py      # Rolling ConversationBuffer singleton
        ├── context.py           # Optional session context manager
        ├── llm.py               # Prompt + LLM call + latency metrics
        ├── copilot.py           # JSON parse + CopilotResult
        └── display.py           # Output layer, terminal now, glasses later
```

No Flask routes exist in current code. Do not add HTTP layer unless explicitly asked.

---

## Runtime Flow

```
[Start]
python backend/main.py

[Loop]
record_chunk(CHUNK_SECONDS)
  → has_speech(audio_bytes)
  → transcribe_audio(audio_bytes)
  → conversation.add_other(transcript)
  → call_llm(conversation.get_all())
  → parse { intent, summary, reply }
  → display reply + intent

[Controls]
Hold SPACE → mute while user speaks
Q          → quit
Ctrl+C     → quit
```

Important: current system captures other person only. User's actual spoken replies are not yet logged.

---

## Hardware-Ready Architecture

Keep these interfaces stable. Hardware migration should replace internals only.

### Audio Input

File: `backend/services/audio.py`

```python
def record_chunk(duration: float = CHUNK_SECONDS) -> bytes
```

Current: `sounddevice` default desktop mic, WAV bytes.

Future replacements:
- phone microphone
- Bluetooth earbuds mic
- AirPods / headset input through OS audio APIs
- beamformed mic array

Rules:
- keep return type as audio bytes unless all consumers are changed together
- do not put STT or VAD in `audio.py`
- do not add device-specific logic to `main.py`

### VAD

File: `backend/services/vad.py`

```python
def has_speech(audio_bytes: bytes, min_size_bytes: int = 8000) -> bool
```

Current: cheap byte-variance heuristic.

Future replacements:
- WebRTC VAD
- Silero VAD
- platform-native voice activity APIs

Rules:
- VAD must be fast
- false negatives are worse than false positives during conversation
- avoid cloud VAD in real-time path

### STT

File: `backend/services/stt.py`

```python
def transcribe_audio(audio_bytes: bytes, filename: str = "audio.webm") -> str
```

Current: Groq Whisper API.

Future replacements:
- local Whisper
- faster-whisper
- whisper.cpp
- phone OS speech API, if latency/privacy acceptable

Rules:
- keep function signature stable
- language is English for MVP
- routes/entrypoint must not know model details

### LLM

File: `backend/services/llm.py`

```python
def call_llm(turns: list) -> LLMResult
```

Current: Groq LLaMA 3.3 70B streaming call.

Future replacements:
- local model on phone
- small on-device model
- quantized model through llama.cpp / MLX / Core ML / NNAPI
- cloud fallback, only if latency acceptable

Rules:
- keep output schema fixed
- preserve truthfulness rule
- preserve low-token output
- keep latency metrics: TTFT and total time

### Copilot Parser

File: `backend/services/copilot.py`

```python
copilot_service.analyze_turns(turns) -> CopilotResult
```

Rules:
- parse and validate LLM JSON here
- `summary` remains internal
- `display_payload()` must expose only `intent` and `reply`

### Display

File: `backend/services/display.py`

Current: terminal ANSI output.

Future replacements:
- phone overlay
- watch display
- glasses optical overlay
- earbuds companion app UI

Rules:
- display only `reply` and `intent`
- never display `summary`
- optimize for single glance

### Conversation Buffer

File: `backend/services/conversation.py`

```python
conversation.add(speaker: str, text: str)
conversation.add_other(text: str)
conversation.add_user(text: str)
conversation.get_all() -> list
conversation.clear()
```

Speaker semantics:
- `"other"` — person user is talking to
- `"user"` — what user actually said, when Mic 2 / self-speech logging exists

Rules:
- in-memory rolling buffer only
- no disk I/O here
- no database here
- no cloud sync here

### Context

File: `backend/services/context.py`

Optional session hints:
- meeting type
- other person name/role
- user goal
- language level

Current: can inject context into prompt, but no CLI/UI setter exists yet.

Rules:
- manual context beats auto-detected context
- low-confidence auto context ignored
- context must not become long-term memory

---

## Real-Time Path vs Learning Path

Keep separate.

```
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

```
LEARNING PATH (future)
User speech + session logs
  ↓
Local cache after session
  ↓
Cloud sync in background
  ↓
Profile learning
  ↓
Updated user_profile.json / device profile
```

Learning path must never block real-time path.

Do not implement learning path unless explicitly asked.

---

## Prompt Contract

System prompt in `services/llm.py` must include:
- user profile
- optional context block
- fixed output schema
- truthfulness rule
- short reply rule
- no extra fields rule

Critical rule that must stay:

> The reply MUST be truthful. Never invent facts about the user.

Output must be raw JSON only. No markdown. No code fences.

---

## Display Contract

Target layout:

```
studying AI, building LLM stuff      ← reply, primary
asking about field of study          ← intent, secondary
```

Constraints:
- reply: 5–9 words target, natural spoken fragment
- intent: max ~6 words
- summary: never shown
- readable in ~0.5 seconds
- no paragraphs, no scripts

---

## Config

File: `backend/config.py`

Current values:
- `GROQ_API_KEY`
- `WHISPER_MODEL`
- `LLM_MODEL`
- `CONVERSATION_MAX_TURNS`
- `USER_PROFILE_PATH`

Rules:
- model names live in config
- services read config
- `main.py` does not hardcode model names
- secrets stay in `.env`, never committed

---

## Running Locally

Windows:

```bash
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Fill `GROQ_API_KEY` in `.env`.

Run:

```bash
python main.py
```

Controls:
- hold `SPACE` while user speaks
- press `Q` to quit
- `Ctrl+C` to quit

---

## Out of Scope Unless Asked

- Flask routes / web frontend
- authentication
- multi-speaker group conversation
- auto-replying
- translation feature
- emotion/psychological analysis
- cloud learning pipeline
- long-term people profiles on device
- hardware-specific SDK integration
- mobile app scaffolding

---

## Engineering Style

- smallest diff wins
- stdlib/native first
- no dependency unless needed
- no abstraction with one implementation unless it protects hardware/model swap boundary
- keep real-time path boring and measurable
- delete stale docs/code instead of supporting two architectures
- add tests/checks only for non-trivial logic
- preserve current service boundaries unless strong reason