# AGENTS.md — Conversation Copilot

This file is the authoritative reference for AI coding assistants working on this codebase.

Read this entire file before making any changes. Every section exists for a reason.

---

## What This Project Is

A real-time AI copilot for live English conversations. The system listens to the other person, detects their intent, and suggests a short honest reply — displayed on smart glasses as an overlay.

**This is not:**
- A translator
- A chatbot
- A voice assistant
- An automatic responder

The user always decides what to say. The AI only reduces the cognitive load of figuring out what to say.

---

## Non-Negotiable Principles

These constraints must never be violated, regardless of what feature is being built:

1. **Truthfulness** — LLM must never invent facts about the user. The truthfulness rule in the system prompt must always be present.

2. **Latency is the primary metric** — Every feature must be evaluated by whether it makes the system faster or slower. Never add processing steps in the real-time path without strong justification.

3. **Output schema is fixed** — Always `{ "intent": string, "summary": string, "reply": string }`. Do not add or remove fields without updating all consumers.

4. **Reply is a short phrase, not a full sentence** — The reply field suggests the key point(s) for the user to say in their own words. It is not a script.

5. **Summary is internal only** — `summary` is used by the LLM for reasoning context. It is never displayed to the user.

6. **Services are swappable, routes are not** — STT and LLM implementations will change (Groq → local). Routes must never contain model-specific code. All model logic stays inside `services/`.

7. **Singleton conversation buffer** — `conversation` in `services/conversation.py` is shared state across all routes. Never instantiate a new buffer inside a route.

---

## System Architecture

### Two Completely Independent Paths

```
REAL-TIME PATH                          LEARNING PATH
──────────────────────────────          ──────────────────────────────
Mic 1 (outward, beamforming)            Mic 2 (toward user's mouth)
  ↓                                       ↓
Noise suppression                       Capture user's speech
  ↓                                       ↓
VAD                                     Log to buffer as "user"
  ↓                                       ↓
STT (local Whisper)                     Cache after session
  ↓                                       ↓
Conversation Buffer                     Push to cloud (background)
  ↓                                       ↓
LLM (local fine-tuned)                  Cloud model analyzes
  ↓                                       ↓
{ intent, summary, reply }              Update user profile
  ↓                                       ↓
Glasses display                         Sync back to device
```

Never mix these two paths. Real-time path must have no dependency on learning path.

### Processing Model

Audio is processed **continuously in rolling chunks** — not only on silence detection.

Every ~N seconds (empirically determined, estimated 1.5–2s):
1. Take latest audio chunk
2. Run STT
3. Append to conversation buffer
4. Run LLM analysis
5. Update display

The display updates while the other person is still speaking. This is intentional — it gives the user time to read and process the suggestion before needing to respond.

---

## Repository Layout

```
personal_translator/
├── README.md
├── AGENTS.md                       ← you are here
├── backend/
│   ├── app.py                      # Flask app factory, blueprint registration
│   ├── config.py                   # All constants and config values
│   ├── requirements.txt
│   ├── .env.example
│   ├── routes/
│   │   ├── transcribe.py           # POST /transcribe
│   │   └── analyze.py              # GET /analyze, POST /log_user, POST /clear
│   └── services/
│       ├── stt.py                  # STT abstraction layer
│       ├── llm.py                  # LLM prompt + streaming
│       └── conversation.py         # ConversationBuffer
└── frontend/
    └── ...                         # Temporary desktop UI, will be replaced
```

---

## Component Reference

### `app.py`
Flask app factory. Registers blueprints only. No business logic here.

---

### `config.py`
Single source of truth for all configuration. Always read values from here, never hardcode.

| Variable | Purpose |
|----------|---------|
| `GROQ_API_KEY` | Groq API key (loaded from .env) |
| `USER_PROFILE` | User's interests and communication style — injected into LLM system prompt |
| `WHISPER_MODEL` | STT model name |
| `LLM_MODEL` | LLM model name |
| `CONVERSATION_MAX_TURNS` | Rolling buffer size |

**When changing models:** update only `WHISPER_MODEL` / `LLM_MODEL` here. Services read from config, routes never touch model names.

**When switching to local models:** update `services/stt.py` and `services/llm.py` internals. Config still holds model names/paths. Routes do not change.

---

### `services/conversation.py`

**Class:** `ConversationBuffer`

Maintains the rolling window of the current conversation session.

```python
conversation.add(speaker: str, text: str)   # "other" or "user"
conversation.get_all() -> list              # returns list of { speaker, text }
conversation.clear()                        # resets buffer
```

**Speaker semantics:**
- `"other"` — the person the user is talking to (transcribed from Mic 1)
- `"user"` — what the user actually said (captured from Mic 2, logged via `/log_user`)

**Implementation:** `collections.deque` with `maxlen=CONVERSATION_MAX_TURNS`

**Rules:**
- Exported as singleton: `conversation = ConversationBuffer()`
- Do not add persistence, disk I/O, or database here
- This is intentionally ephemeral session memory (L1 cache)
- Long-term storage is handled by the cloud learning pipeline, not here

---

### `services/stt.py`

STT abstraction layer. Converts raw audio bytes to transcript string.

```python
def transcribe_audio(audio_bytes: bytes, filename: str = "audio.webm") -> str
```

**Current implementation:** Groq Whisper API

**Migration to local Whisper:**
- Replace the Groq client call inside this function only
- Keep function signature identical — callers do not change
- Options: `faster-whisper`, `whisper.cpp` Python bindings

**Notes:**
- Language hardcoded to `"en"` — English input only for now
- This function handles Mic 1 audio only (other person's speech)
- Mic 2 (user's own speech) may use a separate simpler transcription path when implemented

---

### `services/llm.py`

Builds the system prompt, calls the LLM, streams response tokens.

```python
def stream_analysis(turns: list) -> Generator[str, None, None]
```

Yields raw text tokens. Caller is responsible for SSE formatting.

**Prompt design:**
- System prompt includes `USER_PROFILE` from `config.py`
- Conversation formatted as `Other: ...` / `You: ...`
- LLM instructed to return **only raw JSON** — no markdown, no code fences
- Temperature: 0.3 (low, reduces hallucination)
- max_tokens: 300 (enough for JSON output, prevents runaway generation)

**Output contract:**
```json
{
  "intent": "short phrase, max ~6 words, English",
  "summary": "one sentence context, English — NOT displayed to user",
  "reply": "short phrase with key point(s) to say, English"
}
```

**Critical — do not remove this rule from the system prompt:**
> "The reply MUST be truthful. Never invent facts about the user."

**Migration to local LLM:**
- Replace Groq client with local inference (llama-cpp-python, ollama, etc.)
- Keep `stream_analysis` signature and generator behavior identical
- Routes do not change

---

### `routes/transcribe.py`

**`POST /transcribe`**

Receives audio file → runs STT → appends to buffer as `"other"` → returns transcript.

Request: `multipart/form-data`, field name `audio`
Response: `{ "transcript": "..." }`

Does not trigger analysis automatically. Caller must hit `/analyze` separately.

---

### `routes/analyze.py`

**`GET /analyze`** — SSE stream

Reads buffer → calls `stream_analysis()` → streams tokens as SSE events.

```
data: {"token": "..."}          ← each token as it arrives
data: {"done": true, "full": "{...}"}   ← complete JSON when done
data: {"error": "..."}          ← on exception
```

Frontend/display layer is responsible for assembling tokens into full JSON and parsing `intent` + `reply` for display.

**`POST /log_user`**

Logs what the user actually said. Called after the user speaks their response.

Request: `{ "text": "..." }`

This keeps the conversation buffer accurate for subsequent LLM calls. Without this, the LLM only sees the other person's side of the conversation.

**`POST /clear`**

Resets the conversation buffer. Call at the start of each new conversation session.

---

## Full Conversation Cycle

```
[Session starts]
POST /clear

[Other person speaks]
POST /transcribe  →  STT  →  buffer.add("other", text)

[System analyzes]
GET /analyze  →  buffer.get_all()  →  LLM stream  →  { intent, summary, reply }

[Display updates on glasses]
intent → small text below
reply  → large text primary
summary → not displayed

[User speaks their reply]
POST /log_user  →  buffer.add("user", text)

[Other person speaks again]
→ repeat from POST /transcribe
```

---

## Display Constraints

The glasses display has strict constraints that affect what the LLM should generate:

- **Reply:** Short phrase, key point(s) only. Not a full sentence. User speaks in their own words.
- **Intent:** Short phrase, ~4–6 words max. Describes what the other person wants.
- **Summary:** Never displayed. Keep it in the JSON for LLM reasoning continuity but strip it before display.

**Target layout:**
```
studying AI and software           ← reply  (large, primary)
Asking your background             ← intent (small, below)
```

The display must be readable in a single glance (~0.5 seconds) while maintaining eye contact with the other person.

---

## Storage Architecture

```
On-device
├── User profile (JSON)     — loaded into every LLM system prompt call
└── Conversation cache      — raw session data pending cloud sync

Cloud
├── Conversation history    — permanent record
├── People profiles         — NOT stored on device; lazy loaded when needed
└── Learning pipeline       — background job, runs powerful model
                              extracts facts → updates user profile → syncs back
```

**User profile update rules (cloud pipeline):**
- Durable facts are kept: interests, background, skills, communication style
- Outdated facts are overwritten
- Ephemeral facts are ignored: emotions, one-time events, temporary states
- People encountered are stored in cloud only, never on device in MVP

---

## Out of Scope (MVP)

Do not implement these unless explicitly instructed:

- Multi-speaker / group conversation (1-on-1 only)
- People profiles on device
- Fine-tuning pipeline
- Long-term memory on device beyond user profile
- Emotion or psychological analysis
- Hardware integration (glasses, mic array)
- User authentication
- Multi-language support (English input only)

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `GROQ_API_KEY` | Groq API key for Whisper STT and LLaMA LLM |

See `.env.example` for template.

---

## Running Locally

```bash
cd backend
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
cp .env.example .env         # fill in GROQ_API_KEY
python app.py                # starts on http://localhost:5000
```
