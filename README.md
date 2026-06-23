# Personal Translator — Conversation Copilot

A real-time AI copilot that helps the user understand intent and respond faster in live English conversations.

Not a translator. Not a chatbot. A **cognitive co-processor** that reduces conversation latency.

---

## Core Problem

When conversing in a second language, the brain must:

1. Hear the words
2. Decode language
3. Understand intent
4. Plan a response
5. Translate to target language
6. Speak

Steps 3–5 can take 2–5 seconds. This is **Conversation Latency** — the real problem this project solves.

---

## How It Works

The system listens continuously to the other person's speech. As they speak, it processes audio in rolling chunks and updates the display in real-time — so by the time they finish talking, the user already knows what to say.

**Output per utterance:**
```json
{
  "intent": "Asking about your field of study",
  "summary": "They want to know your academic background.",
  "reply": "studying AI and software"
}
```

`intent` and `reply` are displayed on the glasses. `summary` is used internally by the LLM for reasoning but never shown to the user.

---

## Display Design

Target display: **optical waveguide glasses** (similar to waveguide tech in Xreal, etc.)

- User maintains full eye contact with the other person
- Information appears as an overlay, only visible to the user
- Designed to be read in a single glance — no scrolling, no reading paragraphs

**Layout:**
```
studying AI and software           ← reply (large, primary)
Asking your background             ← intent (small, secondary)
```

**Reply format:** Short phrase capturing the key point(s) to say. Not a full sentence — the user speaks in their own words. This keeps responses natural and truthful.

---

## Architecture Overview

### Two Independent Paths

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REAL-TIME PATH (latency critical)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Mic 1 (outward, beamforming)
  ↓
Noise suppression (RNNoise / DeepFilterNet)
  ↓
VAD — Voice Activity Detection (silero-VAD)
  ↓
Whisper STT (local, runs on device)
  ↓
Conversation Buffer (rolling window, last N turns)
  ↓
LLM — Intent + Reply generation (local fine-tuned model)
  ↓
JSON: { intent, summary, reply }
  ↓
Glasses display (optical waveguide overlay)


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LEARNING PATH (background, no latency requirement)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Mic 2 (toward user's mouth)
  ↓
Capture user's own speech
  ↓
Log to conversation buffer as "user"
  ↓
Cache conversation locally after session ends
  ↓
Push to cloud (background)
  ↓
Powerful cloud model analyzes conversation
  ↓
Extract facts worth storing about user
  ↓
Update user profile
  ↓
Sync updated profile back to device
```

---

## Audio Hardware Design

**Dual microphone setup:**

| Mic | Placement | Purpose |
|-----|-----------|---------|
| Mic 1 | Outward-facing on frame | Capture other person's speech |
| Mic 2 | Toward user's mouth | Capture user's own speech |

Two channels are fully independent — no complex voice filtering needed because separation happens at hardware level.

Mic 1 uses **beamforming** (2–3 mic array) to focus audio pickup toward the front and reject surrounding noise.

**Software audio pipeline (Mic 1 only):**
```
Raw audio → Noise suppression → VAD → STT
```

---

## Processing Model

The system processes audio **continuously in rolling chunks**, not just on silence detection.

Every ~N seconds (exact value TBD via empirical testing, estimate 1.5–2s), the system:
1. Takes latest audio chunk from Mic 1
2. Runs STT
3. Updates conversation buffer
4. Runs LLM analysis
5. Updates glasses display

This means intent and reply start appearing while the other person is still speaking. By the time they finish, the user is already ready to respond.

---

## Storage Architecture

```
On-device (lightweight, always available)
├── User profile        — facts about the user, used in every LLM call
└── Conversation cache  — temporary, pending cloud sync

Cloud (no size constraint, background only)
├── Conversation history
├── People profiles     — lazy loaded only when needed, not stored on device
└── Learning pipeline   — analyzes conversations, updates user profile
```

**User profile update flow:**
- Cloud model reads cached conversation after session
- Extracts durable facts: interests, background, communication patterns
- Overwrites outdated facts, adds new ones
- Syncs updated profile to device

---

## MVP Scope

**Phase 1 — Desktop proof of concept (current)**
- Web frontend as temporary display
- Groq Whisper API for STT
- Groq LLaMA for LLM
- Manual audio recording (push to transcribe)
- Validate: intent detection quality, reply usefulness, latency

**Phase 2 — Mobile**
- Move to mobile interface
- Begin continuous audio processing

**Phase 3 — Glasses integration**
- Optical waveguide display
- Dual mic hardware
- On-device STT (faster-whisper or whisper.cpp)
- Beamforming audio pipeline

**Phase 4 — Personalized local model**
- Fine-tuned small model trained on user's conversation history
- Replies increasingly match user's actual thinking and style
- Fully offline capable

---

## Current Tech Stack

| Layer | MVP (now) | Production target |
|-------|-----------|-------------------|
| STT | Groq Whisper API | Local Whisper (faster-whisper / whisper.cpp) |
| LLM | Groq LLaMA 3.3 70B | Local fine-tuned small model |
| Noise suppression | — | RNNoise / DeepFilterNet |
| VAD | — | silero-VAD |
| Backend | Python / Flask | Python (optimized) |
| Display | Web frontend | Optical waveguide glasses overlay |
| Storage | In-memory only | On-device profile + cloud learning |

---

## Core Principles

**Latency first** — Every design decision is evaluated by one question: does this reduce time-to-response?

**Truthfulness** — The LLM must never invent facts about the user. Reply suggestions must reflect reality.

**Glanceability** — Display is designed to be understood in under 0.5 seconds. No long text.

**User decides** — The system suggests. The user chooses what to say. AI is a co-processor, not a replacement.

**1-on-1 only (MVP)** — Multi-speaker / group conversation is out of scope for now.

---

## Project Structure

```
personal_translator/
├── README.md
├── AGENTS.md
├── backend/
│   ├── app.py                  # Flask app factory
│   ├── config.py               # Models, user profile, constants
│   ├── requirements.txt
│   ├── .env.example
│   ├── routes/
│   │   ├── transcribe.py       # POST /transcribe
│   │   └── analyze.py          # GET /analyze (SSE), POST /log_user, POST /clear
│   └── services/
│       ├── stt.py              # Speech-to-text abstraction
│       ├── llm.py              # LLM prompt + streaming
│       └── conversation.py     # ConversationBuffer singleton
└── frontend/
    └── ...                     # Temporary desktop UI (replaced by glasses)
```

---

## Setup

```bash
cd backend
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
cp .env.example .env         # Add GROQ_API_KEY
python app.py                # Starts on port 5000
```
