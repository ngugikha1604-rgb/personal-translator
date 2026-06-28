import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# STT provider: "faster_whisper" (local) or "groq" (cloud)
STT_PROVIDER = os.getenv("STT_PROVIDER", "faster_whisper")
FASTER_WHISPER_MODEL = os.getenv("FASTER_WHISPER_MODEL", "tiny.en")
FASTER_WHISPER_DEVICE = os.getenv("FASTER_WHISPER_DEVICE", "cpu")
FASTER_WHISPER_COMPUTE_TYPE = os.getenv("FASTER_WHISPER_COMPUTE_TYPE", "int8")
FASTER_WHISPER_CPU_THREADS = int(os.getenv("FASTER_WHISPER_CPU_THREADS", "4"))

# Groq models
WHISPER_MODEL = "whisper-large-v3-turbo"
LLM_MODEL = "llama-3.3-70b-versatile"
LLM_MAX_TOKENS = 300
LLM_TEMPERATURE = 0.3

# Legacy Ollama models — kept for reference but no longer used in main flow
# SMALL_MODEL = os.getenv("SMALL_MODEL", "qwen2.5:0.5b")
# LARGE_MODEL = os.getenv("LARGE_MODEL", "qwen3:4b")

CONVERSATION_MAX_TURNS = 10
USER_PROFILE_PATH = os.path.join(os.path.dirname(__file__), "user_profile.json")
