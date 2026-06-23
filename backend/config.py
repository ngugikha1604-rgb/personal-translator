import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

WHISPER_MODEL = "whisper-large-v3-turbo"
LLM_MODEL = "llama-3.3-70b-versatile"
CONVERSATION_MAX_TURNS = 10
USER_PROFILE_PATH = os.path.join(os.path.dirname(__file__), "user_profile.json")
