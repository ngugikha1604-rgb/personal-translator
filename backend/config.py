import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

USER_PROFILE = {
    "interests": ["AI", "Programming", "Competitive Programming"],
    "communication_style": ["logical", "concise", "truthful"]
}

WHISPER_MODEL = "whisper-large-v3-turbo"
LLM_MODEL = "llama-3.3-70b-versatile"
CONVERSATION_MAX_TURNS = 10
