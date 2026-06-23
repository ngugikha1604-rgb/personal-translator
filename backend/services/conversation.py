from collections import deque
from config import CONVERSATION_MAX_TURNS


class ConversationBuffer:
    def __init__(self, max_turns=CONVERSATION_MAX_TURNS):
        self.buffer = deque(maxlen=max_turns)

    def add(self, speaker: str, text: str):
        self.buffer.append({"speaker": speaker, "text": text})

    def add_other(self, text: str):
        self.add("other", text)

    def add_user(self, text: str):
        self.add("user", text)

    def get_all(self) -> list:
        return list(self.buffer)

    def clear(self):
        self.buffer.clear()


# Shared singleton for the whole app
conversation = ConversationBuffer()

