"""
utterance_filter.py — Lightweight utterance classifier.

Runs before LLM call. Reduces unnecessary model invocations.
Backchannel utterances (okay, yeah, right) go to conversation buffer
but skip LLM. Noise is dropped entirely.

Heuristic implementation — zero dependencies, <1ms.
Upgrade path: replace with small ML classifier when needed.
"""

# Single-word backchannel markers — any of these alone is not a real turn
_BACKCHANNEL_WORDS = frozenset({
    "okay", "ok", "yeah", "yep", "yup", "uhuh", "mm",
    "mhm", "uh-huh", "hmm", "huh", "ah", "oh", "right",
    "sure", "cool", "nice", "great", "good", "fine",
    "alright", "k", "mkay", "mmm",
})

# Words that indicate a substantive statement (negates backchannel guess)
_VERB_INDICATORS = frozenset({
    "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did",
    "can", "could", "will", "would", "shall", "should",
    "may", "might", "must", "need", "want", "think",
    "know", "see", "say", "tell", "ask", "go", "make",
    "take", "come", "get", "give", "use", "work",
})


def classify_utterance(transcript: str) -> str:
    """Classify a transcript chunk.

    Returns one of:
        "backchannel"  — filler/acknowledgment, buffer but skip LLM
        "question"     — contains question syntax
        "statement"    — substantive turn, run LLM
        "noise"        — empty or garbage, drop entirely
    """
    text = transcript.strip()
    if not text:
        return "noise"

    words = text.split()

    # Single-word backchannels
    if len(words) == 1:
        return "backchannel" if words[0].lower().strip(".!?") in _BACKCHANNEL_WORDS else "statement"

    # Very short utterances (2 words) without a verb — likely backchannel
    if len(words) <= 3:
        # Check if any word looks like a verb
        lower_words = {w.lower().strip(".!?") for w in words}
        if not lower_words & _VERB_INDICATORS:
            # Could still be a substantive 2-word turn like "I agree"
            # Check if starts with personal pronoun + verb-like word
            first_word = words[0].lower()
            second_word_raw = words[1].lower().strip(".!?")
            if first_word in ("i", "we", "you", "he", "she", "they", "it"):
                if second_word_raw in _VERB_INDICATORS or second_word_raw.endswith("ing") or second_word_raw.endswith("ed"):
                    return "statement"
            return "backchannel"

    # Questions: contains question mark or starts with wh-word
    if "?" in text:
        return "question"
    first = words[0].lower().strip(".!?")
    if first in ("what", "when", "where", "why", "who", "whom",
                 "whose", "which", "how", "is", "are", "was",
                 "were", "do", "does", "did", "can", "could",
                 "will", "would", "shall", "should", "have",
                 "has", "had", "am"):
        return "question"

    # Everything else is a statement
    return "statement"