import re

# --- Synonyms mapping (converted from your JS version) ---
synonym_map = {
    "automobile": "car",
    "vehicle": "car",
    "climate": "weather",
    "temperature": "weather",
    "instructor": "teacher",
    "student": "learner",
    "start": "begin",
    "end": "finish",
    "study": "research",
    "write": "compose"
}

# --- Basic stopwords (similar to your JS implementation) ---
stopwords = set("""
the is are a an of in on at to for with by and or but from this that these those be being been have has had
""".split())
# -----------------------------------------------------
#                 TEXT CLEANING
# -----------------------------------------------------
def clean_text(text: str):
    """Convert text into normalized tokens (lowercase, no punctuation, synonyms applied)."""
    
    # Lowercase
    text = text.lower()
    
    # Remove punctuation
    text = re.sub(r"[^\w\s]", "", text)
    
    # Tokenize
    words = text.split()
    
    # Remove stopwords + apply synonyms
    words = [synonym_map.get(w, w) for w in words if w not in stopwords]
    
    return words
# -----------------------------------------------------
#               TEXT COMPARISON LOGIC
# -----------------------------------------------------
def compare_texts(text1: str, text2: str):
    """Returns similarity ratio between two texts."""
    words1 = set(clean_text(text1))
    words2 = set(clean_text(text2))

    if not words1 and not words2:
        return {"similarity": 0}

    shared = len(words1 & words2)
    union = len(words1 | words2)

    similarity = shared / union if union > 0 else 0

    return {"similarity": similarity}
