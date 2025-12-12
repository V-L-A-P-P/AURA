import re
from typing import List


def is_noisy_chunk(text: str) -> bool:
    tokens = re.findall(r"\w+", text)
    if not tokens:
        return True

    noise = sum(
        1 for t in tokens
        if re.fullmatch(r"\d+|\d+\.\d+|п|№|таблица|приложение", t.lower())
    )

    return noise / len(tokens) > 0.4


def create_overlap(sentences: List[str], overlap_words: int) -> str:
    if not sentences:
        return ""

    words = " ".join(sentences).split()
    return " ".join(words[-overlap_words:]) if len(words) > overlap_words else " ".join(words)
