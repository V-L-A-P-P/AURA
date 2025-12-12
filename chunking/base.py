import logging
import re
from typing import List

from utils.config import CHUNK_SIZE, CHUNK_OVERLAP
from .filters import is_noisy_chunk, create_overlap

logger = logging.getLogger(__name__)


def split_text_to_chunks(
    text: str,
    max_words: int = CHUNK_SIZE,
    overlap_words: int = CHUNK_OVERLAP,
    min_words: int | None = None,
) -> List[str]:
    """
    Split text into chunks by word count.
    Used as a simple baseline / fallback strategy.
    """

    if max_words <= 0:
        raise ValueError("CHUNK_SIZE must be greater than 0")
    if overlap_words >= max_words:
        raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")

    words = text.split()
    total_words = len(words)

    if min_words is None:
        min_words = max_words // 2

    # If the text is short enough, return it as a single chunk
    if total_words <= min_words:
        return [text.strip()]

    chunks: List[str] = []
    start = 0

    while start < total_words:
        end = min(start + max_words, total_words)
        chunk = " ".join(words[start:end]).strip()

        if chunk:
            chunks.append(chunk)

        if end >= total_words:
            break

        start += max_words - overlap_words

    logger.info(
        f"Base chunking: created {len(chunks)} chunks "
        f"({total_words} words, max_words={max_words}, overlap={overlap_words})"
    )

    return chunks


def split_text_to_chunks_advanced(
    text: str,
    max_words: int = CHUNK_SIZE,
    overlap_words: int = CHUNK_OVERLAP,
    min_words: int | None = None,
) -> List[str]:
    """
    Sentence-based chunking with:
    - basic web-noise removal
    - noisy chunk filtering
    - overlap between chunks
    """

    if not text or not text.strip():
        return []

    # Remove common web noise
    text = re.sub(
        r"(©|cookie|javascript|feedback|contacts|all rights reserved).*",
        "",
        text,
        flags=re.I,
    )
    text = re.sub(r"\s+", " ", text).strip()

    # Sentence splitting
    sentence_endings = r"(?<=[.!?…])\s+(?=[A-ZА-Я])"
    sentences = [s.strip() for s in re.split(sentence_endings, text) if s.strip()]

    if not sentences:
        return [text]

    chunks: List[str] = []
    current_chunk: List[str] = []
    current_size = 0

    for sentence in sentences:
        word_count = len(sentence.split())

        if current_size + word_count > max_words and current_chunk:
            chunk_text = " ".join(current_chunk)

            if not is_noisy_chunk(chunk_text):
                chunks.append(chunk_text)

            overlap_text = create_overlap(current_chunk, overlap_words)
            current_chunk = [overlap_text] if overlap_text else []
            current_size = len(overlap_text.split()) if overlap_text else 0

        current_chunk.append(sentence)
        current_size += word_count

    if current_chunk:
        chunk_text = " ".join(current_chunk)
        if not is_noisy_chunk(chunk_text):
            chunks.append(chunk_text)

    if min_words:
        chunks = [c for c in chunks if len(c.split()) >= min_words]

    logger.debug(f"Advanced chunking: created {len(chunks)} chunks")

    return chunks
