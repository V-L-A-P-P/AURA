"""
websites_embed.py — build embeddings and FAISS index for document chunks.
Uses preprocessed chunks from web_chunks.json.
"""

import json
import logging

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from utils.config import PROCESSED_DIR, EMBEDDINGS_DIR, EMBEDDING_MODEL


logger = logging.getLogger(__name__)


def build_kb_embeddings(batch_size: int = 32) -> None:
    """Build embeddings and FAISS index for chunked documents."""

    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

    chunks_path = PROCESSED_DIR / "web_chunks.json"
    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunks file not found: {chunks_path}")

    logger.info("Loading chunks")
    with open(chunks_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    if not metadata:
        raise ValueError("Chunks file is empty")

    texts = [item["text"] for item in metadata]
    logger.info("Encoding %d chunks", len(texts))

    model = SentenceTransformer(EMBEDDING_MODEL)

    vectors = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    faiss.normalize_L2(vectors)

    np.save(EMBEDDINGS_DIR / "kb_vectors.npy", vectors)

    with open(EMBEDDINGS_DIR / "kb_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    faiss.write_index(index, str(EMBEDDINGS_DIR / "kb_index.faiss"))

    if index.ntotal != len(metadata):
        logger.warning(
            "Index size mismatch: index=%d, metadata=%d",
            index.ntotal,
            len(metadata),
        )

    logger.info(
        "Embeddings built: chunks=%d, dim=%d",
        index.ntotal,
        vectors.shape[1],
    )


def validate_embeddings() -> bool:
    """Validate consistency of saved embeddings and index."""

    files = [
        EMBEDDINGS_DIR / "kb_vectors.npy",
        EMBEDDINGS_DIR / "kb_metadata.json",
        EMBEDDINGS_DIR / "kb_index.faiss",
    ]

    for path in files:
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")

    vectors = np.load(EMBEDDINGS_DIR / "kb_vectors.npy")

    with open(EMBEDDINGS_DIR / "kb_metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)

    index = faiss.read_index(str(EMBEDDINGS_DIR / "kb_index.faiss"))

    if len(vectors) != len(metadata) or index.ntotal != len(vectors):
        logger.error("Embedding consistency check failed")
        return False

    logger.info("Embedding consistency check passed")
    return True


if __name__ == "__main__":
    build_kb_embeddings(batch_size=32)
    validate_embeddings()
