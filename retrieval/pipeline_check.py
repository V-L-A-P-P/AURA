"""
validate_pipeline.py — consistency checks for RAG pipeline artifacts
"""

import json
import logging

import faiss

from utils.config import EMBEDDINGS_DIR, PROCESSED_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------

def validate_chunks_and_embeddings() -> bool:
    """Validate consistency between chunks, metadata and FAISS index"""

    logger.info("Validating chunks and embeddings...")

    chunks_path = PROCESSED_DIR / "web_chunks.json"
    meta_path = EMBEDDINGS_DIR / "kb_metadata.json"
    index_path = EMBEDDINGS_DIR / "kb_index.faiss"

    for p in [chunks_path, meta_path, index_path]:
        if not p.exists():
            logger.error(f"Missing required file: {p}")
            return False

    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    index = faiss.read_index(str(index_path))

    issues: list[str] = []

    if len(chunks) != len(meta):
        issues.append(
            f"Chunks ({len(chunks)}) != metadata ({len(meta)})"
        )

    if index.ntotal != len(meta):
        issues.append(
            f"FAISS index ({index.ntotal}) != metadata ({len(meta)})"
        )

    chunk_ids = [c["chunk_id"] for c in chunks]
    meta_ids = [m["chunk_id"] for m in meta]

    if len(chunk_ids) != len(set(chunk_ids)):
        issues.append("Duplicate chunk_id in web_chunks.json")

    if len(meta_ids) != len(set(meta_ids)):
        issues.append("Duplicate chunk_id in kb_metadata.json")

    if set(chunk_ids) != set(meta_ids):
        issues.append("chunk_id mismatch between chunks and metadata")

    if issues:
        logger.error("Chunk/embedding consistency errors:")
        for i in issues:
            logger.error(f"  - {i}")
        return False

    logger.info("Chunks and embeddings are consistent")
    return True


# ---------------------------------------------------------------------

def validate_questions() -> bool:
    """Validate processed questions file"""

    logger.info("Validating questions...")

    questions_path = PROCESSED_DIR / "questions.json"
    if not questions_path.exists():
        logger.error(f"Missing questions file: {questions_path}")
        return False

    try:
        with open(questions_path, "r", encoding="utf-8") as f:
            questions = json.load(f)

        if not questions:
            logger.error("Questions file is empty")
            return False

        for i, q in enumerate(questions):
            if "q_id" not in q or "query" not in q:
                logger.error(f"Invalid question at index {i}")
                return False

        logger.info(f"Questions OK ({len(questions)} items)")
        return True

    except Exception as e:
        logger.error(f"Failed to read questions: {e}")
        return False


# ---------------------------------------------------------------------

if __name__ == "__main__":
    logger.info("Running pipeline validation")

    ok_chunks = validate_chunks_and_embeddings()
    ok_questions = validate_questions()

    if ok_chunks and ok_questions:
        logger.info("Pipeline validation PASSED")
    else:
        logger.error("Pipeline validation FAILED")
