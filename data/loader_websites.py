"""
loader_websites.py — load and preprocess website documents.
Splits texts into chunks and generates metadata.
"""

import json
import logging
from pathlib import Path

import pandas as pd

from utils.config import (
    RAW_DIR,
    PROCESSED_DIR,
    CHUNKS_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    MIN_SIZE,
)
from chunking.base import (
    split_text_to_chunks_advanced,
    split_text_to_chunks
)

from chunking.semantic import semantic_chunk
from chunking.recursive import recursive_chunking


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_websites(csv_path: Path | None = None, for_test=False) -> pd.DataFrame:
    """
    Load websites from CSV.
    Required columns: web_id, url, kind, title, text
    """
    if csv_path is None:
        csv_path = RAW_DIR / "websites_updated.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"Websites file not found: {csv_path}")
    if for_test:
        df = pd.read_csv(csv_path, dtype=str).iloc[:10]
    else:
        df = pd.read_csv(csv_path, dtype=str)

    required_cols = {"web_id", "url", "kind", "title", "text"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = (
        df.dropna(subset=["web_id", "text"])
          .drop_duplicates(subset=["web_id"])
    )

    df["web_id"] = df["web_id"].astype(str).str.strip()
    df["url"] = df["url"].astype(str).str.strip()
    df["title"] = df["title"].astype(str).str.strip()
    df["text"] = df["text"].astype(str).str.strip()

    logger.info(f"Loaded {len(df)} documents")

    return df


def process_and_chunk(
    df: pd.DataFrame,
    chunking_type: str = "semantic",
    chunks_dir: Path | None = None,
    meta_output_path: Path | None = None,
    max_words: int | None = None,
    overlap_words: int | None = None,
    min_words: int | None = None,
):
    """
    Chunk documents and save chunk files + metadata.
    """
    if chunks_dir is None:
        chunks_dir = CHUNKS_DIR
    if meta_output_path is None:
        meta_output_path = PROCESSED_DIR / "web_chunks.json"

    chunks_dir.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    metadata: list[dict] = []
    total_chunks = 0

    for _, row in df.iterrows():
        web_id = row["web_id"]
        title = row["title"]
        url = row["url"]
        text = row["text"]

        if chunking_type == "base":
            chunks = split_text_to_chunks(
                text,
                max_words=max_words or CHUNK_SIZE,
                overlap_words=overlap_words or CHUNK_OVERLAP,
                min_words=min_words or MIN_SIZE,
            )

        elif chunking_type == "advanced":
            chunks = split_text_to_chunks_advanced(
                text,
                max_words=max_words or CHUNK_SIZE,
                overlap_words=overlap_words or CHUNK_OVERLAP,
                min_words=min_words or MIN_SIZE,
            )

        elif chunking_type == "semantic":
            chunks = semantic_chunk(
                text,
                max_words=max_words or CHUNK_SIZE,
                min_words=min_words or MIN_SIZE,
            )

        elif chunking_type == "recursive":
            chunks = recursive_chunking(
                text,
                chunk_size=max_words or CHUNK_SIZE,
                overlap=overlap_words or CHUNK_OVERLAP,
            )

        else:
            raise ValueError(
                f"Invalid chunking type: {chunking_type}. "
                f"Use one of: base, advanced, semantic, recursive"
            )

        if not chunks:
            logger.warning(f"Document {web_id} produced no chunks")
            continue

        for idx, chunk in enumerate(chunks):
            chunk_id = f"{web_id}_{idx}"
            chunk_file = chunks_dir / f"{chunk_id}.txt"

            with open(chunk_file, "w", encoding="utf-8") as f:
                f.write(chunk)

            metadata.append(
                {
                    "web_id": web_id,
                    "chunk_id": chunk_id,
                    "title": title,
                    "url": url,
                    "file": str(chunk_file),
                    "text": chunk,
                }
            )

        total_chunks += len(chunks)

    with open(meta_output_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved {total_chunks} chunks to {meta_output_path}")

    return metadata


if __name__ == "__main__":
    df_docs = load_websites(for_test=True)
    process_and_chunk(
        df_docs,
        chunking_type="base",
        max_words=600,
        overlap_words=100,

    )
