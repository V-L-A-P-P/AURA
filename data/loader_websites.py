"""
loader_websites.py — загрузка и предварительная обработка файла websites_updated.csv.
Разбивает тексты сайтов на чанки и создаёт метаданные.
"""

import pandas as pd
from pathlib import Path
import json
import logging
from utils.config import RAW_DIR, PROCESSED_DIR, CHUNKS_DIR
from utils.chunk_utils import split_text_to_chunks

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_websites(csv_path: Path = None) -> pd.DataFrame:
    if csv_path is None:
        csv_path = RAW_DIR / "websites_updated.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"❌ Документы-файл не найден: {csv_path}")
    df = pd.read_csv(csv_path, dtype=str)
    required = ["web_id", "url", "kind", "title", "text"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"❌ В файле сайтов нет столбца «{col}»")
    df = df.dropna(subset=["web_id", "text"]).drop_duplicates(subset=["web_id"])
    df["web_id"] = df["web_id"].astype(str).str.strip()
    df["url"]    = df["url"].astype(str).str.strip()
    df["title"]  = df["title"].astype(str).str.strip()
    df["text"]   = df["text"].astype(str).str.strip()
    logger.info(f"🔹 Загружено {len(df)} документов.")
    return df

def process_and_chunk(
    df: pd.DataFrame,
    chunks_dir: Path = None,
    meta_output_path: Path = None,
    max_words: int = 200,
    overlap_words: int = 50,
    min_words: int = 30
):
    if chunks_dir is None:
        chunks_dir = CHUNKS_DIR
    if meta_output_path is None:
        meta_output_path = PROCESSED_DIR / "web_chunks.json"

    chunks_dir.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    metadata = []
    total_chunks = 0
    for _, row in df.iterrows():
        web_id = row["web_id"]
        title  = row["title"]
        url    = row["url"]
        text   = row["text"]
        chunks = split_text_to_chunks(text, max_words=max_words, overlap_words=overlap_words)
        if not chunks:
            logger.warning(f"⚠️ Документ {web_id} не дал ни одного чанка.")
            continue
        for idx, chunk in enumerate(chunks):
            chunk_id = f"{web_id}_{idx}"
            chunk_file = chunks_dir / f"{chunk_id}.txt"
            with open(chunk_file, "w", encoding="utf-8") as f:
                f.write(chunk)
            metadata.append({
                "web_id": web_id,
                "chunk_id": chunk_id,
                "title": title,
                "url": url,
                "file": str(chunk_file),
                "text": chunk
            })
        total_chunks += len(chunks)

    with open(meta_output_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    logger.info(f"✅ Сохранены {total_chunks} чанков документов в {meta_output_path}")
    return metadata

if __name__ == "__main__":
    df_docs = load_websites()
    process_and_chunk(df_docs)
