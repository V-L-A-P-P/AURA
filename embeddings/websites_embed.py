"""
websites_embed.py — создание эмбеддингов и FAISS-индекса для документов (websites).

Шаги:
 1. Загружается метаданные чанков (web_chunks.json) из PROCESSED_DIR.
 2. Чанки кодируются через SentenceTransformer.
 3. Сохраняются:
    - векторный файл: kb_vectors.npy
    - метаданные: kb_metadata.json
    - индекс FAISS:     kb_index.faiss
"""

from pathlib import Path
import json
import numpy as np
import faiss
import logging
from sentence_transformers import SentenceTransformer
from utils.config import PROCESSED_DIR, EMBEDDINGS_DIR, EMBEDDING_MODEL

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def build_kb_embeddings(batch_size: int = 16):
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

    metadata_path = PROCESSED_DIR / "web_chunks.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"❌ Метаданные чанков не найдены: {metadata_path}")
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    texts = [rec["text"] for rec in metadata if ("text" in rec and rec["text"])]
    if not texts:
        raise ValueError("❌ Нет текстов для кодирования.")

    logger.info(f"🔹 Кодируем {len(texts)} чанков.")
    model = SentenceTransformer(EMBEDDING_MODEL)

    logger.info(f"🔢 Кодируем {len(texts)} чанков (batch_size={batch_size}) …")
    vectors = model.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)
    vectors = vectors.astype('float32')
    logger.info(f"✅ Получено {vectors.shape[0]} векторов размерности {vectors.shape[1]}")

    vec_file  = EMBEDDINGS_DIR / "kb_vectors.npy"
    meta_file = EMBEDDINGS_DIR / "kb_metadata.json"
    np.save(vec_file, vectors)
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    logger.info(f"✅ Сохранены файлы: {vec_file}, {meta_file}")

    dim   = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)
    idx_file = EMBEDDINGS_DIR / "kb_index.faiss"
    faiss.write_index(index, str(idx_file))
    logger.info(f"📦 FAISS-индекс сохранён: {idx_file}")

if __name__ == "__main__":
    try:
        build_kb_embeddings(batch_size=16)
    except Exception as e:
        logger.error(f"❌ Ошибка при создании эмбеддингов документов: {e}")
