"""
websites_embed.py — создание эмбеддингов и FAISS-индекса для готовых чанков.
Загружает предварительно созданные чанки из web_chunks.json.
"""

from pathlib import Path
import json
import numpy as np
import faiss
import logging
from sentence_transformers import SentenceTransformer
from utils.config import (
    PROCESSED_DIR,
    EMBEDDINGS_DIR,
    EMBEDDING_MODEL
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def build_kb_embeddings(batch_size: int = 32):
    """
    Создает эмбеддинги и FAISS-индекс из готовых чанков.
    """
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

    # Загружаем готовые чанки из loader_websites
    chunks_path = PROCESSED_DIR / "web_chunks.json"
    if not chunks_path.exists():
        raise FileNotFoundError(f"❌ Файл с чанками не найден: {chunks_path}")

    logger.info(f"🔍 Загружаем чанки: {chunks_path}")
    with open(chunks_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    if not metadata:
        raise ValueError("❌ Файл чанков пуст")

    # Извлекаем тексты чанков
    texts = [item["text"] for item in metadata]

    logger.info(f"🔹 Загружено {len(texts)} чанков для кодирования")

    # Загружаем модель эмбеддингов
    logger.info(f"🔍 Загружаем модель: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    # Создаем эмбеддинги
    logger.info(f"🧠 Кодируем эмбеддинги (batch_size={batch_size})...")
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True  # Автоматическая нормализация
    )
    vectors = vectors.astype('float32')

    # Дополнительная нормализация для гарантии
    faiss.normalize_L2(vectors)

    logger.info(f"✅ Получено {vectors.shape[0]} векторов (размерность {vectors.shape[1]})")

    # Сохраняем векторы
    vec_file = EMBEDDINGS_DIR / "kb_vectors.npy"
    np.save(vec_file, vectors)
    logger.info(f"💾 Векторы сохранены: {vec_file}")

    # Сохраняем метаданные (уже в нужном формате из loader_websites)
    meta_file = EMBEDDINGS_DIR / "kb_metadata.json"
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    logger.info(f"💾 Метаданные сохранены: {meta_file}")

    # Создаем и сохраняем FAISS индекс
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner Product для normalized векторов = cosine similarity
    index.add(vectors)

    idx_file = EMBEDDINGS_DIR / "kb_index.faiss"
    faiss.write_index(index, str(idx_file))
    logger.info(f"📦 FAISS-индекс сохранён: {idx_file}")

    # Выводим статистику
    logger.info("📊 Статистика созданных эмбеддингов:")
    logger.info(f"   - Количество чанков: {len(metadata)}")
    logger.info(f"   - Размерность векторов: {dim}")
    logger.info(f"   - Размер индекса: {index.ntotal}")
    logger.info(f"   - Тип индекса: {type(index).__name__}")

    # Проверяем согласованность
    if index.ntotal != len(metadata):
        logger.warning(f"⚠️ Несоответствие: индекс {index.ntotal} vs метаданные {len(metadata)}")
    else:
        logger.info("✅ Размеры индекса и метаданных согласованы")

def validate_embeddings():
    """
    Проверяет корректность созданных эмбеддингов.
    """
    logger.info("🔍 Проверка созданных эмбеддингов...")

    # Проверяем существование файлов
    required_files = [
        EMBEDDINGS_DIR / "kb_vectors.npy",
        EMBEDDINGS_DIR / "kb_metadata.json",
        EMBEDDINGS_DIR / "kb_index.faiss"
    ]

    for file_path in required_files:
        if not file_path.exists():
            raise FileNotFoundError(f"❌ Файл не найден: {file_path}")

    # Загружаем и проверяем данные
    vectors = np.load(EMBEDDINGS_DIR / "kb_vectors.npy")

    with open(EMBEDDINGS_DIR / "kb_metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)

    index = faiss.read_index(str(EMBEDDINGS_DIR / "kb_index.faiss"))

    # Проверяем согласованность размеров
    issues = []
    if len(vectors) != len(metadata):
        issues.append(f"Векторы ({len(vectors)}) != метаданные ({len(metadata)})")

    if index.ntotal != len(vectors):
        issues.append(f"Индекс ({index.ntotal}) != векторы ({len(vectors)})")

    # Проверяем размерность
    if vectors.shape[0] == 0:
        issues.append("Нулевое количество векторов")

    if vectors.shape[1] == 0:
        issues.append("Нулевая размерность векторов")

    if issues:
        error_msg = "Проблемы в эмбеддингах: " + "; ".join(issues)
        logger.error(f"❌ {error_msg}")
        return False
    else:
        logger.info("✅ Все проверки эмбеддингов пройдены")
        return True

if __name__ == "__main__":
    try:
        logger.info("🚀 Запуск создания эмбеддингов из готовых чанков...")

        # Создаем эмбеддинги
        build_kb_embeddings(batch_size=32)

        # Проверяем результат
        if validate_embeddings():
            logger.info("🎉 Эмбеддинги успешно созданы и проверены!")
        else:
            logger.error("💥 Обнаружены проблемы в созданных эмбеддингах")

    except Exception as e:
        logger.error(f"❌ Ошибка в websites_embed: {e}")
        raise