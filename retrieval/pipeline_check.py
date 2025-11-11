"""
validate_pipeline.py — проверка согласованности данных между этапами pipeline
"""

import json
import numpy as np
import faiss
import pandas as pd
import logging
from pathlib import Path
from utils.config import EMBEDDINGS_DIR, PROCESSED_DIR, RAW_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def validate_chunk_consistency():
    """Проверяет согласованность чанков между этапами"""
    logger.info("🔍 Проверка согласованности чанков...")

    # 1. Проверяем исходные данные
    websites_path = RAW_DIR / "websites_updated.csv"
    if not websites_path.exists():
        logger.error(f"❌ Исходный файл не найден: {websites_path}")
        return False

    df_websites = pd.read_csv(websites_path, dtype=str)
    logger.info(f"📊 Исходные сайты: {len(df_websites)} записей")

    # 2. Проверяем чанки
    chunks_path = PROCESSED_DIR / "web_chunks.json"
    if not chunks_path.exists():
        logger.error(f"❌ Файл чанков не найден: {chunks_path}")
        return False

    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks_data = json.load(f)

    logger.info(f"📊 Чанки: {len(chunks_data)} записей")

    # 3. Проверяем метаданные эмбеддингов
    meta_path = EMBEDDINGS_DIR / "kb_metadata.json"
    if not meta_path.exists():
        logger.error(f"❌ Метаданные эмбеддингов не найдены: {meta_path}")
        return False

    with open(meta_path, "r", encoding="utf-8") as f:
        embedding_meta = json.load(f)

    logger.info(f"📊 Метаданные эмбеддингов: {len(embedding_meta)} записей")

    # 4. Проверяем FAISS индекс
    index_path = EMBEDDINGS_DIR / "kb_index.faiss"
    if not index_path.exists():
        logger.error(f"❌ FAISS индекс не найден: {index_path}")
        return False

    index = faiss.read_index(str(index_path))
    logger.info(f"📊 FAISS индекс: {index.ntotal} векторов")

    # 5. Проверяем согласованность размеров
    issues = []

    if len(chunks_data) != len(embedding_meta):
        issues.append(f"Размер чанков ({len(chunks_data)}) != размер метаданных ({len(embedding_meta)})")

    if index.ntotal != len(embedding_meta):
        issues.append(f"Размер индекса ({index.ntotal}) != размер метаданных ({len(embedding_meta)})")

    # 6. Проверяем уникальность chunk_id
    chunk_ids = [chunk["chunk_id"] for chunk in chunks_data]
    embedding_chunk_ids = [meta["chunk_id"] for meta in embedding_meta]

    if len(chunk_ids) != len(set(chunk_ids)):
        issues.append("Найдены дублирующиеся chunk_id в чанках")

    if len(embedding_chunk_ids) != len(set(embedding_chunk_ids)):
        issues.append("Найдены дублирующиеся chunk_id в метаданных эмбеддингов")

    # 7. Вывод результатов
    if issues:
        logger.error("❌ Найдены проблемы согласованности:")
        for issue in issues:
            logger.error(f"   - {issue}")
        return False
    else:
        logger.info("✅ Все проверки согласованности пройдены")
        return True


def validate_questions():
    """Проверяет корректность вопросов"""
    logger.info("🔍 Проверка вопросов...")

    questions_path = PROCESSED_DIR / "questions.json"
    if not questions_path.exists():
        logger.error(f"❌ Файл вопросов не найден: {questions_path}")
        return False

    try:
        with open(questions_path, "r", encoding="utf-8") as f:
            questions = json.load(f)

        logger.info(f"📊 Вопросы: {len(questions)} записей")

        # Проверяем наличие обязательных полей
        for i, q in enumerate(questions):
            if "q_id" not in q or "query" not in q:
                logger.error(f"❌ Вопрос {i} не содержит обязательных полей")
                return False

        logger.info("✅ Вопросы корректны")
        return True

    except Exception as e:
        logger.error(f"❌ Ошибка при проверке вопросов: {e}")
        return False


if __name__ == "__main__":
    logger.info("🚀 Запуск проверки pipeline...")

    chunks_ok = validate_chunk_consistency()
    questions_ok = validate_questions()

    if chunks_ok and questions_ok:
        logger.info("🎉 Все проверки пройдены! Pipeline готов к работе.")
    else:
        logger.error("💥 Обнаружены проблемы в pipeline. Необходимо исправить перед запуском.")