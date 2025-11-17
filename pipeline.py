"""
run_pipeline.py — основной скрипт запуска всего pipeline RAG системы.
Выполняет все этапы последовательно.
"""

import logging
from pathlib import Path
import sys

# Добавляем корневую директорию в путь для импортов
sys.path.append(str(Path(__file__).parent))

from data.loader_websites import load_websites, process_and_chunk
from data.loader_questions import load_questions, save_processed_questions
from embeddings.websites_embed import build_kb_embeddings
from retrieval.retriever import run_batch_questions

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_full_pipeline():
    """Запускает полный pipeline RAG системы"""

    logger.info("🚀 ЗАПУСК ПОЛНОГО PIPELINE RAG СИСТЕМЫ")
    logger.info("=" * 50)

    try:

        # ЭТАП 1: Обработка документов
        logger.info("📂 ЭТАП 1: Загрузка и чанкинг документов...")
        df_docs = load_websites()
        process_and_chunk(df_docs, max_words=600, overlap_words=100)
        logger.info("✅ ЭТАП 1 завершен")

        # ЭТАП 2: Создание эмбеддингов
        logger.info("🧠 ЭТАП 2: Создание эмбеддингов...")
        build_kb_embeddings(batch_size=32)
        logger.info("✅ ЭТАП 2 завершен")

        # ЭТАП 3: Подготовка вопросов
        logger.info("❓ ЭТАП 3: Подготовка вопросов...")
        df_questions = load_questions()
        save_processed_questions(df_questions)
        logger.info("✅ ЭТАП 3 завершен")

        # ЭТАП 4: Поиск релевантных документов
        logger.info("🔍 ЭТАП 4: Поиск релевантных документов...")
        run_batch_questions(top_k=5)
        logger.info("✅ ЭТАП 4 завершен")

        logger.info("=" * 50)
        logger.info("🎉 ВСЕ ЭТАПЫ PIPELINE УСПЕШНО ЗАВЕРШЕНЫ!")

    except Exception as e:
        logger.error(f"❌ Ошибка в pipeline: {e}")
        raise


def run_retrieval_only():
    """Запускает только этап поиска (если данные уже подготовлены)"""
    logger.info("🔍 ЗАПУСК ТОЛЬКО ПОИСКА...")
    run_batch_questions(top_k=5)
    logger.info("✅ Поиск завершен")


if __name__ == "__main__":
    # Запуск полного pipeline
    run_full_pipeline()

    # Или только поиск, если данные уже готовы:
    # run_retrieval_only()