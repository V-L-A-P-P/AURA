"""
config.py — централизованная конфигурация для RAG-проекта.
Определяет пути, параметры моделей, чанкинга и индексации.
"""

from pathlib import Path
import os
from dotenv import load_dotenv

# === Загрузка переменных окружения (.env) ===
# Позволяет переопределять параметры без изменения кода
load_dotenv()

# === Основные директории проекта ===
# BASE_DIR — корень rag_project/
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CHUNKS_DIR = DATA_DIR / "chunks"

EMBEDDINGS_DIR = BASE_DIR / "embeddings"
RETRIEVAL_DIR = BASE_DIR / "retrieval"
LLM_DIR = BASE_DIR / "llm"

# === Параметры чанкинга ===
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))        # длина чанка в символах
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 100))   # перекрытие между чанками

# === Настройки моделей ===
# Можно указать другие модели через .env без изменения кода
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "cointegrated/rubert-tiny2"#ai-forever/sbert_large_nlu_ru"
)
GENERATION_MODEL = os.getenv(
    "GENERATION_MODEL",
    "unsloth/Llama-3.2-3B-Instruct"
)

# === Настройки FAISS / индекса ===
FAISS_INDEX_PATH = EMBEDDINGS_DIR / "vectors.faiss"
FAISS_METADATA_PATH = EMBEDDINGS_DIR / "metadata.csv"

# === Логирование и отладка ===
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

# === Проверка и автоматическое создание директорий ===
for directory in [DATA_DIR, RAW_DIR, PROCESSED_DIR, CHUNKS_DIR, EMBEDDINGS_DIR, RETRIEVAL_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
