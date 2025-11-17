"""
config.py — централизованная конфигурация для RAG-проекта.
Определяет пути, параметры моделей, чанкинга и индексации.
"""

from pathlib import Path
import os
from dotenv import load_dotenv

# === Загрузка переменных окружения (.env) ===
load_dotenv()

# === Основные директории проекта ===
BASE_DIR       = Path(__file__).resolve().parent.parent
DATA_DIR       = BASE_DIR / "data"
RAW_DIR        = DATA_DIR / "raw"
PROCESSED_DIR  = DATA_DIR / "processed"
CHUNKS_DIR     = DATA_DIR / "chunks"
EMBEDDINGS_DIR = BASE_DIR / "embeddings"
RETRIEVAL_DIR  = BASE_DIR / "retrieval"
LLM_DIR        = BASE_DIR / "llm"

# === Параметры чанкинга ===
CHUNK_SIZE     = int(os.getenv("CHUNK_SIZE",     600))#y egora-500  # число слов или символов (проверьте единицу)
CHUNK_OVERLAP  = int(os.getenv("CHUNK_OVERLAP",  100))

# === Настройки моделей ===
EMBEDDING_MODEL   = os.getenv("EMBEDDING_MODEL",   "ai-forever/sbert_large_nlu_ru")#"cointegrated/rubert-tiny2")
GENERATION_MODEL  = os.getenv("GENERATION_MODEL",  "BAAI/bge-m3")

# === Настройки FAISS / индексов ===
# Можно указать базовые пути, но чаще используются отдельные файлы per-mode
QA_INDEX_PATH        = EMBEDDINGS_DIR / "qa_q_index.faiss"
QA_COMBINED_INDEX    = EMBEDDINGS_DIR / "qa_combined_index.faiss"
KB_INDEX_PATH        = EMBEDDINGS_DIR / "kb_index.faiss"
QA_METADATA_PATH     = EMBEDDINGS_DIR / "qa_metadata.json"
KB_METADATA_PATH     = EMBEDDINGS_DIR / "kb_metadata.json"
HYBRID_ALPHA     = 0.3#float(os.getenv("HYBRID_ALPHA",  0.9))
MAX_CANDIDATES = int(os.getenv("MAX_CANDIDATES", 100))
RERANK_MODEL_NAME = os.getenv("RERANK_MODEL_NAME", "DiTy/cross-encoder-russian-msmarco")
RERANK_ALPHA = float(os.getenv("RERANK_ALPHA", 0.4))
RERANK_MODEL_NAME_2 = "AAI/bge-reranker-v2-m3"  # например
TOP_N_FOR_SECOND_RERANK = 10
RERANK2_ALPHA = 0.3

# === Логирование и отладка ===
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
DEBUG     = os.getenv("DEBUG",     "false").lower() == "true"

# === Автоматическое создание директорий, если не существуют ===
for directory in [DATA_DIR, RAW_DIR, PROCESSED_DIR, CHUNKS_DIR, EMBEDDINGS_DIR, RETRIEVAL_DIR, LLM_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
