"""
config.py — centralized configuration for the RAG project.
Defines paths, model parameters, chunking, and retrieval settings.
"""

from pathlib import Path
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# === Project directories ===
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CHUNKS_DIR = DATA_DIR / "chunks"

EMBEDDINGS_DIR = BASE_DIR / "embeddings"
RETRIEVAL_DIR = BASE_DIR / "retrieval"
LLM_DIR = BASE_DIR / "llm"

# === Chunking parameters ===
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 600))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 100))
MIN_SIZE = int(os.getenv("MIN_SIZE", 30))  # FIXED: was incorrectly using CHUNK_OVERLAP

# === Models ===
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "ai-forever/sbert_large_nlu_ru",
)

GENERATION_MODEL = os.getenv(
    "GENERATION_MODEL",
    "BAAI/bge-m3",
)

# === FAISS / retrieval ===
KB_INDEX_PATH = EMBEDDINGS_DIR / "kb_index.faiss"
KB_METADATA_PATH = EMBEDDINGS_DIR / "kb_metadata.json"

QA_INDEX_PATH = EMBEDDINGS_DIR / "qa_q_index.faiss"
QA_COMBINED_INDEX = EMBEDDINGS_DIR / "qa_combined_index.faiss"
QA_METADATA_PATH = EMBEDDINGS_DIR / "qa_metadata.json"

HYBRID_ALPHA = float(os.getenv("HYBRID_ALPHA", 0.3))
MAX_CANDIDATES = int(os.getenv("MAX_CANDIDATES", 100))

# === Reranking ===
RERANK_MODEL_NAME = os.getenv(
    "RERANK_MODEL_NAME",
    "DiTy/cross-encoder-russian-msmarco",
)
RERANK_ALPHA = float(os.getenv("RERANK_ALPHA", 0.4))

RERANK_MODEL_NAME_2 = os.getenv(
    "RERANK_MODEL_NAME_2",
    "BAAI/bge-reranker-v2-m3",
)
TOP_N_FOR_SECOND_RERANK = int(os.getenv("TOP_N_FOR_SECOND_RERANK", 10))
RERANK2_ALPHA = float(os.getenv("RERANK2_ALPHA", 0.3))

# === Logging / debug ===
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

# Ensure required directories exist
for directory in (
    DATA_DIR,
    RAW_DIR,
    PROCESSED_DIR,
    CHUNKS_DIR,
    EMBEDDINGS_DIR,
    RETRIEVAL_DIR,
    LLM_DIR,
):
    directory.mkdir(parents=True, exist_ok=True)
