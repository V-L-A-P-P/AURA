"""
retriever.py — модуль извлечения релевантных документов по вопросам.

Сценарий:
 1. Загружаем индекс документов и метаданные из EMBEDDINGS_DIR.
 2. Загружаем файл вопросов (processed/questions.json) с полями q_id, query.
 3. Для каждого вопроса: кодируем query → эмбеддинг → ищем top_k документов.
 4. При необходимости применяем порог distance_threshold.
 5. Сохраняем результат в CSV: q_id, web_id_1 … web_id_top_k.
"""

import json
import numpy as np
import faiss
import pandas as pd
from pathlib import Path
import logging
from sentence_transformers import SentenceTransformer

from utils.config import (
    EMBEDDINGS_DIR,
    PROCESSED_DIR,
    EMBEDDING_MODEL,
    # можно добавить порог:
    # QA_DISTANCE_THRESHOLD если потребуется
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Retriever:
    def __init__(self, index_path: Path = None, meta_path: Path = None, model_name: str = EMBEDDING_MODEL):
        if index_path is None:
            index_path = EMBEDDINGS_DIR / "kb_index.faiss"
        if meta_path is None:
            meta_path = EMBEDDINGS_DIR / "kb_metadata.json"

        if not index_path.exists():
            raise FileNotFoundError(f"❌ Индекс не найден: {index_path}")
        if not meta_path.exists():
            raise FileNotFoundError(f"❌ Метаданные не найдены: {meta_path}")

        logger.info(f"🔍 Загрузка индекса: {index_path}")
        self.index = faiss.read_index(str(index_path))

        logger.info(f"🔍 Загрузка метаданных: {meta_path}")
        with open(meta_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        logger.info(f"🔍 Загрузка модели эмбеддингов: {model_name}")
        self.model = SentenceTransformer(model_name)

    def search(self, query: str, top_k: int = 5, distance_threshold: float = None) -> list:
        """
        Кодирует query → находит top_k ближайших документов по индексу.
        Если указан distance_threshold — отфильтровывает результаты с расстоянием больше порога.
        Возвращает список web_id (без дубликатов), длинной ≤ top_k.
        """
        if not query or not query.strip():
            return []

        q_vec = self.model.encode([query], convert_to_numpy=True).astype('float32')
        distances, indices = self.index.search(q_vec, top_k)

        result_web_ids = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            if distance_threshold is not None and dist > distance_threshold:
                logger.debug(f"⚠️ Результат {idx} отфильтрован по порогу: dist={dist:.4f} > {distance_threshold}")
                continue
            web_id = self.metadata[idx].get("web_id")
            if web_id and web_id not in result_web_ids:
                result_web_ids.append(web_id)
            if len(result_web_ids) >= top_k:
                break

        return result_web_ids

def run_batch_questions(
    questions_path: Path = None,
    output_path: Path = None,
    top_k: int = 5,
    distance_threshold: float = None
):
    if questions_path is None:
        questions_path = PROCESSED_DIR / "questions.json"
    if output_path is None:
        output_path = PROCESSED_DIR / "questions_top5_web_ids.csv"

    logger.info(f"📄 Загрузка вопросов: {questions_path}")
    # ожидание: файл JSON с массивом {q_id, query}
    df_q = pd.read_json(questions_path, orient="records", dtype=str)
    retriever = Retriever()

    results = []
    for _, row in df_q.iterrows():
        q_id   = row["q_id"]
        query  = row["query"]
        web_ids = retriever.search(query, top_k=top_k, distance_threshold=distance_threshold)
        web_ids_padded = web_ids + [""] * (top_k - len(web_ids))
        results.append({"q_id": q_id, **{f"web_id_{i+1}": web_ids_padded[i] for i in range(top_k)}})

    df_res = pd.DataFrame(results)
    df_res.to_csv(output_path, index=False, encoding="utf-8")
    logger.info(f"✅ Сохранены результаты поиска: {output_path}")

if __name__ == "__main__":
    try:
        # Например, можно указать порог: distance_threshold=0.7
        run_batch_questions(top_k=5, distance_threshold=None)
    except Exception as e:
        logger.error(f"❌ Ошибка при пакетном поиске вопросов: {e}")
