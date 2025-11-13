"""
retriever.py — модуль извлечения релевантных документов по вопросам,
с расширенным выводом: возвращает web_id, chunk_id и текст чанка для каждого результата.
Улучшено: проверка согласованности, оптимизация кандидатов, вычисление комбинированного скоринга с возможностью реранкинга.
"""

import json
import numpy as np
import faiss
import pandas as pd
import logging
import pickle
from pathlib import Path
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import torch

from utils.config import (
    EMBEDDINGS_DIR,
    PROCESSED_DIR,
    EMBEDDING_MODEL,
    HYBRID_ALPHA,
    MAX_CANDIDATES,
    RERANK_MODEL_NAME,
    RERANK_ALPHA
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Retriever:
    def __init__(
        self,
        index_path: Path = None,
        meta_path: Path = None,
        model_name: str = EMBEDDING_MODEL
    ):
        if index_path is None:
            index_path = EMBEDDINGS_DIR / "kb_index.faiss"
        if meta_path is None:
            meta_path = EMBEDDINGS_DIR / "kb_metadata.json"

        if not index_path.exists():
            raise FileNotFoundError(f"❌ Индекс не найден: {index_path}")
        if not meta_path.exists():
            raise FileNotFoundError(f"❌ Метаданные не найдены: {meta_path}")

        logger.info(f"🔍 Загружаем FAISS-индекс: {index_path}")
        self.index = faiss.read_index(str(index_path))

        logger.info(f"🔍 Загружаем метаданные: {meta_path}")
        with open(meta_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        self._validate_data_consistency()

        logger.info(f"🔍 Загружаем модель эмбеддингов: {model_name}")
        self.model = SentenceTransformer(model_name)

        self._load_or_create_tfidf()

        # Инициализация рерангера
        self.reranker = None
        if RERANK_MODEL_NAME:
            try:
                logger.info(f"🔍 Загружаем CrossEncoder-рерангер: {RERANK_MODEL_NAME}")
                device = "cuda" if torch.cuda.is_available() else "cpu"
                self.reranker = CrossEncoder(RERANK_MODEL_NAME, device=device, max_length=512)
            except Exception as e:
                logger.error(f"❌ Ошибка загрузки рерангера: {e}")
                self.reranker = None

    def _validate_data_consistency(self):
        index_size = self.index.ntotal
        metadata_size = len(self.metadata)
        if index_size != metadata_size:
            logger.warning(
                f"⚠️ Несоответствие размеров: индекс содержит {index_size} векторов, "
                f"метаданные содержат {metadata_size} записей"
            )
            self.valid_size = min(index_size, metadata_size)
        else:
            self.valid_size = index_size
        logger.info(f"📊 Размер валидных данных: {self.valid_size}")

    def _load_or_create_tfidf(self):
        tfidf_path = EMBEDDINGS_DIR / "tfidf_model.pkl"
        if tfidf_path.exists():
            logger.info("🔍 Загружаем TF-IDF из кэша...")
            with open(tfidf_path, "rb") as f:
                data = pickle.load(f)
                self.tfidf = data["vectorizer"]
                self.tfidf_matrix = data["matrix"]
            logger.info("✅ TF-IDF загружен")
        else:
            logger.info("🔧 Обучаем TF-IDF модель...")
            texts = [rec["text"] for rec in self.metadata[:self.valid_size]]
            self.tfidf = TfidfVectorizer(max_features=50000, lowercase=True, analyzer='word')
            self.tfidf_matrix = self.tfidf.fit_transform(texts)
            with open(tfidf_path, "wb") as f:
                pickle.dump({"vectorizer": self.tfidf, "matrix": self.tfidf_matrix}, f)
            logger.info(f"✅ TF-IDF обучен и сохранен: {tfidf_path}")

    def search(
        self,
        query: str,
        top_k: int = 5,
        candidate_factor: int = 4
    ) -> list[dict]:
        if not query or not query.strip():
            logger.warning("Получен пустой запрос")
            return []

        if top_k <= 0:
            raise ValueError("top_k должен быть положительным числом")

        if self.valid_size == 0:
            logger.error("Нет доступных данных для поиска")
            return []

        try:
            q_vec = self.model.encode([query], convert_to_numpy=True).astype('float32')
            faiss.normalize_L2(q_vec)

            num_candidates = min(top_k * candidate_factor, MAX_CANDIDATES, self.valid_size)
            D_vec, I_vec = self.index.search(q_vec, num_candidates)

            q_tfidf = self.tfidf.transform([query])
            scores_tfidf = (self.tfidf_matrix * q_tfidf.T).toarray().flatten()
            top_tfidf_idx = np.argsort(-scores_tfidf)[:num_candidates]

            candidate_idxs = list(I_vec[0].tolist()) + list(top_tfidf_idx.tolist())
            candidate_idxs = list(dict.fromkeys(candidate_idxs))  # удаляем дубликаты

            web_id_scores = {}
            rerank_scores = {}

            # Если рерангер доступен — вычисляем скор
            if self.reranker:
                # Создаём пары (query, text) для всех кандидатов
                pairs = [(query, self.metadata[idx]["text"]) for idx in candidate_idxs]
                try:
                    scores = self.reranker.predict(pairs)
                except Exception as e:
                    logger.error(f"❌ Ошибка при предсказании рерангера: {e}")
                    scores = [0.0] * len(pairs)
                for idx, score in zip(candidate_idxs, scores):
                    rerank_scores[idx] = float(score)

            for idx in candidate_idxs:
                if idx < 0 or idx >= self.valid_size:
                    continue
                rec = self.metadata[idx]
                web_id   = rec.get("web_id")
                chunk_id = rec.get("chunk_id")
                text     = rec.get("text")

                if web_id is None:
                    continue

                vec_score   = D_vec[0][list(I_vec[0]).index(idx)] if idx in I_vec[0] else 0.0
                tfidf_score = scores_tfidf[idx]
                rerank_score = rerank_scores.get(idx, 0.0)

                combined = HYBRID_ALPHA * vec_score + (1.0 - HYBRID_ALPHA) * tfidf_score
                if self.reranker:
                    combined = (1.0 - RERANK_ALPHA) * combined + RERANK_ALPHA * rerank_score

                if web_id not in web_id_scores or combined > web_id_scores[web_id]["score"]:
                    web_id_scores[web_id] = {
                        "chunk_id":   chunk_id,
                        "chunk_text": text,
                        "score":       combined
                    }

            sorted_items = sorted(web_id_scores.items(),
                                  key=lambda x: x[1]["score"],
                                  reverse=True)

            results = []
            for web_id, info in sorted_items[:top_k]:
                results.append({
                    "web_id":     web_id,
                    "chunk_id":   info["chunk_id"],
                    "chunk_text": info["chunk_text"],
                    "score":      info["score"]
                })

            logger.debug(f"🔍 Поиск завершен: найдено {len(results)} результатов")
            return results

        except Exception as e:
            logger.error(f"❌ Ошибка при поиске для запроса '{query}': {e}")
            return []

    def get_stats(self) -> dict:
        return {
            "index_size":     self.index.ntotal,
            "metadata_size":  len(self.metadata),
            "valid_size":     self.valid_size,
            "embedding_dim":   self.index.d,
            "vocabulary_size": len(self.tfidf.vocabulary_) if hasattr(self.tfidf, 'vocabulary_') else 0
        }

def run_batch_questions(
    questions_path: Path = None,
    output_path: Path = None,
    top_k: int = 5
):
    if questions_path is None:
        questions_path = PROCESSED_DIR / "questions.json"
    if output_path is None:
        output_path = PROCESSED_DIR / "questions_top5_web_ids_with_chunks.csv"

    logger.info(f"📄 Загрузка вопросов: {questions_path}")

    try:
        df_q = pd.read_json(questions_path, orient="records", dtype=str)
    except Exception as e:
        logger.error(f"❌ Ошибка загрузки вопросов: {e}")
        return

    retriever = Retriever()

    stats = retriever.get_stats()
    logger.info(f"📊 Статистика Retriever: {stats}")

    all_results = []
    processed   = 0

    for _, row in df_q.iterrows():
        q_id  = row["q_id"]
        query = row["query"]

        try:
            hits = retriever.search(query, top_k=top_k)
            processed += 1

            for rank, hit in enumerate(hits, start=1):
                all_results.append({
                    "q_id":      q_id,
                    "rank":      rank,
                    "web_id":    hit["web_id"],
                    "chunk_id":  hit["chunk_id"],
                    "chunk_text": hit["chunk_text"][:200],
                    "score":     hit["score"]
                })

            for rank in range(len(hits)+1, top_k+1):
                all_results.append({
                    "q_id":      q_id,
                    "rank":      rank,
                    "web_id":    "",
                    "chunk_id": "",
                    "chunk_text": "",
                    "score":    ""
                })

        except Exception as e:
            logger.error(f"❌ Ошибка при обработке вопроса {q_id}: {e}")
            for rank in range(1, top_k+1):
                all_results.append({
                    "q_id":      q_id,
                    "rank":      rank,
                    "web_id":    "",
                    "chunk_id": "",
                    "chunk_text": "",
                    "score":    ""
                })

    df_res = pd.DataFrame(all_results)

    try:
        df_res.to_csv(output_path, index=False, encoding="utf-8")
        logger.info(f"✅ Сохранены результаты с чанками: {output_path}")
        logger.info(f"📈 Обработано вопросов: {processed}/{len(df_q)}")
    except Exception as e:
        logger.error(f"❌ Ошибка сохранения результатов: {e}")

if __name__ == "__main__":
    try:
        run_batch_questions(top_k=5)
    except Exception as e:
        logger.error(f"❌ Ошибка при запуске извлечения: {e}")
