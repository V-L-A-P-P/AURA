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
    RERANK_ALPHA,

    # === NEW === параметры второго реранкера
    RERANK_MODEL_NAME_2,
    TOP_N_FOR_SECOND_RERANK,
    RERANK2_ALPHA,
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

        # === первый реранкер ===
        self.reranker = None
        if RERANK_MODEL_NAME:
            try:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                logger.info(f"🔍 Загружаем CrossEncoder-реранкер: {RERANK_MODEL_NAME}")
                self.reranker = CrossEncoder(RERANK_MODEL_NAME, device=device, max_length=512)
            except Exception as e:
                logger.error(f"❌ Ошибка загрузки реранкера: {e}")
                self.reranker = None

        # === второй реранкер (новый) ===
        self.reranker2 = None
        if RERANK_MODEL_NAME_2:
            try:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                logger.info(f"🔍 Загружаем второй CrossEncoder-реранкер: {RERANK_MODEL_NAME_2}")
                self.reranker2 = CrossEncoder(RERANK_MODEL_NAME_2, device=device, max_length=512)
            except Exception as e:
                logger.error(f"❌ Ошибка загрузки второго реранкера: {e}")
                self.reranker2 = None

    def _validate_data_consistency(self):
        index_size = self.index.ntotal
        metadata_size = len(self.metadata)
        if index_size != metadata_size:
            logger.warning(
                f"⚠️ Несоответствие размеров: индекс содержит {index_size} векторов, "
                f"метаданные — {metadata_size} записей"
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
            logger.info(f"✅ TF-IDF обучен и сохранён: {tfidf_path}")

    def search(self, query: str, top_k: int = 5, candidate_factor: int = 3) -> list[dict]:
        if not query or not query.strip():
            logger.warning("Получен пустой запрос")
            return []

        if top_k <= 0:
            raise ValueError("top_k должен быть положительным числом")

        if self.valid_size == 0:
            logger.error("Нет доступных данных для поиска")
            return []

        try:
            # === эмбеддинг ===
            q_vec = self.model.encode([query], convert_to_numpy=True).astype('float32')
            faiss.normalize_L2(q_vec)

            num_candidates = min(top_k * candidate_factor, MAX_CANDIDATES, self.valid_size)
            D_vec, I_vec = self.index.search(q_vec, num_candidates)

            # === TF-IDF ===
            q_tfidf = self.tfidf.transform([query])
            scores_tfidf = (self.tfidf_matrix * q_tfidf.T).toarray().flatten()
            top_tfidf_idx = np.argsort(-scores_tfidf)[:num_candidates]

            # объединение FAISS + TF-IDF
            candidate_idxs = list(dict.fromkeys(I_vec[0].tolist() + top_tfidf_idx.tolist()))

            faiss_scores = D_vec[0]
            pos_map = {val: pos for pos, val in enumerate(I_vec[0])}

            faiss_subset = np.array([
                faiss_scores[pos_map[i]] if i in pos_map else 0.0
                for i in candidate_idxs
            ], float)
            tfidf_subset = np.array([scores_tfidf[i] for i in candidate_idxs], float)

            def _minmax(arr):
                mn, mx = arr.min(), arr.max()
                return (arr - mn) / (mx - mn) if mx != mn else np.zeros_like(arr)

            faiss_norm = _minmax(faiss_subset)
            tfidf_norm = _minmax(tfidf_subset)

            # === Реранк #1 ===
            if self.reranker:
                pairs = [(query, self.metadata[idx]["text"]) for idx in candidate_idxs]
                try:
                    s = self.reranker.predict(pairs)
                except Exception as e:
                    logger.error(f"❌ Ошибка первого реранкера: {e}")
                    s = [0.0] * len(pairs)

                rerank1_arr = np.array(s, float)
                rerank1_norm = _minmax(rerank1_arr)
            else:
                rerank1_norm = np.zeros(len(candidate_idxs), float)

            # === Реранк #2 (тяжёлый) ===
            if self.reranker2:
                top_n = np.argsort(-rerank1_norm)[:TOP_N_FOR_SECOND_RERANK]
                top_ids = [candidate_idxs[i] for i in top_n]

                pairs2 = [(query, self.metadata[idx]["text"]) for idx in top_ids]
                try:
                    s2 = self.reranker2.predict(pairs2)
                except Exception as e:
                    logger.error(f"❌ Ошибка второго реранкера: {e}")
                    s2 = [0.0] * len(top_ids)

                # Заполняем массив оценок второго реранка
                rerank2_full = np.zeros(len(candidate_idxs), float)
                for local_pos, global_id in zip(top_n, top_ids):
                    rerank2_full[local_pos] = float(s2[top_n.tolist().index(local_pos)])

                rerank2_norm = _minmax(rerank2_full)
            else:
                rerank2_norm = np.zeros(len(candidate_idxs), float)

            # === финальное объединение ===
            web_id_scores = {}

            for pos, idx in enumerate(candidate_idxs):
                if idx < 0 or idx >= self.valid_size:
                    continue

                rec = self.metadata[idx]
                web_id = rec.get("web_id")
                if web_id is None:
                    continue

                vec_s = faiss_norm[pos]
                tfidf_s = tfidf_norm[pos]
                r1 = rerank1_norm[pos]
                r2 = rerank2_norm[pos]

                combined = (
                    HYBRID_ALPHA * vec_s +
                    (1 - HYBRID_ALPHA) * tfidf_s
                )

                combined = (1 - RERANK_ALPHA) * combined + RERANK_ALPHA * r1
                combined = (1 - RERANK2_ALPHA) * combined + RERANK2_ALPHA * r2

                if web_id not in web_id_scores or combined > web_id_scores[web_id]["score"]:
                    web_id_scores[web_id] = {
                        "chunk_id": rec.get("chunk_id"),
                        "text": rec.get("text"),
                        "score": combined
                    }

            sorted_items = sorted(web_id_scores.items(), key=lambda x: x[1]["score"], reverse=True)

            return [
                {
                    "web_id": web_id,
                    "chunk_id": info["chunk_id"],
                    "text": info["text"],
                    "score": info["score"]
                }
                for web_id, info in sorted_items[:top_k]
            ]

        except Exception as e:
            logger.error(f"❌ Ошибка при поиске: {e}")
            return []

    def get_stats(self) -> dict:
        return {
            "index_size": self.index.ntotal,
            "metadata_size": len(self.metadata),
            "valid_size": self.valid_size,
            "embedding_dim": self.index.d,
            "vocabulary_size": len(self.tfidf.vocabulary_) if hasattr(self.tfidf, 'vocabulary_') else 0
        }


def run_batch_questions(questions_path: Path = None, output_path: Path = None, top_k: int = 5):
    if questions_path is None:
        questions_path = PROCESSED_DIR / "questions.json"
    if output_path is None:
        output_path = PROCESSED_DIR / "questions_top5_web_ids_with_chunks.csv"

    logger.info(f"📄 Загружаем вопросы: {questions_path}")

    try:
        df_q = pd.read_json(questions_path, orient="records", dtype=str)
    except Exception as e:
        logger.error(f"❌ Ошибка загрузки вопросов: {e}")
        return

    retriever = Retriever()

    stats = retriever.get_stats()
    logger.info(f"📊 Статистика Retriever: {stats}")

    all_results = []
    processed = 0

    for _, row in df_q.iterrows():
        q_id = row["q_id"]
        query = row["query"]

        try:
            hits = retriever.search(query, top_k=top_k)
            processed += 1

            for rank, hit in enumerate(hits, start=1):
                all_results.append({
                    "q_id": q_id,
                    "rank": rank,
                    "web_id": hit["web_id"],
                    "chunk_id": hit["chunk_id"],
                    "text": hit["text"][:200],
                    "score": hit["score"]
                })

            for rank in range(len(hits) + 1, top_k + 1):
                all_results.append({
                    "q_id": q_id,
                    "rank": rank,
                    "web_id": "",
                    "chunk_id": "",
                    "text": "",
                    "score": ""
                })

        except Exception as e:
            logger.error(f"❌ Ошибка обработки {q_id}: {e}")
            for rank in range(1, top_k + 1):
                all_results.append({
                    "q_id": q_id,
                    "rank": rank,
                    "web_id": "",
                    "chunk_id": "",
                    "text": "",
                    "score": ""
                })

    df_res = pd.DataFrame(all_results)

    try:
        df_res.to_csv(output_path, index=False, encoding="utf-8")
        logger.info(f"✅ Сохранены результаты: {output_path}")
        logger.info(f"📈 Обработано вопросов: {processed}/{len(df_q)}")
    except Exception as e:
        logger.error(f"❌ Ошибка сохранения результатов: {e}")


if __name__ == "__main__":
    try:
        run_batch_questions(top_k=5)
    except Exception as e:
        logger.error(f"❌ Ошибка запуска извлечения: {e}")
