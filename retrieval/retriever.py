import json
import numpy as np
import faiss
import pandas as pd
import logging
import pickle
from pathlib import Path
from tqdm import tqdm


import torch
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

from graph.graph_store import load_graph
from graph.graph_expander import expand_chunks

from utils.config import (
    EMBEDDINGS_DIR,
    PROCESSED_DIR,
    EMBEDDING_MODEL,
    HYBRID_ALPHA,
    MAX_CANDIDATES,
    RERANK_MODEL_NAME,
    RERANK_ALPHA,
    RERANK_MODEL_NAME_2,
    TOP_N_FOR_SECOND_RERANK,
    RERANK2_ALPHA,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Retriever:
    def __init__(
        self,
        index_path: Path | None = None,
        meta_path: Path | None = None,
        model_name: str = EMBEDDING_MODEL,
    ):
        # --- graph ---
        try:
            self.graph = load_graph()
            logger.info("Graph loaded")
        except Exception as e:
            logger.warning(f"Graph unavailable: {e}")
            self.graph = None

        index_path = index_path or (EMBEDDINGS_DIR / "kb_index.faiss")
        meta_path = meta_path or (EMBEDDINGS_DIR / "kb_metadata.json")

        if not index_path.exists():
            raise FileNotFoundError(f"Index not found: {index_path}")
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata not found: {meta_path}")

        self.index = faiss.read_index(str(index_path))

        with open(meta_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        self._validate_data_consistency()

        self.model = SentenceTransformer(model_name)

        self._load_or_create_tfidf()

        # --- reranker 1 ---
        self.reranker = None
        if RERANK_MODEL_NAME:
            try:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                self.reranker = CrossEncoder(
                    RERANK_MODEL_NAME,
                    device=device,
                    max_length=512,
                )
            except Exception as e:
                logger.warning(f"Reranker-1 disabled: {e}")

        # --- reranker 2 ---
        self.reranker2 = None
        if RERANK_MODEL_NAME_2:
            try:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                self.reranker2 = CrossEncoder(
                    RERANK_MODEL_NAME_2,
                    device=device,
                    max_length=512,
                )
            except Exception as e:
                logger.warning(f"Reranker-2 disabled: {e}")

    # ------------------------------------------------------------------

    def _validate_data_consistency(self):
        index_size = self.index.ntotal
        meta_size = len(self.metadata)
        self.valid_size = min(index_size, meta_size)

        if index_size != meta_size:
            logger.warning(
                f"Index/metadata mismatch: {index_size} vs {meta_size}"
            )

    # ------------------------------------------------------------------

    def _load_or_create_tfidf(self):
        tfidf_path = EMBEDDINGS_DIR / "tfidf_model.pkl"

        if tfidf_path.exists():
            with open(tfidf_path, "rb") as f:
                data = pickle.load(f)

            self.tfidf = data["vectorizer"]
            self.tfidf_matrix = data["matrix"]

            if self.tfidf_matrix.shape[0] != self.valid_size:
                logger.warning("TF-IDF cache mismatch, rebuilding")
                tfidf_path.unlink()
                self._load_or_create_tfidf()
                return

            logger.info("TF-IDF loaded from cache")
            return

        texts = [rec["text"] for rec in self.metadata[:self.valid_size]]

        self.tfidf = TfidfVectorizer(
            max_features=50_000,
            lowercase=True,
        )
        self.tfidf_matrix = self.tfidf.fit_transform(texts)

        with open(tfidf_path, "wb") as f:
            pickle.dump(
                {
                    "vectorizer": self.tfidf,
                    "matrix": self.tfidf_matrix,
                },
                f,
            )

        logger.info("TF-IDF trained")

    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 5, candidate_factor: int = 3):
        if not query or not query.strip():
            return []

        # --- dense search ---
        q_vec = self.model.encode([query], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(q_vec)

        n_candidates = min(
            top_k * candidate_factor,
            MAX_CANDIDATES,
            self.valid_size,
        )

        D, I = self.index.search(q_vec, n_candidates)

        # --- sparse search ---
        q_tfidf = self.tfidf.transform([query])
        tfidf_scores = (self.tfidf_matrix @ q_tfidf.T).toarray().ravel()
        top_sparse = np.argsort(-tfidf_scores)[:n_candidates]

        candidate_idxs = list(
            dict.fromkeys(I[0].tolist() + top_sparse.tolist())
        )

        # --- graph expansion ---
        if self.graph:
            seed_chunks = [
                self.metadata[i]["chunk_id"]
                for i in candidate_idxs
                if 0 <= i < self.valid_size
            ]

            expanded = expand_chunks(
                seed_chunks,
                self.graph,
                hops=1,
                max_nodes=80,
            )

            chunk_to_idx = {
                rec["chunk_id"]: i
                for i, rec in enumerate(self.metadata)
            }

            expanded_idxs = [
                chunk_to_idx[c]
                for c in expanded
                if c in chunk_to_idx
            ]

            before = len(candidate_idxs)
            candidate_idxs = list(
                dict.fromkeys(candidate_idxs + expanded_idxs)
            )
            logger.debug(f"Graph expansion: {before} → {len(candidate_idxs)}")

        # --- score vectors ---
        pos_map = {idx: pos for pos, idx in enumerate(I[0])}

        faiss_raw = np.array(
            [D[0][pos_map[i]] if i in pos_map else 0.0 for i in candidate_idxs],
            float,
        )

        tfidf_raw = np.array(
            [
                tfidf_scores[i] if 0 <= i < len(tfidf_scores) else 0.0
                for i in candidate_idxs
            ],
            float,
        )

        def _norm(x):
            mn, mx = x.min(), x.max()
            return (x - mn) / (mx - mn) if mx > mn else np.zeros_like(x)

        faiss_n = _norm(faiss_raw)
        tfidf_n = _norm(tfidf_raw)

        # --- rerank 1 ---
        if self.reranker:
            pairs = [(query, self.metadata[i]["text"]) for i in candidate_idxs]
            try:
                r1 = np.array(self.reranker.predict(pairs), float)
                r1 = _norm(r1)
            except Exception:
                r1 = np.zeros(len(candidate_idxs))
        else:
            r1 = np.zeros(len(candidate_idxs))

        # --- rerank 2 ---
        if self.reranker2:
            top_n = np.argsort(-r1)[:TOP_N_FOR_SECOND_RERANK]
            pairs2 = [(query, self.metadata[candidate_idxs[i]]["text"]) for i in top_n]

            try:
                r2_scores = self.reranker2.predict(pairs2)
            except Exception:
                r2_scores = [0.0] * len(top_n)

            r2 = np.zeros(len(candidate_idxs))
            for i, pos in enumerate(top_n):
                r2[pos] = r2_scores[i]

            r2 = _norm(r2)
        else:
            r2 = np.zeros(len(candidate_idxs))

        # --- final fusion ---
        results = {}

        for pos, idx in enumerate(candidate_idxs):
            if idx >= self.valid_size:
                continue

            rec = self.metadata[idx]
            web_id = rec["web_id"]

            score = (
                HYBRID_ALPHA * faiss_n[pos]
                + (1 - HYBRID_ALPHA) * tfidf_n[pos]
            )
            score = (1 - RERANK_ALPHA) * score + RERANK_ALPHA * r1[pos]
            score = (1 - RERANK2_ALPHA) * score + RERANK2_ALPHA * r2[pos]

            if web_id not in results or score > results[web_id]["score"]:
                results[web_id] = {
                    "web_id": web_id,
                    "chunk_id": rec["chunk_id"],
                    "text": rec["text"],
                    "score": float(score),
                }

        return sorted(
            results.values(),
            key=lambda x: x["score"],
            reverse=True,
        )[:top_k]

    # ------------------------------------------------------------------

    def get_stats(self):
        return {
            "index_size": self.index.ntotal,
            "metadata_size": len(self.metadata),
            "valid_size": self.valid_size,
            "embedding_dim": self.index.d,
            "vocabulary_size": len(self.tfidf.vocabulary_),
        }


# ======================================================================

def run_batch_questions(
    questions_path: Path | None = None,
    output_path: Path | None = None,
    top_k: int = 5,
):
    questions_path = questions_path or (PROCESSED_DIR / "questions.json")
    output_path = output_path or (
        PROCESSED_DIR / "questions_top5_web_ids_with_chunks.csv"
    )

    df_q = pd.read_json(questions_path, dtype=str)

    retriever = Retriever()

    rows = []

    for _, row in tqdm(
            df_q.iterrows(),
            total=len(df_q),
            desc="Retrieval",
    ):

        hits = retriever.search(row["query"], top_k=top_k)

        for rank in range(top_k):
            if rank < len(hits):
                h = hits[rank]
                rows.append(
                    {
                        "q_id": row["q_id"],
                        "rank": rank + 1,
                        "web_id": h["web_id"],
                        "chunk_id": h["chunk_id"],
                        "text": h["text"][:200],
                        "score": h["score"],
                    }
                )
            else:
                rows.append(
                    {
                        "q_id": row["q_id"],
                        "rank": rank + 1,
                        "web_id": "",
                        "chunk_id": "",
                        "text": "",
                        "score": "",
                    }
                )

    pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8")

if __name__ == "__main__":
    try:
        run_batch_questions(top_k=5)
    except Exception as e:
        logger.error(f"❌ Launch error: {e}")