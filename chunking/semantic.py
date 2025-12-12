import numpy as np
import nltk
from sentence_transformers import SentenceTransformer, util

nltk.download("punkt", quiet=True)

MODEL = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")


def semantic_chunk(
    text: str,
    max_words: int = 600,
    min_words: int = 30,
    sim_threshold: float = 0.75,
):
    sentences = nltk.sent_tokenize(text)
    embeddings = MODEL.encode(sentences, normalize_embeddings=True)

    chunks, current, current_len = [], [], 0
    current_embeds = []

    def centroid():
        return np.mean(current_embeds, axis=0)

    for sent, emb in zip(sentences, embeddings):
        wc = len(sent.split())

        if current:
            sim = util.cos_sim(emb, centroid()).item()
            if current_len + wc > max_words or (sim < sim_threshold and current_len >= min_words):
                chunks.append(" ".join(current))
                current, current_embeds, current_len = [], [], 0

        current.append(sent)
        current_embeds.append(emb)
        current_len += wc

    if current:
        chunks.append(" ".join(current))

    return chunks
