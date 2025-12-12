import json
import pickle
import logging
from pathlib import Path

import faiss
import networkx as nx
import numpy as np

from utils.config import EMBEDDINGS_DIR


TOP_K = 8
SIM_THRESHOLD = 0.7

logger = logging.getLogger(__name__)


def build_chunk_graph() -> nx.Graph:
    """
    Build a graph of chunks based on embedding similarity and document structure.
    """

    logger.info("Loading embeddings and metadata")

    vectors = np.load(EMBEDDINGS_DIR / "kb_vectors.npy")
    with open(EMBEDDINGS_DIR / "kb_metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    G = nx.Graph()

    # Add chunk nodes
    for rec in meta:
        G.add_node(
            rec["chunk_id"],
            web_id=rec["web_id"],
            text=rec["text"],
        )

    logger.info("Building semantic edges using FAISS KNN")

    # KNN search over all vectors
    D, I = index.search(vectors, TOP_K + 1)

    for i, (scores, idxs) in enumerate(zip(D, I)):
        src = meta[i]["chunk_id"]

        # Skip self-match (first neighbor)
        for score, j in zip(scores[1:], idxs[1:]):
            if score < SIM_THRESHOLD:
                continue

            tgt = meta[j]["chunk_id"]

            G.add_edge(
                src,
                tgt,
                weight=float(score),
                type="semantic",
            )

    logger.info("Adding intra-document edges")

    # Strengthen connections between sequential chunks from the same document
    by_web: dict[str, list[str]] = {}
    for rec in meta:
        by_web.setdefault(rec["web_id"], []).append(rec["chunk_id"])

    for chunks in by_web.values():
        for a, b in zip(chunks, chunks[1:]):
            if G.has_edge(a, b):
                G[a][b]["weight"] += 0.1
            else:
                G.add_edge(a, b, weight=0.3, type="intra_doc")

    logger.info(
        "Graph built: %d nodes, %d edges",
        G.number_of_nodes(),
        G.number_of_edges(),
    )

    return G


def build_and_save_graph(path: Path) -> nx.Graph:
    """
    Build the chunk graph and persist it to disk.
    """
    logger.info("Building and saving chunk graph")

    G = build_chunk_graph()
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "wb") as f:
        pickle.dump(G, f)

    logger.info("Graph saved to %s", path)
    return G


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    GRAPH_PATH = Path(__file__).parent / "graph_index.pkl"

    build_and_save_graph(GRAPH_PATH)
