"""
run_pipeline.py — main entry point for the RAG pipeline.
Runs all pipeline stages sequentially.
"""

import logging

from utils.config import BASE_DIR
from graph.build_graph import build_and_save_graph
from data.loader_websites import load_websites, process_and_chunk
from data.loader_questions import load_questions, save_processed_questions
from embeddings.websites_embed import build_kb_embeddings
from retrieval.retriever import run_batch_questions


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

GRAPH_PATH = BASE_DIR / "graph" / "graph_index.pkl"


def run_full_pipeline() -> None:
    """Run the full RAG pipeline from raw data to retrieval results."""

    logger.info("Starting full RAG pipeline")

    try:
        # Stage 1: Load and chunk documents
        logger.info("Stage 1: Loading and chunking documents")
        df_docs = load_websites()
        process_and_chunk(df_docs, max_words=600, overlap_words=100)

        # Stage 2: Build embeddings and FAISS index
        logger.info("Stage 2: Building embeddings")
        build_kb_embeddings(batch_size=32)

        # Stage 3: Prepare questions
        logger.info("Stage 3: Preparing questions")
        df_questions = load_questions()
        save_processed_questions(df_questions)

        # Stage 4: Build and save chunk graph
        logger.info("Stage 4: Building chunk graph")
        build_and_save_graph(GRAPH_PATH)

        # Stage 5: Retrieval
        logger.info("Stage 5: Running retrieval")
        run_batch_questions(top_k=5)

        logger.info("Pipeline finished successfully")

    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        raise


def run_retrieval_only() -> None:
    """Run retrieval only (assumes all data is already prepared)."""
    logger.info("Running retrieval only")
    run_batch_questions(top_k=5)
    logger.info("Retrieval finished")


if __name__ == "__main__":
    run_full_pipeline()
    # Alternatively:
    # run_retrieval_only()
