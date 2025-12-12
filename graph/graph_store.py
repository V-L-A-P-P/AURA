import logging
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)

GRAPH_PATH = Path(__file__).parent / "graph_index.pkl"


def load_graph():
    if not GRAPH_PATH.exists():
        raise FileNotFoundError("Graph index not found. Build it first.")

    logger.info("Loading chunk graph")

    with open(GRAPH_PATH, "rb") as f:
        return pickle.load(f)
