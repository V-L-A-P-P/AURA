import logging
from typing import Iterable, List

logger = logging.getLogger(__name__)


def expand_chunks(
    seed_chunk_ids: Iterable[str],
    graph,
    hops: int = 1,
    max_nodes: int = 50,
) -> List[str]:
    """
    Expand a set of seed chunks using graph neighborhood traversal.

    Parameters:
    - seed_chunk_ids : initial chunk_ids (seeds)
    - graph          : NetworkX graph with chunk_id nodes
    - hops           : number of BFS hops
    - max_nodes      : hard limit to prevent graph explosion

    Returns:
    - List of expanded chunk_ids (including seeds)
    """

    expanded = set(seed_chunk_ids)
    frontier = set(seed_chunk_ids)

    for _ in range(hops):
        next_frontier = set()

        for cid in frontier:
            if cid not in graph:
                continue

            for neighbor in graph.neighbors(cid):
                if neighbor not in expanded:
                    next_frontier.add(neighbor)

        expanded |= next_frontier
        frontier = next_frontier

        if len(expanded) >= max_nodes:
            logger.debug(
                "Graph expansion stopped: reached max_nodes=%d",
                max_nodes,
            )
            break

        if not frontier:
            break

    logger.debug(
        "Graph expansion completed: %d → %d chunks",
        len(seed_chunk_ids),
        len(expanded),
    )

    return list(expanded)
