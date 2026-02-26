"""
Shared utilities for the intelligence layer.

Provides the canonical node-ID generator and a batched Cypher helper
used across all post-ingestion passes.
"""
from typing import Any, Dict, List

from src.core.id_generator import generate_node_id as _generate_node_id


def generate_node_id(node_type: str, *identifiers: Any) -> str:
    """Deterministic node ID from type + identifiers.

    This is the single source of truth — NodeFactory, EnhancedGraphLinker,
    CrossFileResolver, TestCoverageMapper, ConfigScanner, and PerformanceProfiler
    must all use this function (or an equivalent instance method that delegates here).
    """
    return _generate_node_id(node_type, *identifiers)


async def batch_cypher(
    session: Any,
    query: str,
    params_list: List[Dict[str, Any]],
    *,
    chunk_size: int = 500,
    extra_params: Dict[str, Any] | None = None,
) -> int:
    """Execute a Cypher query in chunked UNWIND batches.

    Args:
        session: Neo4j async session.
        query: Cypher query that references ``$items`` as the UNWIND variable.
               Example::
                   UNWIND $items AS item
                   MATCH (n {id: item.id}) SET n.x = item.x
        params_list: List of dicts — each dict becomes one ``item``.
        chunk_size: Max items per round-trip (default 500).
        extra_params: Additional top-level parameters merged into every call
                      (e.g. ``{"repo_hash": "abc"}``).

    Returns:
        Total number of items processed (sum of chunk sizes sent).
    """
    total = 0
    merged = extra_params or {}
    for start in range(0, len(params_list), chunk_size):
        chunk = params_list[start : start + chunk_size]
        await session.run(query, {"items": chunk, **merged})
        total += len(chunk)
    return total
