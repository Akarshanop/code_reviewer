"""
Subgraph compression utilities for LLM-ready context packets.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Any, List, Set, Tuple


class SubgraphCompressionEngine:
    """Compress graph candidates into a compact, structured subgraph packet."""

    def compress(
        self,
        graph_results: List[Dict[str, Any]],
        graph_candidates: List[Dict[str, Any]],
        max_nodes: int = 50,
    ) -> Dict[str, Any]:
        node_weights: Dict[str, float] = defaultdict(float)
        node_payloads: Dict[str, Dict[str, Any]] = {}
        edges: List[Dict[str, Any]] = []

        for candidate in graph_candidates:
            node_id = str(candidate.get("id", ""))
            if not node_id:
                continue
            score = float(candidate.get("score", 0.0) or 0.0)
            node_weights[node_id] += score
            node_payloads[node_id] = {
                "id": node_id,
                "name": candidate.get("name", ""),
                "type": candidate.get("type", ""),
                "repository": candidate.get("repository", ""),
                "file": candidate.get("file", ""),
            }

        for grouped in graph_results:
            source = grouped.get("source") or {}
            source_id = str(source.get("id", ""))
            relationships = grouped.get("relationships", {}) or {}
            for rel_type, targets in relationships.items():
                for item in targets:
                    target = item.get("target", {}) if isinstance(item, dict) else {}
                    target_id = str(target.get("id", ""))
                    if not source_id or not target_id:
                        continue
                    node_weights[source_id] += 0.5
                    node_weights[target_id] += 0.5
                    node_payloads.setdefault(source_id, {"id": source_id, "name": source.get("name", ""), "type": source.get("type", "")})
                    node_payloads.setdefault(target_id, {"id": target_id, "name": target.get("name", ""), "type": target.get("type", "")})
                    edges.append(
                        {
                            "source": source_id,
                            "target": target_id,
                            "type": rel_type,
                            "path_length": item.get("path_length", 1) if isinstance(item, dict) else 1,
                        }
                    )

        ranked_nodes = sorted(node_weights.items(), key=lambda kv: kv[1], reverse=True)
        keep_ids: Set[str] = {node_id for node_id, _ in ranked_nodes[:max_nodes]}
        kept_nodes = [node_payloads[nid] for nid in keep_ids if nid in node_payloads]
        kept_edges = [e for e in edges if e["source"] in keep_ids and e["target"] in keep_ids]

        return {
            "nodes": kept_nodes,
            "edges": kept_edges,
            "stats": {
                "input_candidates": len(graph_candidates),
                "input_edges": len(edges),
                "compressed_nodes": len(kept_nodes),
                "compressed_edges": len(kept_edges),
            },
        }
