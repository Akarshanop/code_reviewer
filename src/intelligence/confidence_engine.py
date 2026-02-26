"""
Confidence-Aware Query Engine

Every query to the graph or vector store returns not just results,
but also confidence scores and evidence trails.

Agents MUST use this engine instead of raw database queries.
The engine enforces:
    - Results always include confidence
    - Results always include evidence trail
    - Agents are told when confidence is below threshold
    - Observable reasoning: every query logs why it concluded what it did

Trust-Preserving Silence:
    If confidence < threshold the engine returns an explicit
    "insufficient certainty" marker instead of speculative results.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

from src.core.logger import get_logger
from src.intelligence.evidence_links import (
    CONFIDENCE_RELIABLE,
    CONFIDENCE_REFUSE,
    format_confidence_for_agent,
)

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Confidence-aware result types
# ---------------------------------------------------------------------------

@dataclass
class ConfidenceResult:
    """A single result item with confidence and evidence."""
    data: Dict[str, Any]
    confidence: float
    evidence: List[str] = field(default_factory=list)
    source: str = ""
    query_used: str = ""
    evidence_type: str = ""

    @property
    def is_reliable(self) -> bool:
        return self.confidence >= CONFIDENCE_RELIABLE

    @property
    def confidence_label(self) -> str:
        return format_confidence_for_agent(self.confidence)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "data": self.data,
            "confidence": round(self.confidence, 3),
            "confidence_label": self.confidence_label,
            "evidence": self.evidence,
            "source": self.source,
            "evidence_type": self.evidence_type,
        }


@dataclass
class ConfidenceQueryResult:
    """Complete result from a confidence-aware query."""
    results: List[ConfidenceResult] = field(default_factory=list)
    overall_confidence: float = 0.0
    query_description: str = ""
    evidence_trail: List[str] = field(default_factory=list)
    reasoning_notes: List[str] = field(default_factory=list)

    has_sufficient_confidence: bool = True
    insufficient_confidence_reason: str = ""

    graph_queries: int = 0
    vector_queries: int = 0
    total_candidates: int = 0
    execution_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "result_count": len(self.results),
            "overall_confidence": round(self.overall_confidence, 3),
            "has_sufficient_confidence": self.has_sufficient_confidence,
            "insufficient_confidence_reason": self.insufficient_confidence_reason,
            "evidence_trail": self.evidence_trail,
            "reasoning_notes": self.reasoning_notes,
            "results": [r.to_dict() for r in self.results],
            "metrics": {
                "graph_queries": self.graph_queries,
                "vector_queries": self.vector_queries,
                "total_candidates": self.total_candidates,
                "execution_time_ms": round(self.execution_time_ms, 2),
            },
        }


# ---------------------------------------------------------------------------
# Confidence engine
# ---------------------------------------------------------------------------

class ConfidenceEngine:
    """
    Wraps all graph and vector queries with confidence scoring.

    Every query goes through this engine, which:
        1. Executes the query
        2. Scores confidence based on result quality
        3. Builds an evidence trail
        4. Returns ConfidenceQueryResult with trust signals
    """

    def __init__(self, min_confidence: float = CONFIDENCE_REFUSE):
        self._min_confidence = min_confidence
        self._stats = {
            "queries_run": 0,
            "high_confidence_results": 0,
            "low_confidence_results": 0,
            "silence_results": 0,
        }

    @property
    def stats(self) -> Dict[str, int]:
        return dict(self._stats)

    # ── helpers ────────────────────────────────────────────────────────

    def _record(self, overall: float) -> None:
        self._stats["queries_run"] += 1
        if overall >= self._min_confidence:
            self._stats["high_confidence_results"] += 1
        else:
            self._stats["low_confidence_results"] += 1

    def _combine_confidences(
        self,
        graph_conf: Optional[float],
        vector_conf: Optional[float],
        *,
        contradictory: bool = False,
    ) -> float:
        """
        Evidence-aware confidence combiner.

        - independent evidence (graph + vector): multiplicative union
        - correlated evidence (single-source): passthrough
        - contradictory evidence: return uncertainty (0.5)
        """
        if contradictory:
            return 0.5
        if graph_conf is not None and vector_conf is not None:
            # Independent evidence approximation: P(A ∪ B)
            return 1 - (1 - graph_conf) * (1 - vector_conf)
        if graph_conf is not None:
            return graph_conf
        if vector_conf is not None:
            return vector_conf
        return 0.0

    def _empty(self, description: str, reason: str, elapsed: float,
               *, graph: bool = False) -> ConfidenceQueryResult:
        self._stats["silence_results"] += 1
        return ConfidenceQueryResult(
            overall_confidence=0.0,
            query_description=description,
            evidence_trail=[f"{'Graph' if graph else 'Vector'} query: {reason}: {description}"],
            reasoning_notes=[reason],
            has_sufficient_confidence=False,
            insufficient_confidence_reason=reason,
            graph_queries=1 if graph else 0,
            vector_queries=0 if graph else 1,
            execution_time_ms=elapsed,
        )

    # ── graph ──────────────────────────────────────────────────────────

    async def graph_query(
        self,
        neo4j_client: Any,
        query: str,
        params: Dict[str, Any],
        description: str = "",
    ) -> ConfidenceQueryResult:
        start = time.time()
        try:
            raw_results = await neo4j_client.execute_query(query, params)
            elapsed = (time.time() - start) * 1000

            if not raw_results:
                return self._empty(description, "No results found in knowledge graph", elapsed, graph=True)

            scored: list[ConfidenceResult] = []
            for record in raw_results:
                rd = dict(record) if not isinstance(record, dict) else record
                confidence = 0.85
                ev = [f"Found in knowledge graph via: {description}"]
                if rd.get("behavioral_labels"):
                    confidence = min(1.0, confidence + 0.05)
                    ev.append(f"Behavioral context: {rd['behavioral_labels']}")
                if rd.get("volatility_score") is not None:
                    ev.append(f"Temporal data: volatility={rd['volatility_score']}")
                scored.append(
                    ConfidenceResult(
                        data=rd,
                        confidence=confidence,
                        evidence=ev,
                        source="graph",
                        query_used=description,
                        evidence_type="graph_traversal",
                    )
                )

            overall = sum(r.confidence for r in scored) / len(scored)
            self._record(overall)
            sufficient = overall >= self._min_confidence

            return ConfidenceQueryResult(
                results=scored, overall_confidence=overall,
                query_description=description,
                evidence_trail=[f"Graph query returned {len(scored)} results (avg confidence: {overall:.2f})"],
                has_sufficient_confidence=sufficient,
                insufficient_confidence_reason="" if sufficient else f"Confidence {overall:.2f} below threshold {self._min_confidence}",
                graph_queries=1, total_candidates=len(raw_results), execution_time_ms=elapsed,
            )
        except Exception as e:
            elapsed = (time.time() - start) * 1000
            logger.error(f"Confidence graph query failed: {e}")
            return self._empty(description, f"Query error: {e}", elapsed, graph=True)

    # ── vector ─────────────────────────────────────────────────────────

    async def vector_query(
        self,
        vector_service: Any,
        query_text: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        description: str = "",
    ) -> ConfidenceQueryResult:
        start = time.time()
        try:
            raw_results = await vector_service.semantic_search(query=query_text, top_k=top_k, filters=filters)
            elapsed = (time.time() - start) * 1000

            if not raw_results:
                return self._empty(description, "No semantic matches found", elapsed)

            scored: list[ConfidenceResult] = []
            scores: list[float] = []
            for result in raw_results:
                rd = result if isinstance(result, dict) else dict(result)
                sim = rd.get("score", rd.get("similarity", 0.5))
                scores.append(sim)
                confidence = min(0.85, sim * 0.9)
                scored.append(ConfidenceResult(
                    data=rd, confidence=confidence,
                    evidence=[f"Semantic match (similarity={sim:.3f}) via: {description}"],
                    source="vector", query_used=description,
                    evidence_type="vector_similarity",
                ))

            overall = sum(r.confidence for r in scored) / len(scored)
            reasoning: list[str] = []
            if len(scores) >= 2:
                gap = max(scores) - sorted(scores, reverse=True)[1]
                reasoning.append(f"{'Clear top result' if gap > 0.2 else 'Ambiguous results'} (score gap: {gap:.3f})")

            self._record(overall)
            sufficient = overall >= self._min_confidence

            return ConfidenceQueryResult(
                results=scored, overall_confidence=overall,
                query_description=description,
                evidence_trail=[f"Vector search returned {len(scored)} results (avg confidence: {overall:.2f})"],
                reasoning_notes=reasoning,
                has_sufficient_confidence=sufficient,
                insufficient_confidence_reason="" if sufficient else f"Confidence {overall:.2f} below threshold",
                vector_queries=1, total_candidates=len(raw_results), execution_time_ms=elapsed,
            )
        except Exception as e:
            elapsed = (time.time() - start) * 1000
            logger.error(f"Confidence vector query failed: {e}")
            return self._empty(description, f"Query error: {e}", elapsed)

    # ── hybrid ─────────────────────────────────────────────────────────

    async def hybrid_query(
        self,
        neo4j_client: Any,
        vector_service: Any,
        graph_query: str,
        graph_params: Dict[str, Any],
        vector_query_text: str,
        vector_top_k: int = 10,
        vector_filters: Optional[Dict[str, Any]] = None,
        alpha: float = 0.7,
        description: str = "",
    ) -> ConfidenceQueryResult:
        g = await self.graph_query(neo4j_client, graph_query, graph_params, f"[graph] {description}")
        v = await self.vector_query(vector_service, vector_query_text, vector_top_k, vector_filters, f"[vector] {description}")

        all_results = g.results + v.results
        all_evidence = g.evidence_trail + v.evidence_trail
        reasoning = list(g.reasoning_notes + v.reasoning_notes)

        contradiction = False
        if g.results and v.results:
            graph_ids = {str(r.data.get("id")) for r in g.results if r.data.get("id") is not None}
            vector_ids = {str(r.data.get("id")) for r in v.results if r.data.get("id") is not None}
            if graph_ids and vector_ids and graph_ids.isdisjoint(vector_ids):
                contradiction = True
                reasoning.append("Contradictory evidence: graph and vector candidate sets do not overlap")
            overall = self._combine_confidences(
                g.overall_confidence,
                v.overall_confidence,
                contradictory=contradiction,
            )
            reasoning.append(f"Hybrid evidence combine (alpha hint={alpha}) → {overall:.2f}")
        elif g.results:
            overall = g.overall_confidence
            reasoning.append("Graph-only results (no vector matches)")
        elif v.results:
            overall = v.overall_confidence
            reasoning.append("Vector-only results (no graph matches)")
        else:
            overall = 0.0
            reasoning.append("No results from either graph or vector search")

        sufficient = overall >= self._min_confidence
        return ConfidenceQueryResult(
            results=all_results, overall_confidence=overall,
            query_description=description,
            evidence_trail=all_evidence, reasoning_notes=reasoning,
            has_sufficient_confidence=sufficient,
            insufficient_confidence_reason="" if sufficient else "Hybrid search confidence below threshold",
            graph_queries=g.graph_queries, vector_queries=v.vector_queries,
            total_candidates=g.total_candidates + v.total_candidates,
            execution_time_ms=g.execution_time_ms + v.execution_time_ms,
        )


confidence_engine = ConfidenceEngine()
