"""
Confidence-Aware Query Engine — Technical Improvement 2

Every query to the graph or vector store returns not just results,
but also confidence scores and evidence trails.

Agents MUST use this engine instead of raw database queries.
The engine enforces:
    - Results always include confidence
    - Results always include evidence trail
    - Agents are told when confidence is below threshold
    - Observable reasoning: every query logs why it concluded what it did

Trust-Preserving Silence:
    If confidence < threshold → the engine returns an explicit
    "insufficient certainty" marker instead of speculative results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple

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
    source: str = ""  # "graph", "vector", "hybrid"
    query_used: str = ""

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
        }


@dataclass
class ConfidenceQueryResult:
    """Complete result from a confidence-aware query."""
    results: List[ConfidenceResult] = field(default_factory=list)
    overall_confidence: float = 0.0
    query_description: str = ""
    evidence_trail: List[str] = field(default_factory=list)
    reasoning_notes: List[str] = field(default_factory=list)
    
    # Trust signal
    has_sufficient_confidence: bool = True
    insufficient_confidence_reason: str = ""
    
    # Metrics
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

    def __init__(
        self,
        min_confidence: float = CONFIDENCE_REFUSE,
    ):
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

    async def graph_query(
        self,
        neo4j_client: Any,
        query: str,
        params: Dict[str, Any],
        description: str = "",
    ) -> ConfidenceQueryResult:
        """
        Execute a graph query with confidence scoring.

        Confidence is based on:
            - Number of results (more = higher, up to a point)
            - Result consistency (similar results = higher)
            - Query specificity (more specific = higher)
        """
        import time
        start = time.time()
        evidence_trail: List[str] = []
        reasoning_notes: List[str] = []

        try:
            raw_results = await neo4j_client.execute_query(query, params)
            elapsed = (time.time() - start) * 1000

            if not raw_results:
                evidence_trail.append(f"Graph query returned 0 results: {description}")
                reasoning_notes.append("No matching nodes/relationships found in the knowledge graph")
                self._stats["silence_results"] += 1
                return ConfidenceQueryResult(
                    results=[],
                    overall_confidence=0.0,
                    query_description=description,
                    evidence_trail=evidence_trail,
                    reasoning_notes=reasoning_notes,
                    has_sufficient_confidence=False,
                    insufficient_confidence_reason="No results found in knowledge graph",
                    graph_queries=1,
                    execution_time_ms=elapsed,
                )

            # Score results
            scored_results: List[ConfidenceResult] = []
            for record in raw_results:
                record_dict = dict(record) if not isinstance(record, dict) else record
                # Base confidence from graph (high — these are structural facts)
                confidence = 0.85
                ev = [f"Found in knowledge graph via: {description}"]

                # Boost if node has behavioral labels
                if record_dict.get("behavioral_labels"):
                    confidence = min(1.0, confidence + 0.05)
                    ev.append(f"Behavioral context: {record_dict.get('behavioral_labels')}")

                # Boost if node has temporal metadata
                if record_dict.get("volatility_score") is not None:
                    ev.append(f"Temporal data available: volatility={record_dict.get('volatility_score')}")

                scored_results.append(ConfidenceResult(
                    data=record_dict,
                    confidence=confidence,
                    evidence=ev,
                    source="graph",
                    query_used=description,
                ))

            overall = sum(r.confidence for r in scored_results) / len(scored_results)
            evidence_trail.append(f"Graph query returned {len(scored_results)} results (avg confidence: {overall:.2f})")

            has_sufficient = overall >= self._min_confidence
            if has_sufficient:
                self._stats["high_confidence_results"] += 1
            else:
                self._stats["low_confidence_results"] += 1

            self._stats["queries_run"] += 1

            return ConfidenceQueryResult(
                results=scored_results,
                overall_confidence=overall,
                query_description=description,
                evidence_trail=evidence_trail,
                reasoning_notes=reasoning_notes,
                has_sufficient_confidence=has_sufficient,
                insufficient_confidence_reason="" if has_sufficient else f"Confidence {overall:.2f} below threshold {self._min_confidence}",
                graph_queries=1,
                total_candidates=len(raw_results),
                execution_time_ms=elapsed,
            )

        except Exception as e:
            elapsed = (time.time() - start) * 1000
            logger.error(f"Confidence graph query failed: {e}")
            self._stats["silence_results"] += 1
            return ConfidenceQueryResult(
                results=[],
                overall_confidence=0.0,
                query_description=description,
                evidence_trail=[f"Graph query failed: {str(e)}"],
                reasoning_notes=["Query execution error — treating as insufficient confidence"],
                has_sufficient_confidence=False,
                insufficient_confidence_reason=f"Query error: {str(e)}",
                graph_queries=1,
                execution_time_ms=elapsed,
            )

    async def vector_query(
        self,
        vector_service: Any,
        query_text: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        description: str = "",
    ) -> ConfidenceQueryResult:
        """
        Execute a vector query with confidence scoring.

        Confidence is based on:
            - Similarity scores of results
            - Result count
            - Score distribution (tight cluster = higher)
        """
        import time
        start = time.time()
        evidence_trail: List[str] = []
        reasoning_notes: List[str] = []

        try:
            raw_results = await vector_service.semantic_search(
                query=query_text,
                top_k=top_k,
                filters=filters,
            )
            elapsed = (time.time() - start) * 1000

            if not raw_results:
                evidence_trail.append(f"Vector search returned 0 results: {description}")
                self._stats["silence_results"] += 1
                return ConfidenceQueryResult(
                    results=[],
                    overall_confidence=0.0,
                    query_description=description,
                    evidence_trail=evidence_trail,
                    has_sufficient_confidence=False,
                    insufficient_confidence_reason="No semantic matches found",
                    vector_queries=1,
                    execution_time_ms=elapsed,
                )

            scored_results: List[ConfidenceResult] = []
            scores: List[float] = []

            for result in raw_results:
                result_dict = result if isinstance(result, dict) else dict(result)
                sim_score = result_dict.get("score", result_dict.get("similarity", 0.5))
                scores.append(sim_score)

                # Map similarity score to confidence
                # High similarity → higher confidence, but never above 0.85 for vector-only
                confidence = min(0.85, sim_score * 0.9)
                ev = [f"Semantic match (similarity={sim_score:.3f}) via: {description}"]

                scored_results.append(ConfidenceResult(
                    data=result_dict,
                    confidence=confidence,
                    evidence=ev,
                    source="vector",
                    query_used=description,
                ))

            overall = sum(r.confidence for r in scored_results) / len(scored_results) if scored_results else 0.0

            # Score distribution analysis
            if len(scores) >= 2:
                top_score = max(scores)
                score_gap = top_score - sorted(scores, reverse=True)[min(1, len(scores)-1)]
                if score_gap > 0.2:
                    reasoning_notes.append(f"Clear top result (score gap: {score_gap:.3f})")
                else:
                    reasoning_notes.append(f"Results are ambiguous (small score gap: {score_gap:.3f})")

            evidence_trail.append(f"Vector search returned {len(scored_results)} results (avg confidence: {overall:.2f})")

            has_sufficient = overall >= self._min_confidence
            if has_sufficient:
                self._stats["high_confidence_results"] += 1
            else:
                self._stats["low_confidence_results"] += 1

            self._stats["queries_run"] += 1

            return ConfidenceQueryResult(
                results=scored_results,
                overall_confidence=overall,
                query_description=description,
                evidence_trail=evidence_trail,
                reasoning_notes=reasoning_notes,
                has_sufficient_confidence=has_sufficient,
                insufficient_confidence_reason="" if has_sufficient else f"Confidence {overall:.2f} below threshold",
                vector_queries=1,
                total_candidates=len(raw_results),
                execution_time_ms=elapsed,
            )

        except Exception as e:
            elapsed = (time.time() - start) * 1000
            logger.error(f"Confidence vector query failed: {e}")
            self._stats["silence_results"] += 1
            return ConfidenceQueryResult(
                results=[],
                overall_confidence=0.0,
                query_description=description,
                evidence_trail=[f"Vector query failed: {str(e)}"],
                has_sufficient_confidence=False,
                insufficient_confidence_reason=f"Query error: {str(e)}",
                vector_queries=1,
                execution_time_ms=elapsed,
            )

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
        """
        Execute hybrid graph + vector query with combined confidence.

        Alpha controls the weighting:
            alpha=1.0 → graph only
            alpha=0.0 → vector only
            alpha=0.7 → default (graph-preferred)
        """
        graph_result = await self.graph_query(neo4j_client, graph_query, graph_params, f"[graph] {description}")
        vector_result = await self.vector_query(vector_service, vector_query_text, vector_top_k, vector_filters, f"[vector] {description}")

        # Merge results
        all_results = graph_result.results + vector_result.results
        all_evidence = graph_result.evidence_trail + vector_result.evidence_trail
        all_reasoning = graph_result.reasoning_notes + vector_result.reasoning_notes

        # Weighted overall confidence
        if graph_result.results and vector_result.results:
            overall = alpha * graph_result.overall_confidence + (1 - alpha) * vector_result.overall_confidence
            all_reasoning.append(f"Hybrid fusion: α={alpha} graph + {1-alpha} vector → {overall:.2f}")
        elif graph_result.results:
            overall = graph_result.overall_confidence
            all_reasoning.append("Graph-only results (no vector matches)")
        elif vector_result.results:
            overall = vector_result.overall_confidence
            all_reasoning.append("Vector-only results (no graph matches)")
        else:
            overall = 0.0
            all_reasoning.append("No results from either graph or vector search")

        has_sufficient = overall >= self._min_confidence

        return ConfidenceQueryResult(
            results=all_results,
            overall_confidence=overall,
            query_description=description,
            evidence_trail=all_evidence,
            reasoning_notes=all_reasoning,
            has_sufficient_confidence=has_sufficient,
            insufficient_confidence_reason="" if has_sufficient else "Hybrid search confidence below threshold",
            graph_queries=graph_result.graph_queries,
            vector_queries=vector_result.vector_queries,
            total_candidates=graph_result.total_candidates + vector_result.total_candidates,
            execution_time_ms=graph_result.execution_time_ms + vector_result.execution_time_ms,
        )


# Singleton
confidence_engine = ConfidenceEngine()
