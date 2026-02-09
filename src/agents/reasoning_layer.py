"""
Reasoning Layer — Problem 5 Fix (Part 1 of 2)

This is the INNER layer of the two-layer agent architecture.
The Reasoning Layer has:
    - Full graph access (Neo4j, Qdrant)
    - Full evidence visibility
    - Behavioral, temporal, and domain intelligence
    - Confidence scoring on every conclusion

The Reasoning Layer NEVER shapes output for end users.
That is the Policy Layer's job.

Output contract:
    ReasoningResult — structured, evidence-rich, confidence-scored
    The Policy Layer then transforms this based on role/RBAC.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

from src.core.logger import get_logger
from src.intelligence.evidence_links import (
    EvidenceWeightedLink,
    CONFIDENCE_RELIABLE,
    CONFIDENCE_REFUSE,
    format_confidence_for_agent,
)
from src.intelligence.behavioral_layer import BehavioralClassification
from src.intelligence.temporal_metadata import TemporalMetadata

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Reasoning result types
# ---------------------------------------------------------------------------

@dataclass
class ReasoningEvidence:
    """A single piece of evidence supporting a conclusion."""
    source: str  # "graph_traversal", "vector_search", "temporal_analysis", etc.
    description: str
    confidence: float = 0.0
    node_ids: List[str] = field(default_factory=list)
    raw_data: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "description": self.description,
            "confidence": round(self.confidence, 3),
            "node_count": len(self.node_ids),
        }


@dataclass
class ReasoningConclusion:
    """A single conclusion from reasoning, with evidence trail."""
    statement: str
    confidence: float
    evidence: List[ReasoningEvidence] = field(default_factory=list)
    uncertainty: str = ""  # What we're NOT sure about
    behavioral_context: List[str] = field(default_factory=list)  # Behavioral labels
    temporal_context: Optional[Dict[str, Any]] = None
    domain_context: List[str] = field(default_factory=list)  # Domain names

    @property
    def confidence_label(self) -> str:
        return format_confidence_for_agent(self.confidence)

    @property
    def is_reliable(self) -> bool:
        return self.confidence >= CONFIDENCE_RELIABLE

    @property
    def should_refuse(self) -> bool:
        return self.confidence < CONFIDENCE_REFUSE

    def to_dict(self) -> Dict[str, Any]:
        return {
            "statement": self.statement,
            "confidence": round(self.confidence, 3),
            "confidence_label": self.confidence_label,
            "evidence": [e.to_dict() for e in self.evidence],
            "uncertainty": self.uncertainty,
            "behavioral_context": self.behavioral_context,
            "temporal_context": self.temporal_context,
            "domain_context": self.domain_context,
        }


@dataclass
class ReasoningResult:
    """
    Complete output from the Reasoning Layer.
    
    This is the contract between Reasoning and Policy layers.
    Policy Layer reads this and applies RBAC / redaction / shaping.
    """
    # Core content
    conclusions: List[ReasoningConclusion] = field(default_factory=list)
    overall_confidence: float = 0.0

    # Rich context (Policy Layer decides what to expose)
    affected_files: List[str] = field(default_factory=list)
    affected_classes: List[str] = field(default_factory=list)
    affected_functions: List[str] = field(default_factory=list)
    affected_routes: List[str] = field(default_factory=list)
    affected_domains: List[str] = field(default_factory=list)

    # Impact assessment
    impact_score: float = 0.0
    risk_level: str = "low"
    behavioral_flags: List[str] = field(default_factory=list)

    # Temporal awareness
    volatility_summary: Optional[Dict[str, Any]] = None
    recently_modified: List[str] = field(default_factory=list)

    # Links with evidence
    evidence_links: List[Dict[str, Any]] = field(default_factory=list)

    # Observable reasoning (internal trace)
    reasoning_trace: List[str] = field(default_factory=list)
    tools_used: List[str] = field(default_factory=list)
    graph_queries_run: int = 0
    vector_queries_run: int = 0
    execution_time_seconds: float = 0.0

    # Trust signal
    trust_preserving_silence: bool = False  # If True, system couldn't answer confidently
    silence_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "conclusions": [c.to_dict() for c in self.conclusions],
            "overall_confidence": round(self.overall_confidence, 3),
            "affected_files": self.affected_files,
            "affected_classes": self.affected_classes,
            "affected_functions": self.affected_functions,
            "affected_routes": self.affected_routes,
            "affected_domains": self.affected_domains,
            "impact_score": round(self.impact_score, 3),
            "risk_level": self.risk_level,
            "behavioral_flags": self.behavioral_flags,
            "volatility_summary": self.volatility_summary,
            "reasoning_trace": self.reasoning_trace,
            "tools_used": self.tools_used,
            "trust_preserving_silence": self.trust_preserving_silence,
            "silence_reason": self.silence_reason,
        }


# ---------------------------------------------------------------------------
# Reasoning Layer implementation
# ---------------------------------------------------------------------------

class ReasoningLayer:
    """
    The inner reasoning engine with full graph access.
    
    This layer:
        1. Receives a query + context
        2. Resolves domains
        3. Traverses graph with behavioral/temporal awareness
        4. Scores confidence on every conclusion
        5. Returns ReasoningResult (never shapes for end user)
    """

    def __init__(self) -> None:
        self._stats = {
            "reasoning_runs": 0,
            "silence_invoked": 0,
            "avg_confidence": 0.0,
        }

    @property
    def stats(self) -> Dict[str, int]:
        return dict(self._stats)

    def build_result(
        self,
        conclusions: List[ReasoningConclusion],
        affected_files: List[str] = None,
        affected_classes: List[str] = None,
        affected_functions: List[str] = None,
        affected_routes: List[str] = None,
        affected_domains: List[str] = None,
        impact_score: float = 0.0,
        risk_level: str = "low",
        behavioral_flags: List[str] = None,
        volatility_summary: Optional[Dict[str, Any]] = None,
        recently_modified: List[str] = None,
        evidence_links: List[Dict[str, Any]] = None,
        reasoning_trace: List[str] = None,
        tools_used: List[str] = None,
        graph_queries_run: int = 0,
        vector_queries_run: int = 0,
        execution_time_seconds: float = 0.0,
    ) -> ReasoningResult:
        """Build a ReasoningResult with trust-preserving silence check."""

        # Compute overall confidence
        if conclusions:
            overall = sum(c.confidence for c in conclusions) / len(conclusions)
        else:
            overall = 0.0

        # Trust-preserving silence
        silence = False
        silence_reason = ""
        if not conclusions:
            silence = True
            silence_reason = "No conclusions could be drawn from available evidence"
        elif overall < CONFIDENCE_REFUSE:
            silence = True
            silence_reason = f"Overall confidence ({overall:.2f}) is below the reliability threshold ({CONFIDENCE_REFUSE})"

        if silence:
            self._stats["silence_invoked"] += 1

        result = ReasoningResult(
            conclusions=conclusions,
            overall_confidence=overall,
            affected_files=affected_files or [],
            affected_classes=affected_classes or [],
            affected_functions=affected_functions or [],
            affected_routes=affected_routes or [],
            affected_domains=affected_domains or [],
            impact_score=impact_score,
            risk_level=risk_level,
            behavioral_flags=behavioral_flags or [],
            volatility_summary=volatility_summary,
            recently_modified=recently_modified or [],
            evidence_links=evidence_links or [],
            reasoning_trace=reasoning_trace or [],
            tools_used=tools_used or [],
            graph_queries_run=graph_queries_run,
            vector_queries_run=vector_queries_run,
            execution_time_seconds=execution_time_seconds,
            trust_preserving_silence=silence,
            silence_reason=silence_reason,
        )

        self._stats["reasoning_runs"] += 1
        total_runs = self._stats["reasoning_runs"]
        self._stats["avg_confidence"] = (
            (self._stats["avg_confidence"] * (total_runs - 1) + overall) / total_runs
        )

        return result

    def create_silence_result(self, reason: str) -> ReasoningResult:
        """
        Explicitly create a trust-preserving silence result.
        Used when the system cannot confidently answer.
        """
        self._stats["silence_invoked"] += 1
        return ReasoningResult(
            trust_preserving_silence=True,
            silence_reason=reason,
        )


# Singleton
reasoning_layer = ReasoningLayer()
