"""
Evidence-Weighted Link Model — Problem 4 Fix

All cross-language and cross-repository links are no longer assumed correct.
Every link carries:
    confidence_score  - Probabilistic strength [0, 1]
    evidence_types    - What evidence supports this link
    link_origin       - Which system component created the link

Agents MUST reason probabilistically about links:
    - confidence >= 0.8 → treat as reliable
    - 0.5 <= confidence < 0.8 → flag uncertainty in output
    - confidence < 0.5 → refuse to assert the link

Evidence Types (ordered by strength):
    AST_MATCH         (0.95) - AST-level import resolution
    OPENAPI_MATCH     (0.90) - OpenAPI/Swagger spec confirms the link
    TYPE_SIGNATURE    (0.85) - Matching type signatures across boundaries
    COMMIT_HISTORY    (0.80) - Co-changed files in git history
    PATH_CONVENTION   (0.70) - URL/path pattern matching (e.g. /api/users → UsersController)
    LITERAL_STRING    (0.60) - Literal string match (e.g. endpoint URL in frontend)
    NAMING_CONVENTION (0.50) - Name similarity (e.g. UserService → UserDTO)
    HEURISTIC         (0.40) - General heuristic / best guess
    LLM_INFERRED      (0.35) - LLM suggested the link (lowest trust)
"""

from __future__ import annotations

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any, Tuple

from src.core.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Evidence types with base confidence
# ---------------------------------------------------------------------------

class EvidenceType(str, Enum):
    AST_MATCH = "ast_match"
    OPENAPI_MATCH = "openapi_match"
    TYPE_SIGNATURE = "type_signature"
    COMMIT_HISTORY = "commit_history"
    PATH_CONVENTION = "path_convention"
    LITERAL_STRING = "literal_string"
    NAMING_CONVENTION = "naming_convention"
    HEURISTIC = "heuristic"
    LLM_INFERRED = "llm_inferred"


# Base confidence per evidence type (single evidence)
_EVIDENCE_BASE_CONFIDENCE: Dict[EvidenceType, float] = {
    EvidenceType.AST_MATCH: 0.95,
    EvidenceType.OPENAPI_MATCH: 0.90,
    EvidenceType.TYPE_SIGNATURE: 0.85,
    EvidenceType.COMMIT_HISTORY: 0.80,
    EvidenceType.PATH_CONVENTION: 0.70,
    EvidenceType.LITERAL_STRING: 0.60,
    EvidenceType.NAMING_CONVENTION: 0.50,
    EvidenceType.HEURISTIC: 0.40,
    EvidenceType.LLM_INFERRED: 0.35,
}


# Confidence thresholds for agent reasoning
CONFIDENCE_RELIABLE = 0.80
CONFIDENCE_UNCERTAIN = 0.50
CONFIDENCE_REFUSE = 0.50  # Below this, agent should refuse to assert


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LinkEvidence:
    """A single piece of evidence supporting a cross-language/repo link."""
    evidence_type: EvidenceType
    description: str
    source_artifact: str = ""  # e.g. "import statement in App.jsx:12"
    confidence_modifier: float = 0.0  # Optional adjustment (-0.2 to +0.2)

    @property
    def base_confidence(self) -> float:
        base = _EVIDENCE_BASE_CONFIDENCE.get(self.evidence_type, 0.3)
        return max(0.0, min(1.0, base + self.confidence_modifier))


@dataclass
class EvidenceWeightedLink:
    """
    A link between two code artifacts with explicit evidence and confidence.

    The combined confidence is computed from all evidence pieces using
    probabilistic OR: P(any evidence correct) = 1 - ∏(1 - p_i)
    This models: "if any of my evidence sources is correct, the link holds."
    """
    source_id: str
    target_id: str
    link_type: str  # e.g. "CALLS_ROUTE", "IMPORTS", "USES_DTO"
    evidence: List[LinkEvidence] = field(default_factory=list)
    link_origin: str = ""  # e.g. "edge_linker", "enhanced_graph_linker", "llm"
    created_at: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def confidence_score(self) -> float:
        """
        Combined confidence using probabilistic OR.
        If evidence = [0.7, 0.6], combined = 1 - (1-0.7)*(1-0.6) = 1 - 0.12 = 0.88
        """
        if not self.evidence:
            return 0.0
        complement_product = 1.0
        for ev in self.evidence:
            complement_product *= (1.0 - ev.base_confidence)
        return round(1.0 - complement_product, 4)

    @property
    def is_reliable(self) -> bool:
        return self.confidence_score >= CONFIDENCE_RELIABLE

    @property
    def is_uncertain(self) -> bool:
        return CONFIDENCE_REFUSE <= self.confidence_score < CONFIDENCE_RELIABLE

    @property
    def should_refuse(self) -> bool:
        return self.confidence_score < CONFIDENCE_REFUSE

    @property
    def strongest_evidence(self) -> Optional[LinkEvidence]:
        if not self.evidence:
            return None
        return max(self.evidence, key=lambda e: e.base_confidence)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "link_type": self.link_type,
            "confidence_score": self.confidence_score,
            "is_reliable": self.is_reliable,
            "evidence": [
                {
                    "type": e.evidence_type.value,
                    "description": e.description,
                    "confidence": e.base_confidence,
                }
                for e in self.evidence
            ],
            "evidence_count": len(self.evidence),
            "link_origin": self.link_origin,
        }

    def to_neo4j_properties(self) -> Dict[str, Any]:
        """Flatten for Neo4j edge properties."""
        return {
            "confidence_score": self.confidence_score,
            "evidence_types": [e.evidence_type.value for e in self.evidence],
            "evidence_count": len(self.evidence),
            "link_origin": self.link_origin,
            "is_reliable": self.is_reliable,
        }


# ---------------------------------------------------------------------------
# Evidence-aware link factory
# ---------------------------------------------------------------------------

class EvidenceLinkFactory:
    """
    Creates evidence-weighted links with proper evidence tracking.
    Used by ingestion pipeline to replace deterministic link creation.
    """

    def __init__(self) -> None:
        self._stats = {
            "links_created": 0,
            "high_confidence": 0,
            "medium_confidence": 0,
            "low_confidence": 0,
        }

    @property
    def stats(self) -> Dict[str, int]:
        return dict(self._stats)

    def create_link(
        self,
        source_id: str,
        target_id: str,
        link_type: str,
        evidence: List[LinkEvidence],
        link_origin: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> EvidenceWeightedLink:
        """Create an evidence-weighted link."""
        link = EvidenceWeightedLink(
            source_id=source_id,
            target_id=target_id,
            link_type=link_type,
            evidence=evidence,
            link_origin=link_origin,
            metadata=metadata or {},
        )

        self._stats["links_created"] += 1
        if link.is_reliable:
            self._stats["high_confidence"] += 1
        elif link.is_uncertain:
            self._stats["medium_confidence"] += 1
        else:
            self._stats["low_confidence"] += 1

        return link

    def create_ast_link(
        self,
        source_id: str,
        target_id: str,
        link_type: str,
        import_path: str = "",
        link_origin: str = "import_resolver",
    ) -> EvidenceWeightedLink:
        """Create a high-confidence link from AST import resolution."""
        evidence = [LinkEvidence(
            evidence_type=EvidenceType.AST_MATCH,
            description=f"AST import resolution: {import_path}" if import_path else "AST match",
            source_artifact=import_path,
        )]
        return self.create_link(source_id, target_id, link_type, evidence, link_origin)

    def create_path_convention_link(
        self,
        source_id: str,
        target_id: str,
        link_type: str,
        source_path: str = "",
        target_path: str = "",
        link_origin: str = "edge_linker",
    ) -> EvidenceWeightedLink:
        """Create a medium-confidence link from URL/path matching."""
        evidence = [LinkEvidence(
            evidence_type=EvidenceType.PATH_CONVENTION,
            description=f"Path match: {source_path} → {target_path}",
        )]
        return self.create_link(source_id, target_id, link_type, evidence, link_origin)

    def create_naming_convention_link(
        self,
        source_id: str,
        target_id: str,
        link_type: str,
        source_name: str = "",
        target_name: str = "",
        link_origin: str = "enhanced_graph_linker",
    ) -> EvidenceWeightedLink:
        """Create a low-confidence link from naming conventions."""
        evidence = [LinkEvidence(
            evidence_type=EvidenceType.NAMING_CONVENTION,
            description=f"Name similarity: {source_name} ↔ {target_name}",
        )]
        return self.create_link(source_id, target_id, link_type, evidence, link_origin)

    def create_commit_history_link(
        self,
        source_id: str,
        target_id: str,
        link_type: str,
        co_change_count: int = 0,
        link_origin: str = "temporal_analyzer",
    ) -> EvidenceWeightedLink:
        """Create a link from git co-change history."""
        modifier = min(0.15, co_change_count * 0.02)  # More co-changes → higher confidence
        evidence = [LinkEvidence(
            evidence_type=EvidenceType.COMMIT_HISTORY,
            description=f"Co-changed {co_change_count} times in git history",
            confidence_modifier=modifier,
        )]
        return self.create_link(source_id, target_id, link_type, evidence, link_origin)

    def merge_links(
        self,
        existing: EvidenceWeightedLink,
        new_evidence: List[LinkEvidence],
    ) -> EvidenceWeightedLink:
        """Add new evidence to an existing link (raises confidence)."""
        all_evidence = list(existing.evidence) + new_evidence
        return EvidenceWeightedLink(
            source_id=existing.source_id,
            target_id=existing.target_id,
            link_type=existing.link_type,
            evidence=all_evidence,
            link_origin=existing.link_origin,
            metadata=existing.metadata,
        )


def format_confidence_for_agent(confidence: float) -> str:
    """
    Human-readable confidence qualifier for agent output.

    Used in agent responses to communicate link reliability.
    """
    if confidence >= 0.9:
        return "confirmed"
    if confidence >= CONFIDENCE_RELIABLE:
        return "high confidence"
    if confidence >= 0.65:
        return "likely"
    if confidence >= CONFIDENCE_UNCERTAIN:
        return "uncertain — treat with caution"
    return "insufficient certainty"


# Singleton
evidence_link_factory = EvidenceLinkFactory()
