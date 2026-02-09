"""
Policy Layer — Problem 5 Fix (Part 2 of 2)

This is the OUTER layer of the two-layer agent architecture.
The Policy Layer enforces:
    - RBAC (role-based access control)
    - Content redaction (strip file names, class names for sales)
    - Abstraction level (translate technical details into business language)
    - Output shaping (format for the target audience)

Policy Layer NEVER accesses the graph directly.
It only transforms ReasoningResult objects.

Role Rules:
    admin       → Full access: all files, classes, architecture details
    architect   → Full technical detail, no admin operations
    developer   → Code-level detail, no admin operations
    qa          → Test-oriented view, behavioral summaries
    sales       → Business-only: feasibility, risk, complexity, reuse signals
                  NEVER: file names, class names, internal architecture
    viewer      → Read-only codebase view
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Set

from src.core.logger import get_logger
from src.agents.reasoning_layer import ReasoningResult, ReasoningConclusion

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Role definitions
# ---------------------------------------------------------------------------

class UserRole(str, Enum):
    ADMIN = "admin"
    ARCHITECT = "architect"
    DEVELOPER = "developer"
    QA = "qa"
    SALES = "sales"
    VIEWER = "viewer"


# What each role is allowed to see
_ROLE_PERMISSIONS: Dict[UserRole, Set[str]] = {
    UserRole.ADMIN: {
        "files", "classes", "functions", "routes", "domains",
        "behavioral", "temporal", "evidence_links", "reasoning_trace",
        "impact_score", "risk_level", "architecture", "code_snippets",
    },
    UserRole.ARCHITECT: {
        "files", "classes", "functions", "routes", "domains",
        "behavioral", "temporal", "evidence_links", "reasoning_trace",
        "impact_score", "risk_level", "architecture", "code_snippets",
    },
    UserRole.DEVELOPER: {
        "files", "classes", "functions", "routes", "domains",
        "behavioral", "temporal", "evidence_links",
        "impact_score", "risk_level", "code_snippets",
    },
    UserRole.QA: {
        "domains", "behavioral", "temporal",
        "impact_score", "risk_level", "test_intents",
        "routes",  # QA needs to know what to test
    },
    UserRole.SALES: {
        "domains", "impact_score", "risk_level",
        "feasibility", "complexity", "reuse_signals",
    },
    UserRole.VIEWER: {
        "files", "classes", "functions", "routes", "domains",
    },
}

# Patterns to redact for sales role
_CODE_ARTIFACT_PATTERNS = [
    re.compile(r"\b[A-Z][a-z]+[A-Z]\w+\b"),  # CamelCase class names
    re.compile(r"\b\w+\.(java|py|js|ts|jsx|tsx|vue|go|rs)\b"),  # File names
    re.compile(r"/[a-zA-Z_/]+\.\w{1,4}\b"),  # File paths
    re.compile(r"\b(src|lib|app|backend|frontend)/\S+"),  # Project paths
    re.compile(r"`[^`]+`"),  # Inline code references
]


# ---------------------------------------------------------------------------
# Policy output types
# ---------------------------------------------------------------------------

@dataclass
class PolicyOutput:
    """
    The final output after Policy Layer processing.
    This is what the end user sees.
    """
    # User-facing content
    answer: str
    confidence_label: str  # "high confidence", "uncertain", etc.
    risk_level: str
    
    # Visible metadata (role-filtered)
    affected_domains: List[str] = field(default_factory=list)
    impact_summary: str = ""
    
    # Technical details (only for technical roles)
    affected_files: List[str] = field(default_factory=list)
    affected_classes: List[str] = field(default_factory=list)
    affected_routes: List[str] = field(default_factory=list)
    behavioral_flags: List[str] = field(default_factory=list)
    
    # Evidence (for transparency)
    evidence_summary: List[str] = field(default_factory=list)
    uncertainty_notes: List[str] = field(default_factory=list)
    
    # Trust signal
    is_silence: bool = False
    silence_reason: str = ""
    
    # Internal metadata (not shown to user, for logging)
    role_applied: str = ""
    redactions_applied: int = 0
    original_confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "answer": self.answer,
            "confidence": self.confidence_label,
            "risk_level": self.risk_level,
            "affected_domains": self.affected_domains,
        }
        
        if self.is_silence:
            result["notice"] = "The system could not provide a confident answer."
            result["reason"] = self.silence_reason
            
        if self.affected_files:
            result["affected_files"] = self.affected_files
        if self.affected_classes:
            result["affected_classes"] = self.affected_classes
        if self.affected_routes:
            result["affected_routes"] = self.affected_routes
        if self.behavioral_flags:
            result["behavioral_flags"] = self.behavioral_flags
        if self.evidence_summary:
            result["evidence"] = self.evidence_summary
        if self.uncertainty_notes:
            result["uncertainty"] = self.uncertainty_notes
        if self.impact_summary:
            result["impact_summary"] = self.impact_summary
            
        return result


# ---------------------------------------------------------------------------
# Policy Layer implementation
# ---------------------------------------------------------------------------

class PolicyLayer:
    """
    Transforms ReasoningResult into role-appropriate PolicyOutput.
    
    This layer:
        1. Checks user role permissions
        2. Redacts sensitive information for restricted roles
        3. Translates technical conclusions to appropriate abstraction level
        4. Shapes output format for the target audience
    """

    def __init__(self) -> None:
        self._stats = {
            "policies_applied": 0,
            "redactions": 0,
            "silence_preserved": 0,
        }

    @property
    def stats(self) -> Dict[str, int]:
        return dict(self._stats)

    def apply_policy(
        self,
        reasoning: ReasoningResult,
        role: UserRole | str,
        raw_answer: str = "",
    ) -> PolicyOutput:
        """
        Apply role-based policy to a ReasoningResult.

        Args:
            reasoning: Output from the Reasoning Layer.
            role: User's role (string or UserRole enum).
            raw_answer: The raw LLM-generated answer text to be filtered.

        Returns:
            PolicyOutput: Role-appropriate output for the end user.
        """
        if isinstance(role, str):
            try:
                role = UserRole(role.lower())
            except ValueError:
                role = UserRole.VIEWER  # Default to most restrictive

        permissions = _ROLE_PERMISSIONS.get(role, _ROLE_PERMISSIONS[UserRole.VIEWER])

        # Handle trust-preserving silence
        if reasoning.trust_preserving_silence:
            self._stats["silence_preserved"] += 1
            self._stats["policies_applied"] += 1
            return PolicyOutput(
                answer=self._format_silence(reasoning.silence_reason, role),
                confidence_label="insufficient certainty",
                risk_level="unknown",
                is_silence=True,
                silence_reason=reasoning.silence_reason,
                role_applied=role.value,
                original_confidence=reasoning.overall_confidence,
            )

        # Build the answer
        answer = raw_answer or self._build_answer_from_conclusions(reasoning.conclusions, role)

        # Apply redactions for restricted roles
        redaction_count = 0
        if role == UserRole.SALES:
            answer, redaction_count = self._redact_code_artifacts(answer)

        # Filter metadata by permissions
        affected_files = reasoning.affected_files if "files" in permissions else []
        affected_classes = reasoning.affected_classes if "classes" in permissions else []
        affected_routes = reasoning.affected_routes if "routes" in permissions else []
        behavioral_flags = reasoning.behavioral_flags if "behavioral" in permissions else []
        affected_domains = reasoning.affected_domains if "domains" in permissions else []

        # Build evidence summary (technical roles see more)
        evidence_summary: List[str] = []
        uncertainty_notes: List[str] = []
        for conclusion in reasoning.conclusions:
            if "reasoning_trace" in permissions:
                for ev in conclusion.evidence:
                    evidence_summary.append(f"[{ev.source}] {ev.description}")
            if conclusion.uncertainty:
                uncertainty_notes.append(conclusion.uncertainty)

        # Impact summary (adapted per role)
        impact_summary = self._format_impact(reasoning, role, permissions)

        # Confidence label
        from src.intelligence.evidence_links import format_confidence_for_agent
        confidence_label = format_confidence_for_agent(reasoning.overall_confidence)

        self._stats["policies_applied"] += 1
        self._stats["redactions"] += redaction_count

        return PolicyOutput(
            answer=answer,
            confidence_label=confidence_label,
            risk_level=reasoning.risk_level,
            affected_domains=affected_domains,
            impact_summary=impact_summary,
            affected_files=affected_files,
            affected_classes=affected_classes,
            affected_routes=affected_routes,
            behavioral_flags=behavioral_flags,
            evidence_summary=evidence_summary[:10],
            uncertainty_notes=uncertainty_notes,
            role_applied=role.value,
            redactions_applied=redaction_count,
            original_confidence=reasoning.overall_confidence,
        )

    def _build_answer_from_conclusions(
        self,
        conclusions: List[ReasoningConclusion],
        role: UserRole,
    ) -> str:
        """Build user-facing answer from reasoning conclusions."""
        if not conclusions:
            return "No conclusions could be drawn from the available evidence."

        parts: List[str] = []
        for c in conclusions:
            if c.should_refuse:
                parts.append(f"- {c.statement} (insufficient certainty)")
            else:
                label = c.confidence_label
                parts.append(f"- {c.statement} [{label}]")

        return "\n".join(parts)

    def _redact_code_artifacts(self, text: str) -> tuple[str, int]:
        """
        Remove code-level details from text for non-technical roles.
        Returns (redacted_text, redaction_count).
        """
        count = 0
        result = text

        for pattern in _CODE_ARTIFACT_PATTERNS:
            matches = pattern.findall(result)
            if matches:
                count += len(matches)
                result = pattern.sub("[internal component]", result)

        return result, count

    def _format_silence(self, reason: str, role: UserRole) -> str:
        """Format trust-preserving silence message."""
        if role == UserRole.SALES:
            return (
                "The system does not have enough information to provide a confident "
                "assessment for this query. Please ask your technical team for a "
                "detailed investigation."
            )
        return (
            f"I cannot provide a confident answer to this query. "
            f"Reason: {reason}. "
            f"The system prefers silence over speculation."
        )

    def _format_impact(
        self,
        reasoning: ReasoningResult,
        role: UserRole,
        permissions: Set[str],
    ) -> str:
        """Format impact summary appropriate for the role."""
        if role == UserRole.SALES:
            risk = reasoning.risk_level
            score = reasoning.impact_score
            domains = ", ".join(reasoning.affected_domains[:3]) or "general"
            return (
                f"This affects the {domains} area(s) of the system. "
                f"Risk level: {risk}. "
                f"Impact score: {score:.0%}."
            )

        if role == UserRole.QA:
            return (
                f"Risk: {reasoning.risk_level}. "
                f"Impact score: {reasoning.impact_score:.2f}. "
                f"Behavioral flags: {', '.join(reasoning.behavioral_flags[:5]) or 'none'}. "
                f"Recently modified: {len(reasoning.recently_modified)} files."
            )

        # Technical roles (admin, architect, developer)
        parts = [
            f"Risk: {reasoning.risk_level}",
            f"Impact: {reasoning.impact_score:.2f}",
            f"Files: {len(reasoning.affected_files)}",
            f"Routes: {len(reasoning.affected_routes)}",
        ]
        if reasoning.behavioral_flags:
            parts.append(f"Behavioral: {', '.join(reasoning.behavioral_flags[:5])}")
        if reasoning.volatility_summary:
            parts.append(f"Volatility: {reasoning.volatility_summary.get('max_volatility', 'N/A')}")

        return " | ".join(parts)


# Singleton
policy_layer = PolicyLayer()
