"""
Structured PRD Graph Model — Problem 6 Fix

PRDs are no longer treated as unstructured text.
They are converted into first-class graph objects with:
    Capabilities   - What the feature should do
    Constraints    - Non-functional requirements and boundaries
    Assumptions    - What the PRD takes for granted
    NonGoals       - Explicitly out of scope
    Dependencies   - External systems / existing capabilities required

This enables graph-native reasoning:
    PRD Capability → Existing Capability → Gap → Risk
    PRD Constraint → Existing Architecture → Feasibility
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any

from src.core.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# PRD graph elements
# ---------------------------------------------------------------------------

class PRDPriority(str, Enum):
    MUST_HAVE = "must_have"
    SHOULD_HAVE = "should_have"
    NICE_TO_HAVE = "nice_to_have"
    WONT_HAVE = "wont_have"


class ConstraintType(str, Enum):
    PERFORMANCE = "performance"
    SECURITY = "security"
    SCALABILITY = "scalability"
    COMPLIANCE = "compliance"
    COMPATIBILITY = "compatibility"
    COST = "cost"
    TIMELINE = "timeline"
    OTHER = "other"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class PRDCapability:
    """A discrete capability described in the PRD."""
    id: str
    name: str
    description: str
    priority: PRDPriority = PRDPriority.MUST_HAVE
    acceptance_criteria: List[str] = field(default_factory=list)
    existing_capability_ids: List[str] = field(default_factory=list)  # Mapped to existing
    gap_description: str = ""  # What's missing vs existing
    estimated_effort: str = ""  # e.g. "2-3 sprints"
    risk_level: RiskLevel = RiskLevel.MEDIUM

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "priority": self.priority.value,
            "acceptance_criteria": self.acceptance_criteria,
            "existing_capability_ids": self.existing_capability_ids,
            "gap_description": self.gap_description,
            "estimated_effort": self.estimated_effort,
            "risk_level": self.risk_level.value,
        }


@dataclass
class PRDConstraint:
    """A constraint or non-functional requirement from the PRD."""
    id: str
    description: str
    constraint_type: ConstraintType = ConstraintType.OTHER
    measurable_target: str = ""  # e.g. "< 200ms p99 latency"
    current_state: str = ""  # What the system does today
    feasibility: str = ""  # Assessment against current architecture

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "constraint_type": self.constraint_type.value,
            "measurable_target": self.measurable_target,
            "current_state": self.current_state,
            "feasibility": self.feasibility,
        }


@dataclass
class PRDAssumption:
    """An assumption the PRD makes (may or may not be valid)."""
    id: str
    description: str
    validated: bool = False
    validation_evidence: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "validated": self.validated,
            "validation_evidence": self.validation_evidence,
        }


@dataclass
class PRDNonGoal:
    """Something explicitly out of scope."""
    id: str
    description: str
    rationale: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "rationale": self.rationale,
        }


@dataclass
class PRDDependency:
    """An external dependency required by the PRD."""
    id: str
    name: str
    description: str = ""
    is_internal: bool = True  # Internal codebase vs external service
    existing_in_codebase: bool = False
    blocking: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "is_internal": self.is_internal,
            "existing_in_codebase": self.existing_in_codebase,
            "blocking": self.blocking,
        }


@dataclass
class StructuredPRD:
    """
    A fully structured PRD converted from raw text.
    This is the graph-native representation of a product requirement.
    """
    id: str
    title: str
    raw_text: str
    summary: str = ""
    capabilities: List[PRDCapability] = field(default_factory=list)
    constraints: List[PRDConstraint] = field(default_factory=list)
    assumptions: List[PRDAssumption] = field(default_factory=list)
    non_goals: List[PRDNonGoal] = field(default_factory=list)
    dependencies: List[PRDDependency] = field(default_factory=list)
    risk_assessment: str = ""
    domain_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "summary": self.summary,
            "capabilities": [c.to_dict() for c in self.capabilities],
            "constraints": [c.to_dict() for c in self.constraints],
            "assumptions": [a.to_dict() for a in self.assumptions],
            "non_goals": [n.to_dict() for n in self.non_goals],
            "dependencies": [d.to_dict() for d in self.dependencies],
            "risk_assessment": self.risk_assessment,
            "domain_ids": self.domain_ids,
            "stats": {
                "capability_count": len(self.capabilities),
                "constraint_count": len(self.constraints),
                "assumption_count": len(self.assumptions),
                "must_have_count": sum(1 for c in self.capabilities if c.priority == PRDPriority.MUST_HAVE),
            },
        }

    def to_neo4j_properties(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "summary": self.summary[:500],
            "capability_count": len(self.capabilities),
            "constraint_count": len(self.constraints),
            "risk_assessment": self.risk_assessment[:500],
        }

    def get_capability_gap_analysis(self) -> List[Dict[str, Any]]:
        """
        Generate gap analysis: PRD Capability → Existing → Gap → Risk.
        """
        gaps: List[Dict[str, Any]] = []
        for cap in self.capabilities:
            gaps.append({
                "capability": cap.name,
                "priority": cap.priority.value,
                "has_existing": len(cap.existing_capability_ids) > 0,
                "existing_ids": cap.existing_capability_ids,
                "gap": cap.gap_description or "No gap analysis available",
                "risk": cap.risk_level.value,
                "effort": cap.estimated_effort,
            })
        return gaps


# ---------------------------------------------------------------------------
# PRD extraction prompt (for LLM-based extraction)
# ---------------------------------------------------------------------------

PRD_EXTRACTION_SYSTEM_PROMPT = """You are a Principal Architect extracting structured data from a Product Requirements Document (PRD).

Your task is to decompose the PRD into a structured JSON with these exact sections:

{
  "title": "Feature name",
  "summary": "One-paragraph summary",
  "capabilities": [
    {
      "name": "Capability name",
      "description": "What it does",
      "priority": "must_have | should_have | nice_to_have | wont_have",
      "acceptance_criteria": ["Criterion 1", "Criterion 2"]
    }
  ],
  "constraints": [
    {
      "description": "Constraint text",
      "type": "performance | security | scalability | compliance | compatibility | cost | timeline | other",
      "measurable_target": "e.g. < 200ms p99"
    }
  ],
  "assumptions": [
    {
      "description": "What the PRD assumes"
    }
  ],
  "non_goals": [
    {
      "description": "What is explicitly out of scope",
      "rationale": "Why it's excluded"
    }
  ],
  "dependencies": [
    {
      "name": "Dependency name",
      "description": "What it is",
      "is_internal": true,
      "blocking": false
    }
  ]
}

Rules:
- Extract ONLY what's in the PRD. Do not invent.
- If a section is not mentioned, return an empty array.
- Every capability must have at least one acceptance criterion.
- Constraints must be measurable where possible.
- Return valid JSON only.
"""


def build_structured_prd_from_llm_output(
    prd_id: str,
    raw_text: str,
    llm_json: Dict[str, Any],
) -> StructuredPRD:
    """
    Convert LLM-extracted JSON into a StructuredPRD object.

    Args:
        prd_id: Unique identifier for this PRD.
        raw_text: Original PRD text.
        llm_json: JSON output from LLM extraction.

    Returns:
        StructuredPRD with all elements populated.
    """
    capabilities = []
    for i, cap in enumerate(llm_json.get("capabilities", [])):
        capabilities.append(PRDCapability(
            id=f"{prd_id}_cap_{i}",
            name=cap.get("name", f"Capability {i+1}"),
            description=cap.get("description", ""),
            priority=PRDPriority(cap.get("priority", "must_have")),
            acceptance_criteria=cap.get("acceptance_criteria", []),
        ))

    constraints = []
    for i, con in enumerate(llm_json.get("constraints", [])):
        constraints.append(PRDConstraint(
            id=f"{prd_id}_con_{i}",
            description=con.get("description", ""),
            constraint_type=ConstraintType(con.get("type", "other")),
            measurable_target=con.get("measurable_target", ""),
        ))

    assumptions = []
    for i, asm in enumerate(llm_json.get("assumptions", [])):
        assumptions.append(PRDAssumption(
            id=f"{prd_id}_asm_{i}",
            description=asm.get("description", ""),
        ))

    non_goals = []
    for i, ng in enumerate(llm_json.get("non_goals", [])):
        non_goals.append(PRDNonGoal(
            id=f"{prd_id}_ng_{i}",
            description=ng.get("description", ""),
            rationale=ng.get("rationale", ""),
        ))

    dependencies = []
    for i, dep in enumerate(llm_json.get("dependencies", [])):
        dependencies.append(PRDDependency(
            id=f"{prd_id}_dep_{i}",
            name=dep.get("name", f"Dependency {i+1}"),
            description=dep.get("description", ""),
            is_internal=dep.get("is_internal", True),
            blocking=dep.get("blocking", False),
        ))

    return StructuredPRD(
        id=prd_id,
        title=llm_json.get("title", "Untitled PRD"),
        raw_text=raw_text,
        summary=llm_json.get("summary", ""),
        capabilities=capabilities,
        constraints=constraints,
        assumptions=assumptions,
        non_goals=non_goals,
        dependencies=dependencies,
    )
