"""
QA Projection Layer — Problem 7 Fix

QA reasoning should not think like a developer.
Instead of "test this function," QA reasons about:
    - Async chains (message flows that could drop or duplicate)
    - Failure surfaces (where errors propagate without being caught)
    - Race windows (concurrent access to shared state)
    - Untested paths (code branches with no test coverage signals)
    - Volatility hotspots (frequently-changing code with high impact)

QA output produces:
    - Test intents (what should be tested and why)
    - Risk-ranked scenarios (highest-risk first)
    - NOT raw test code (test code comes later, grounded in intent)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple

from src.core.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# QA-specific abstractions
# ---------------------------------------------------------------------------

class QARiskCategory(str, Enum):
    ASYNC_CHAIN = "async_chain"
    FAILURE_SURFACE = "failure_surface"
    RACE_WINDOW = "race_window"
    UNTESTED_PATH = "untested_path"
    VOLATILITY_HOTSPOT = "volatility_hotspot"
    AUTH_BOUNDARY = "auth_boundary"
    DATA_CONSISTENCY = "data_consistency"
    EXTERNAL_DEPENDENCY = "external_dependency"


class QASeverity(str, Enum):
    CRITICAL = "critical"  # Could cause data loss or security breach
    HIGH = "high"  # Could cause service outage or major bug
    MEDIUM = "medium"  # Could cause degraded experience
    LOW = "low"  # Minor issue, cosmetic or edge case


@dataclass
class TestIntent:
    """
    What should be tested and why — not how.
    
    Test intents are the QA layer's primary output.
    They describe the purpose and reasoning behind a test,
    leaving implementation details for later.
    """
    id: str
    title: str
    description: str
    risk_category: QARiskCategory
    severity: QASeverity
    rationale: str  # Why this needs testing — the evidence trail
    affected_nodes: List[str] = field(default_factory=list)  # Node IDs
    affected_paths: List[str] = field(default_factory=list)  # File paths
    preconditions: List[str] = field(default_factory=list)
    expected_behavior: str = ""
    failure_mode: str = ""  # What goes wrong if this isn't tested
    confidence: float = 0.0  # How confident we are this is a real risk

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "risk_category": self.risk_category.value,
            "severity": self.severity.value,
            "rationale": self.rationale,
            "affected_nodes": self.affected_nodes,
            "affected_paths": self.affected_paths,
            "preconditions": self.preconditions,
            "expected_behavior": self.expected_behavior,
            "failure_mode": self.failure_mode,
            "confidence": round(self.confidence, 3),
        }


@dataclass
class QAScenario:
    """
    A risk-ranked testing scenario combining multiple test intents.
    """
    id: str
    name: str
    description: str
    risk_score: float  # 0-1, composite risk
    test_intents: List[TestIntent] = field(default_factory=list)
    domains_affected: List[str] = field(default_factory=list)
    estimated_test_effort: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "risk_score": round(self.risk_score, 3),
            "test_intents": [t.to_dict() for t in self.test_intents],
            "domains_affected": self.domains_affected,
            "estimated_test_effort": self.estimated_test_effort,
            "intent_count": len(self.test_intents),
            "max_severity": max(
                (t.severity for t in self.test_intents),
                key=lambda s: ["low", "medium", "high", "critical"].index(s.value),
                default=QASeverity.LOW,
            ).value,
        }


@dataclass
class QAProjectionResult:
    """Complete QA projection output."""
    scenarios: List[QAScenario] = field(default_factory=list)
    all_intents: List[TestIntent] = field(default_factory=list)
    coverage_gaps: List[str] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": self.summary,
            "scenario_count": len(self.scenarios),
            "total_test_intents": len(self.all_intents),
            "coverage_gaps": self.coverage_gaps,
            "scenarios": [s.to_dict() for s in self.scenarios],
            "severity_distribution": self._severity_distribution(),
            "category_distribution": self._category_distribution(),
        }

    def _severity_distribution(self) -> Dict[str, int]:
        dist: Dict[str, int] = {}
        for intent in self.all_intents:
            dist[intent.severity.value] = dist.get(intent.severity.value, 0) + 1
        return dist

    def _category_distribution(self) -> Dict[str, int]:
        dist: Dict[str, int] = {}
        for intent in self.all_intents:
            dist[intent.risk_category.value] = dist.get(intent.risk_category.value, 0) + 1
        return dist


# ---------------------------------------------------------------------------
# QA Projection Engine
# ---------------------------------------------------------------------------

class QAProjectionEngine:
    """
    Projects graph intelligence into QA-specific reasoning.

    Instead of exposing raw graph data to QA, this engine translates
    architectural facts into testing concerns:
        - Graph cycles → potential deadlock / race windows
        - Unguarded auth paths → security test intents
        - High-volatility nodes → regression risk scenarios
        - EventBus fan-out → async chain reliability scenarios
        - Missing HANDLED_BY edges → untested error paths
    """

    def __init__(self) -> None:
        self._stats = {"projections_run": 0, "intents_generated": 0}

    @property
    def stats(self) -> Dict[str, int]:
        return dict(self._stats)

    def project_async_chains(
        self,
        eventbus_nodes: List[Dict[str, Any]],
        eventbus_edges: List[Dict[str, Any]],
    ) -> List[TestIntent]:
        """
        Identify async chains that could drop, duplicate, or deadlock.
        """
        intents: List[TestIntent] = []
        # Find producers without matching consumers
        producer_addresses = set()
        consumer_addresses = set()

        for node in eventbus_nodes:
            ntype = node.get("type", "")
            address = node.get("address", "")
            if ntype in ("producer", "publisher", "publish", "send"):
                producer_addresses.add(address)
            elif ntype in ("consumer", "handler", "consume"):
                consumer_addresses.add(address)

        orphan_producers = producer_addresses - consumer_addresses
        for addr in orphan_producers:
            intents.append(TestIntent(
                id=f"async_orphan_{addr[:40]}",
                title=f"Orphan producer on EventBus address: {addr}",
                description=f"Producer publishes to '{addr}' but no consumer is registered",
                risk_category=QARiskCategory.ASYNC_CHAIN,
                severity=QASeverity.HIGH,
                rationale="Messages sent to this address will be silently dropped",
                failure_mode="Silent data loss — messages published but never processed",
                confidence=0.85,
            ))

        # Find fan-out patterns (one producer → many consumers)
        for node in eventbus_nodes:
            if node.get("type") in ("producer", "publisher"):
                addr = node.get("address", "")
                consumers = [
                    e for e in eventbus_edges
                    if e.get("address") == addr and e.get("type") == "PUBLISHES_TO"
                ]
                if len(consumers) >= 3:
                    intents.append(TestIntent(
                        id=f"async_fanout_{addr[:40]}",
                        title=f"High fan-out on EventBus: {addr}",
                        description=f"Address '{addr}' has {len(consumers)} consumers — ordering and completeness are at risk",
                        risk_category=QARiskCategory.ASYNC_CHAIN,
                        severity=QASeverity.HIGH,
                        rationale=f"Fan-out to {len(consumers)} consumers increases failure surface",
                        failure_mode="Partial processing — some consumers may fail while others succeed",
                        confidence=0.80,
                    ))

        return intents

    def project_failure_surfaces(
        self,
        nodes_with_behavior: List[Dict[str, Any]],
    ) -> List[TestIntent]:
        """
        Identify paths where errors propagate without being caught.
        """
        intents: List[TestIntent] = []

        for node in nodes_with_behavior:
            behaviors = node.get("behavioral_labels", [])
            name = node.get("name", "")
            node_id = node.get("id", "")

            if "SideEffecting" in behaviors and "ExternallyVisible" in behaviors:
                intents.append(TestIntent(
                    id=f"failure_surface_{node_id[:40]}",
                    title=f"Side-effecting public endpoint: {name}",
                    description=f"{name} is externally visible AND produces side effects",
                    risk_category=QARiskCategory.FAILURE_SURFACE,
                    severity=QASeverity.HIGH,
                    rationale="External callers can trigger side effects; failures must be handled gracefully",
                    expected_behavior="Endpoint returns appropriate error codes; side effects are rolled back on failure",
                    failure_mode="Partial writes, orphaned state, or inconsistent external state on error",
                    confidence=0.75,
                    affected_nodes=[node_id],
                ))

            if "TransactionBoundary" in behaviors and "AsyncFanOut" in behaviors:
                intents.append(TestIntent(
                    id=f"consistency_risk_{node_id[:40]}",
                    title=f"Transaction + async fan-out: {name}",
                    description=f"{name} manages transactions but also fans out asynchronously",
                    risk_category=QARiskCategory.DATA_CONSISTENCY,
                    severity=QASeverity.CRITICAL,
                    rationale="Transaction boundaries and async fan-out can lead to inconsistent state",
                    failure_mode="Transaction commits but async consumers see stale or missing data",
                    confidence=0.85,
                    affected_nodes=[node_id],
                ))

        return intents

    def project_volatility_hotspots(
        self,
        nodes_with_temporal: List[Dict[str, Any]],
        volatility_threshold: float = 0.5,
    ) -> List[TestIntent]:
        """
        Flag frequently-changing code that needs regression testing.
        """
        intents: List[TestIntent] = []

        for node in nodes_with_temporal:
            volatility = node.get("volatility_score", 0.0)
            if volatility < volatility_threshold:
                continue

            name = node.get("name", node.get("file_path", "unknown"))
            node_id = node.get("id", "")
            change_count = node.get("change_count", 0)

            severity = (
                QASeverity.CRITICAL if volatility >= 0.8 else
                QASeverity.HIGH if volatility >= 0.6 else
                QASeverity.MEDIUM
            )

            intents.append(TestIntent(
                id=f"volatility_{node_id[:40]}",
                title=f"Volatility hotspot: {name}",
                description=f"{name} has volatility score {volatility:.2f} with {change_count} recent changes",
                risk_category=QARiskCategory.VOLATILITY_HOTSPOT,
                severity=severity,
                rationale=f"High change frequency (volatility={volatility:.2f}) indicates active development area",
                failure_mode="Regression bugs due to frequent changes in a critical code path",
                confidence=min(0.9, 0.5 + volatility * 0.4),
                affected_nodes=[node_id],
                affected_paths=[node.get("file_path", "")],
            ))

        return intents

    def project_auth_boundaries(
        self,
        routes: List[Dict[str, Any]],
        auth_nodes: List[Dict[str, Any]],
    ) -> List[TestIntent]:
        """
        Identify routes that might be missing authentication guards.
        """
        intents: List[TestIntent] = []

        guarded_routes: set = set()
        for auth_node in auth_nodes:
            guarded = auth_node.get("guards_routes", [])
            guarded_routes.update(guarded)

        for route in routes:
            route_id = route.get("id", "")
            path = route.get("path", "")
            method = route.get("method", "")

            if route_id not in guarded_routes and method.upper() != "GET":
                intents.append(TestIntent(
                    id=f"auth_gap_{route_id[:40]}",
                    title=f"Potentially unguarded route: {method} {path}",
                    description=f"Route {method} {path} has no detected auth guard",
                    risk_category=QARiskCategory.AUTH_BOUNDARY,
                    severity=QASeverity.CRITICAL if method.upper() in ("POST", "PUT", "DELETE") else QASeverity.HIGH,
                    rationale="No AuthCritical node found guarding this route",
                    expected_behavior="Route should require authentication/authorization",
                    failure_mode="Unauthorized access to protected resources",
                    confidence=0.65,  # Lower — might be a false positive
                    affected_nodes=[route_id],
                ))

        return intents

    def build_projection(
        self,
        eventbus_nodes: List[Dict[str, Any]] = None,
        eventbus_edges: List[Dict[str, Any]] = None,
        nodes_with_behavior: List[Dict[str, Any]] = None,
        nodes_with_temporal: List[Dict[str, Any]] = None,
        routes: List[Dict[str, Any]] = None,
        auth_nodes: List[Dict[str, Any]] = None,
    ) -> QAProjectionResult:
        """
        Build complete QA projection from all available intelligence.
        """
        all_intents: List[TestIntent] = []

        if eventbus_nodes and eventbus_edges:
            all_intents.extend(self.project_async_chains(eventbus_nodes, eventbus_edges))

        if nodes_with_behavior:
            all_intents.extend(self.project_failure_surfaces(nodes_with_behavior))

        if nodes_with_temporal:
            all_intents.extend(self.project_volatility_hotspots(nodes_with_temporal))

        if routes:
            all_intents.extend(self.project_auth_boundaries(routes, auth_nodes or []))

        # Sort by severity (critical first), then by confidence
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        all_intents.sort(
            key=lambda t: (severity_order.get(t.severity.value, 4), -t.confidence)
        )

        # Group into scenarios by category
        by_category: Dict[str, List[TestIntent]] = {}
        for intent in all_intents:
            by_category.setdefault(intent.risk_category.value, []).append(intent)

        scenarios: List[QAScenario] = []
        for category, intents in by_category.items():
            risk_score = max(t.confidence for t in intents) if intents else 0.0
            scenarios.append(QAScenario(
                id=f"scenario_{category}",
                name=category.replace("_", " ").title(),
                description=f"Testing scenario for {category} risks ({len(intents)} test intents)",
                risk_score=risk_score,
                test_intents=intents,
            ))

        scenarios.sort(key=lambda s: -s.risk_score)

        self._stats["projections_run"] += 1
        self._stats["intents_generated"] += len(all_intents)

        return QAProjectionResult(
            scenarios=scenarios,
            all_intents=all_intents,
            summary=f"Generated {len(all_intents)} test intents across {len(scenarios)} risk categories",
        )


# Singleton
qa_projection_engine = QAProjectionEngine()
