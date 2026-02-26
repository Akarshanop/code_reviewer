"""
Behavioral Intelligence Layer — Problem 1 Fix

Transforms structural graph nodes into behaviorally-classified entities.
The system no longer just models "what calls what" but understands "what this code does."

Behavioral Node Labels:
    AuthCritical        - Guards authentication/authorization boundaries
    TransactionBoundary - Manages transactional consistency (DB commits, saga patterns)
    AsyncFanOut         - Publishes to multiple async consumers (EventBus, queues)
    SideEffecting       - Produces external side effects (HTTP calls, file I/O, email)
    Idempotent          - Safe for retry (GET handlers, pure functions, cache reads)
    ExternallyVisible   - Part of the public API surface (routes, exports, SDKs)

Behavioral Edge Labels:
    GUARDS              - A node enforces a security invariant before another executes
    ESCALATES_FAILURE   - Failure in source propagates and amplifies in target
    REQUIRES_CONSISTENCY - Source and target must be atomically consistent
"""

from __future__ import annotations

import re
from enum import Enum
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Dict, List, Optional, Set, Any, Tuple

from src.core.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Behavioral classifications
# ---------------------------------------------------------------------------

class BehavioralNodeLabel(str, Enum):
    """Labels that can be attached to any code node to describe its behavior."""
    AUTH_CRITICAL = "AuthCritical"
    TRANSACTION_BOUNDARY = "TransactionBoundary"
    ASYNC_FAN_OUT = "AsyncFanOut"
    SIDE_EFFECTING = "SideEffecting"
    IDEMPOTENT = "Idempotent"
    EXTERNALLY_VISIBLE = "ExternallyVisible"


class BehavioralEdgeType(str, Enum):
    """Behavioral relationship types between code nodes."""
    GUARDS = "GUARDS"
    ESCALATES_FAILURE = "ESCALATES_FAILURE"
    REQUIRES_CONSISTENCY = "REQUIRES_CONSISTENCY"


@dataclass(frozen=True)
class BehavioralClassification:
    """Result of classifying a single code node."""
    node_id: str
    labels: Set[BehavioralNodeLabel]
    confidence: float  # 0.0–1.0 overall confidence
    evidence: List[str]  # Human-readable evidence trail
    heuristics_used: List[str]  # Which heuristic rules fired

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "labels": [l.value for l in self.labels],
            "confidence": round(self.confidence, 3),
            "evidence": self.evidence,
            "heuristics_used": self.heuristics_used,
        }


@dataclass(frozen=True)
class BehavioralEdge:
    """A behavioral edge between two nodes."""
    source_id: str
    target_id: str
    edge_type: BehavioralEdgeType
    confidence: float
    evidence: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "edge_type": self.edge_type.value,
            "confidence": round(self.confidence, 3),
            "evidence": self.evidence,
        }


# ---------------------------------------------------------------------------
# Heuristic patterns (static analysis, no LLM required)
# ---------------------------------------------------------------------------

# Auth-critical patterns
_AUTH_PATTERNS = [
    re.compile(r"auth", re.IGNORECASE),
    re.compile(r"login|logout|signIn|signOut|signUp", re.IGNORECASE),
    re.compile(r"jwt|token|bearer|oauth|session", re.IGNORECASE),
    re.compile(r"@(Secured|PreAuthorize|RequiresRole|Protected)", re.IGNORECASE),
    re.compile(r"middleware.*auth", re.IGNORECASE),
    re.compile(r"requiresAuth|isAuthenticated|checkPermission", re.IGNORECASE),
    re.compile(r"password|credential|apiKey|secret", re.IGNORECASE),
]

# Transaction boundary patterns
_TRANSACTION_PATTERNS = [
    re.compile(r"@Transactional|BEGIN|COMMIT|ROLLBACK|SAVEPOINT", re.IGNORECASE),
    re.compile(r"transaction|withTransaction|atomic|saga", re.IGNORECASE),
    re.compile(r"\.commit\(|\.rollback\(|\.begin\(", re.IGNORECASE),
    re.compile(r"two.?phase|2pc|compensat", re.IGNORECASE),
    re.compile(r"lock|mutex|semaphore|synchronized", re.IGNORECASE),
]

# Async fan-out patterns
_ASYNC_FANOUT_PATTERNS = [
    re.compile(r"eventBus\.publish|eventBus\.send|emit\(|dispatch\(", re.IGNORECASE),
    re.compile(r"kafka\.produce|rabbitmq\.publish|sns\.publish", re.IGNORECASE),
    re.compile(r"Promise\.all|Promise\.allSettled|asyncio\.gather", re.IGNORECASE),
    re.compile(r"fanOut|broadcast|notify.*all|multicast", re.IGNORECASE),
    re.compile(r"@Produces|@EventListener|@RabbitListener", re.IGNORECASE),
]

# Side-effecting patterns
_SIDE_EFFECT_PATTERNS = [
    re.compile(r"fetch\(|axios\.|http\.|request\(|urllib", re.IGNORECASE),
    re.compile(r"sendEmail|sendSms|sendNotification|push", re.IGNORECASE),
    re.compile(r"fs\.(write|append|unlink|mkdir)|file\.(write|delete)", re.IGNORECASE),
    re.compile(r"\.save\(|\.insert\(|\.update\(|\.delete\(|\.remove\(", re.IGNORECASE),
    re.compile(r"stripe\.|paypal\.|payment", re.IGNORECASE),
    re.compile(r"S3\.put|upload|download", re.IGNORECASE),
]

# Idempotent patterns
_IDEMPOTENT_PATTERNS = [
    re.compile(r"^GET$", re.IGNORECASE),  # HTTP GET handlers
    re.compile(r"@Cacheable|cache\.(get|read)", re.IGNORECASE),
    re.compile(r"pure|readonly|immutable", re.IGNORECASE),
    re.compile(r"idempotent|retry.?safe", re.IGNORECASE),
]

# Externally visible patterns
_EXTERNALLY_VISIBLE_PATTERNS = [
    re.compile(r"@(Get|Post|Put|Delete|Patch|RequestMapping|Route)", re.IGNORECASE),
    re.compile(r"router\.(get|post|put|delete|patch|all|use)", re.IGNORECASE),
    re.compile(r"app\.(get|post|put|delete|patch)", re.IGNORECASE),
    re.compile(r"@(api_view|action)\b", re.IGNORECASE),
    re.compile(r"export\s+(default\s+)?(function|class|const)", re.IGNORECASE),
    re.compile(r"module\.exports", re.IGNORECASE),
    re.compile(r"@app\.(route|get|post|put|delete)", re.IGNORECASE),
]

_PATTERN_MAP: Dict[BehavioralNodeLabel, List[re.Pattern]] = {
    BehavioralNodeLabel.AUTH_CRITICAL: _AUTH_PATTERNS,
    BehavioralNodeLabel.TRANSACTION_BOUNDARY: _TRANSACTION_PATTERNS,
    BehavioralNodeLabel.ASYNC_FAN_OUT: _ASYNC_FANOUT_PATTERNS,
    BehavioralNodeLabel.SIDE_EFFECTING: _SIDE_EFFECT_PATTERNS,
    BehavioralNodeLabel.IDEMPOTENT: _IDEMPOTENT_PATTERNS,
    BehavioralNodeLabel.EXTERNALLY_VISIBLE: _EXTERNALLY_VISIBLE_PATTERNS,
}


# ---------------------------------------------------------------------------
# Naming-convention heuristics
# ---------------------------------------------------------------------------

_NAMING_HINTS: Dict[BehavioralNodeLabel, List[re.Pattern]] = {
    BehavioralNodeLabel.AUTH_CRITICAL: [
        re.compile(r"(Auth|Security|Permission|Guard|Protect)", re.IGNORECASE),
    ],
    BehavioralNodeLabel.TRANSACTION_BOUNDARY: [
        re.compile(r"(Transaction|UnitOfWork|Saga|Atomic)", re.IGNORECASE),
    ],
    BehavioralNodeLabel.ASYNC_FAN_OUT: [
        re.compile(r"(Publisher|Emitter|Broadcaster|Dispatcher|Producer)", re.IGNORECASE),
    ],
    BehavioralNodeLabel.SIDE_EFFECTING: [
        re.compile(r"(Writer|Sender|Uploader|Notifier|Mailer)", re.IGNORECASE),
    ],
    BehavioralNodeLabel.IDEMPOTENT: [
        re.compile(r"(Reader|Getter|Fetcher|Resolver|Query)", re.IGNORECASE),
    ],
    BehavioralNodeLabel.EXTERNALLY_VISIBLE: [
        re.compile(r"(Controller|Handler|Endpoint|Resource|Api|Route)", re.IGNORECASE),
    ],
}


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

class BehavioralClassifier:
    """
    Classifies code nodes into behavioral categories using multiple heuristics.

    Classification sources (in order of confidence):
        1. Annotation / decorator patterns (0.9 confidence)
        2. Code-body pattern matches (0.7 confidence)
        3. Naming conventions (0.5 confidence)
        4. Graph-structural signals (0.6 confidence) — e.g. node is EventBus producer
    """

    def __init__(self) -> None:
        self._stats = {
            "nodes_classified": 0,
            "labels_assigned": 0,
            "edges_inferred": 0,
        }

    @property
    def stats(self) -> Dict[str, int]:
        return dict(self._stats)

    def classify_node(
        self,
        node_id: str,
        name: str = "",
        code_body: str = "",
        file_path: str = "",
        decorators: List[str] | None = None,
        node_labels: List[str] | None = None,
        http_method: str | None = None,
        is_handler: bool = False,
        is_exported: bool = False,
        eventbus_type: str | None = None,
    ) -> BehavioralClassification:
        """
        Classify a single code node.

        Args:
            node_id:        Unique node identifier (Neo4j id).
            name:           Node name (function/class/component name).
            code_body:      Source code of the node.
            decorators:     Decorator / annotation strings on the node.
            node_labels:    Existing Neo4j labels (e.g. ["VertxRoute", "Function"]).
            http_method:    HTTP method if the node is a route (GET, POST, etc.).
            is_handler:     Whether the node is a handler function.
            is_exported:    Whether the node is exported / public.
            eventbus_type:  "producer", "consumer", "publisher", etc.
        """
        labels: Set[BehavioralNodeLabel] = set()
        evidence: List[str] = []
        heuristics: List[str] = []
        confidences: List[float] = []
        transaction_pattern_hits = 0

        decorators = decorators or []
        node_labels = node_labels or []
        combined_text = f"{name} {code_body} {' '.join(decorators)}"

        # --- 1. Annotation / decorator patterns (highest confidence) ---
        decorator_text = " ".join(decorators)
        if decorator_text:
            for label, patterns in _PATTERN_MAP.items():
                for pat in patterns:
                    if pat.search(decorator_text):
                        labels.add(label)
                        match = pat.search(decorator_text)
                        matched = match.group(0) if match else ""
                        evidence.append(
                            f"Decorator match ({label.value}): pattern={pat.pattern!r}, matched={matched!r}, line=1"
                        )
                        heuristics.append(f"decorator:{label.value}")
                        confidences.append(0.9)
                        if label == BehavioralNodeLabel.TRANSACTION_BOUNDARY:
                            transaction_pattern_hits += 1
                        break

        # --- 2. Code body patterns ---
        if code_body:
            for label, patterns in _PATTERN_MAP.items():
                if label in labels:
                    continue  # already classified from decorator
                for pat in patterns:
                    if pat.search(code_body):
                        labels.add(label)
                        match = pat.search(code_body)
                        snippet = code_body[max(0, match.start() - 20):match.end() + 20] if match else ""
                        line_no = 1 + (code_body[: match.start()].count("\n") if match else 0)
                        matched = match.group(0) if match else snippet.strip()[:40]
                        evidence.append(
                            f"Code pattern ({label.value}): pattern={pat.pattern!r}, matched={matched!r}, line={line_no}"
                        )
                        heuristics.append(f"code_body:{label.value}")
                        confidences.append(0.7)
                        if label == BehavioralNodeLabel.TRANSACTION_BOUNDARY:
                            transaction_pattern_hits += 1
                        break

        # --- 3. Naming conventions ---
        for label, patterns in _NAMING_HINTS.items():
            if label in labels:
                continue
            for pat in patterns:
                if pat.search(name):
                    labels.add(label)
                    evidence.append(f"Naming convention: {pat.pattern!r} matches '{name}'")
                    heuristics.append(f"naming:{label.value}")
                    confidences.append(0.5)
                    break

        # --- 4. Graph-structural signals ---
        if "VertxRoute" in node_labels or "Route" in node_labels or is_handler:
            labels.add(BehavioralNodeLabel.EXTERNALLY_VISIBLE)
            evidence.append(f"Structural: node has label {node_labels}")
            heuristics.append("structural:ExternallyVisible")
            confidences.append(0.8)

        if http_method and http_method.upper() == "GET":
            labels.add(BehavioralNodeLabel.IDEMPOTENT)
            evidence.append(f"HTTP method GET is idempotent")
            heuristics.append("http_method:Idempotent")
            confidences.append(0.8)

        if eventbus_type in ("producer", "publisher", "publish", "send"):
            labels.add(BehavioralNodeLabel.ASYNC_FAN_OUT)
            evidence.append(f"EventBus type={eventbus_type} → async fan-out")
            heuristics.append("eventbus:AsyncFanOut")
            confidences.append(0.85)

        if is_exported:
            labels.add(BehavioralNodeLabel.EXTERNALLY_VISIBLE)
            evidence.append("Node is exported")
            heuristics.append("exported:ExternallyVisible")
            confidences.append(0.6)

        # --- Resolve conflicts ---
        # A node that is SIDE_EFFECTING cannot also be IDEMPOTENT
        if BehavioralNodeLabel.SIDE_EFFECTING in labels and BehavioralNodeLabel.IDEMPOTENT in labels:
            # side-effecting wins (more dangerous to get wrong)
            labels.discard(BehavioralNodeLabel.IDEMPOTENT)
            evidence.append("Conflict: SideEffecting overrides Idempotent")

        # Negative patterns: test/mock/stub files should not be side-effecting by default.
        lowered_path = file_path.lower()
        if lowered_path and re.search(r"(^|/)(test|tests|mock|mocks|stub|stubs)(/|$)", lowered_path):
            if BehavioralNodeLabel.SIDE_EFFECTING in labels:
                labels.discard(BehavioralNodeLabel.SIDE_EFFECTING)
                evidence.append(f"Negative pattern: suppressed SideEffecting for non-prod path '{file_path}'")

        # Negative auth weighting for parse/validate helpers.
        if BehavioralNodeLabel.AUTH_CRITICAL in labels and re.match(r"^(parse|validate)", name, re.IGNORECASE):
            confidences.append(0.6)
            evidence.append("Negative pattern: parse*/validate* name reduced AuthCritical confidence")

        # Transaction boundaries need stronger evidence (>1 hit) for high confidence.
        if BehavioralNodeLabel.TRANSACTION_BOUNDARY in labels and transaction_pattern_hits <= 1:
            confidences.append(0.5)
            evidence.append("TransactionBoundary confidence capped: only one transaction pattern matched")

        overall_confidence = (
            max(confidences) if confidences else 0.0
        )

        self._stats["nodes_classified"] += 1
        self._stats["labels_assigned"] += len(labels)

        return BehavioralClassification(
            node_id=node_id,
            labels=labels,
            confidence=overall_confidence,
            evidence=evidence,
            heuristics_used=heuristics,
        )

    def propagate_side_effecting_labels(
        self,
        classified_nodes: Dict[str, BehavioralClassification],
        structural_edges: List[Dict[str, Any]],
        max_depth: int = 3,
    ) -> Dict[str, BehavioralClassification]:
        """
        Propagate SideEffecting up the call graph (callee -> caller) with lower confidence.
        """
        reverse_calls: Dict[str, Set[str]] = defaultdict(set)
        for edge in structural_edges:
            edge_type = edge.get("type", "")
            if edge_type not in {"CALLS", "CALLS_FUNCTION"}:
                continue
            src = edge.get("source_id")
            tgt = edge.get("target_id")
            if src and tgt:
                reverse_calls[tgt].add(src)

        propagated: Dict[str, BehavioralClassification] = dict(classified_nodes)
        queue: List[Tuple[str, int]] = []
        for node_id, cls in classified_nodes.items():
            if BehavioralNodeLabel.SIDE_EFFECTING in cls.labels:
                queue.append((node_id, 0))

        visited: Set[Tuple[str, int]] = set()
        while queue:
            callee_id, depth = queue.pop(0)
            if depth >= max_depth:
                continue
            for caller_id in reverse_calls.get(callee_id, set()):
                key = (caller_id, depth + 1)
                if key in visited:
                    continue
                visited.add(key)
                caller_cls = propagated.get(caller_id)
                if not caller_cls:
                    continue
                if BehavioralNodeLabel.SIDE_EFFECTING in caller_cls.labels:
                    continue
                new_labels = set(caller_cls.labels)
                new_labels.add(BehavioralNodeLabel.SIDE_EFFECTING)
                new_evidence = list(caller_cls.evidence) + [
                    f"Propagated SideEffecting from callee={callee_id}, propagation_depth={depth + 1}"
                ]
                propagated[caller_id] = BehavioralClassification(
                    node_id=caller_cls.node_id,
                    labels=new_labels,
                    confidence=max(caller_cls.confidence, 0.6),
                    evidence=new_evidence,
                    heuristics_used=list(caller_cls.heuristics_used) + ["propagation:SideEffecting"],
                )
                queue.append((caller_id, depth + 1))

        return propagated

    def infer_behavioral_edges(
        self,
        classified_nodes: Dict[str, BehavioralClassification],
        structural_edges: List[Dict[str, Any]],
    ) -> List[BehavioralEdge]:
        """
        Infer behavioral edges from node classifications and existing structural edges.

        Rules:
            - AuthCritical → X via structural edge → GUARDS(AuthCritical, X)
            - X → AsyncFanOut via CALLS → ESCALATES_FAILURE(X, AsyncFanOut)
            - TransactionBoundary → Y → REQUIRES_CONSISTENCY(TransactionBoundary, Y)
        """
        behavioral_edges: List[BehavioralEdge] = []
        classified_nodes = self.propagate_side_effecting_labels(
            classified_nodes=classified_nodes,
            structural_edges=structural_edges,
            max_depth=3,
        )

        for edge in structural_edges:
            source_id = edge.get("source_id", "")
            target_id = edge.get("target_id", "")
            edge_type = edge.get("type", "")

            src_cls = classified_nodes.get(source_id)
            tgt_cls = classified_nodes.get(target_id)
            if not src_cls or not tgt_cls:
                continue

            # GUARDS: AuthCritical node feeds into another node
            if BehavioralNodeLabel.AUTH_CRITICAL in src_cls.labels:
                behavioral_edges.append(BehavioralEdge(
                    source_id=source_id,
                    target_id=target_id,
                    edge_type=BehavioralEdgeType.GUARDS,
                    confidence=min(src_cls.confidence, 0.85),
                    evidence=[
                        f"AuthCritical node {source_id} guards {target_id} via {edge_type}",
                    ],
                ))

            # ESCALATES_FAILURE: calling into AsyncFanOut means failure propagates
            if BehavioralNodeLabel.ASYNC_FAN_OUT in tgt_cls.labels:
                if edge_type in ("CALLS", "CALLS_FUNCTION", "HANDLED_BY", "CALLS_ROUTE"):
                    behavioral_edges.append(BehavioralEdge(
                        source_id=source_id,
                        target_id=target_id,
                        edge_type=BehavioralEdgeType.ESCALATES_FAILURE,
                        confidence=min(src_cls.confidence, tgt_cls.confidence, 0.75),
                        evidence=[
                            f"Failure in {source_id} escalates via async fan-out in {target_id}",
                        ],
                    ))

            # REQUIRES_CONSISTENCY: TransactionBoundary must be atomic with downstream
            if BehavioralNodeLabel.TRANSACTION_BOUNDARY in src_cls.labels:
                if BehavioralNodeLabel.SIDE_EFFECTING in tgt_cls.labels:
                    behavioral_edges.append(BehavioralEdge(
                        source_id=source_id,
                        target_id=target_id,
                        edge_type=BehavioralEdgeType.REQUIRES_CONSISTENCY,
                        confidence=min(src_cls.confidence, tgt_cls.confidence, 0.80),
                        evidence=[
                            f"TransactionBoundary {source_id} requires consistency with SideEffecting {target_id}",
                        ],
                    ))

        self._stats["edges_inferred"] = len(behavioral_edges)
        return behavioral_edges


# ---------------------------------------------------------------------------
# Impact reasoning (behavioral-aware)
# ---------------------------------------------------------------------------

def compute_behavioral_impact_score(
    node_classification: BehavioralClassification,
    downstream_count: int = 0,
    upstream_count: int = 0,
) -> Tuple[float, List[str]]:
    """
    Compute impact score that accounts for behavioral classification,
    not just dependency count.

    Returns:
        (score, reasons): score in [0, 1] and list of reasoning evidence.
    """
    base_score = 0.1
    reasons: List[str] = []

    label_weights = {
        BehavioralNodeLabel.AUTH_CRITICAL: 0.30,
        BehavioralNodeLabel.TRANSACTION_BOUNDARY: 0.25,
        BehavioralNodeLabel.ASYNC_FAN_OUT: 0.20,
        BehavioralNodeLabel.SIDE_EFFECTING: 0.15,
        BehavioralNodeLabel.EXTERNALLY_VISIBLE: 0.10,
        BehavioralNodeLabel.IDEMPOTENT: -0.05,  # reduces impact (safe to retry)
    }

    for label in node_classification.labels:
        weight = label_weights.get(label, 0.0)
        base_score += weight
        reasons.append(f"{label.value}: +{weight:.2f}")

    # Structural amplifier (log-scaled to prevent domination)
    import math
    structural = 0.1 * math.log1p(downstream_count + upstream_count)
    base_score += structural
    reasons.append(f"Structural fan: {downstream_count} downstream, {upstream_count} upstream → +{structural:.2f}")

    clamped = max(0.0, min(1.0, base_score))
    return clamped, reasons


# Singleton
behavioral_classifier = BehavioralClassifier()
