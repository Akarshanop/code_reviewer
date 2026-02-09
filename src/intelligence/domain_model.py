"""
Domain-Centric Reasoning Model — Problem 2 Fix

Replaces repo-centric thinking with a Domain → Capability → Repo → CodeNode hierarchy.

At enterprise scale, repositories are implementation details.
Users ask "How does our authentication work?" — not "What's in repo X?"

Domain Hierarchy:
    Domain      - Business domain (e.g. "Authentication", "Payments", "Inventory")
    Capability  - What the domain can do (e.g. "Login", "Token refresh", "Password reset")
    Contract    - API/EventBus contracts between capabilities
    CodeNode    - Actual code artifacts implementing a capability

All user questions resolve through Domain first:
    Query → Domain identification → Capability matching → Repo/Code resolution
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Any, Tuple

from src.core.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Core domain abstractions
# ---------------------------------------------------------------------------

class DomainStatus(str, Enum):
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    PLANNED = "planned"


class ContractType(str, Enum):
    REST_API = "rest_api"
    GRAPHQL = "graphql"
    EVENTBUS = "eventbus"
    GRPC = "grpc"
    INTERNAL = "internal"  # In-process call


@dataclass
class Contract:
    """
    An explicit contract between capabilities.
    Contracts are the connective tissue of enterprise architecture.
    """
    id: str
    name: str
    contract_type: ContractType
    provider_capability_id: str
    consumer_capability_ids: List[str] = field(default_factory=list)
    endpoint: str = ""  # API path, EventBus address, etc.
    method: str = ""  # HTTP method, message type
    schema_ref: str = ""  # OpenAPI schema ref, DTO name
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "contract_type": self.contract_type.value,
            "provider_capability_id": self.provider_capability_id,
            "consumer_capability_ids": self.consumer_capability_ids,
            "endpoint": self.endpoint,
            "method": self.method,
            "confidence": self.confidence,
        }

    def to_neo4j_properties(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "contract_type": self.contract_type.value,
            "provider_capability_id": self.provider_capability_id,
            "endpoint": self.endpoint,
            "method": self.method,
            "confidence": self.confidence,
        }


@dataclass
class Capability:
    """
    A discrete business capability within a domain.
    Maps to one or more code implementations across repositories.
    """
    id: str
    name: str
    description: str = ""
    domain_id: str = ""
    implementing_repos: List[str] = field(default_factory=list)
    implementing_node_ids: List[str] = field(default_factory=list)
    contracts_provided: List[str] = field(default_factory=list)
    contracts_consumed: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    confidence: float = 0.0  

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "domain_id": self.domain_id,
            "implementing_repos": self.implementing_repos,
            "implementing_node_count": len(self.implementing_node_ids),
            "contracts_provided": self.contracts_provided,
            "contracts_consumed": self.contracts_consumed,
            "confidence": self.confidence,
        }

    def to_neo4j_properties(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "domain_id": self.domain_id,
            "keywords": self.keywords,
            "confidence": self.confidence,
        }


@dataclass
class Domain:
    """
    A business domain grouping related capabilities.
    This is the primary reasoning unit for enterprise-scale systems.
    """
    id: str
    name: str
    description: str = ""
    status: DomainStatus = DomainStatus.ACTIVE
    capabilities: List[Capability] = field(default_factory=list)
    owner_team: str = ""
    keywords: List[str] = field(default_factory=list)
    related_domain_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "status": self.status.value,
            "capability_count": len(self.capabilities),
            "capabilities": [c.to_dict() for c in self.capabilities],
            "owner_team": self.owner_team,
            "keywords": self.keywords,
        }

    def to_neo4j_properties(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "status": self.status.value,
            "owner_team": self.owner_team,
            "keywords": self.keywords,
        }


# ---------------------------------------------------------------------------
# Domain inference engine
# ---------------------------------------------------------------------------

# Keyword → domain mapping (heuristic, extensible)
_DOMAIN_KEYWORDS: Dict[str, List[str]] = {
    "authentication": [
        "auth", "login", "logout", "session", "jwt", "token", "oauth",
        "password", "credential", "sso", "mfa", "2fa", "rbac", "permission",
        "role", "access.control", "identity", "signup", "register",
    ],
    "payments": [
        "payment", "billing", "invoice", "subscription", "checkout",
        "stripe", "paypal", "charge", "refund", "transaction", "price",
        "currency", "tax", "discount", "coupon",
    ],
    "user_management": [
        "user", "profile", "account", "preference", "settings",
        "avatar", "onboarding", "notification.preference",
    ],
    "notifications": [
        "notification", "email", "sms", "push", "alert", "webhook",
        "subscribe", "broadcast", "template.email", "mailer",
    ],
    "data_ingestion": [
        "ingest", "import", "parse", "extract", "pipeline",
        "etl", "transform", "load", "crawl", "scrape", "clone",
    ],
    "search_and_discovery": [
        "search", "query", "filter", "sort", "index", "elastic",
        "qdrant", "vector", "embedding", "semantic", "fulltext",
    ],
    "analytics": [
        "analytics", "metric", "dashboard", "report", "chart",
        "insight", "statistics", "kpi", "tracking", "event.track",
    ],
    "content_management": [
        "content", "cms", "article", "blog", "page", "document",
        "media", "upload", "asset", "file.management",
    ],
    "api_gateway": [
        "gateway", "proxy", "rate.limit", "throttle", "cors",
        "middleware", "router", "dispatch", "route", "endpoint",
    ],
    "infrastructure": [
        "deploy", "ci.cd", "docker", "kubernetes", "config",
        "environment", "health.check", "monitoring", "logging",
    ],
}


class DomainInferenceEngine:
    """
    Infers domain membership of code nodes from naming, path, and content signals.

    This replaces repo-centric navigation with domain-first reasoning.
    """

    def __init__(self, custom_domains: Optional[Dict[str, List[str]]] = None):
        self._domain_keywords = dict(_DOMAIN_KEYWORDS)
        if custom_domains:
            self._domain_keywords.update(custom_domains)
        self._compiled_patterns: Dict[str, List[re.Pattern]] = {}
        for domain, keywords in self._domain_keywords.items():
            self._compiled_patterns[domain] = [
                re.compile(r"\b" + kw.replace(".", r"\.?") + r"\b", re.IGNORECASE)
                for kw in keywords
            ]
        self._stats = {"queries_resolved": 0, "nodes_classified": 0}

    @property
    def stats(self) -> Dict[str, int]:
        return dict(self._stats)

    def infer_domain_for_node(
        self,
        node_name: str = "",
        file_path: str = "",
        code_body: str = "",
        node_labels: Optional[List[str]] = None,
    ) -> List[Tuple[str, float, List[str]]]:
        """
        Infer which domain(s) a code node belongs to.

        Returns:
            List of (domain_name, confidence, evidence) sorted by confidence desc.
        """
        combined = f"{node_name} {file_path} {code_body[:500]}"
        results: Dict[str, Tuple[float, List[str]]] = {}

        for domain, patterns in self._compiled_patterns.items():
            matches: List[str] = []
            for pat in patterns:
                if pat.search(combined):
                    matches.append(pat.pattern)
            if matches:
                # More keyword matches → higher confidence
                confidence = min(0.95, 0.3 + 0.15 * len(matches))
                evidence = [f"Keyword match: {m}" for m in matches[:5]]
                results[domain] = (confidence, evidence)

        self._stats["nodes_classified"] += 1

        sorted_results = sorted(results.items(), key=lambda x: -x[1][0])
        return [(domain, conf, ev) for domain, (conf, ev) in sorted_results]

    def resolve_query_to_domains(
        self,
        query: str,
    ) -> List[Tuple[str, float]]:
        """
        Given a user query, identify which business domain(s) it relates to.

        This is the entry point for domain-first resolution:
            User query → Domain → Capabilities → Code

        Returns:
            List of (domain_name, relevance_score) sorted by relevance.
        """
        results: Dict[str, float] = {}
        query_lower = query.lower()

        for domain, patterns in self._compiled_patterns.items():
            score = 0.0
            for pat in patterns:
                matches = pat.findall(query_lower)
                score += len(matches) * 0.2
            if score > 0:
                results[domain] = min(1.0, score)

        self._stats["queries_resolved"] += 1

        sorted_results = sorted(results.items(), key=lambda x: -x[1])
        return sorted_results

    def build_domain_graph_from_nodes(
        self,
        nodes: List[Dict[str, Any]],
    ) -> List[Domain]:
        """
        Build domain hierarchy from existing code nodes.

        Scans all nodes, classifies them into domains, and creates
        capabilities from clusters of related nodes.

        Args:
            nodes: List of node dicts with 'id', 'name', 'file_path', etc.

        Returns:
            List of Domain objects with populated capabilities.
        """
        domain_nodes: Dict[str, List[Dict[str, Any]]] = {}

        for node in nodes:
            inferred = self.infer_domain_for_node(
                node_name=node.get("name", ""),
                file_path=node.get("file_path", node.get("relative_path", "")),
            )
            if inferred:
                top_domain, confidence, evidence = inferred[0]
                if confidence >= 0.3:
                    domain_nodes.setdefault(top_domain, []).append({
                        **node,
                        "_domain_confidence": confidence,
                        "_domain_evidence": evidence,
                    })

        domains: List[Domain] = []
        for domain_name, d_nodes in domain_nodes.items():
            # Create capabilities from node clusters
            capabilities: List[Capability] = []
            repos_in_domain: Set[str] = set()

            for n in d_nodes:
                repo = n.get("repository_name", n.get("repository", ""))
                repos_in_domain.add(repo)

            cap_id = f"cap_{domain_name}_primary"
            capabilities.append(Capability(
                id=cap_id,
                name=f"{domain_name.replace('_', ' ').title()} (auto-inferred)",
                description=f"Auto-inferred capability from {len(d_nodes)} code nodes",
                domain_id=f"domain_{domain_name}",
                implementing_repos=list(repos_in_domain),
                implementing_node_ids=[n.get("id", "") for n in d_nodes],
                keywords=[domain_name],
                confidence=sum(n.get("_domain_confidence", 0) for n in d_nodes) / max(len(d_nodes), 1),
            ))

            domain = Domain(
                id=f"domain_{domain_name}",
                name=domain_name.replace("_", " ").title(),
                description=f"Auto-inferred domain covering {len(d_nodes)} code artifacts across {len(repos_in_domain)} repositories",
                capabilities=capabilities,
                keywords=[domain_name] + list(repos_in_domain),
            )
            domains.append(domain)

        return domains


# Singleton
domain_engine = DomainInferenceEngine()
