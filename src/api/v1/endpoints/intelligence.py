"""
Intelligence Layer API Endpoints

Exposes the behavioral, temporal, domain, and confidence intelligence
to API consumers for advanced querying and analytics.
"""
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.core.logger import get_logger
from src.intelligence.behavioral_layer import behavioral_classifier
from src.intelligence.temporal_metadata import temporal_analyzer, classify_temporal_risk
from src.intelligence.domain_model import domain_engine
from src.intelligence.evidence_links import evidence_link_factory
from src.intelligence.qa_projection import qa_projection_engine
from src.database.neo4j_client import neo4j_client

logger = get_logger(__name__)
router = APIRouter()


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class DomainQueryRequest(BaseModel):
    """Resolve a user query to business domains."""
    query: str = Field(..., min_length=1, max_length=5000)


class BehavioralSummaryRequest(BaseModel):
    """Get behavioral classification summary for a repository or file."""
    repository_name: Optional[str] = None
    file_path: Optional[str] = None


class VolatilityRequest(BaseModel):
    """Get volatility hotspots."""
    repository_name: Optional[str] = None
    min_volatility: float = Field(default=0.3, ge=0.0, le=1.0)
    limit: int = Field(default=20, ge=1, le=100)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/domains/resolve")
async def resolve_domains(request: DomainQueryRequest):
    """
    Resolve a user query to business domains.
    
    This is the entry point for domain-first reasoning:
        Query → Domain identification → Capability matching → Repo/Code resolution
    """
    try:
        domains = domain_engine.resolve_query_to_domains(request.query)
        return {
            "query": request.query,
            "domains": [
                {"name": name, "relevance": round(score, 3)}
                for name, score in domains
            ],
            "primary_domain": domains[0][0] if domains else None,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/behavioral/summary")
async def get_behavioral_summary(repository_name: Optional[str] = None):
    """
    Get behavioral classification summary across the knowledge graph.
    
    Returns distribution of behavioral labels and high-risk nodes.
    """
    try:
        query = """
        MATCH (n)
        WHERE n.behavioral_labels IS NOT NULL
        """
        if repository_name:
            query += " AND n.repository_name = $repo"
        query += """
        RETURN n.behavioral_labels AS labels, 
               labels(n)[0] AS node_type,
               n.name AS name,
               n.id AS id,
               n.behavioral_confidence AS confidence
        LIMIT 500
        """

        params = {"repo": repository_name} if repository_name else {}
        results = await neo4j_client.execute_query(query, params)

        label_counts: Dict[str, int] = {}
        high_risk_nodes: List[Dict[str, Any]] = []

        for record in (results or []):
            labels = record.get("labels", [])
            if isinstance(labels, list):
                for label in labels:
                    label_counts[label] = label_counts.get(label, 0) + 1
                if "AuthCritical" in labels or "TransactionBoundary" in labels:
                    high_risk_nodes.append({
                        "id": record.get("id"),
                        "name": record.get("name"),
                        "type": record.get("node_type"),
                        "labels": labels,
                        "confidence": record.get("confidence"),
                    })

        return {
            "total_classified": sum(label_counts.values()),
            "label_distribution": label_counts,
            "high_risk_nodes": high_risk_nodes[:20],
            "classifier_stats": behavioral_classifier.stats,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/temporal/hotspots")
async def get_volatility_hotspots(
    repository_name: Optional[str] = None,
    min_volatility: float = 0.3,
    limit: int = 20,
):
    """
    Get the most volatile (frequently-changing) code nodes.
    
    High-volatility nodes represent active regression risk.
    """
    try:
        query = """
        MATCH (n)
        WHERE n.volatility_score IS NOT NULL
          AND n.volatility_score >= $min_volatility
        """
        if repository_name:
            query += " AND n.repository_name = $repo"
        query += """
        RETURN n.id AS id,
               n.name AS name,
               labels(n)[0] AS node_type,
               n.file_name AS file_name,
               n.relative_path AS path,
               n.volatility_score AS volatility_score,
               n.change_count AS change_count,
               n.behavioral_labels AS behavioral_labels,
               n.domain AS domain
        ORDER BY n.volatility_score DESC
        LIMIT $limit
        """

        params = {"min_volatility": min_volatility, "limit": limit}
        if repository_name:
            params["repo"] = repository_name

        results = await neo4j_client.execute_query(query, params)

        hotspots = []
        for record in (results or []):
            volatility = record.get("volatility_score", 0)
            days_ago = 0  # Would need last_modified_at for precise calculation
            risk_level, explanation = classify_temporal_risk(volatility, days_ago)

            hotspots.append({
                "id": record.get("id"),
                "name": record.get("name"),
                "type": record.get("node_type"),
                "path": record.get("path"),
                "volatility_score": volatility,
                "change_count": record.get("change_count"),
                "risk_level": risk_level,
                "risk_explanation": explanation,
                "behavioral_labels": record.get("behavioral_labels", []),
                "domain": record.get("domain"),
            })

        return {
            "total_hotspots": len(hotspots),
            "min_volatility_threshold": min_volatility,
            "hotspots": hotspots,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/evidence/links")
async def get_evidence_weighted_links(
    min_confidence: float = 0.0,
    link_type: Optional[str] = None,
    limit: int = 50,
):
    """
    Query evidence-weighted links in the knowledge graph.
    
    Shows confidence scores and evidence types for cross-language links.
    """
    try:
        query = """
        MATCH (a)-[r]->(b)
        WHERE r.confidence_score IS NOT NULL
          AND r.confidence_score >= $min_confidence
        """
        if link_type:
            query = f"""
            MATCH (a)-[r:{link_type}]->(b)
            WHERE r.confidence_score IS NOT NULL
              AND r.confidence_score >= $min_confidence
            """
        query += """
        RETURN a.id AS source_id,
               a.name AS source_name,
               type(r) AS link_type,
               r.confidence_score AS confidence,
               r.evidence_type AS evidence_type,
               r.link_origin AS link_origin,
               b.id AS target_id,
               b.name AS target_name
        ORDER BY r.confidence_score DESC
        LIMIT $limit
        """

        params = {"min_confidence": min_confidence, "limit": limit}
        results = await neo4j_client.execute_query(query, params)

        links = []
        for record in (results or []):
            links.append({
                "source": {"id": record.get("source_id"), "name": record.get("source_name")},
                "target": {"id": record.get("target_id"), "name": record.get("target_name")},
                "link_type": record.get("link_type"),
                "confidence": record.get("confidence"),
                "evidence_type": record.get("evidence_type"),
                "link_origin": record.get("link_origin"),
            })

        return {
            "total_links": len(links),
            "min_confidence_filter": min_confidence,
            "links": links,
            "factory_stats": evidence_link_factory.stats,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_intelligence_stats():
    """Get statistics from all intelligence layer components."""
    return {
        "behavioral_classifier": behavioral_classifier.stats,
        "temporal_analyzer": temporal_analyzer.stats,
        "domain_engine": domain_engine.stats,
        "evidence_link_factory": evidence_link_factory.stats,
        "qa_projection_engine": qa_projection_engine.stats,
    }
