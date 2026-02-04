"""
Admin endpoints for system management and monitoring
"""
from typing import Dict, Any, Optional, List
from datetime import datetime
import asyncio

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel

from src.database.neo4j_client import neo4j_client
from src.database.qdrant_client import qdrant_client
from src.database.supabase_session import supabase_client
from src.core.logger import get_logger
from config.settings import settings

logger = get_logger(__name__)
router = APIRouter()


# =============================================================================
# Request/Response Models
# =============================================================================

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    services: Dict[str, Dict[str, Any]]


class SystemStatsResponse(BaseModel):
    repositories: int
    total_nodes: int
    total_relationships: int
    vector_count: int
    collections: List[str]


class GraphConnectivityResponse(BaseModel):
    labels: Dict[str, int]
    relationships: Dict[str, int]
    relationship_types_top: Dict[str, int]
    routes_with_full_path: int
    qdrant_collection: Dict[str, Any]


class ReindexRequest(BaseModel):
    repository: Optional[str] = None
    force: bool = False


class CacheInvalidateRequest(BaseModel):
    cache_type: str = "all"  # all, graph, vector, embedding


# =============================================================================
# Health Check Endpoints
# =============================================================================

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Comprehensive health check for all services.
    """
    timestamp = datetime.utcnow().isoformat()
    services = {}
    overall_status = "healthy"
    
    # Check Neo4j
    try:
        neo4j_healthy = await neo4j_client.health_check()
        services["neo4j"] = {
            "status": "healthy" if neo4j_healthy else "unhealthy",
            "url": settings.NEO4J_URI
        }
        if not neo4j_healthy:
            overall_status = "degraded"
    except Exception as e:
        services["neo4j"] = {
            "status": "unhealthy",
            "error": str(e)
        }
        overall_status = "unhealthy"
    
    # Check Qdrant
    try:
        qdrant_healthy = await qdrant_client.health_check()
        collections = await qdrant_client.list_collections()
        services["qdrant"] = {
            "status": "healthy" if qdrant_healthy else "unhealthy",
            "url": settings.QDRANT_URL,
            "collections": collections
        }
        if not qdrant_healthy:
            overall_status = "degraded"
    except Exception as e:
        services["qdrant"] = {
            "status": "unhealthy",
            "error": str(e)
        }
        overall_status = "unhealthy"
    
    # Check Supabase
    try:
        supabase_healthy = await supabase_client.health_check()
        services["supabase"] = {
            "status": "healthy" if supabase_healthy else "unhealthy",
            "url": settings.SUPABASE_URL
        }
        if not supabase_healthy:
            overall_status = "degraded"
    except Exception as e:
        services["supabase"] = {
            "status": "unhealthy",
            "error": str(e)
        }
        overall_status = "unhealthy"
    
    return HealthResponse(
        status=overall_status,
        timestamp=timestamp,
        services=services
    )

    

@router.get("/live")
async def liveness_check():
    """
    Kubernetes-style liveness check.
    Returns 200 if the service is alive.
    """
    return {"status": "alive", "timestamp": datetime.utcnow().isoformat()}


# =============================================================================
# System Statistics Endpoints
# =============================================================================

@router.get("/stats", response_model=SystemStatsResponse)
async def get_system_stats():
    """
    Get system-wide statistics.
    """
    try:
        # Get Neo4j stats
        graph_stats = await neo4j_client.get_graph_stats()
        
        # Get Qdrant stats
        collections = await qdrant_client.list_collections()
        total_vectors = 0
        for collection in collections:
            info = await qdrant_client.get_collection_info(collection)
            total_vectors += info.get("vectors_count", 0)
        
        return SystemStatsResponse(
            repositories=len(settings.REPOSITORIES),
            total_nodes=graph_stats.get("total_nodes", 0),
            total_relationships=graph_stats.get("total_relationships", 0),
            vector_count=total_vectors,
            collections=collections
        )
    except Exception as e:
        logger.error("Failed to get system stats", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")


@router.get("/graph-connectivity", response_model=GraphConnectivityResponse)
async def get_graph_connectivity_stats():
    """Get connectivity-focused metrics for validating the code graph."""
    label_keys = [
        "Repository",
        "File",
        "VertxRoute",
        "ReactComponent",
        "ApiCall",
        "Function",
    ]
    rel_keys = [
        "CONTAINS",
        "IMPORTS",
        "DECLARES",
        "CALLS",
        "MOUNTS",
        "HANDLED_BY",
        "CALLS_ROUTE",
    ]

    try:
        # Neo4j metrics
        await neo4j_client.initialize()

        labels: Dict[str, int] = {}
        for label in label_keys:
            recs = await neo4j_client.execute_query(
                f"MATCH (n:{label}) RETURN count(n) AS c"
            )
            labels[label] = int(recs[0]["c"]) if recs else 0

        relationships: Dict[str, int] = {}
        for rel in rel_keys:
            recs = await neo4j_client.execute_query(
                f"MATCH ()-[r:{rel}]->() RETURN count(r) AS c"
            )
            relationships[rel] = int(recs[0]["c"]) if recs else 0

        routes_full_recs = await neo4j_client.execute_query(
            "MATCH (r:VertxRoute) WHERE r.full_path IS NOT NULL RETURN count(r) AS c"
        )
        routes_with_full_path = int(routes_full_recs[0]["c"]) if routes_full_recs else 0

        rel_type_recs = await neo4j_client.execute_query(
            """
            MATCH ()-[r]->()
            RETURN type(r) AS t, count(r) AS c
            ORDER BY c DESC
            LIMIT 15
            """
        )
        relationship_types_top = {r["t"]: int(r["c"]) for r in rel_type_recs}

        # Qdrant metrics
        await qdrant_client.initialize()
        coll_name = settings.QDRANT_COLLECTION_NAME
        base_info = await qdrant_client.get_collection_info(coll_name)

        try:
            raw_info = await qdrant_client.client.get_collection(coll_name)
            base_info.update(
                {
                    "points_count": getattr(raw_info, "points_count", None),
                    "vectors_count": getattr(raw_info, "vectors_count", None),
                    "indexed_vectors_count": getattr(raw_info, "indexed_vectors_count", None),
                    "status": str(getattr(raw_info, "status", None)),
                    "vector_size": raw_info.config.params.vectors.size,
                    "distance": str(raw_info.config.params.vectors.distance),
                }
            )
        except Exception as e:
            base_info["raw_info_error"] = str(e)

        return GraphConnectivityResponse(
            labels=labels,
            relationships=relationships,
            relationship_types_top=relationship_types_top,
            routes_with_full_path=routes_with_full_path,
            qdrant_collection=base_info,
        )
    except Exception as e:
        logger.error("Failed to get graph connectivity stats", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get connectivity stats: {str(e)}")


@router.get("/config")
async def get_configuration():
    """
    Get current system configuration (non-sensitive).
    """
    return {
        "environment": settings.ENVIRONMENT,
        "debug": settings.DEBUG,
        "repositories_configured": len(settings.REPOSITORIES),
        "llm_model": settings.LLM_MODEL,
        "embedding_model": settings.EMBEDDING_MODEL,
        "features": {
            "prd_analysis": settings.ENABLE_PRD_ANALYSIS,
            "qa_generation": settings.ENABLE_QA_GENERATION,
            "developer_assistant": settings.ENABLE_DEVELOPER_ASSISTANT,
            "pr_analysis": settings.ENABLE_PR_ANALYSIS,
        }
    }


# =============================================================================
# Cache Management Endpoints
# =============================================================================

@router.post("/cache/invalidate")
async def invalidate_cache(request: CacheInvalidateRequest):
    """
    Invalidate system caches.
    """
    try:
        invalidated = []
        
        if request.cache_type in ["all", "graph"]:
            # Clear Neo4j query cache if supported
            await neo4j_client.execute_query("CALL db.clearQueryCaches()")
            invalidated.append("graph")
        
        if request.cache_type in ["all", "embedding"]:
            # Clear embedding cache
            from src.services.embedding_service import embedding_service
            embedding_service.clear_cache()
            invalidated.append("embedding")
        
        return {
            "message": "Cache invalidated successfully",
            "caches_cleared": invalidated
        }
    except Exception as e:
        logger.error("Failed to invalidate cache", error=str(e))
        raise HTTPException(status_code=500, detail=f"Cache invalidation failed: {str(e)}")


# =============================================================================
# Reindexing Endpoints
# =============================================================================

@router.post("/reindex")
async def trigger_reindex(
    request: ReindexRequest,
    background_tasks: BackgroundTasks
):
    """
    Trigger a reindexing operation for the knowledge graph.
    """
    from src.services.data_ingestion_service import DataIngestionService
    
    job_id = f"reindex_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
    
    async def run_reindex():
        try:
            service = DataIngestionService()
            
            if request.repository:
                # Reindex specific repository
                repos = [r for r in settings.REPOSITORIES if r.name == request.repository]
                if not repos:
                    logger.error(f"Repository not found: {request.repository}")
                    return
            else:
                repos = settings.REPOSITORIES
            
            # Run full pipeline
            await service.parse_repositories([r.dict() for r in repos])
            await service.build_knowledge_graph()
            await service.generate_embeddings()
            
            logger.info(f"Reindex job {job_id} completed successfully")
        except Exception as e:
            logger.error(f"Reindex job {job_id} failed", error=str(e))
    
    background_tasks.add_task(run_reindex)
    
    return {
        "job_id": job_id,
        "message": "Reindex operation started",
        "repository": request.repository or "all"
    }


# =============================================================================
# Audit Endpoints
# =============================================================================

@router.get("/audit/recent")
async def get_recent_audit_logs(limit: int = 100):
    """
    Get recent audit logs from Supabase.
    """
    try:
        logs = await supabase_client.get_audit_logs(limit=limit)
        return {
            "logs": logs,
            "count": len(logs)
        }
    except Exception as e:
        logger.error("Failed to fetch audit logs", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to fetch audit logs: {str(e)}")


@router.get("/metrics")
async def get_metrics():
    """
    Get Prometheus-style metrics.
    """
    # Basic metrics for monitoring
    metrics = []
    
    try:
        graph_stats = await neo4j_client.get_graph_stats()
        metrics.append(f'code_intel_graph_nodes_total {graph_stats.get("total_nodes", 0)}')
        metrics.append(f'code_intel_graph_relationships_total {graph_stats.get("total_relationships", 0)}')
    except:
        pass
    
    try:
        collections = await qdrant_client.list_collections()
        for collection in collections:
            info = await qdrant_client.get_collection_info(collection)
            metrics.append(f'code_intel_vector_count{{collection="{collection}"}} {info.get("vectors_count", 0)}')
    except:
        pass
    
    metrics.append(f'code_intel_repositories_total {len(settings.REPOSITORIES)}')
    
    return "\n".join(metrics)