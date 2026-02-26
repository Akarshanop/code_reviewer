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
from src.schemas.admin_schema import (
    HealthResponse, SystemStatsResponse, GraphConnectivityResponse,
    ReindexRequest, CacheInvalidateRequest,
    DatabaseResetRequest, DatabaseResetResponse,
)

logger = get_logger(__name__)
router = APIRouter()



# Health Check Endpoints

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
    
    # Check Qdrant (initialize first so we actually try to connect)
    try:
        try:
            await qdrant_client.initialize()
        except Exception as init_err:
            logger.warning(f"Qdrant initialize during health check: {init_err}")
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
            "url": settings.QDRANT_URL,
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
            total_vectors += info.get("vectors_count") or info.get("points_count") or 0
        
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
            
            # Run full pipeline: parse → build graph → generate embeddings
            repo_dicts = [r.model_dump() if hasattr(r, 'model_dump') else r.dict() for r in repos]
            parse_result = await service.parse_repositories(repo_dicts)
            parsed_data = parse_result.get("parsed_data", {})
            
            if parsed_data:
                graph_result = await service.build_knowledge_graph(parsed_data)
                
                # Collect nodes from graph result for embedding
                nodes = graph_result.get("nodes", []) if isinstance(graph_result, dict) else []
                if nodes:
                    await service.generate_embeddings(nodes)
            else:
                logger.warning(f"Reindex job {job_id}: no parsed data to build graph from")
            
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
# Database Reset Endpoint
# =============================================================================

@router.post("/database/reset", response_model=DatabaseResetResponse)
async def reset_databases(request: DatabaseResetRequest):
    """
    Wipe all data from Neo4j and/or Qdrant so you can run a fresh ingestion.

    **WARNING** — this is destructive and irreversible.
    You must send `"confirm": true` in the request body.

    Example body:
        { "targets": ["neo4j", "qdrant"], "confirm": true }
    """
    if not request.confirm:
        raise HTTPException(
            status_code=400,
            detail='Safety check: set "confirm": true in the request body to proceed.',
        )

    errors: list[str] = []
    neo4j_result: Dict[str, Any] | None = None
    qdrant_result: Dict[str, Any] | None = None

    # ------------------------------------------------------------------
    # 1. Neo4j — delete all nodes and relationships
    # ------------------------------------------------------------------
    if "neo4j" in request.targets:
        try:
            await neo4j_client.initialize()

            # Grab counts before deletion for the response
            before = await neo4j_client.get_graph_stats()

            # DETACH DELETE removes every node and its relationships.
            # We do it in batches to avoid OOM on very large graphs.
            deleted_total = 0
            batch_size = 5000
            while True:
                result = await neo4j_client.execute_query(
                    f"MATCH (n) WITH n LIMIT {batch_size} DETACH DELETE n RETURN count(*) AS deleted"
                )
                deleted = result[0]["deleted"] if result else 0
                deleted_total += deleted
                if deleted < batch_size:
                    break

            # Re-create constraints & indexes so the DB is ready for re-ingestion
            await neo4j_client.ensure_constraints()

            neo4j_result = {
                "deleted_nodes": before.get("total_nodes", 0),
                "deleted_relationships": before.get("total_relationships", 0),
                "batches_executed": (deleted_total // batch_size) + 1,
                "constraints_recreated": True,
            }
            logger.info(
                "Neo4j database reset complete",
                deleted_nodes=neo4j_result["deleted_nodes"],
                deleted_relationships=neo4j_result["deleted_relationships"],
            )
        except Exception as e:
            msg = f"Neo4j reset failed: {e}"
            logger.error(msg, exc_info=True)
            errors.append(msg)

    # ------------------------------------------------------------------
    # 2. Qdrant — drop and recreate the collection
    # ------------------------------------------------------------------
    if "qdrant" in request.targets:
        try:
            await qdrant_client.initialize()

            collection_name = settings.QDRANT_COLLECTION_NAME

            # Get stats before deletion
            before_info = await qdrant_client.get_collection_info(collection_name)
            vectors_before = before_info.get("vectors_count", 0)

            # Drop the collection entirely
            await qdrant_client.client.delete_collection(collection_name)
            logger.info(f"Qdrant collection '{collection_name}' deleted")

            # Recreate with the same schema so ingestion can write immediately
            qdrant_client._initialized = False  # force re-init
            await qdrant_client.initialize()

            qdrant_result = {
                "collection": collection_name,
                "vectors_deleted": vectors_before,
                "collection_recreated": True,
            }
            logger.info(
                "Qdrant database reset complete",
                collection=collection_name,
                vectors_deleted=vectors_before,
            )
        except Exception as e:
            msg = f"Qdrant reset failed: {e}"
            logger.error(msg, exc_info=True)
            errors.append(msg)

    status = "success" if not errors else ("partial" if (neo4j_result or qdrant_result) else "failed")

    return DatabaseResetResponse(
        status=status,
        neo4j=neo4j_result,
        qdrant=qdrant_result,
        errors=errors,
    )


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
    except Exception as e:
        logger.debug("Neo4j stats unavailable for metrics", error=str(e))
    
    try:
        collections = await qdrant_client.list_collections()
        for collection in collections:
            info = await qdrant_client.get_collection_info(collection)
            metrics.append(f'code_intel_vector_count{{collection="{collection}"}} {info.get("vectors_count", 0)}')
    except Exception as e:
        logger.debug("Qdrant stats unavailable for metrics", error=str(e))
    
    metrics.append(f'code_intel_repositories_total {len(settings.REPOSITORIES)}')
    
    return "\n".join(metrics)