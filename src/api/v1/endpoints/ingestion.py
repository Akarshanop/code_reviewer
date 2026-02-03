"""
Enterprise Data Ingestion Endpoint
Manages repository cloning, parsing, and knowledge graph construction
"""
import uuid
from typing import Dict, Any, List, Optional
from datetime import datetime
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

from src.core.logger import get_logger
from src.services.data_ingestion_service import DataIngestionService
from src.data_ingestion.git_service import git_service
from config.settings import settings

logger = get_logger(__name__)
router = APIRouter()

# In-memory job storage for ingestion tasks
_ingestion_jobs: Dict[str, Dict[str, Any]] = {}


class IngestionRequest(BaseModel):
    """Request for triggering ingestion"""
    repositories: Optional[List[str]] = Field(
        default=None,
        description="Specific repositories to ingest. If None, ingest all configured repos"
    )
    force_refresh: Optional[bool] = Field(
        default=False,
        description="Force re-ingestion even if repo hasn't changed"
    )
    parse_only: Optional[List[str]] = Field(
        default=None,
        description="Only parse specific file types: java, typescript, python"
    )
    skip_embeddings: Optional[bool] = Field(
        default=False,
        description="Skip embedding generation (faster for testing)"
    )


class IngestionStatus(BaseModel):
    """Status of ingestion job"""
    job_id: str
    status: str 
    progress: float
    current_step: str
    repositories: List[str]
    stats: Dict[str, Any]
    errors: List[str]
    created_at: datetime
    updated_at: datetime


class RepositoryStatus(BaseModel):
    """Status of a single repository"""
    name: str
    url: str
    last_commit: Optional[str]
    last_ingestion: Optional[datetime]
    file_count: int
    node_count: int
    status: str


@router.post("/trigger")
async def trigger_ingestion(
    request: IngestionRequest = None,
    background_tasks: BackgroundTasks = None
):
    """
    Trigger repository ingestion pipeline.
    
    Pipeline steps:
    1. Clone/pull repositories
    2. Parse source files (Java, TypeScript)
    3. Build knowledge graph in Neo4j
    4. Generate embeddings in Qdrant
    
    Returns job_id for tracking progress.
    """
    try:
        request = request or IngestionRequest()
        
        job_id = str(uuid.uuid4())
        
        # Determine repositories to process
        if request.repositories:
            repos = request.repositories
        else:
            repos = [r.name for r in settings.get_repositories()]
        
        logger.info(
            "Starting ingestion job",
            job_id=job_id,
            repositories=repos,
            force_refresh=request.force_refresh
        )
        
        # Initialize job status
        _ingestion_jobs[job_id] = {
            "status": "pending",
            "progress": 0.0,
            "current_step": "initializing",
            "repositories": repos,
            "stats": {
                "files_processed": 0,
                "nodes_created": 0,
                "edges_created": 0,
                "embeddings_generated": 0
            },
            "errors": [],
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        
        # Start background task
        background_tasks.add_task(
            _run_ingestion_pipeline,
            job_id,
            repos,
            request.force_refresh,
            request.parse_only,
            request.skip_embeddings
        )
        
        return {
            "status": "accepted",
            "job_id": job_id,
            "message": f"Ingestion started for {len(repos)} repositories",
            "status_url": f"/api/v1/ingestion/status/{job_id}"
        }
        
    except Exception as e:
        logger.error(f"Ingestion trigger failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@router.get("/status/{job_id}", response_model=IngestionStatus)
async def get_ingestion_status(job_id: str):
    """Get status of ingestion job"""
    if job_id not in _ingestion_jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    job = _ingestion_jobs[job_id]
    
    return IngestionStatus(
        job_id=job_id,
        status=job["status"],
        progress=job["progress"],
        current_step=job["current_step"],
        repositories=job["repositories"],
        stats=job["stats"],
        errors=job["errors"],
        created_at=job["created_at"],
        updated_at=job["updated_at"]
    )


@router.get("/repositories")
async def list_repositories():
    """List all configured repositories with their status"""
    try:
        repos = settings.get_repositories()
        repo_statuses = []
        
        for repo in repos:
            # Get repo stats from git service
            repo_state = git_service.get_repository_state(repo.name)
            
            status = RepositoryStatus(
                name=repo.name,
                url=repo.url,
                last_commit=repo_state.get("last_commit") if repo_state else None,
                last_ingestion=repo_state.get("last_ingestion") if repo_state else None,
                file_count=repo_state.get("file_count", 0) if repo_state else 0,
                node_count=repo_state.get("node_count", 0) if repo_state else 0,
                status=repo_state.get("status", "not_cloned") if repo_state else "not_cloned"
            )
            repo_statuses.append(status)
        
        return {
            "status": "success",
            "repositories": [r.dict() for r in repo_statuses],
            "total": len(repo_statuses)
        }
        
    except Exception as e:
        logger.error(f"Failed to list repositories: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/repositories/{repo_name}/refresh")
async def refresh_repository(repo_name: str, background_tasks: BackgroundTasks):
    """Force refresh a single repository"""
    try:
        repos = settings.get_repositories()
        repo = next((r for r in repos if r.name == repo_name), None)
        
        if not repo:
            raise HTTPException(status_code=404, detail=f"Repository {repo_name} not found")
        
        job_id = str(uuid.uuid4())
        
        _ingestion_jobs[job_id] = {
            "status": "pending",
            "progress": 0.0,
            "current_step": "initializing",
            "repositories": [repo_name],
            "stats": {},
            "errors": [],
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        
        background_tasks.add_task(
            _run_ingestion_pipeline,
            job_id,
            [repo_name],
            True,  # force refresh
            None,
            False
        )
        
        return {
            "status": "accepted",
            "job_id": job_id,
            "message": f"Refresh started for {repo_name}",
            "status_url": f"/api/v1/ingestion/status/{job_id}"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Repository refresh failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_ingestion_stats():
    """Get overall ingestion statistics"""
    try:
        from src.database.neo4j_client import neo4j_client
        from src.database.qdrant_client import qdrant_client
        
        # Get Neo4j stats
        neo4j_stats = {}
        try:
            async with neo4j_client.get_session() as session:
                result = await session.run(
                    """
                    MATCH (n)
                    RETURN labels(n)[0] as label, count(n) as count
                    """
                )
                records = await result.data()
                neo4j_stats["nodes_by_type"] = {r["label"]: r["count"] for r in records}
                neo4j_stats["total_nodes"] = sum(r["count"] for r in records)
        except Exception as e:
            logger.warning(f"Failed to get Neo4j stats: {e}")
            neo4j_stats = {"error": str(e)}
        
        # Get Qdrant stats
        qdrant_stats = {}
        try:
            collections = await qdrant_client.list_collections()
            qdrant_stats["collections"] = collections
            
            total_vectors = 0
            for collection in collections:
                info = await qdrant_client.get_collection_info(collection)
                total_vectors += info.get("vectors_count", 0)
            qdrant_stats["total_vectors"] = total_vectors
        except Exception as e:
            logger.warning(f"Failed to get Qdrant stats: {e}")
            qdrant_stats = {"error": str(e)}
        
        # Get job stats
        completed_jobs = len([j for j in _ingestion_jobs.values() if j["status"] == "completed"])
        failed_jobs = len([j for j in _ingestion_jobs.values() if j["status"] == "failed"])
        running_jobs = len([j for j in _ingestion_jobs.values() if j["status"] == "running"])
        
        return {
            "status": "success",
            "neo4j": neo4j_stats,
            "qdrant": qdrant_stats,
            "jobs": {
                "total": len(_ingestion_jobs),
                "completed": completed_jobs,
                "failed": failed_jobs,
                "running": running_jobs
            },
            "repositories_configured": len(settings.get_repositories())
        }
        
    except Exception as e:
        logger.error(f"Failed to get ingestion stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/cache")
async def clear_cache():
    """Clear all cached data and reset ingestion state"""
    try:
        # Clear job history (keep last 10)
        if len(_ingestion_jobs) > 10:
            sorted_jobs = sorted(
                _ingestion_jobs.items(),
                key=lambda x: x[1]["created_at"],
                reverse=True
            )
            _ingestion_jobs.clear()
            for job_id, job in sorted_jobs[:10]:
                _ingestion_jobs[job_id] = job
        
        return {
            "status": "success",
            "message": "Cache cleared successfully",
            "jobs_retained": len(_ingestion_jobs)
        }
        
    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Background task implementation

async def _run_ingestion_pipeline(
    job_id: str,
    repositories: List[str],
    force_refresh: bool,
    parse_only: Optional[List[str]],
    skip_embeddings: bool
):
    """
    Run the full ingestion pipeline in background.
    
    Steps:
    1. Clone/update repositories
    2. Parse source files
    3. Build knowledge graph
    4. Generate embeddings
    """
    try:
        job = _ingestion_jobs[job_id]
        ingestion_service = DataIngestionService()
        
        # Step 1: Clone repositories
        job["status"] = "cloning"
        job["current_step"] = "Cloning repositories"
        job["progress"] = 0.1
        job["updated_at"] = datetime.utcnow()
        
        clone_results = await git_service.clone_all_repositories(
            repository_filter=repositories,
            force_refresh=force_refresh
        )
        
        if not clone_results.get("successful"):
            job["errors"].append("No repositories cloned successfully")
            if not clone_results.get("partial"):
                job["status"] = "failed"
                job["updated_at"] = datetime.utcnow()
                return
        
        # Step 2: Parse source files
        job["status"] = "parsing"
        job["current_step"] = "Parsing source files"
        job["progress"] = 0.3
        job["updated_at"] = datetime.utcnow()
        
        parse_results = await ingestion_service.parse_repositories(
            repositories=clone_results.get("repositories", []),
            file_types=parse_only
        )
        
        job["stats"]["files_processed"] = parse_results.get("files_processed", 0)
        
        # Step 3: Build knowledge graph
        job["status"] = "building_graph"
        job["current_step"] = "Building knowledge graph"
        job["progress"] = 0.5
        job["updated_at"] = datetime.utcnow()
        
        graph_results = await ingestion_service.build_knowledge_graph(
            parsed_data=parse_results.get("parsed_data", {})
        )
        
        job["stats"]["nodes_created"] = graph_results.get("nodes_created", 0)
        job["stats"]["edges_created"] = graph_results.get("edges_created", 0)
        
        # Step 4: Generate embeddings
        if not skip_embeddings:
            job["status"] = "embedding"
            job["current_step"] = "Generating embeddings"
            job["progress"] = 0.8
            job["updated_at"] = datetime.utcnow()
            
            embedding_results = await ingestion_service.generate_embeddings(
                nodes=graph_results.get("nodes", [])
            )
            
            job["stats"]["embeddings_generated"] = embedding_results.get("embeddings_generated", 0)
        
        # Complete
        job["status"] = "completed"
        job["current_step"] = "Completed"
        job["progress"] = 1.0
        job["updated_at"] = datetime.utcnow()
        
        logger.info(
            "Ingestion pipeline completed",
            job_id=job_id,
            stats=job["stats"]
        )
        
    except Exception as e:
        logger.error(f"Ingestion pipeline failed: {e}", exc_info=True)
        _ingestion_jobs[job_id]["status"] = "failed"
        _ingestion_jobs[job_id]["errors"].append(str(e))
        _ingestion_jobs[job_id]["updated_at"] = datetime.utcnow()