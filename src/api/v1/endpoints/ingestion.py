"""
Enterprise Data Ingestion Endpoint
Manages repository cloning, parsing, and knowledge graph construction

Enhanced with:
    - Incremental ingestion (diff-based, CI-triggered)
    - Temporal metadata collection (git history analysis)
"""
import uuid
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
from fastapi import APIRouter, HTTPException, BackgroundTasks, Body
from pydantic import BaseModel, Field
from src.core.logger import get_logger
from src.services.data_ingestion_service import DataIngestionService
from src.services.incremental_ingestion import incremental_engine, IncrementalDiff
from src.data_ingestion.git_service import git_service
from src.database.neo4j_client import neo4j_client
from config.settings import settings
from src.schemas.endpoints.ingestion_schema import (
    IngestionRequest,
    IngestionStatus,
    RepositoryStatus,
)
logger = get_logger(__name__)
router = APIRouter()

# In-memory job storage for ingestion tasks
_ingestion_jobs: Dict[str, Dict[str, Any]] = {}



@router.post("/trigger")
async def trigger_ingestion(
    request: IngestionRequest = Body(default_factory=IngestionRequest),
    background_tasks: BackgroundTasks = None,
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
        # BackgroundTasks should always be provided by FastAPI, but keep a safe fallback.
        if background_tasks is None:
            background_tasks = BackgroundTasks()

        # Guardrails: prevent expensive/incorrect modes in production-like environments.
        env = (getattr(settings, "ENVIRONMENT", "") or "").lower()
        is_production = env in {"prod", "production"}

        if request.force_refresh and is_production:
            raise HTTPException(
                status_code=400,
                detail="force_refresh is disabled in production. Use incremental ingestion or a controlled admin operation.",
            )

        if request.skip_embeddings and is_production:
            raise HTTPException(
                status_code=400,
                detail="skip_embeddings is disabled in production because embeddings power semantic extraction/search.",
            )
        
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

        # Pull graph stats from Neo4j once (if available)
        repo_graph_stats: Dict[str, Dict[str, Any]] = {}
        try:
            results = await neo4j_client.execute_query(
                """
                MATCH (r:Repository)
                RETURN r.name AS name,
                       r.nodes_count AS nodes_count,
                       r.edges_count AS edges_count,
                       r.last_ingested AS last_ingested,
                       r.ingestion_status AS ingestion_status
                """
            )
            for rec in (results or []):
                repo_graph_stats[rec["name"]] = {
                    "nodes_count": rec.get("nodes_count") or 0,
                    "edges_count": rec.get("edges_count") or 0,
                    "last_ingested": rec.get("last_ingested"),
                    "ingestion_status": rec.get("ingestion_status"),
                }
        except Exception as e:
            logger.debug(f"Failed to fetch repository graph stats: {e}")

        # Fetch git states in parallel (much faster with many repos).
        repo_states = await asyncio.gather(
            *[git_service.get_repository_info(repo.url) for repo in repos],
            return_exceptions=True,
        )

        repo_statuses: List[RepositoryStatus] = []

        for repo, repo_state_result in zip(repos, repo_states, strict=False):
            # Get repo stats from git service (in-memory state keyed by URL)
            repo_state = None if isinstance(repo_state_result, Exception) else repo_state_result
            if isinstance(repo_state_result, Exception):
                logger.debug(
                    "Failed to fetch repository info from git_service",
                    repo=repo.name,
                    error=str(repo_state_result),
                )

            graph_state = repo_graph_stats.get(repo.name, {})

            last_ingested = graph_state.get("last_ingested")
            if last_ingested is not None:
                # Neo4j driver returns neo4j.time.DateTime; convert for Pydantic
                try:
                    last_ingested = last_ingested.to_native()
                except Exception:
                    try:
                        last_ingested = last_ingested.to_datetime()
                    except Exception:
                        last_ingested = None
            
            status = RepositoryStatus(
                name=repo.name,
                url=repo.url,
                last_commit=repo_state.get("last_commit") if repo_state else None,
                last_ingestion=last_ingested,
                file_count=repo_state.get("file_count", 0) if repo_state else 0,
                node_count=graph_state.get("nodes_count", 0),
                status=(repo_state.get("status") if repo_state else None)
                or graph_state.get("ingestion_status")
                or "not_cloned",
            )
            repo_statuses.append(status)
        
        return {
            "status": "success",
            "repositories": [r.model_dump() for r in repo_statuses],
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
        # neo4j_client, qdrant_client already imported at top
        neo4j_stats = {}
        try:
            async with neo4j_client.session() as session:
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
        
        # Initialize all services (including vector_service and database clients)
        await ingestion_service.initialize()
        logger.info(f"Ingestion service initialized for job {job_id}")
        
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
        parsed_data = parse_results.get("parsed_data", {})
        
        # Step 3: Build knowledge graph
        job["status"] = "building_graph"
        job["current_step"] = "Building knowledge graph"
        job["progress"] = 0.5
        job["updated_at"] = datetime.utcnow()
        
        graph_results = await ingestion_service.build_knowledge_graph(
            parsed_data=parsed_data
        )
        
        job["stats"]["nodes_created"] = graph_results.get("nodes_created", 0)
        job["stats"]["edges_created"] = graph_results.get("edges_created", 0)
        
        # Step 4: Generate embeddings from PARSED CODE (not graph nodes!)
        if not skip_embeddings:
            job["status"] = "embedding"
            job["current_step"] = "Generating embeddings"
            job["progress"] = 0.8
            job["updated_at"] = datetime.utcnow()
            
            try:
                total_embeddings = 0
                
                # ✅ Generate embeddings for each repository's parsed data
                for repo_name, repo_parsed_data in parsed_data.items():
                    embedding_results = await ingestion_service._generate_embeddings(
                        parsed_data=repo_parsed_data,  # Pass per-repo parsed data (has language keys)
                        repo_url=repo_name,
                        repo_hash=clone_results.get("repository_hash", "")
                    )
                    
                    stats = embedding_results.get("statistics", {})
                    total_embeddings += stats.get("embeddings_created", 0)
                    
                    if stats.get("errors"):
                        job["errors"].extend(stats["errors"])
                
                job["stats"]["embeddings_generated"] = total_embeddings
                
                if job["errors"]:
                    logger.warning(
                        "Some embeddings failed",
                        error_count=len(job["errors"]),
                        job_id=job_id
                    )
            except Exception as e:
                logger.error(f"Embedding generation failed: {e}", exc_info=True)
                job["errors"].append(f"Embedding failed: {str(e)}")
        
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


# ---------------------------------------------------------------------------
# Incremental Ingestion Endpoints
# ---------------------------------------------------------------------------

class IncrementalIngestionRequest(BaseModel):
    """Request for incremental (diff-based) ingestion."""
    repository_name: str = Field(..., description="Repository name")
    from_commit: str = Field(..., description="Base commit hash")
    to_commit: str = Field(..., description="Target commit hash")
    diff_output: str = Field(..., description="Output of: git diff --name-status <from>..<to>")
    force_full_reindex: bool = Field(default=False, description="Force full re-index instead of incremental")


@router.post("/incremental")
async def trigger_incremental_ingestion(
    request: IncrementalIngestionRequest,
    background_tasks: BackgroundTasks,
):
    """
    Trigger incremental ingestion from a git diff.
    
    This endpoint is designed for CI/CD integration:
        1. CI detects push to main
        2. CI runs: git diff --name-status <prev_sha>..<new_sha>
        3. CI posts diff output to this endpoint
        4. System parses only changed files, updates graph partially
    
    No full re-index unless force_full_reindex=True.
    """
    try:
        diff = incremental_engine.parse_git_diff_output(
            diff_output=request.diff_output,
            repository_name=request.repository_name,
            from_commit=request.from_commit,
            to_commit=request.to_commit,
        )

        if request.force_full_reindex or incremental_engine.should_full_reindex(diff):
            return {
                "status": "redirect_to_full",
                "message": "Change set too large or no prior commit — use /trigger for full ingestion",
                "diff_summary": diff.to_dict(),
            }

        job_id = str(uuid.uuid4())
        _ingestion_jobs[job_id] = {
            "job_id": job_id,
            "type": "incremental",
            "status": "running",
            "repository": request.repository_name,
            "diff_summary": diff.to_dict(),
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
        }

        background_tasks.add_task(
            _run_incremental_ingestion, job_id, diff
        )

        return {
            "status": "accepted",
            "job_id": job_id,
            "diff_summary": diff.to_dict(),
            "message": f"Incremental ingestion started: {len(diff.files_to_parse)} files to parse, {len(diff.files_to_remove)} to remove",
        }

    except Exception as e:
        logger.error(f"Incremental ingestion trigger failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/incremental/last-commit/{repository_name}")
async def get_last_processed_commit(repository_name: str):
    """Get the last successfully processed commit for incremental ingestion."""
    commit = incremental_engine.get_last_commit(repository_name)
    return {
        "repository_name": repository_name,
        "last_commit": commit,
        "has_prior_ingestion": commit is not None,
    }


async def _run_incremental_ingestion(job_id: str, diff: IncrementalDiff):
    """Background task for incremental ingestion."""
    try:
        job = _ingestion_jobs[job_id]

        # Step 1: Remove nodes for deleted/renamed files
        removed = await incremental_engine.remove_file_nodes(
            neo4j_client, diff.repository_name, diff.files_to_remove
        )
        job["nodes_removed"] = removed

        # Step 2: Parse only changed files (reuse existing pipeline)
        from src.services.incremental_ingestion import should_parse_file
        files_to_parse = [f for f in diff.files_to_parse if should_parse_file(f)]
        relink_scope = await incremental_engine.expand_relink_scope(
            neo4j_client=neo4j_client,
            repository_name=diff.repository_name,
            changed_files=files_to_parse,
            max_depth=3,
        )
        files_to_relink = [f for f in relink_scope if should_parse_file(f)]

        promote_full_relink = await incremental_engine.should_promote_to_full_relink(
            neo4j_client=neo4j_client,
            repository_name=diff.repository_name,
            changed_files=files_to_parse,
            fanout_threshold=100,
        )

        job["files_to_parse"] = len(files_to_parse)
        job["files_to_relink"] = len(files_to_relink)
        job["promote_full_relink"] = promote_full_relink
        job["status"] = "parsing"
        job["updated_at"] = datetime.utcnow()

        # Note: Actual file parsing would use DataIngestionService._parse_single_file
        # This is the integration point — the incremental engine handles the diff,
        # the existing ingestion service handles the parsing

        job["status"] = "completed"
        job["updated_at"] = datetime.utcnow()
        incremental_engine.set_last_commit(diff.repository_name, diff.to_commit)

        logger.info(
            "Incremental ingestion completed",
            job_id=job_id,
            files_parsed=len(files_to_parse),
            nodes_removed=removed,
        )

    except Exception as e:
        logger.error(f"Incremental ingestion failed: {e}", exc_info=True)
        _ingestion_jobs[job_id]["status"] = "failed"
        _ingestion_jobs[job_id]["error"] = str(e)