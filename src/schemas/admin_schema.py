from pydantic import BaseModel
from typing import Dict, Any, List, Optional


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


class DatabaseResetRequest(BaseModel):
    """Request body for the database reset endpoint."""
    targets: List[str] = ["neo4j", "qdrant"]  # which databases to wipe
    confirm: bool = False  # safety flag — must be True to proceed


class DatabaseResetResponse(BaseModel):
    """Response for the database reset endpoint."""
    status: str
    neo4j: Optional[Dict[str, Any]] = None
    qdrant: Optional[Dict[str, Any]] = None
    errors: List[str] = []