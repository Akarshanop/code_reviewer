from typing import Dict, Any, List, Optional
from datetime import datetime

from pydantic import BaseModel, Field


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
