"""
API Dependencies - FastAPI dependency injection
"""
from typing import Annotated
from fastapi import Depends

from src.database.neo4j_client import neo4j_client, Neo4jClient
from src.database.qdrant_client import qdrant_client, QdrantClient
from src.database.supabase_session import supabase_client, SupabaseClient
from src.services.search_service import SearchService
from src.services.embedding_service import embedding_service, EmbeddingService
from src.agents.orchestrator import agent_orchestrator, AgentOrchestrator


# Database dependencies
async def get_neo4j() -> Neo4jClient:
    """Get Neo4j client dependency"""
    await neo4j_client.initialize()
    return neo4j_client


async def get_qdrant() -> QdrantClient:
    """Get Qdrant client dependency"""
    await qdrant_client.initialize()
    return qdrant_client


async def get_supabase() -> SupabaseClient:
    """Get Supabase client dependency"""
    await supabase_client.initialize()
    return supabase_client


# Service dependencies
async def get_search_service() -> SearchService:
    """Get search service dependency"""
    service = SearchService()
    await service.initialize()
    return service


async def get_embedding_service() -> EmbeddingService:
    """Get embedding service dependency"""
    await embedding_service.initialize()
    return embedding_service


async def get_orchestrator() -> AgentOrchestrator:
    """Get agent orchestrator dependency"""
    await agent_orchestrator.initialize()
    return agent_orchestrator


# Type aliases for cleaner endpoint signatures
Neo4jDep = Annotated[Neo4jClient, Depends(get_neo4j)]
QdrantDep = Annotated[QdrantClient, Depends(get_qdrant)]
SupabaseDep = Annotated[SupabaseClient, Depends(get_supabase)]
SearchServiceDep = Annotated[SearchService, Depends(get_search_service)]
EmbeddingServiceDep = Annotated[EmbeddingService, Depends(get_embedding_service)]
OrchestratorDep = Annotated[AgentOrchestrator, Depends(get_orchestrator)]