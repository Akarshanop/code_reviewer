import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from qdrant_client.models import Filter, FieldCondition, MatchValue

from src.database.qdrant_client import qdrant_client
from src.services.embedding_service import embedding_service
from src.core.logger import get_logger
from src.core.exceptions import InternalServerException

logger = get_logger(__name__)

class VectorService:
    """Service for vector operations and semantic search"""
    
    def __init__(self):
        self.qdrant = qdrant_client
        self.embedding = embedding_service
    
    async def initialize(self):
        """Initialize vector service"""
        await self.qdrant.initialize()
        await self.embedding.initialize()
    
    async def semantic_search(
        self,
        query: str,
        top_k: int = 10,
        filter_type: Optional[str] = None,
        filter_repository: Optional[str] = None,
        filter_language: Optional[str] = None,
        score_threshold: float = 0.5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Perform semantic search on indexed code artifacts"""
        if filters:
            filter_type = filter_type or filters.get("type")
            filter_repository = filter_repository or filters.get("repository")
            filter_language = filter_language or filters.get("language")

        # Generate query embedding
        query_embedding = await self.embedding.generate_embedding(query)
        
        if not query_embedding:
            raise InternalServerException("Failed to generate query embedding")
        
        # Perform search
        results = await self.qdrant.search_by_text(
            query=query,
            query_vector=query_embedding,
            limit=top_k,
            filter_type=filter_type,
            filter_repository=filter_repository,
            filter_language=filter_language,
            score_threshold=score_threshold
        )
        
        # Format results
        formatted_results = []
        for result in results:
            payload = result["payload"]
            graph_node_id = payload.get("graph_node_id")
            formatted_results.append({
                "id": graph_node_id or result["id"],
                "vector_id": result["id"],
                "score": result["score"],
                "text": payload.get("text", ""),
                "type": payload.get("type", "unknown"),
                "name": payload.get("name", ""),
                "repository": payload.get("repository", "unknown"),
                "repository_hash": payload.get("repository_hash", ""),
                "file": payload.get("file", ""),
                "relative_path": payload.get("relative_path", ""),
                "line": payload.get("line", 0),
                "graph_node_id": graph_node_id,
                "metadata": payload.get("metadata", {})
            })
        
        logger.debug(
            "Semantic search completed",
            query=query[:100],
            results_count=len(formatted_results),
            top_score=formatted_results[0]["score"] if formatted_results else 0
        )
        
        return formatted_results
    
    async def find_similar_code(
        self,
        code_snippet: str,
        top_k: int = 5,
        filter_repository: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Find similar code snippets"""
        return await self.semantic_search(
            query=code_snippet,
            top_k=top_k,
            filter_type="code",
            filter_repository=filter_repository,
            score_threshold=0.7
        )
    
    async def find_related_components(
        self,
        component_description: str,
        component_type: str,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """Find components related to a description"""
        query = f"{component_type} component: {component_description}"
        
        return await self.semantic_search(
            query=query,
            top_k=top_k,
            filter_type=component_type,
            score_threshold=0.6
        )
    
    async def hybrid_search(
        self,
        query: str,
        graph_results: List[Dict[str, Any]],
        top_k: int = 10,
        alpha: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Perform hybrid search combining vector and graph results"""
        if alpha < 0 or alpha > 1:
            raise ValueError("Alpha must be between 0 and 1")
        
        # Get vector search results
        vector_results = await self.semantic_search(query, top_k=top_k * 2)
        
        # Combine results using reciprocal rank fusion
        combined_results = self._fuse_results(
            vector_results=vector_results,
            graph_results=graph_results,
            alpha=alpha
        )
        
        # Return top k results
        return combined_results[:top_k]
    
    def _fuse_results(
        self,
        vector_results: List[Dict[str, Any]],
        graph_results: List[Dict[str, Any]],
        alpha: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Fuse vector and graph search results"""
        # Assign scores
        vector_scores = {}
        for i, result in enumerate(vector_results):
            vector_scores[result["id"]] = {
                "score": result["score"],
                "rank": i + 1,
                "data": result
            }
        
        graph_scores = {}
        for i, result in enumerate(graph_results):
            result_id = result.get("id") or f"graph_{i}"
            graph_scores[result_id] = {
                "score": result.get("score", 1.0 / (i + 2)),  # Default scoring
                "rank": i + 1,
                "data": result
            }
        
        # Combine scores using weighted combination
        all_ids = set(vector_scores.keys()) | set(graph_scores.keys())
        combined_scores = []
        
        for result_id in all_ids:
            vector_info = vector_scores.get(result_id, {"score": 0, "rank": len(vector_scores) + 1})
            graph_info = graph_scores.get(result_id, {"score": 0, "rank": len(graph_results) + 1})
            
            # Normalize scores
            vector_score_norm = 1.0 / vector_info["rank"]
            graph_score_norm = 1.0 / graph_info["rank"]
            
            # Combined score
            combined_score = (alpha * vector_score_norm) + ((1 - alpha) * graph_score_norm)
            
            # Get data from whichever source has it
            data = vector_info.get("data") or graph_info.get("data")
            
            if data:
                combined_scores.append({
                    "id": result_id,
                    "combined_score": combined_score,
                    "vector_score": vector_info["score"],
                    "graph_rank": graph_info["rank"],
                    "data": data
                })
        
        # Sort by combined score
        combined_scores.sort(key=lambda x: x["combined_score"], reverse=True)
        
        # Extract data
        return [item["data"] for item in combined_scores]
    
    async def delete_artifacts(
        self,
        repository: str = None,
        artifact_type: str = None
    ) -> Dict[str, Any]:
        """Delete indexed artifacts by filter"""
        must_conditions = []
        
        if repository:
            must_conditions.append(
                FieldCondition(
                    key="repository",
                    match=MatchValue(value=repository)
                )
            )
        
        if artifact_type:
            must_conditions.append(
                FieldCondition(
                    key="type",
                    match=MatchValue(value=artifact_type)
                )
            )
        
        if not must_conditions:
            raise ValueError("At least one filter condition must be specified")
        
        filter_conditions = Filter(must=must_conditions)
        
        deleted = await self.qdrant.delete_vectors(filter_conditions)
        
        return {
            "deleted": deleted,
            "filters": {
                "repository": repository,
                "type": artifact_type
            }
        }
    
    async def get_index_stats(self) -> Dict[str, Any]:
        """Get vector index statistics"""
        collection_info = await self.qdrant.get_collection_info()
        
        return {
            "collection": collection_info,
            "model": await self.embedding.get_model_info(),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def close(self):
        """Close service resources"""
        await self.qdrant.close()
        await self.embedding.close()
    
    async def __aenter__(self):
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

# Singleton instance
vector_service = VectorService()