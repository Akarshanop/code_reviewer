import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from qdrant_client.models import PointStruct, Filter, FieldCondition, MatchValue
import asyncio

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
    
    async def index_code_artifacts(
        self,
        artifacts: List[Dict[str, Any]],
        batch_size: int = 100
    ) -> Dict[str, Any]:
        """Index code artifacts (classes, methods, components, etc.)"""
        if not artifacts:
            logger.warning("No artifacts to index")
            return {"indexed": 0, "errors": []}
        
        total_indexed = 0
        errors = []
        
        logger.info(
            "Starting artifact indexing",
            total_artifacts=len(artifacts),
            batch_size=batch_size
        )
        
        for i in range(0, len(artifacts), batch_size):
            batch = artifacts[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            
            try:
                # Extract texts for embedding
                texts = []
                metadata_list = []
                artifact_names = []
                
                for artifact in batch:
                    text = self._create_index_text(artifact)
                    
                    # ✅ Validate: ensure we have content to embed
                    if not text or len(text.strip()) < 5:
                        logger.warning(
                            f"Artifact has minimal content",
                            artifact_type=artifact.get("type"),
                            artifact_name=artifact.get("name"),
                            text_length=len(text) if text else 0
                        )
                    
                    texts.append(text)
                    artifact_names.append(artifact.get("name", "unknown"))
                    
                    # Extract file_name and relative_path from file path
                    file_path = artifact.get("file", "")
                    file_name = file_path.split('/')[-1] if file_path else ""
                    
                    # REQUIRED metadata fields per spec
                    metadata = {
                        "type": artifact.get("type", "unknown"),
                        "language": artifact.get("language", "unknown"),
                        "repository": artifact.get("repository", "unknown"),
                        "file": file_path,
                        "line": artifact.get("line", 0),
                        "name": artifact.get("name", ""),
                        "signature": artifact.get("signature", ""),
                        "description": artifact.get("description", ""),
                        "metadata": artifact.get("metadata", {}),
                        "created_at": datetime.utcnow().isoformat(),
                        # REQUIRED FIELDS for enterprise-grade system
                        "repository_name": artifact.get("repository_name", ""),
                        "file_name": file_name,
                        "relative_path": file_path,
                        "entry_file": artifact.get("entry_file", ""),
                        "repo_type": artifact.get("repo_type", ""),
                    }
                    metadata_list.append(metadata)
                
                # Generate embeddings
                logger.debug(
                    f"Generating embeddings for batch {batch_num}",
                    batch_size=len(batch),
                    text_samples=artifact_names[:3]
                )
                
                points_data = await self.embedding.batch_generate_with_ids(texts, metadata_list)
                
                if not points_data:
                    error_msg = f"Batch {batch_num}: Failed to generate embeddings (empty result)"
                    logger.error(error_msg, batch_index=batch_num)
                    errors.append(error_msg)
                    continue
                
                # Convert to Qdrant PointStruct
                points = []
                for point_data in points_data:
                    point = PointStruct(
                        id=point_data["id"],
                        vector=point_data["vector"],
                        payload=point_data["payload"]
                    )
                    points.append(point)
                
                # ✅ Validate embeddings before upsert
                if not points:
                    error_msg = f"Batch {batch_num}: No valid points to upsert"
                    logger.error(error_msg)
                    errors.append(error_msg)
                    continue
                
                # Upsert to Qdrant
                logger.debug(
                    f"Upserting batch {batch_num} to Qdrant",
                    point_count=len(points)
                )
                
                await self.qdrant.upsert_vectors(points)
                
                total_indexed += len(batch)
                logger.info(
                    f"Indexed batch {batch_num} successfully",
                    batch_size=len(batch),
                    total_indexed=total_indexed,
                    batch_artifacts=artifact_names[:3]
                )
                
            except Exception as e:
                error_msg = f"Batch {batch_num} failed: {str(e)}"
                logger.error(
                    f"Failed to index batch {batch_num}",
                    error=str(e),
                    batch_index=batch_num,
                    exc_info=True
                )
                errors.append(error_msg)
        
        logger.info(
            "Artifact indexing completed",
            total_indexed=total_indexed,
            total_artifacts=len(artifacts),
            error_count=len(errors),
            success_rate=f"{(total_indexed/len(artifacts)*100):.1f}%" if artifacts else "N/A"
        )
        
        return {
            "indexed": total_indexed,
            "errors": errors,
            "success": len(errors) == 0 and total_indexed > 0
        }
    
    def _create_index_text(self, artifact: Dict[str, Any]) -> str:
        """
        Create text for embedding from artifact.
        
        FIX 6: Prefer the rich 'text' field that _generate_embeddings already
        builds (contains type + name + key attributes).  Fall back to the
        'code' field.  Both are concatenated with name/signature so the
        embedding captures the same semantic content that was intended during
        ingestion.  Code limit raised from 500→1500 chars so meaningful
        function bodies aren't truncated to a useless prefix.
        """
        parts = []
        
        # Add name and type (always first for retrieval anchoring)
        artifact_type = artifact.get('type', 'artifact')
        artifact_name = artifact.get('name', '')
        parts.append(f"{artifact_type}: {artifact_name}")
        
        # Language / file context (helps discriminate across repos)
        if artifact.get("language"):
            parts.append(f"Language: {artifact['language']}")
        if artifact.get("file"):
            parts.append(f"File: {artifact['file']}")
        
        # Add signature if available
        if artifact.get("signature"):
            parts.append(f"Signature: {artifact['signature']}")
        
        # Add description if available
        if artifact.get("description"):
            parts.append(f"Description: {artifact['description']}")
        
        # Prefer the ingestion-built 'text' field (richer summary).
        # Fall back to 'code' (raw snippet).
        text_field = artifact.get("text") or ""
        code_field = artifact.get("code") or ""
        
        # Use whichever is more informative; if both exist, concatenate.
        if text_field and code_field and text_field != code_field:
            combined = f"{text_field}\n{code_field}"
        else:
            combined = text_field or code_field
        
        if combined:
            # FIX 1: Increase limit to 2500 chars now that we embed actual code bodies
            if len(combined) > 2500:
                combined = combined[:2500] + "..."
            parts.append(combined)
        
        # Add metadata as context
        if artifact.get("metadata") and isinstance(artifact["metadata"], dict):
            for key, value in artifact["metadata"].items():
                if isinstance(value, str) and len(value) < 100:
                    parts.append(f"{key}: {value}")
        
        return "\n".join(parts)
    
    async def semantic_search(
        self,
        query: str,
        top_k: int = 10,
        filter_type: Optional[str] = None,
        filter_repository: Optional[str] = None,
        filter_language: Optional[str] = None,
        score_threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """Perform semantic search on indexed code artifacts"""
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
            formatted_results.append({
                "id": result["id"],
                "score": result["score"],
                "text": result["payload"].get("text", ""),
                "type": result["payload"].get("type", "unknown"),
                "name": result["payload"].get("name", ""),
                "repository": result["payload"].get("repository", "unknown"),
                "file": result["payload"].get("file", ""),
                "line": result["payload"].get("line", 0),
                "metadata": result["payload"].get("metadata", {})
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