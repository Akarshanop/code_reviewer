import logging
from typing import List, Dict, Any, Optional, Union
from qdrant_client import AsyncQdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import numpy as np
from datetime import datetime

from config.settings import settings
from src.core.exceptions import InternalServerException
from src.core.logger import get_logger

logger = get_logger(__name__)

class QdrantClient:
    """Enterprise Qdrant client with connection pooling and retry logic"""
    
    def __init__(self):
        self.client = None
        self._initialized = False
        self._collection_name = settings.QDRANT_COLLECTION_NAME
        self._vector_size = settings.QDRANT_VECTOR_SIZE
        
    async def initialize(self):
        """Initialize Qdrant client"""
        if not self._initialized:
            try:
                logger.info(f"Creating Qdrant client for URL: {settings.QDRANT_URL}")
                self.client = AsyncQdrantClient(
                    url=settings.QDRANT_URL,
                    api_key=settings.QDRANT_API_KEY,
                    timeout=10.0,  # Reduced timeout for faster failure
                    prefer_grpc=False,  # Use HTTP REST API for Qdrant Cloud
                )
                logger.info("Qdrant client created, verifying connectivity...")
                
                await self.verify_connectivity()
                logger.info("Connectivity verified, ensuring collection...")
                
                await self.ensure_collection()
                logger.info("Collection ensured")
                
                self._initialized = True
                logger.info(
                    "Qdrant client initialized",
                    url=settings.QDRANT_URL,
                    collection=self._collection_name
                )
                
            except Exception as e:
                logger.error("Failed to initialize Qdrant client", error=str(e))
                raise
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((ConnectionError, UnexpectedResponse))
    )
    async def verify_connectivity(self):
        """Verify Qdrant connectivity"""
        try:
            await self.client.get_collections()
            logger.info("Qdrant connectivity verified")
        except Exception as e:
            logger.error("Qdrant connectivity check failed", error=str(e))
            raise
    
    async def ensure_collection(self):
        """Create collection if it doesn't exist, or recreate if vector size changed"""
        collections = await self.client.get_collections()
        collection_names = [c.name for c in collections.collections]
        
        if self._collection_name in collection_names:
            # Check if vector size matches
            collection_info = await self.client.get_collection(self._collection_name)
            existing_size = collection_info.config.params.vectors.size
            
            if existing_size != self._vector_size:
                logger.warning(
                    f"Vector size mismatch (existing={existing_size}, required={self._vector_size}). "
                    f"Recreating collection..."
                )
                await self.client.delete_collection(self._collection_name)
                collection_names.remove(self._collection_name)
        
        if self._collection_name not in collection_names:
            await self.client.create_collection(
                collection_name=self._collection_name,
                vectors_config=models.VectorParams(
                    size=self._vector_size,
                    distance=models.Distance.COSINE
                )
            )
            
            # Create payload index for faster filtering
            await self.client.create_payload_index(
                collection_name=self._collection_name,
                field_name="type",
                field_schema=models.PayloadSchemaType.KEYWORD
            )
            
            await self.client.create_payload_index(
                collection_name=self._collection_name,
                field_name="repository",
                field_schema=models.PayloadSchemaType.KEYWORD
            )
            
            await self.client.create_payload_index(
                collection_name=self._collection_name,
                field_name="language",
                field_schema=models.PayloadSchemaType.KEYWORD
            )
            
            logger.info(f"Created Qdrant collection: {self._collection_name}")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=5)
    )
    async def upsert_vectors(
        self,
        vectors: List[models.PointStruct],
        batch_size: int = 100
    ) -> bool:
        """Upsert vectors in batches"""
        if not vectors:
            return True
        
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            try:
                await self.client.upsert(
                    collection_name=self._collection_name,
                    points=batch,
                    wait=True
                )
                logger.debug(
                    f"Upserted vectors batch {i//batch_size + 1}",
                    batch_size=len(batch),
                    total_vectors=len(vectors)
                )
            except Exception as e:
                logger.error(f"Failed to upsert vectors batch {i//batch_size + 1}", error=str(e))
                raise
        
        logger.info(
            "Vectors upserted successfully",
            total_vectors=len(vectors),
            collection=self._collection_name
        )
        return True
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=5)
    )
    async def search(
        self,
        vector: List[float],
        limit: int = 10,
        filter_conditions: Optional[models.Filter] = None,
        score_threshold: Optional[float] = None,
        with_payload: bool = True,
        with_vectors: bool = False
    ) -> List[models.ScoredPoint]:
        """Search for similar vectors using query_points API"""
        start_time = datetime.utcnow()
        
        try:
            # Use query_points API (replaces deprecated search method)
            results = await self.client.query_points(
                collection_name=self._collection_name,
                query=vector,
                query_filter=filter_conditions,
                limit=limit,
                score_threshold=score_threshold,
                with_payload=with_payload,
                with_vectors=with_vectors
            )
            
            duration = (datetime.utcnow() - start_time).total_seconds() * 1000
            logger.debug(
                "Vector search completed",
                duration_ms=round(duration, 2),
                results_count=len(results.points) if hasattr(results, 'points') else 0,
                filter=str(filter_conditions)[:100] if filter_conditions else None
            )
            
            # Return points from QueryResponse
            return results.points if hasattr(results, 'points') else []
            
        except Exception as e:
            logger.error("Vector search failed", error=str(e))
            raise
    
    async def search_by_text(
        self,
        query: str,
        query_vector: List[float],
        limit: int = 10,
        filter_type: Optional[str] = None,
        filter_repository: Optional[str] = None,
        filter_language: Optional[str] = None,
        score_threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """Search with text-based filtering"""
        # Build filter conditions
        must_conditions = []
        
        if filter_type:
            must_conditions.append(
                models.FieldCondition(
                    key="type",
                    match=models.MatchValue(value=filter_type)
                )
            )
        
        if filter_repository:
            must_conditions.append(
                models.FieldCondition(
                    key="repository",
                    match=models.MatchValue(value=filter_repository)
                )
            )
        
        if filter_language:
            must_conditions.append(
                models.FieldCondition(
                    key="language",
                    match=models.MatchValue(value=filter_language)
                )
            )
        
        filter_conditions = None
        if must_conditions:
            filter_conditions = models.Filter(must=must_conditions)
        
        # Perform search
        results = await self.search(
            vector=query_vector,
            limit=limit,
            filter_conditions=filter_conditions,
            score_threshold=score_threshold
        )
        
        # Format results
        formatted_results = []
        for point in results:
            formatted_results.append({
                "id": point.id,
                "score": point.score,
                "payload": point.payload,
                "vector": point.vector if hasattr(point, 'vector') else None
            })
        
        return formatted_results
    
    async def batch_search(
        self,
        vectors: List[List[float]],
        limit_per_vector: int = 5,
        filter_conditions: Optional[models.Filter] = None
    ) -> List[List[models.ScoredPoint]]:
        """Search with multiple vectors"""
        if not vectors:
            return []
        
        # Use batch search if available, otherwise fall back to sequential
        try:
            results = await self.client.search_batch(
                collection_name=self._collection_name,
                requests=[
                    models.SearchRequest(
                        vector=vector,
                        filter=filter_conditions,
                        limit=limit_per_vector,
                        with_payload=True
                    )
                    for vector in vectors
                ]
            )
            return results
        except Exception:
            # Fall back to sequential search
            logger.warning("Batch search not available, falling back to sequential")
            all_results = []
            for vector in vectors:
                results = await self.search(
                    vector=vector,
                    limit=limit_per_vector,
                    filter_conditions=filter_conditions
                )
                all_results.append(results)
            return all_results
    
    async def delete_vectors(
        self,
        filter_conditions: models.Filter,
        wait: bool = True
    ) -> bool:
        """Delete vectors by filter"""
        try:
            await self.client.delete(
                collection_name=self._collection_name,
                points_selector=models.FilterSelector(filter=filter_conditions),
                wait=wait
            )
            logger.info("Vectors deleted", filter=str(filter_conditions)[:100])
            return True
        except Exception as e:
            logger.error("Failed to delete vectors", error=str(e))
            raise
    
    async def list_collections(self) -> List[str]:
        """List all collection names"""
        try:
            collections = await self.client.get_collections()
            return [c.name for c in collections.collections]
        except Exception as e:
            logger.error("Failed to list collections", error=str(e))
            return []
    
    async def get_collection_info(self, collection_name: str = None) -> Dict[str, Any]:
        """Get collection statistics and information"""
        try:
            coll_name = collection_name or self._collection_name
            collection_info = await self.client.get_collection(coll_name)
            
            # Get collection statistics
            stats = {
                "name": coll_name,
                "vectors_count": collection_info.vectors_count,
                "segments_count": collection_info.segments_count,
                "status": str(collection_info.status),
                "config": {
                    "vector_size": self._vector_size,
                }
            }
            
            return stats
            
        except Exception as e:
            logger.error("Failed to get collection info", error=str(e))
            return {"error": str(e)}
    
    async def health_check(self) -> bool:
        """Check if Qdrant is healthy"""
        try:
            if not self._initialized:
                logger.warning("Qdrant client not initialized")
                return False
            if not self.client:
                logger.warning("Qdrant client is None")
                return False
            await self.client.get_collections()
            return True
        except Exception as e:
            logger.error(f"Qdrant health check failed: {e}")
            return False
    
    async def close(self):
        """Close Qdrant client"""
        if self.client:
            await self.client.close()
            self._initialized = False
            logger.info("Qdrant client closed")
    
    async def __aenter__(self):
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

# Singleton instance
qdrant_client = QdrantClient()