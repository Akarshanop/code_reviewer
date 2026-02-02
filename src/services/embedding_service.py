"""
Embedding Service using HuggingFace Inference API
Cloud-based - no local model download required
"""
import logging
from typing import List, Dict, Any, Optional
import numpy as np
from tenacity import retry, stop_after_attempt, wait_exponential
import asyncio
import hashlib
import httpx

from config.settings import settings
from src.core.logger import get_logger
from src.core.exceptions import InternalServerException

logger = get_logger(__name__)

class EmbeddingService:
    """Service for generating embeddings using HuggingFace Inference API (Cloud)"""
    
    def __init__(self):
        self.model_name = settings.EMBEDDING_MODEL
        self.vector_size = settings.QDRANT_VECTOR_SIZE
        self.hf_token = settings.HF_TOKEN
        self._initialized = False
        self._api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{self.model_name}"
        self._client: Optional[httpx.AsyncClient] = None
        
    async def initialize(self):
        """Initialize HTTP client for HuggingFace Inference API"""
        if not self._initialized:
            try:
                if not self.hf_token:
                    raise ValueError("HF_TOKEN is required for HuggingFace Inference API")
                
                logger.info(f"Initializing HuggingFace Inference API for model: {self.model_name}")
                
                # Create async HTTP client
                self._client = httpx.AsyncClient(
                    timeout=httpx.Timeout(120.0, connect=10.0),
                    headers={
                        "Authorization": f"Bearer {self.hf_token}",
                        "Content-Type": "application/json"
                    }
                )
                
                # Test the API with a simple request
                test_embedding = await self.generate_embeddings(["test"])
                actual_size = len(test_embedding[0])
                
                if self.vector_size != actual_size:
                    logger.warning(
                        f"Vector size mismatch: config={self.vector_size}, actual={actual_size}. "
                        f"Using actual size: {actual_size}"
                    )
                    self.vector_size = actual_size
                
                self._initialized = True
                logger.info(
                    "HuggingFace Inference API initialized",
                    model=self.model_name,
                    vector_size=self.vector_size,
                    api_url=self._api_url
                )
                
            except Exception as e:
                logger.error("Failed to initialize embedding service", error=str(e))
                raise
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=4, max=30)
    )
    async def _call_hf_api(self, texts: List[str]) -> List[List[float]]:
        """Call HuggingFace Inference API to get embeddings"""
        if not self._client:
            await self.initialize()
        
        try:
            response = await self._client.post(
                self._api_url,
                json={
                    "inputs": texts,
                    "options": {
                        "wait_for_model": True  # Wait if model is loading
                    }
                }
            )
            
            if response.status_code == 503:
                # Model is loading, retry
                error_data = response.json()
                estimated_time = error_data.get("estimated_time", 60)
                logger.info(f"Model loading on HuggingFace, waiting {estimated_time}s...")
                await asyncio.sleep(min(estimated_time, 60))
                raise ConnectionError("Model loading, will retry")
            
            response.raise_for_status()
            result = response.json()
            
            # Handle different response formats from HuggingFace
            if isinstance(result, list):
                if len(result) > 0:
                    if isinstance(result[0], list):
                        if isinstance(result[0][0], list):
                            # Nested: [[[floats]]] - use mean pooling
                            embeddings = [self._pool_embeddings(emb) for emb in result]
                        else:
                            # Direct: [[floats]]
                            embeddings = result
                    else:
                        embeddings = [result]
                else:
                    embeddings = []
            else:
                raise ValueError(f"Unexpected response format: {type(result)}")
            
            return embeddings
            
        except httpx.HTTPStatusError as e:
            error_text = e.response.text[:200] if e.response else "Unknown error"
            logger.error(f"HuggingFace API error {e.response.status_code}: {error_text}")
            raise
        except Exception as e:
            logger.error(f"Failed to call HuggingFace API: {e}")
            raise
    
    def _pool_embeddings(self, token_embeddings: List[List[float]]) -> List[float]:
        """Pool token embeddings using mean pooling"""
        if not token_embeddings:
            return []
        arr = np.array(token_embeddings)
        pooled = np.mean(arr, axis=0)
        return pooled.tolist()
    
    def _normalize_embedding(self, embedding: List[float]) -> List[float]:
        """L2 normalize an embedding"""
        arr = np.array(embedding)
        norm = np.linalg.norm(arr)
        if norm > 0:
            arr = arr / norm
        return arr.tolist()
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=5)
    )
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts using HuggingFace Inference API"""
        if not texts:
            return []
        
        if not self._initialized and self._client is None:
            await self.initialize()
        
        try:
            # Process in batches to avoid API limits
            batch_size = 32  # HF API handles up to 32 texts per request
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                batch_embeddings = await self._call_hf_api(batch)
                
                # Normalize embeddings
                normalized = [self._normalize_embedding(emb) for emb in batch_embeddings]
                all_embeddings.extend(normalized)
            
            logger.debug(
                "Embeddings generated via HuggingFace Inference API",
                text_count=len(texts),
                embedding_dim=len(all_embeddings[0]) if all_embeddings else 0
            )
            
            return all_embeddings
            
        except Exception as e:
            logger.error("Failed to generate embeddings", error=str(e), text_count=len(texts))
            raise InternalServerException(f"Embedding generation failed: {str(e)}")
    
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        embeddings = await self.generate_embeddings([text])
        return embeddings[0] if embeddings else []
    
    def generate_id(self, text: str, prefix: str = "") -> str:
        """Generate deterministic ID for embedding"""
        content_hash = hashlib.md5(text.encode()).hexdigest()[:16]
        return f"{prefix}_{content_hash}" if prefix else content_hash
    
    async def batch_generate_with_ids(
        self,
        texts: List[str],
        metadata_list: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate embeddings with IDs and metadata"""
        if len(texts) != len(metadata_list):
            raise ValueError("Texts and metadata lists must have the same length")
        
        # Generate embeddings
        embeddings = await self.generate_embeddings(texts)
        
        # Create points with metadata
        points = []
        for i, (text, embedding, metadata) in enumerate(zip(texts, embeddings, metadata_list)):
            point_id = self.generate_id(text, metadata.get("type", "embed"))
            
            point = {
                "id": point_id,
                "vector": embedding,
                "payload": {
                    "text": text,
                    "type": metadata.get("type", "unknown"),
                    "language": metadata.get("language", "unknown"),
                    "repository": metadata.get("repository", "unknown"),
                    "file": metadata.get("file", ""),
                    "line": metadata.get("line", 0),
                    "metadata": metadata.get("metadata", {}),
                    "created_at": metadata.get("created_at")
                }
            }
            points.append(point)
        
        return points
    
    async def calculate_similarity(
        self,
        embedding1: List[float],
        embedding2: List[float]
    ) -> float:
        """Calculate cosine similarity between two embeddings"""
        if not embedding1 or not embedding2:
            return 0.0
        
        if len(embedding1) != len(embedding2):
            logger.warning(
                "Embedding dimension mismatch",
                dim1=len(embedding1),
                dim2=len(embedding2)
            )
            return 0.0
        
        # Convert to numpy arrays for efficient calculation
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        # Cosine similarity
        similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        
        # Ensure value is between -1 and 1
        similarity = max(-1.0, min(1.0, similarity))
        
        return float(similarity)
    
    async def find_most_similar(
        self,
        query_embedding: List[float],
        candidate_embeddings: List[List[float]],
        candidate_data: List[Any],
        top_k: int = 5,
        threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """Find most similar embeddings from a list of candidates"""
        if not candidate_embeddings or not query_embedding:
            return []
        
        similarities = []
        for i, candidate_embedding in enumerate(candidate_embeddings):
            similarity = await self.calculate_similarity(query_embedding, candidate_embedding)
            if similarity >= threshold:
                similarities.append({
                    "index": i,
                    "similarity": similarity,
                    "data": candidate_data[i] if i < len(candidate_data) else None
                })
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        
        # Return top k results
        return similarities[:top_k]
    
    async def get_model_info(self) -> Dict[str, Any]:
        """Get embedding model information"""
        return {
            "model_name": self.model_name,
            "vector_size": self.vector_size,
            "initialized": self._initialized,
            "api_type": "huggingface_inference_api",
            "api_url": self._api_url,
            "normalized": True,
            "source": "Cloud (HuggingFace Servers)"
        }
    
    def clear_cache(self):
        """Clear any cached data"""
        logger.info("Embedding cache cleared")
    
    async def close(self):
        """Clean up resources"""
        if self._client:
            await self._client.aclose()
            self._client = None
            self._initialized = False
            logger.info("Embedding service closed")
    
    async def __aenter__(self):
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

# Singleton instance
embedding_service = EmbeddingService()