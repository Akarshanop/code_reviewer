"""
Embedding Service using Google Vertex AI Text Embedding API
Cloud-based - uses Google Cloud service account for authentication
"""
from __future__ import annotations

import logging
from typing import List, Dict, Any, Optional
import numpy as np
from tenacity import retry, stop_after_attempt, wait_exponential
import asyncio
import hashlib
import json
import os

from config.settings import settings
from src.core.logger import get_logger
from src.core.exceptions import InternalServerException

logger = get_logger(__name__)

try:
    import google.auth
    from google.auth.transport.requests import Request
    from google.oauth2.service_account import Credentials
    from google.cloud import aiplatform
    from google.cloud.aiplatform.gapic.services.prediction_service import PredictionServiceAsyncClient
    from google.api_core.gapic_v1 import client_info as client_info_lib
except ImportError:
    logger.warning("Google Cloud libraries not installed. Run: pip install google-cloud-aiplatform")
    Credentials = Any
    PredictionServiceAsyncClient = Any


class EmbeddingService:
    """Service for generating embeddings using Google Vertex AI Text Embedding API"""
    
    def __init__(self):
        self.model_name = settings.EMBEDDING_MODEL  # text-embedding-004
        self.vector_size = settings.QDRANT_VECTOR_SIZE  # Will be updated to correct size
        self.project_id = settings.GOOGLE_CLOUD_PROJECT_ID
        self.region = settings.GOOGLE_CLOUD_REGION
        self.service_account_file = settings.GOOGLE_CLOUD_SERVICE_ACCOUNT_FILE
        self._initialized = False
        self._credentials: Optional[Credentials] = None
        self._client: Optional[PredictionServiceAsyncClient] = None
        
        # Load service account credentials
        if not self.project_id:
            self._load_project_from_service_account()
    
    def _load_project_from_service_account(self):
        """Load project ID from service account JSON file"""
        try:
            if os.path.exists(self.service_account_file):
                with open(self.service_account_file, 'r') as f:
                    sa_data = json.load(f)
                    self.project_id = sa_data.get("project_id")
                    logger.info(f"Loaded project ID from service account: {self.project_id}")
            else:
                logger.warning(f"Service account file not found: {self.service_account_file}")
        except Exception as e:
            logger.error(f"Failed to load service account: {e}")
    
    async def initialize(self):
        """Initialize Google Vertex AI client"""
        if not self._initialized:
            try:
                if not self.project_id:
                    raise ValueError("GOOGLE_CLOUD_PROJECT_ID is required or service account file must exist")
                
                logger.info(f"Initializing Google Vertex AI embeddings for model: {self.model_name}")
                logger.info(f"Project: {self.project_id}, Region: {self.region}")
                
                # Initialize Vertex AI with service account credentials
                if os.path.exists(self.service_account_file):
                    aiplatform.init(
                        project=self.project_id,
                        location=self.region,
                        credentials=self._get_credentials()
                    )
                else:
                    # Use default credentials (Application Default Credentials)
                    aiplatform.init(
                        project=self.project_id,
                        location=self.region
                    )
                
                logger.info("Google Vertex AI initialized successfully")
                
                # Test with a simple embedding request
                test_embedding = await self.generate_embeddings(["test"])
                if test_embedding:
                    actual_size = len(test_embedding[0])
                    if self.vector_size != actual_size:
                        logger.info(
                            f"Updating vector size from config {self.vector_size} to actual {actual_size}"
                        )
                        self.vector_size = actual_size
                
                self._initialized = True
                logger.info(
                    "Google Vertex AI Embedding Service initialized",
                    model=self.model_name,
                    vector_size=self.vector_size,
                    project_id=self.project_id,
                    region=self.region
                )
                
            except Exception as e:
                logger.error("Failed to initialize embedding service", error=str(e))
                raise
    
    def _get_credentials(self) -> Credentials:
        """Get credentials from service account file"""
        if not self._credentials:
            try:
                self._credentials = Credentials.from_service_account_file(
                    self.service_account_file,
                    scopes=["https://www.googleapis.com/auth/cloud-platform"]
                )
            except Exception as e:
                logger.error(f"Failed to load credentials: {e}")
                raise
        return self._credentials
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=4, max=30)
    )
    async def _call_vertex_api(self, texts: List[str]) -> List[List[float]]:
        """Call Google Vertex AI Text Embedding API to get embeddings"""
        if not self._initialized:
            await self.initialize()
        
        try:
            from google.cloud import aiplatform
            
            # Use the TextEmbeddingModel for synchronous calls (async wrapper)
            model = aiplatform.TextEmbeddingModel.from_pretrained(self.model_name)
            
            # Generate embeddings
            embeddings = model.get_embeddings(texts)
            
            # Extract vectors from Embedding objects
            embedding_vectors = [emb.values for emb in embeddings]
            
            logger.debug(
                "Embeddings generated via Vertex AI",
                text_count=len(texts),
                embedding_dim=len(embedding_vectors[0]) if embedding_vectors else 0
            )
            
            return embedding_vectors
            
        except Exception as e:
            logger.error(f"Failed to call Vertex AI API: {e}")
            raise
    
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
        """Generate embeddings for a list of texts using Google Vertex AI"""
        if not texts:
            return []
        
        if not self._initialized:
            await self.initialize()
        
        try:
            # Process in batches (Vertex AI supports up to 100 texts per request)
            batch_size = 50
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                batch_embeddings = await self._call_vertex_api(batch)
                
                # Normalize embeddings for consistent similarity calculations
                normalized = [self._normalize_embedding(emb) for emb in batch_embeddings]
                all_embeddings.extend(normalized)
            
            logger.debug(
                "Embeddings generated via Google Vertex AI",
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
            "api_type": "google_vertex_ai",
            "project_id": self.project_id,
            "region": self.region,
            "normalized": True,
            "source": "Google Cloud Vertex AI"
        }
    
    def clear_cache(self):
        """Clear any cached data"""
        logger.info("Embedding cache cleared")
    
    async def close(self):
        """Clean up resources"""
        self._initialized = False
        logger.info("Embedding service closed")
    
    async def __aenter__(self):
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


# Singleton instance
embedding_service = EmbeddingService()
