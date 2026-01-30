import logging

from qdrant_client import AsyncQdrantClient
from fastapi import Depends
from config.settings import settings

logger = logging.getLogger(__name__)

async def get_qdrant_client():
    client = AsyncQdrantClient(
        url=settings.QDRANT_URL, 
        api_key=settings.QDRANT_API_KEY,
    )
    try:
        yield client
    finally:
        await client.close()

async def test_qdrant_connection(client=Depends(get_qdrant_client)):
    collections = await client.get_collections()
    return collections