from fastapi import APIRouter, Depends, HTTPException
from src.database.neo4j_client import get_neo4j_driver, test_neo4j_connection
from src.database.qdrant_client import get_qdrant_client, test_qdrant_connection
from src.database.supabase_session import get_supabase_client, test_supabase_connection

router = APIRouter()

@router.get("/health")
async def health_check(
    neo4j=Depends(get_neo4j_driver),
    qdrant=Depends(get_qdrant_client),
    supabase=Depends(get_supabase_client)
):
    try:
        await test_neo4j_connection(neo4j)
        await test_qdrant_connection(qdrant)
        await test_supabase_connection(supabase)
        return {"status": "healthy"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")