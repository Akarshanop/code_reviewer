from fastapi import APIRouter, Depends
from src.database.neo4j_client import get_neo4j_driver
from src.database.qdrant_client import get_qdrant_client
from src.database.postgres_session import get_db

router = APIRouter()

@router.get("/health")
async def health_check(neo4j=Depends(get_neo4j_driver), qdrant=Depends(get_qdrant_client), db=Depends(get_db)):
    neo4j.run("MATCH (n) RETURN count(n) as count")
    qdrant.get_collections()
    db.execute("SELECT 1")
    console.log("Health check passed")
    return {"status": "healthy"}
