from fastapi import APIRouter, HTTPException
from src.data_ingestion.git_service import GitService

router = APIRouter()

@router.post("/trigger")
async def trigger_ingestion():
    try:
        git_service = GitService()
        result = await git_service.ingest_repos()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")