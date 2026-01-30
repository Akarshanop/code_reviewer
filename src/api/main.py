import logging
import sys
from pathlib import Path

from fastapi import FastAPI

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.api.v1.endpoints import admin
from src.core.logger import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

app=FastAPI(title="Code Intelligence Platform", version="0.1.0")

app.include_router(admin.router,prefix="/api/v1/admin",tags=["admin"])

@app.on_event("startup")
async def startup_event():
    logger.info("Application startup")

@app.get("/")
async def root():
    return {"message": "Welcome to the Code Intelligence Platform API"}
