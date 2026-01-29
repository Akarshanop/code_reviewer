import logging
from fastapi import FastAPI
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
