"""
Enterprise Code Intelligence Platform - Main API Application
FastAPI application with comprehensive middleware and route configuration
"""
import time
import uuid
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

from src.core.logger import setup_logging, get_logger
from src.api.v1.endpoints import admin, ingestion, developer, chat, prd_qa
from src.agents.orchestrator import agent_orchestrator
from src.database.neo4j_client import neo4j_client
from src.database.qdrant_client import qdrant_client
from src.database.supabase_session import supabase_client
from config.settings import settings

setup_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management - startup and shutdown"""
    logger.info("🔄 Starting Code Intelligence Platform...")
    
    try:
        # Initialize databases
        logger.info("Initializing Neo4j connection...")
        await neo4j_client.initialize()
        
        logger.info("Initializing Qdrant connection...")
        await qdrant_client.initialize()
        
        logger.info("Initializing Supabase connection...")
        await supabase_client.initialize()
        
        # Initialize orchestrator
        logger.info("Initializing agent orchestrator...")
        await agent_orchestrator.initialize()
        
        logger.info("✅ All services initialized successfully")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        # Continue anyway to allow health checks
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down services...")
    
    try:
        await neo4j_client.close()
        await qdrant_client.close()
        await supabase_client.close()
    except Exception as e:
        logger.error(f"Shutdown error: {e}")
    
    logger.info("🛑 Application shutdown complete")


app = FastAPI(
    title="Enterprise Code Intelligence Platform",
    version="2.0.0",
    description="""
    AI-powered engineering advisor for multi-repository systems.
    
    ## Features
    - **PRD Analysis**: Analyze Product Requirements Documents and generate architectural blueprints
    - **QA Generation**: Generate comprehensive test strategies from blueprints
    - **Developer Assistant**: Interactive chat for code questions and debugging
    - **Data Ingestion**: Ingest and index multiple repositories
    
    ## Supported Languages
    - Java (Vert.x patterns)
    - TypeScript/React
    - Python
    """,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS if hasattr(settings, 'CORS_ORIGINS') else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID", "X-Response-Time"]
)

# Compression
app.add_middleware(GZipMiddleware, minimum_size=1000)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests with timing information"""
    request_id = str(uuid.uuid4())
    start_time = time.time()
    
    # Skip logging for health checks
    if request.url.path in ["/health", "/favicon.ico"]:
        response = await call_next(request)
        return response
    
    logger.info(
        f"Request started: {request.method} {request.url.path}",
        extra={
            "request_id": request_id,
            "method": request.method,
            "url": str(request.url),
            "client": request.client.host if request.client else None
        }
    )
    
    try:
        response = await call_next(request)
    except Exception as e:
        logger.error(
            f"Request failed: {request.method} {request.url.path}",
            extra={"request_id": request_id, "error": str(e)}
        )
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error", "request_id": request_id}
        )
    
    process_time = (time.time() - start_time) * 1000
    
    logger.info(
        f"Request completed: {request.method} {request.url.path} {response.status_code}",
        extra={
            "request_id": request_id,
            "method": request.method,
            "url": str(request.url),
            "status_code": response.status_code,
            "duration_ms": round(process_time, 2)
        }
    )
    
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Response-Time"] = str(round(process_time, 2))
    return response


# -------------------- ROUTERS --------------------

app.include_router(admin.router, prefix="/api/v1/admin", tags=["Admin"])
app.include_router(ingestion.router, prefix="/api/v1/ingestion", tags=["Data Ingestion"])
app.include_router(developer.router, prefix="/api/v1/developer", tags=["Developer Tools"])
app.include_router(chat.router, prefix="/api/v1/chat", tags=["Chat & AI"])
app.include_router(prd_qa.router, prefix="/api/v1/prd-qa", tags=["PRD & QA"])


# -------------------- ROOT ENDPOINTS --------------------

@app.get("/")
async def root():
    """Root endpoint with service information"""
    return {
        "service": "Enterprise Code Intelligence Platform",
        "version": "2.0.0",
        "status": "operational",
        "documentation": "/docs",
        "endpoints": {
            "chat": "/api/v1/chat/ask",
            "prd_analysis": "/api/v1/chat/prd",
            "qa_generation": "/api/v1/chat/qa",
            "ingestion": "/api/v1/ingestion/trigger",
            "developer": "/api/v1/developer/analyze"
        }
    }


@app.get("/health")
async def health():
    """Health check endpoint"""
    health_status = {
        "status": "healthy",
        "timestamp": time.time(),
        "version": "2.0.0",
        "services": {}
    }
    
    # Check Neo4j
    try:
        if neo4j_client._initialized:
            health_status["services"]["neo4j"] = "connected"
        else:
            health_status["services"]["neo4j"] = "not_initialized"
    except:
        health_status["services"]["neo4j"] = "error"
    
    # Check Qdrant
    try:
        if qdrant_client._initialized:
            health_status["services"]["qdrant"] = "connected"
        else:
            health_status["services"]["qdrant"] = "not_initialized"
    except:
        health_status["services"]["qdrant"] = "error"
    
    # Check Supabase
    try:
        if supabase_client._initialized:
            health_status["services"]["supabase"] = "connected"
        else:
            health_status["services"]["supabase"] = "not_initialized"
    except:
        health_status["services"]["supabase"] = "error"
    
    # Overall status
    service_statuses = list(health_status["services"].values())
    if all(s == "connected" for s in service_statuses):
        health_status["status"] = "healthy"
    elif any(s == "connected" for s in service_statuses):
        health_status["status"] = "degraded"
    else:
        health_status["status"] = "unhealthy"
    
    return health_status


@app.get("/stats")
async def get_stats():
    """Get platform statistics"""
    try:
        stats = agent_orchestrator.get_stats()
        return {
            "status": "success",
            "orchestrator_stats": stats,
            "timestamp": time.time()
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_config=None,
        timeout_keep_alive=30
    )
