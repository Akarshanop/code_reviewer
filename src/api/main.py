# main.py - ENTERPRISE VERSION
import logging
import sys
import time
import uuid
import asyncio
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

# Add project root to path (3 levels up from src/api/main.py)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.logger import setup_logging, get_logger
from src.api.v1.endpoints import admin, ingestion

setup_logging()
logger = get_logger(__name__)

# Try to import optional endpoints - don't fail if they don't exist
try:
    from src.api.v1.endpoints import developer
except ImportError:
    logger.warning("Developer endpoint not available")
    developer = None

try:
    from src.api.v1.endpoints import chat
except ImportError:
    logger.warning("Chat endpoint not available")
    chat = None

try:
    from src.api.v1.endpoints import prd_qa
except ImportError:
    logger.warning("PRD QA endpoint not available")
    prd_qa = None

try:
    from src.agents.orchestrator import AgentOrchestrator
    orchestrator_available = True
except (ImportError, ModuleNotFoundError) as e:
    logger.warning(f"AgentOrchestrator not available: {e}")
    orchestrator_available = False
    AgentOrchestrator = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management"""
    logger.info("🔄 Starting Code Intelligence Platform...")
    
    try:
        # Initialize agent orchestrator if available
        if orchestrator_available and AgentOrchestrator:
            try:
                app.state.agent_orchestrator = AgentOrchestrator()
                await app.state.agent_orchestrator.initialize()
                logger.info("✅ Agent orchestrator initialized")
            except Exception as e:
                logger.warning(f"Agent orchestrator initialization failed: {e}")
        
        logger.info("✅ Application started successfully")
        yield
        
    except Exception as e:
        logger.error(f"❌ Application startup failed: {e}")
        raise
    finally:
        logger.info("🛑 Application shutdown complete")


app = FastAPI(
    title="Enterprise Code Intelligence Platform",
    version="2.0.0",
    description="AI-powered engineering advisor for multi-repository systems",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID", "X-Response-Time"]
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = str(uuid.uuid4())
    start_time = time.time()
    
    # Log request
    logger.info(
        f"Request started: {request.method} {request.url.path}",
        extra={
            "request_id": request_id,
            "method": request.method,
            "url": str(request.url),
            "client": request.client.host if request.client else None
        }
    )
    
    response = await call_next(request)
    
    # Calculate processing time
    process_time = (time.time() - start_time) * 1000
    
    # Log response
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
    
    # Add headers
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Response-Time"] = str(round(process_time, 2))
    
    return response

# Include routers
app.include_router(admin.router, prefix="/api/v1/admin", tags=["admin"])
app.include_router(ingestion.router, prefix="/api/v1/trigger", tags=["ingestion"])

if developer:
    app.include_router(developer.router, prefix="/api/v1/developer", tags=["developer"])

if chat:
    app.include_router(chat.router, prefix="/api/v1/chat", tags=["chat"])

if prd_qa:
    app.include_router(prd_qa.router, prefix="/api/v1/prd-qa", tags=["prd-qa"])

@app.get("/")
async def root():
    return {
        "service": "Enterprise Code Intelligence Platform",
        "version": "2.0.0",
        "status": "operational",
        "documentation": "/docs"
    }

@app.get("/health")
async def health():
    """Quick health check without external dependencies"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "version": "2.0.0"
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
