import sys
import time
import uuid
import os
import asyncio
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.exceptions import RequestValidationError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import settings
from src.core.logger import setup_logging, get_logger
from src.api.v1.endpoints import admin, ingestion, developer, chat, prd_qa, intelligence
from src.agents.orchestrator import agent_orchestrator
from src.database.neo4j_client import neo4j_client
from src.database.qdrant_client import qdrant_client
from src.database.supabase_session import supabase_client

setup_logging()
logger = get_logger(__name__)

# Set Google Cloud credentials for Vertex AI / Gemini
if settings.LLM_PROVIDER == "gemini" and settings.GOOGLE_CLOUD_SERVICE_ACCOUNT_FILE:
    if os.path.exists(settings.GOOGLE_CLOUD_SERVICE_ACCOUNT_FILE):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.abspath(
            settings.GOOGLE_CLOUD_SERVICE_ACCOUNT_FILE
        )
        logger.info(
            "Google Cloud credentials configured for Gemini",
            project_id=settings.GOOGLE_CLOUD_PROJECT_ID
        )
    else:
        logger.warning(
            "GOOGLE_CLOUD_SERVICE_ACCOUNT_FILE not found",
            path=settings.GOOGLE_CLOUD_SERVICE_ACCOUNT_FILE
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🔄 Starting Code Intelligence Platform...")
    # Use the global singleton orchestrator to avoid double initialization/state.
    app.state.agent_orchestrator = agent_orchestrator
    await agent_orchestrator.initialize()
    logger.info("✅ Agent orchestrator initialized")
    yield
    logger.info("🛑 Application shutdown complete")



app = FastAPI(
    title="Enterprise Code Intelligence Platform",
    version="2.0.0",
    description="AI-powered engineering advisor for multi-repository systems",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS middleware MUST be added before other middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=[
        "Content-Type",
        "Authorization",
        "X-Requested-With",
        "Accept",
        "Origin"
    ],
    expose_headers=["X-Request-ID", "X-Response-Time", "Content-Length"],
    max_age=3600
)

app.add_middleware(GZipMiddleware, minimum_size=1000)


# Validation error handler - log details for 422 errors
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.error(
        "Validation error",
        path=request.url.path,
        method=request.method,
        errors=exc.errors(),
        body=str(exc.body)[:500] if exc.body else None
    )
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors()}
    )



# @app.middleware("http")
# async def cors_preflight_handler(request: Request, call_next):
#     """Handle CORS preflight requests explicitly."""
#     if request.method == "OPTIONS":
#         origin = request.headers.get("origin", "")
#         allowed_origins = [
#             "http://localhost:3000",
#             "http://localhost:5173",
#             "http://127.0.0.1:3000",
#             "http://127.0.0.1:5173"
#         ]
        
#         if origin in allowed_origins:
#             return Response(
#                 status_code=200,
#                 headers={
#                     "Access-Control-Allow-Origin": origin,
#                     "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS, PATCH",
#                     "Access-Control-Allow-Headers": "Content-Type, Authorization, X-Requested-With, Accept, Origin",
#                     "Access-Control-Allow-Credentials": "true",
#                     "Access-Control-Max-Age": "3600"
#                 }
#             )
    
#     response = await call_next(request)
#     return response


@app.middleware("http")
async def log_requests(request: Request, call_next):
    from src.core.logger import trace_id_var

    request_id = str(uuid.uuid4())
    trace_id_var.set(request_id)
    start_time = time.time()

    logger.info(
        "request_started",
        request_id=request_id,
        method=request.method,
        path=request.url.path,
        url=str(request.url),
        client=request.client.host if request.client else None,
    )

    response = await call_next(request)

    process_time = (time.time() - start_time) * 1000

    logger.info(
        "request_completed",
        request_id=request_id,
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        duration_ms=round(process_time, 2),
    )

    response.headers["X-Request-ID"] = request_id
    response.headers["X-Trace-ID"] = request_id
    response.headers["X-Response-Time"] = str(round(process_time, 2))
    return response


# -------------------- ROUTERS --------------------

app.include_router(admin.router, prefix="/api/v1/admin", tags=["admin"])
app.include_router(ingestion.router, prefix="/api/v1/ingestion", tags=["ingestion"])
app.include_router(developer.router, prefix="/api/v1/developer", tags=["developer"])
app.include_router(chat.router, prefix="/api/v1/chat", tags=["chat"])
app.include_router(prd_qa.router, prefix="/api/v1/prd-qa", tags=["prd-qa"])
app.include_router(intelligence.router, prefix="/api/v1/intelligence", tags=["intelligence"])


# -------------------- SYSTEM ENDPOINTS --------------------

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
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "version": "2.0.0"
    }

@app.get("/ready")
async def ready():
    checks = await asyncio.gather(
    neo4j_client.health_check(),
    qdrant_client.health_check(),
    supabase_client.health_check(),
    return_exceptions=True
    )
    all_healthy = all(
        check is True for check in checks
        if not isinstance(check, Exception)
    )


# -------------------- LOCAL RUN --------------------

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
