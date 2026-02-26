"""
Enterprise Chat Endpoint for AI-Powered Code Intelligence
Provides interactive conversations with developer assistant agent

Enhanced with:
    - Role-aware endpoint (two-layer: Reasoning → Policy)
    - PRD → QA → PDF pipeline
    - Confidence scoring in responses
    - Trust-preserving silence
"""
import uuid
import os
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel, Field
import json
import asyncio
import redis.asyncio as redis

from src.core.exceptions import (
    ValidationException,
    InternalServerException,
    AgentException as AgentExceptionHTTP,
    LLMServiceException,
    ResourceNotFoundException,
)
from src.schemas.chat_schema import ChatResponse, ChatRequest, PRDRequest, QARequest, JobStatus
from src.core.logger import get_logger
from src.agents.orchestrator import agent_orchestrator
from src.database.supabase_session import supabase_client
from config.settings import settings

logger = get_logger(__name__)
router = APIRouter()


class RoleAwareChatRequest(BaseModel):
    """Chat request with role-based output shaping."""
    question: str = Field(..., min_length=1, max_length=10000)
    user_role: str = Field(default="developer", description="User role: admin, architect, developer, qa, sales, viewer")
    chat_history: Optional[List[dict]] = Field(default_factory=list)
    context_files: Optional[List[str]] = Field(default_factory=list)
    session_id: Optional[str] = None


class PRDGenerateRequest(BaseModel):
    """Request for PRD + QA generation pipeline."""
    client_requirements: str = Field(..., min_length=20, max_length=50000, description="Client requirements text")
    business_context: Optional[str] = Field(default="", description="Business context and background")
    constraints: Optional[str] = Field(default="", description="Performance, security, timeline constraints")
    prd_format_template: Optional[str] = Field(default="", description="Custom PRD format template")

# Fallback in-memory store (Redis is primary for production)
_job_store: Dict[str, Dict[str, Any]] = {}
_redis_client: Optional[redis.Redis] = None


def _get_redis_client() -> Optional[redis.Redis]:
    global _redis_client
    if _redis_client is None and settings.REDIS_URL:
        _redis_client = redis.from_url(settings.REDIS_URL, decode_responses=True)
    return _redis_client


def _serialize_job(job: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in (job or {}).items():
        if isinstance(v, datetime):
            out[k] = v.isoformat()
        elif isinstance(v, (dict, list)):
            out[k] = json.dumps(v, default=str)
        elif v is None:
            out[k] = ""
        else:
            out[k] = str(v)
    return out


def _deserialize_job(raw: Dict[str, Any]) -> Dict[str, Any]:
    if not raw:
        return {}
    job: Dict[str, Any] = dict(raw)
    for dt_field in ("created_at", "updated_at"):
        if job.get(dt_field):
            try:
                job[dt_field] = datetime.fromisoformat(job[dt_field])
            except Exception:
                job[dt_field] = datetime.utcnow()
        else:
            job[dt_field] = datetime.utcnow()
    for num_field in ("progress",):
        try:
            job[num_field] = float(job.get(num_field, 0.0))
        except Exception:
            job[num_field] = 0.0
    if job.get("result"):
        try:
            job["result"] = json.loads(job["result"])
        except Exception:
            pass
    return job


async def _create_job(job_id: str, job: Dict[str, Any]) -> None:
    redis_cli = _get_redis_client()
    if redis_cli:
        key = f"chat_job:{job_id}"
        await redis_cli.hset(key, mapping=_serialize_job(job))
        await redis_cli.expire(key, settings.REDIS_TTL)
        await redis_cli.sadd("chat_jobs:all", job_id)
        return
    _job_store[job_id] = job


async def _get_job(job_id: str) -> Optional[Dict[str, Any]]:
    redis_cli = _get_redis_client()
    if redis_cli:
        key = f"chat_job:{job_id}"
        raw = await redis_cli.hgetall(key)
        return _deserialize_job(raw) if raw else None
    return _job_store.get(job_id)


async def _update_job(job_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    existing = await _get_job(job_id)
    if not existing:
        return None
    existing.update(updates or {})
    existing["updated_at"] = datetime.utcnow()
    await _create_job(job_id, existing)
    return existing


async def _list_jobs() -> List[Dict[str, Any]]:
    redis_cli = _get_redis_client()
    if redis_cli:
        job_ids = await redis_cli.smembers("chat_jobs:all")
        jobs: List[Dict[str, Any]] = []
        for job_id in job_ids:
            job = await _get_job(job_id)
            if job:
                jobs.append(job)
        return jobs
    return list(_job_store.values())

@router.post("/ask", response_model=ChatResponse)
async def chat_with_ai(request: ChatRequest):
    """
    Interactive chat with AI assistant — role-aware.
    
    Routes to the correct agent based on user_role:
    - sales → Sales advisor (ELI5, no code)
    - qa → Developer assistant (test-oriented view via policy)
    - developer/architect/admin → Developer assistant (full detail)
    
    All responses go through the Policy Layer for RBAC and output shaping.
    """
    try:
        logger.info(
            "Processing chat request",
            question_length=len(request.question),
            user_role=request.user_role,
            agent_type=request.agent_type,
        )
        
        # Ensure orchestrator is initialized
        await agent_orchestrator.initialize()
        
        # Convert chat history to list of dicts
        history = [
            {"role": msg.role, "content": msg.content}
            for msg in (request.chat_history or [])
        ]
        
        # Determine effective role
        user_role = request.user_role or "developer"
        
        # Route through policy layer for role-aware processing
        policy_result = await agent_orchestrator.process_with_policy(
            question=request.question,
            user_role=user_role,
            chat_history=history,
            context_files=request.context_files,
        )
        
        # Generate session ID if not provided
        session_id = request.session_id or str(uuid.uuid4())
        
        answer = policy_result.get("answer", "I couldn't process your question. Please try again.")
        tools_used = policy_result.get("tools_used", [])
        execution_time = policy_result.get("execution_time_seconds", 0.0)
        is_silence = policy_result.get("is_silence", False)
        
        # Store conversation in Supabase if available
        if supabase_client and not is_silence:
            try:
                await _store_conversation(
                    session_id=session_id,
                    question=request.question,
                    answer=answer,
                    metadata={
                        "tools_used": tools_used,
                        "execution_time": execution_time,
                        "role_applied": policy_result.get("role_applied", user_role),
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to store conversation: {e}")
        
        # Generate follow-up suggestions
        followups = _generate_followups(request.question, answer)
        
        # Map confidence label to numeric for ChatResponse schema
        confidence_label = policy_result.get("confidence", "unknown")
        confidence_numeric = 0.9  # default
        if confidence_label == "low confidence":
            confidence_numeric = 0.4
        elif confidence_label == "uncertain":
            confidence_numeric = 0.5
        elif confidence_label == "moderate confidence":
            confidence_numeric = 0.7
        elif confidence_label == "high confidence":
            confidence_numeric = 0.9
        elif confidence_label == "insufficient certainty":
            confidence_numeric = 0.2
        
        return ChatResponse(
            status="silence" if is_silence else "success",
            answer=answer,
            confidence=confidence_numeric,
            sources=[],
            followups=followups,
            tools_used=tools_used,
            execution_time_seconds=execution_time,
            session_id=session_id,
        )
        
    except Exception as e:
        logger.error(f"Chat failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")


@router.post("/ask/role-aware")
async def chat_with_role_policy(request: RoleAwareChatRequest):
    """
    Role-aware chat endpoint with two-layer architecture.
    
    This endpoint applies RBAC-based output shaping:
        - admin/architect/developer: Full technical detail
        - qa: Test-oriented view with behavioral summaries
        - sales: Business-only (no file names, class names, architecture)
        - viewer: Read-only codebase view
    
    Responses include confidence scoring and trust-preserving silence.
    """
    try:
        await agent_orchestrator.initialize()

        history = [
            {"role": msg.get("role", "user"), "content": msg.get("content", "")}
            for msg in (request.chat_history or [])
        ]

        result = await agent_orchestrator.process_with_policy(
            question=request.question,
            user_role=request.user_role,
            chat_history=history,
            context_files=request.context_files,
        )

        session_id = request.session_id or str(uuid.uuid4())

        return {
            "status": "silence" if result.get("is_silence") else "success",
            "answer": result.get("answer", ""),
            "confidence": result.get("confidence", "unknown"),
            "risk_level": result.get("risk_level", "unknown"),
            "affected_domains": result.get("affected_domains", []),
            "impact_summary": result.get("impact_summary", ""),
            "affected_files": result.get("affected_files", []),
            "evidence": result.get("evidence", []),
            "uncertainty": result.get("uncertainty", []),
            "tools_used": result.get("tools_used", []),
            "execution_time_seconds": result.get("execution_time_seconds", 0.0),
            "role_applied": result.get("role_applied", request.user_role),
            "session_id": session_id,
        }

    except Exception as e:
        logger.error(f"Role-aware chat failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")


@router.post("/prd")
async def analyze_prd(request: PRDRequest, background_tasks: BackgroundTasks):
    """
    Analyze a PRD and generate architectural blueprint.
    
    Returns implementation roadmap with:
    - Component identification
    - API endpoints needed
    - Database changes
    - Risk assessment
    - Effort estimation
    """
    try:
        logger.info(
            "Starting PRD analysis",
            prd_length=len(request.prd_text),
            async_mode=request.async_mode
        )
        
        # Ensure orchestrator is initialized
        await agent_orchestrator.initialize()
        
        if request.async_mode:
            # Create job for background processing
            job_id = str(uuid.uuid4())
            await _create_job(job_id, {
                "status": "pending",
                "progress": 0.0,
                "result": None,
                "error": None,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            })
            
            # Add background task
            background_tasks.add_task(
                _process_prd_async,
                job_id,
                request.prd_text,
                request.human_prompt,
                request.repository_filter,
                request.architectural_constraints,
                request.priority
            )
            
            return {
                "status": "accepted",
                "job_id": job_id,
                "message": "PRD analysis started in background",
                "status_url": f"/api/v1/chat/jobs/{job_id}/status"
            }
        
        # Synchronous processing
        result = await agent_orchestrator.process_prd(
            prd_text=request.prd_text,
            human_prompt=request.human_prompt or "",
            repository_filter=request.repository_filter,
            architectural_constraints=request.architectural_constraints,
            priority=request.priority
        )
        
        if result.success:
            return {
                "status": "success",
                "blueprint": result.output.get("blueprint", {}),
                "validation_results": result.output.get("validation_results", []),
                "context_sources": result.output.get("context_sources", 0),
                "execution_time_seconds": result.execution_time_seconds
            }
        else:
            return {
                "status": "failed",
                "errors": result.errors,
                "partial_output": result.output,
                "execution_time_seconds": result.execution_time_seconds
            }
        
    except Exception as e:
        logger.error(f"PRD analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"PRD analysis failed: {str(e)}")


@router.post("/qa")
async def generate_qa_plan(request: QARequest):
    """
    Generate comprehensive QA and testing strategy.
    
    Returns:
    - Unit test specifications
    - Integration test scenarios
    - E2E test cases
    - Race condition analysis
    - Test data requirements
    """
    try:
        logger.info(
            "Starting QA generation",
            test_focus=request.test_focus
        )
        
        # Ensure orchestrator is initialized
        await agent_orchestrator.initialize()
        
        result = await agent_orchestrator.process_qa(
            prd_text=request.prd_text,
            design_blueprint=request.design_blueprint,
            test_focus=request.test_focus,
            test_categories=request.test_categories
        )
        
        if result.success:
            return {
                "status": "success",
                "test_strategy": result.output.get("test_strategy", {}),
                "validation_results": result.output.get("validation_results", []),
                "execution_time_seconds": result.execution_time_seconds
            }
        else:
            return {
                "status": "failed",
                "errors": result.errors,
                "partial_output": result.output,
                "execution_time_seconds": result.execution_time_seconds
            }
        
    except Exception as e:
        logger.error(f"QA generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"QA generation failed: {str(e)}")


@router.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    """Retrieve conversation history by session ID"""
    try:
        if not supabase_client:
            raise HTTPException(status_code=503, detail="Database not available")
        
        # Fetch from Supabase
        messages = await supabase_client.get_chat_history(
            session_id=conversation_id,
            limit=100
        )
        
        return {
            "status": "success",
            "conversation_id": conversation_id,
            "messages": messages,
            "message_count": len(messages)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Conversation retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions")
async def list_chat_sessions(limit: int = 50):
    """List all chat sessions with titles"""
    try:
        if not supabase_client:
            raise HTTPException(status_code=503, detail="Database not available")
        
        await supabase_client.initialize()
        sessions = await supabase_client.get_all_chat_sessions(limit=limit)
        
        return {
            "status": "success",
            "sessions": sessions,
            "count": len(sessions)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to list sessions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/sessions/{session_id}")
async def delete_chat_session(session_id: str):
    """Delete a chat session and all its messages"""
    try:
        if not supabase_client:
            raise HTTPException(status_code=503, detail="Database not available")
        
        await supabase_client.initialize()
        success = await supabase_client.delete_chat_session(session_id)
        
        if success:
            return {"status": "success", "message": f"Session {session_id} deleted"}
        else:
            raise HTTPException(status_code=500, detail="Failed to delete session")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/jobs/{job_id}/status", response_model=JobStatus)
async def get_job_status(job_id: str):
    """Get status of async job"""
    try:
        job = await _get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        
        return JobStatus(
            job_id=job_id,
            status=job["status"],
            progress=job["progress"],
            result=job.get("result"),
            error=job.get("error"),
            created_at=job["created_at"],
            updated_at=job["updated_at"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Job status retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/jobs/{job_id}")
async def cancel_job(job_id: str):
    """Cancel a running job"""
    try:
        job = await _get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        
        if job["status"] in ["completed", "failed"]:
            return {"status": "already_finished", "job_id": job_id}
        
        await _update_job(job_id, {"status": "cancelled"})
        
        return {"status": "cancelled", "job_id": job_id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Job cancellation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate-prd")
async def generate_prd_with_qa(request: PRDGenerateRequest):
    """
    Generate enterprise-grade PRD + QA Test Plan (both as PDFs).

    This endpoint:
    1. Runs the PRD Agent (searches codebase, generates structured PRD)
    2. Auto-runs the QA Agent (generates test plan from PRD)
    3. Generates both documents as PDFs
    4. Returns download URLs for both PDFs

    The PRD Agent is strictly codebase-grounded — zero hallucination.
    """
    try:
        logger.info(
            "Starting PRD+QA generation pipeline",
            requirements_length=len(request.client_requirements),
        )

        await agent_orchestrator.initialize()

        result = await agent_orchestrator.generate_prd_with_qa(
            client_requirements=request.client_requirements,
            business_context=request.business_context or "",
            constraints=request.constraints or "",
            prd_format_template=request.prd_format_template or "",
        )

        if not result.get("success"):
            return {
                "status": "failed",
                "error": result.get("error", "Unknown error"),
                "errors": result.get("errors", []),
            }

        # Build download URLs
        prd_pdf = result.get("prd_pdf_path", "")
        qa_pdf = result.get("qa_pdf_path", "")

        prd_filename = os.path.basename(prd_pdf) if prd_pdf else ""
        qa_filename = os.path.basename(qa_pdf) if qa_pdf else ""

        return {
            "status": "success",
            "feature_name": result.get("feature_name", ""),
            "prd_pdf_url": f"/api/v1/chat/download-pdf/{prd_filename}" if prd_filename else None,
            "qa_pdf_url": f"/api/v1/chat/download-pdf/{qa_filename}" if qa_filename else None,
            "prd_data": result.get("prd_data", {}),
            "qa_data": result.get("qa_data", {}),
            "prd_tools_used": result.get("prd_tools_used", []),
            "qa_tools_used": result.get("qa_tools_used", []),
            "execution_time_seconds": result.get("execution_time_seconds", 0.0),
        }

    except ValidationException:
        raise
    except Exception as e:
        logger.error(f"PRD+QA generation failed: {e}", exc_info=True)
        raise AgentExceptionHTTP(
            message=f"PRD generation pipeline failed: {str(e)}",
            agent_type="prd_agent"
        )


@router.get("/download-pdf/{filename}")
async def download_pdf(filename: str):
    """Download a generated PDF document (PRD or QA Test Plan)."""
    try:
        from src.services.pdf_service import PDF_OUTPUT_DIR

        filepath = PDF_OUTPUT_DIR / filename

        if not filepath.exists():
            raise ResourceNotFoundException("PDF", filename)

        # Security: ensure the file is within the output directory
        if not str(filepath.resolve()).startswith(str(PDF_OUTPUT_DIR.resolve())):
            raise HTTPException(status_code=403, detail="Access denied")

        return FileResponse(
            path=str(filepath),
            filename=filename,
            media_type="application/pdf",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"PDF download failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_orchestrator_stats():
    """Get orchestrator statistics"""
    try:
        stats = agent_orchestrator.get_stats()
        jobs = await _list_jobs()
        return {
            "status": "success",
            "stats": stats,
            "active_jobs": len([j for j in jobs if j.get("status") == "running"]),
            "total_jobs": len(jobs)
        }
    except Exception as e:
        logger.error(f"Stats retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Helper functions

async def _process_prd_async(
    job_id: str,
    prd_text: str,
    human_prompt: str,
    repository_filter: List[str],
    architectural_constraints: Dict[str, Any],
    priority: str
):
    """Background task for PRD processing"""
    try:
        await _update_job(job_id, {"status": "running", "progress": 0.1})
        
        result = await agent_orchestrator.process_prd(
            prd_text=prd_text,
            human_prompt=human_prompt or "",
            repository_filter=repository_filter,
            architectural_constraints=architectural_constraints,
            priority=priority
        )
        
        await _update_job(job_id, {
            "status": "completed" if result.success else "failed",
            "progress": 1.0,
            "result": {
                "blueprint": result.output.get("blueprint", {}),
                "validation_results": result.output.get("validation_results", []),
                "execution_time_seconds": result.execution_time_seconds
            },
            "error": result.errors[0] if result.errors else None
        })
        
    except Exception as e:
        logger.error(f"Background PRD processing failed: {e}")
        await _update_job(job_id, {"status": "failed", "error": str(e)})


async def _store_conversation(
    session_id: str,
    question: str,
    answer: str,
    metadata: Dict[str, Any]
):
    """Store conversation in Supabase"""
    if not supabase_client:
        return
    
    await supabase_client.store_chat_message(
        session_id=session_id,
        role="user",
        content=question,
        metadata=metadata
    )
    
    await supabase_client.store_chat_message(
        session_id=session_id,
        role="assistant",
        content=answer,
        metadata=metadata
    )


def _generate_followups(question: str, answer: str) -> List[str]:
    """Generate follow-up question suggestions"""
    followups = []
    
    question_lower = question.lower()
    
    if "how" in question_lower or "what" in question_lower:
        followups.append("Can you show me the related tests?")
        followups.append("What are the potential edge cases?")
    
    if "api" in question_lower or "endpoint" in question_lower:
        followups.append("What services does this API depend on?")
        followups.append("Can you show me the request/response schema?")
    
    if "bug" in question_lower or "error" in question_lower or "fix" in question_lower:
        followups.append("What other components might be affected?")
        followups.append("Are there similar patterns elsewhere in the codebase?")
    
    if "component" in question_lower or "module" in question_lower:
        followups.append("What are the dependencies of this component?")
        followups.append("How is this component used throughout the codebase?")
    
    # Default follow-ups
    if not followups:
        followups = [
            "Can you explain more about how this works?",
            "What are the potential risks or issues?",
            "Are there any related components I should look at?"
        ]
    
    return followups[:3]  # Return max 3 suggestions
