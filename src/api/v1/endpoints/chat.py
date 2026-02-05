"""
Enterprise Chat Endpoint for AI-Powered Code Intelligence
Provides interactive conversations with developer assistant agent
"""
import uuid
from typing import Dict, Any, List, Optional
from datetime import datetime
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import json
import asyncio

from src.core.logger import get_logger
from src.agents.orchestrator import agent_orchestrator
from src.database.supabase_session import supabase_client
from config.settings import settings

logger = get_logger(__name__)
router = APIRouter()

# In-memory job storage (use Redis in production)
_job_store: Dict[str, Dict[str, Any]] = {}


class ChatMessage(BaseModel):
    """Individual chat message"""
    role: str = Field(..., description="Message role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")
    timestamp: Optional[datetime] = Field(default_factory=datetime.utcnow)
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class ChatRequest(BaseModel):
    """Request for chat endpoint"""
    question: str = Field(..., min_length=1, max_length=10000, description="User question")
    chat_history: Optional[List[ChatMessage]] = Field(default_factory=list)
    context_type: Optional[str] = Field(default="auto", description="Context type: auto, code, architecture, debug")
    repository_filter: Optional[List[str]] = Field(default_factory=list)
    context_files: Optional[List[str]] = Field(default_factory=list, description="Specific files for context")
    session_id: Optional[str] = Field(default=None, description="Session ID for conversation persistence")


class ChatResponse(BaseModel):
    """Response from chat endpoint"""
    status: str
    answer: str
    confidence: float = Field(default=0.85)
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    followups: List[str] = Field(default_factory=list)
    tools_used: List[str] = Field(default_factory=list)
    execution_time_seconds: float = Field(default=0.0)
    session_id: Optional[str] = None


class PRDRequest(BaseModel):
    """Request for PRD analysis"""
    prd_text: str = Field(..., min_length=50, max_length=50000)
    human_prompt: Optional[str] = Field(default="")
    repository_filter: Optional[List[str]] = Field(default_factory=list)
    architectural_constraints: Optional[Dict[str, Any]] = Field(default_factory=dict)
    priority: Optional[str] = Field(default="medium")
    async_mode: Optional[bool] = Field(default=False, description="Run in background")


class QARequest(BaseModel):
    """Request for QA generation"""
    prd_text: str = Field(..., min_length=50)
    design_blueprint: Dict[str, Any] = Field(...)
    test_focus: Optional[str] = Field(default="all")
    test_categories: Optional[List[str]] = Field(default_factory=lambda: ["unit", "integration", "e2e"])


class JobStatus(BaseModel):
    """Status of async job"""
    job_id: str
    status: str  # pending, running, completed, failed
    progress: float = 0.0
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: datetime
    updated_at: datetime


@router.post("/ask", response_model=ChatResponse)
async def chat_with_ai(request: ChatRequest):
    """
    Interactive chat with AI developer assistant.
    
    The assistant can:
    - Search the codebase semantically
    - Analyze impact of changes
    - Find components and their dependencies
    - Explain API flows
    - Provide debugging assistance
    """
    try:
        logger.info(
            "Processing chat request",
            question_length=len(request.question),
            has_history=len(request.chat_history) >= 0
        )
        
        # Ensure orchestrator is initialized
        await agent_orchestrator.initialize()
        
        # Convert chat history to list of dicts
        history = [
            {"role": msg.role, "content": msg.content}
            for msg in (request.chat_history or [])
        ]
        
        # Process through developer agent
        result = await agent_orchestrator.process_developer_question(
            question=request.question,
            chat_history=history,
            context_files=request.context_files
        )
        
        # Generate session ID if not provided
        session_id = request.session_id or str(uuid.uuid4())
        
        # Store conversation in Supabase if available
        if supabase_client and result.success:
            try:
                await _store_conversation(
                    session_id=session_id,
                    question=request.question,
                    answer=result.output.get("answer", ""),
                    metadata={
                        "tools_used": result.output.get("tools_used", []),
                        "execution_time": result.execution_time_seconds
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to store conversation: {e}")
        
        # Generate follow-up suggestions
        followups = _generate_followups(request.question, result.output.get("answer", ""))
        
        return ChatResponse(
            status="success" if result.success else "partial",
            answer=result.output.get("answer", "I couldn't process your question. Please try again."),
            confidence=0.9 if result.success else 0.5,
            sources=[],  # TODO: Extract sources from context
            followups=followups,
            tools_used=result.output.get("tools_used", []),
            execution_time_seconds=result.execution_time_seconds,
            session_id=session_id
        )
        
    except Exception as e:
        logger.error(f"Chat failed: {e}", exc_info=True)
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
            _job_store[job_id] = {
                "status": "pending",
                "progress": 0.0,
                "result": None,
                "error": None,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }
            
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
        if job_id not in _job_store:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        
        job = _job_store[job_id]
        
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
        if job_id not in _job_store:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        
        job = _job_store[job_id]
        
        if job["status"] in ["completed", "failed"]:
            return {"status": "already_finished", "job_id": job_id}
        
        job["status"] = "cancelled"
        job["updated_at"] = datetime.utcnow()
        
        return {"status": "cancelled", "job_id": job_id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Job cancellation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_orchestrator_stats():
    """Get orchestrator statistics"""
    try:
        stats = agent_orchestrator.get_stats()
        return {
            "status": "success",
            "stats": stats,
            "active_jobs": len([j for j in _job_store.values() if j["status"] == "running"]),
            "total_jobs": len(_job_store)
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
        _job_store[job_id]["status"] = "running"
        _job_store[job_id]["progress"] = 0.1
        _job_store[job_id]["updated_at"] = datetime.utcnow()
        
        result = await agent_orchestrator.process_prd(
            prd_text=prd_text,
            human_prompt=human_prompt or "",
            repository_filter=repository_filter,
            architectural_constraints=architectural_constraints,
            priority=priority
        )
        
        _job_store[job_id]["status"] = "completed" if result.success else "failed"
        _job_store[job_id]["progress"] = 1.0
        _job_store[job_id]["result"] = {
            "blueprint": result.output.get("blueprint", {}),
            "validation_results": result.output.get("validation_results", []),
            "execution_time_seconds": result.execution_time_seconds
        }
        _job_store[job_id]["error"] = result.errors[0] if result.errors else None
        _job_store[job_id]["updated_at"] = datetime.utcnow()
        
    except Exception as e:
        logger.error(f"Background PRD processing failed: {e}")
        _job_store[job_id]["status"] = "failed"
        _job_store[job_id]["error"] = str(e)
        _job_store[job_id]["updated_at"] = datetime.utcnow()


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
