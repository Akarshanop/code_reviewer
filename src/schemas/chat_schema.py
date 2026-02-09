from datetime import datetime
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional


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