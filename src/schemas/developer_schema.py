from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional

class DeveloperQuestion(BaseModel):
    """Request for developer assistant"""
    question: str = Field(..., min_length=3, max_length=5000)
    chat_history: Optional[List[Dict[str, str]]] = Field(default_factory=list)
    context_type: Optional[str] = Field(default="auto", description="Context type: auto, code, architecture, debug")
    repository_filter: Optional[List[str]] = Field(default_factory=list)
    context_files: Optional[List[str]] = Field(default_factory=list, description="Specific files for context")


class DeveloperResponse(BaseModel):
    """Response from developer assistant"""
    status: str
    answer: str
    confidence: float
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    tools_used: List[str] = Field(default_factory=list)
    execution_time_seconds: float = 0.0


class ModuleScanRequest(BaseModel):
    """Request for module scanning"""
    module_path: str = Field(..., description="Path to module to scan")
    repository: Optional[str] = Field(default=None)
    scan_depth: Optional[int] = Field(default=2, ge=1, le=5)
    include_dependencies: Optional[bool] = Field(default=True)


class PRAnalysisRequest(BaseModel):
    """Request for PR analysis"""
    pr_number: int = Field(..., gt=0)
    repository: str = Field(...)
    include_test_analysis: Optional[bool] = Field(default=True)
    include_security_scan: Optional[bool] = Field(default=True)
