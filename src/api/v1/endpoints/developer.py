"""
Enterprise Developer Assistance Endpoint
Provides code understanding, debugging, module scanning, and PR analysis
"""
from typing import Dict, Any, List, Optional
from datetime import datetime
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.core.logger import get_logger
from src.agents.orchestrator import agent_orchestrator
from src.agents.tools.module_scanner import ModuleScanner
from src.agents.tools.pr_analyzer import PRAnalyzer
from src.services.search_service import SearchService
from src.services.graph_service import GraphService

logger = get_logger(__name__)
router = APIRouter()

# Initialize services
search_service = SearchService()
graph_service = GraphService()


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


@router.post("/ask", response_model=DeveloperResponse)
async def ask_developer(request: DeveloperQuestion):
    """
    Ask the developer assistant a question about the codebase.
    
    The assistant can:
    - Search code semantically
    - Analyze component relationships
    - Explain API flows
    - Debug issues
    - Find similar patterns
    """
    try:
        start_time = datetime.utcnow()
        
        logger.info(
            "Processing developer question",
            question_length=len(request.question),
            context_type=request.context_type
        )
        
        # Ensure orchestrator is initialized
        await agent_orchestrator.initialize()
        
        # Convert history to proper format
        history = [
            {"role": msg.get("role", "user"), "content": msg.get("content", "")}
            for msg in (request.chat_history or [])
        ]
        
        # Process through developer agent
        result = await agent_orchestrator.process_developer_question(
            question=request.question,
            chat_history=history,
            context_files=request.context_files
        )
        
        execution_time = (datetime.utcnow() - start_time).total_seconds()
        
        return DeveloperResponse(
            status="success" if result.success else "partial",
            answer=result.output.get("answer", "I couldn't generate a response."),
            confidence=0.9 if result.success else 0.5,
            sources=[],  # TODO: Extract from tool results
            tools_used=result.output.get("tools_used", []),
            execution_time_seconds=execution_time
        )
        
    except Exception as e:
        logger.error(f"Developer question failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Question processing failed: {str(e)}")


@router.post("/scan/{module_name}")
async def scan_module(module_name: str, request: ModuleScanRequest = None):
    """
    Scan a module for issues, complexity, and improvement suggestions.
    
    Returns:
    - Code quality score
    - Complexity metrics
    - Dependency analysis
    - Improvement suggestions
    """
    try:
        logger.info(f"Starting module scan: {module_name}")
        
        scanner = ModuleScanner()
        
        # Perform scan
        scan_results = await scanner.scan_module(
            module_name=module_name,
            module_path=request.module_path if request else None,
            repository=request.repository if request else None,
            scan_depth=request.scan_depth if request else 2,
            include_dependencies=request.include_dependencies if request else True
        )
        
        return {
            "status": "success",
            "module": module_name,
            "scan_timestamp": datetime.utcnow().isoformat(),
            "metrics": {
                "complexity_score": scan_results.get("complexity_score", 0),
                "quality_score": scan_results.get("quality_score", 0),
                "test_coverage": scan_results.get("test_coverage", "unknown"),
                "lines_of_code": scan_results.get("loc", 0),
                "dependencies_count": len(scan_results.get("dependencies", []))
            },
            "issues": scan_results.get("issues", []),
            "suggestions": scan_results.get("suggestions", []),
            "dependencies": scan_results.get("dependencies", []),
            "files_analyzed": scan_results.get("files_analyzed", 0)
        }
        
    except Exception as e:
        logger.error(f"Module scan failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Module scan failed: {str(e)}")


@router.post("/analyze-pr")
async def analyze_pr(request: PRAnalysisRequest):
    """
    Analyze a pull request for code quality, impact, and risks.
    
    Returns:
    - Quality score
    - Breaking changes
    - Test coverage impact
    - Security concerns
    - Recommendations
    """
    try:
        logger.info(f"Starting PR analysis: {request.repository}#{request.pr_number}")
        
        analyzer = PRAnalyzer()
        
        # Analyze PR
        analysis = await analyzer.analyze_pr(
            pr_number=request.pr_number,
            repository=request.repository,
            include_tests=request.include_test_analysis,
            include_security=request.include_security_scan
        )
        
        return {
            "status": "success",
            "pr_number": request.pr_number,
            "repository": request.repository,
            "analysis_timestamp": datetime.utcnow().isoformat(),
            "summary": {
                "quality_score": analysis.get("quality_score", 0),
                "risk_level": analysis.get("risk_level", "unknown"),
                "files_changed": analysis.get("files_changed", 0),
                "additions": analysis.get("additions", 0),
                "deletions": analysis.get("deletions", 0)
            },
            "breaking_changes": analysis.get("breaking_changes", []),
            "test_analysis": analysis.get("test_analysis", {}) if request.include_test_analysis else None,
            "security_issues": analysis.get("security_issues", []) if request.include_security_scan else None,
            "recommendations": analysis.get("recommendations", []),
            "impacted_components": analysis.get("impacted_components", [])
        }
        
    except Exception as e:
        logger.error(f"PR analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"PR analysis failed: {str(e)}")


@router.get("/search")
async def search_codebase(
    query: str,
    search_type: str = "hybrid",
    top_k: int = 10,
    repository: Optional[str] = None,
    file_type: Optional[str] = None
):
    """
    Search the codebase with semantic and graph-based search.
    
    Search types:
    - hybrid: Combined semantic + graph search (default)
    - semantic: Vector similarity search
    - graph: Relationship-based search
    """
    try:
        filters = {}
        if repository:
            filters["repository"] = repository
        if file_type:
            filters["type"] = file_type
        
        results = await search_service.search_codebase(
            query=query,
            search_type=search_type,
            top_k=top_k,
            filters=filters
        )
        
        return {
            "status": "success",
            "query": query,
            "search_type": search_type,
            "results": results.get("results", {}).get("combined", []),
            "total_results": results.get("summary", {}).get("combined_results", 0),
            "duration_ms": results.get("duration_ms", 0)
        }
        
    except Exception as e:
        logger.error(f"Search failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.get("/component/{component_name}")
async def get_component_info(component_name: str, include_related: bool = True):
    """
    Get detailed information about a specific component.
    
    Returns:
    - Component definition
    - Dependencies
    - Dependents
    - Related components
    - Usage examples
    """
    try:
        result = await graph_service.find_related_entities(
            entity_name=component_name,
            max_depth=2 if include_related else 1
        )
        
        return {
            "status": "success",
            "component": component_name,
            "exists": bool(result.get("entity")),
            "entity": result.get("entity"),
            "relationships": result.get("relationships", {}),
            "related_count": sum(len(v) for v in result.get("relationships", {}).values())
        }
        
    except Exception as e:
        logger.error(f"Component lookup failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Component lookup failed: {str(e)}")


@router.get("/api-flow/{endpoint_path:path}")
async def get_api_flow(endpoint_path: str, include_eventbus: bool = True):
    """
    Trace the complete flow of an API endpoint.
    
    Returns:
    - Route definition
    - Handler chain
    - Service calls
    - EventBus interactions
    - Database operations
    """
    try:
        # Normalize path
        if not endpoint_path.startswith("/"):
            endpoint_path = "/" + endpoint_path
        
        result = await graph_service.trace_api_flow(
            endpoint=endpoint_path,
            include_eventbus=include_eventbus
        )
        
        return {
            "status": "success",
            "endpoint": endpoint_path,
            "flow": result.get("flow", []),
            "eventbus_interactions": result.get("eventbus", []) if include_eventbus else None,
            "total_steps": len(result.get("flow", []))
        }
        
    except Exception as e:
        logger.error(f"API flow trace failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"API flow trace failed: {str(e)}")
