"""
Enterprise PRD/QA Workflow Endpoint
Provides combined PRD analysis and QA test generation pipeline
"""
import uuid
from typing import Dict, Any, List, Optional
from datetime import datetime
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

from src.core.logger import get_logger
from src.agents.orchestrator import agent_orchestrator

logger = get_logger(__name__)
router = APIRouter()

# Job tracking
_prd_qa_jobs: Dict[str, Dict[str, Any]] = {}


class PRDQARequest(BaseModel):
    """Request for combined PRD + QA analysis"""
    prd_text: str = Field(..., min_length=50, max_length=50000, description="Product Requirements Document")
    human_prompt: Optional[str] = Field(default="", description="Additional instructions")
    repository_filter: Optional[List[str]] = Field(default_factory=list)
    test_focus: Optional[str] = Field(default="all", description="Test focus: all, unit, integration, e2e")
    include_race_conditions: Optional[bool] = Field(default=True, description="Include race condition analysis")
    architectural_constraints: Optional[Dict[str, Any]] = Field(default_factory=dict)
    async_mode: Optional[bool] = Field(default=False)


class PRDQAResponse(BaseModel):
    """Response from PRD + QA analysis"""
    status: str
    job_id: Optional[str] = None
    blueprint: Optional[Dict[str, Any]] = None
    test_strategy: Optional[Dict[str, Any]] = None
    validation_results: List[Dict[str, Any]] = Field(default_factory=list)
    execution_time_seconds: float = 0.0
    errors: List[str] = Field(default_factory=list)


@router.post("/analyze", response_model=PRDQAResponse)
async def analyze_prd_qa(request: PRDQARequest, background_tasks: BackgroundTasks):
    """
    Combined PRD analysis and QA test generation.
    
    This endpoint runs:
    1. PRD Analysis → Architectural Blueprint
    2. QA Generation → Test Strategy based on Blueprint
    
    Returns both outputs in a single response.
    """
    try:
        logger.info(
            "Starting combined PRD/QA analysis",
            prd_length=len(request.prd_text),
            test_focus=request.test_focus
        )
        
        # Ensure orchestrator is initialized
        await agent_orchestrator.initialize()
        
        if request.async_mode:
            # Create background job
            job_id = str(uuid.uuid4())
            
            _prd_qa_jobs[job_id] = {
                "status": "pending",
                "progress": 0.0,
                "blueprint": None,
                "test_strategy": None,
                "errors": [],
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }
            
            background_tasks.add_task(
                _run_prd_qa_pipeline,
                job_id,
                request
            )
            
            return PRDQAResponse(
                status="accepted",
                job_id=job_id,
                errors=[]
            )
        
        # Synchronous processing
        start_time = datetime.utcnow()
        errors = []
        
        # Step 1: PRD Analysis
        prd_result = await agent_orchestrator.process_prd(
            prd_text=request.prd_text,
            human_prompt=request.human_prompt or "",
            repository_filter=request.repository_filter,
            architectural_constraints=request.architectural_constraints
        )
        
        if not prd_result.success:
            return PRDQAResponse(
                status="failed",
                blueprint=prd_result.output,
                errors=prd_result.errors,
                execution_time_seconds=(datetime.utcnow() - start_time).total_seconds()
            )
        
        blueprint = prd_result.output.get("blueprint", {})
        
        # Step 2: QA Generation
        qa_result = await agent_orchestrator.process_qa(
            prd_text=request.prd_text,
            design_blueprint=blueprint,
            test_focus=request.test_focus,
            test_categories=["unit", "integration", "e2e"] if request.test_focus == "all" else [request.test_focus]
        )
        
        execution_time = (datetime.utcnow() - start_time).total_seconds()
        
        if qa_result.errors:
            errors.extend(qa_result.errors)
        
        return PRDQAResponse(
            status="success" if qa_result.success else "partial",
            blueprint=blueprint,
            test_strategy=qa_result.output.get("test_strategy", {}),
            validation_results=prd_result.output.get("validation_results", []) + qa_result.output.get("validation_results", []),
            execution_time_seconds=execution_time,
            errors=errors
        )
        
    except Exception as e:
        logger.error(f"PRD/QA analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.get("/jobs/{job_id}")
async def get_prd_qa_job_status(job_id: str):
    """Get status of async PRD/QA job"""
    if job_id not in _prd_qa_jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    job = _prd_qa_jobs[job_id]
    
    return {
        "job_id": job_id,
        "status": job["status"],
        "progress": job["progress"],
        "blueprint": job.get("blueprint"),
        "test_strategy": job.get("test_strategy"),
        "errors": job["errors"],
        "created_at": job["created_at"].isoformat(),
        "updated_at": job["updated_at"].isoformat()
    }


@router.post("/validate-blueprint")
async def validate_blueprint(blueprint: Dict[str, Any]):
    """Validate an existing architectural blueprint"""
    try:
        validation_results = []
        
        # Check required sections
        required_sections = ["components", "api_changes", "database_changes"]
        for section in required_sections:
            if section not in blueprint:
                validation_results.append({
                    "check": f"required_section_{section}",
                    "status": "failed",
                    "message": f"Missing required section: {section}"
                })
            else:
                validation_results.append({
                    "check": f"required_section_{section}",
                    "status": "passed",
                    "message": f"Section {section} present"
                })
        
        # Check component definitions
        components = blueprint.get("components", [])
        for comp in components:
            if not comp.get("name"):
                validation_results.append({
                    "check": "component_name",
                    "status": "failed",
                    "message": "Component missing name"
                })
            if not comp.get("type"):
                validation_results.append({
                    "check": "component_type",
                    "status": "warning",
                    "message": f"Component {comp.get('name', 'unknown')} missing type"
                })
        
        # Check API changes
        api_changes = blueprint.get("api_changes", [])
        for api in api_changes:
            if not api.get("path") or not api.get("method"):
                validation_results.append({
                    "check": "api_definition",
                    "status": "warning",
                    "message": "API change missing path or method"
                })
        
        passed = sum(1 for r in validation_results if r["status"] == "passed")
        failed = sum(1 for r in validation_results if r["status"] == "failed")
        warnings = sum(1 for r in validation_results if r["status"] == "warning")
        
        return {
            "status": "valid" if failed == 0 else "invalid",
            "summary": {
                "passed": passed,
                "failed": failed,
                "warnings": warnings
            },
            "validation_results": validation_results
        }
        
    except Exception as e:
        logger.error(f"Blueprint validation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/estimate-effort")
async def estimate_implementation_effort(blueprint: Dict[str, Any]):
    """Estimate implementation effort based on blueprint"""
    try:
        # Simple effort estimation based on components
        components = blueprint.get("components", [])
        api_changes = blueprint.get("api_changes", [])
        database_changes = blueprint.get("database_changes", [])
        
        # Base estimates (in hours)
        component_hours = len(components) * 8  # 8 hours per component
        api_hours = len(api_changes) * 4  # 4 hours per API
        db_hours = len(database_changes) * 6  # 6 hours per DB change
        
        # Testing overhead (40% of development time)
        testing_hours = (component_hours + api_hours + db_hours) * 0.4
        
        # Review and integration (20% of development time)
        integration_hours = (component_hours + api_hours + db_hours) * 0.2
        
        total_hours = component_hours + api_hours + db_hours + testing_hours + integration_hours
        
        return {
            "status": "success",
            "estimate": {
                "component_development_hours": component_hours,
                "api_development_hours": api_hours,
                "database_development_hours": db_hours,
                "testing_hours": round(testing_hours, 1),
                "integration_hours": round(integration_hours, 1),
                "total_hours": round(total_hours, 1),
                "estimated_days": round(total_hours / 8, 1),
                "estimated_sprints": round(total_hours / 40, 1)  # 40 hours per sprint
            },
            "breakdown": {
                "components": len(components),
                "api_changes": len(api_changes),
                "database_changes": len(database_changes)
            },
            "confidence": "medium",
            "assumptions": [
                "8 hours per component (average complexity)",
                "4 hours per API endpoint",
                "6 hours per database change",
                "40% testing overhead",
                "20% integration and review overhead"
            ]
        }
        
    except Exception as e:
        logger.error(f"Effort estimation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Background task

async def _run_prd_qa_pipeline(job_id: str, request: PRDQARequest):
    """Background pipeline for PRD + QA analysis"""
    try:
        job = _prd_qa_jobs[job_id]
        job["status"] = "running"
        job["progress"] = 0.1
        job["updated_at"] = datetime.utcnow()
        
        # Step 1: PRD Analysis
        job["progress"] = 0.2
        prd_result = await agent_orchestrator.process_prd(
            prd_text=request.prd_text,
            human_prompt=request.human_prompt or "",
            repository_filter=request.repository_filter,
            architectural_constraints=request.architectural_constraints
        )
        
        if not prd_result.success:
            job["status"] = "failed"
            job["errors"] = prd_result.errors
            job["updated_at"] = datetime.utcnow()
            return
        
        job["blueprint"] = prd_result.output.get("blueprint", {})
        job["progress"] = 0.5
        job["updated_at"] = datetime.utcnow()
        
        # Step 2: QA Generation
        qa_result = await agent_orchestrator.process_qa(
            prd_text=request.prd_text,
            design_blueprint=job["blueprint"],
            test_focus=request.test_focus
        )
        
        job["test_strategy"] = qa_result.output.get("test_strategy", {})
        job["status"] = "completed"
        job["progress"] = 1.0
        job["errors"] = qa_result.errors
        job["updated_at"] = datetime.utcnow()
        
    except Exception as e:
        logger.error(f"Background PRD/QA pipeline failed: {e}")
        _prd_qa_jobs[job_id]["status"] = "failed"
        _prd_qa_jobs[job_id]["errors"].append(str(e))
        _prd_qa_jobs[job_id]["updated_at"] = datetime.utcnow()
