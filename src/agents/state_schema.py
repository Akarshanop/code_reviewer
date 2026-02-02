"""
Agent State Schemas for LangGraph Workflows
Defines state structures for PRD, QA, and Developer agents
"""
from typing import TypedDict, List, Dict, Any, Optional, Annotated, Sequence
from langchain_core.messages import BaseMessage
from operator import add


class AgentState(TypedDict):
    """Base state for all agents"""
    prd_text: str
    human_prompt: str
    retrieved_context: Dict[str, Any]
    analysis_output: Dict[str, Any]
    errors: List[str]
    metadata: Dict[str, Any]


class PRDState(TypedDict):
    """State for PRD analysis workflow"""
    prd_text: str
    human_prompt: str
    retrieved_context: Dict[str, Any]
    analysis_output: Dict[str, Any]
    repository_filter: List[str]
    priority: str
    architectural_constraints: Dict[str, Any]
    validation_results: List[Dict[str, Any]]
    errors: List[str]


class QAState(TypedDict):
    """State for QA test generation workflow"""
    prd_text: str
    human_prompt: str
    design_blueprint: Dict[str, Any]
    retrieved_context: Dict[str, Any]
    analysis_output: Dict[str, Any]
    test_focus: str
    test_categories: List[str]
    validation_results: List[Dict[str, Any]]
    errors: List[str]


class DeveloperState(TypedDict):
    """State for Developer assistant workflow"""
    messages: Annotated[Sequence[BaseMessage], add]
    retrieved_context: Dict[str, Any]
    tool_results: Dict[str, Any]
    final_answer: str
    current_task: str
    context_files: List[str]
    errors: List[str]


class ImpactAnalysisState(TypedDict):
    """State for impact analysis workflow"""
    file_path: str
    change_description: str
    change_type: str  # modify, delete, add
    affected_components: List[Dict[str, Any]]
    risk_assessment: Dict[str, Any]
    recommendations: List[str]
    errors: List[str]


class FeatureEstimationState(TypedDict):
    """State for feature estimation workflow"""
    feature_description: str
    similar_features: List[Dict[str, Any]]
    complexity_factors: Dict[str, Any]
    effort_estimate: Dict[str, Any]
    confidence_score: float
    errors: List[str]