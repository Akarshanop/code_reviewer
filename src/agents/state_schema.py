"""
Agent State Schemas for LangGraph Workflows
Defines state structures for PRD, QA, and Developer agents.

Each tool-calling agent workflow uses a state with:
    - messages: Annotated list for LangGraph message accumulation
    - Domain-specific fields for tracking context and output
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
    """
    State for PRD Agent workflow (tool-calling LangGraph agent).

    Fields:
        messages: LangGraph-managed message accumulation (Human, AI, Tool)
        client_requirements: Original client requirements text
        business_context: Business background and stakeholders
        constraints: Performance, security, timeline, scale constraints
        retrieved_context: Codebase artifacts found via tools
        prd_output: Final structured PRD JSON output
        repository_filter: Limit analysis to specific repos
        errors: List of errors encountered during processing
    """
    messages: Annotated[Sequence[BaseMessage], add]
    client_requirements: str
    business_context: str
    constraints: str
    retrieved_context: Dict[str, Any]
    prd_output: Dict[str, Any]
    repository_filter: List[str]
    errors: List[str]


class QAState(TypedDict):
    """
    State for QA Agent workflow (tool-calling LangGraph agent).

    Fields:
        messages: LangGraph-managed message accumulation (Human, AI, Tool)
        prd_data: The PRD document that QA is being generated from
        retrieved_context: Codebase artifacts found via tools
        qa_output: Final structured QA Test Plan JSON output
        test_focus: Area of primary testing focus
        test_categories: Which testing types to cover
        errors: List of errors encountered during processing
    """
    messages: Annotated[Sequence[BaseMessage], add]
    prd_data: Dict[str, Any]
    retrieved_context: Dict[str, Any]
    qa_output: Dict[str, Any]
    test_focus: str
    test_categories: List[str]
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
