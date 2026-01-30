from typing import TypedDict, List, Dict, Any, Optional
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    prd_text: str
    human_prompt: str
    retrieved_context: Dict[str, Any]
    analysis_output: Dict[str, Any]

class DeveloperState(TypedDict):
    messages: List[BaseMessage]
    retrieved_context: Dict[str, Any]
    tool_results: Dict[str, Any]
    final_answer: str

class QAState(AgentState):
    design_blueprint: Dict[str, Any]

class PRDState(AgentState):
    pass