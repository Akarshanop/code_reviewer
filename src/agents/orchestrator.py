"""
Enterprise Agent Orchestrator for Code Intelligence Platform
Manages PRD, QA, and Developer agent workflows with proper error handling and monitoring

Enhanced with:
    - Two-layer architecture (Reasoning Layer → Policy Layer)
    - Role-based output shaping
    - Trust-preserving silence
    - Observable reasoning traces
"""
import asyncio
from typing import Dict, Any, List, Optional
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
import json
import traceback

from langchain_core.messages import HumanMessage

from src.core.logger import get_logger
from src.core.exceptions import AgentException
from src.agents.workflows.prd_flow import prd_workflow
from src.agents.workflows.qa_flow import qa_workflow
from src.agents.workflows.developer_flow import developer_app
from src.agents.policy_layer import policy_layer, PolicyOutput, UserRole
from src.agents.reasoning_layer import reasoning_layer, ReasoningResult, ReasoningConclusion, ReasoningEvidence
from config.settings import settings

logger = get_logger(__name__)


class AgentType(Enum):
    """Supported agent types"""
    PRD = "prd"
    QA = "qa"
    DEVELOPER = "developer"


@dataclass
class AgentConfig:
    """Configuration for agent execution"""
    max_tokens: int = 4096
    temperature: float = 0.2
    model: str = "gemini-2.5-flash"
    timeout_seconds: int = 300
    max_iterations: int = 10


@dataclass
class AgentResult:
    """Result from agent execution"""
    success: bool
    agent_type: AgentType
    output: Dict[str, Any]
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    execution_time_seconds: float = 0.0


class AgentOrchestrator:
    """
    Enterprise orchestrator for managing AI agent workflows.
    Handles PRD analysis, QA generation, and developer assistance.
    """
    
    def __init__(self):
        self.semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_AGENTS)
        self._initialized = False
        self._stats = {
            "total_executions": 0,
            "successful": 0,
            "failed": 0,
            "by_agent_type": {
                AgentType.PRD.value: {"total": 0, "success": 0},
                AgentType.QA.value: {"total": 0, "success": 0},
                AgentType.DEVELOPER.value: {"total": 0, "success": 0}
            }
        }
    
    async def initialize(self):
        """Initialize the orchestrator and verify workflows are compiled"""
        if self._initialized:
            return
        
        try:
            # Verify workflows are available
            assert prd_workflow is not None, "PRD workflow not compiled"
            assert qa_workflow is not None, "QA workflow not compiled"
            assert developer_app is not None, "Developer workflow not compiled"
            
            self._initialized = True
            logger.info(
                "Agent orchestrator initialized",
                max_concurrent=settings.MAX_CONCURRENT_AGENTS
            )
        except Exception as e:
            logger.error(f"Failed to initialize orchestrator: {e}")
            raise
    
    async def process_prd(
        self,
        prd_text: str,
        human_prompt: str = "",
        repository_filter: List[str] = None,
        architectural_constraints: Dict[str, Any] = None,
        priority: str = "medium"
    ) -> AgentResult:
        """
        Process a PRD and generate architectural blueprint.
        
        Args:
            prd_text: The Product Requirements Document text
            human_prompt: Additional instructions from the user
            repository_filter: List of repositories to focus on
            architectural_constraints: Constraints for the architecture
            priority: Priority level (low, medium, high)
        
        Returns:
            AgentResult with the architectural blueprint
        """
        start_time = datetime.utcnow()
        
        async with self.semaphore:
            self._stats["total_executions"] += 1
            self._stats["by_agent_type"][AgentType.PRD.value]["total"] += 1
            
            try:
                logger.info(
                    "Starting PRD analysis",
                    prd_length=len(prd_text),
                    has_constraints=bool(architectural_constraints)
                )
                
                # Prepare initial state
                initial_state = {
                    "prd_text": prd_text,
                    "human_prompt": human_prompt or "",
                    "retrieved_context": {},
                    "analysis_output": {},
                    "repository_filter": repository_filter or [],
                    "priority": priority,
                    "architectural_constraints": architectural_constraints or {},
                    "validation_results": [],
                    "errors": []
                }
                
                try:
                    result = await asyncio.wait_for(
                        prd_workflow.ainvoke(initial_state),
                        timeout=settings.AGENT_TIMEOUT_SECONDS
                    )
                except asyncio.TimeoutError:
                    raise AgentException("PRD analysis timed out")
                
                execution_time = (datetime.utcnow() - start_time).total_seconds()
                
                # Check for errors
                errors = result.get("errors", [])
                has_output = bool(result.get("analysis_output"))
                
                if has_output and not any("error" in str(result.get("analysis_output", {})).lower()):
                    self._stats["successful"] += 1
                    self._stats["by_agent_type"][AgentType.PRD.value]["success"] += 1
                    
                    logger.info(
                        "PRD analysis completed successfully",
                        execution_time=execution_time,
                        validation_checks=len(result.get("validation_results", []))
                    )
                    
                    return AgentResult(
                        success=True,
                        agent_type=AgentType.PRD,
                        output={
                            "blueprint": result.get("analysis_output", {}),
                            "validation_results": result.get("validation_results", []),
                            "context_sources": len(result.get("retrieved_context", {}))
                        },
                        errors=errors,
                        metadata={
                            "prd_length": len(prd_text),
                            "repositories": repository_filter or [],
                            "priority": priority
                        },
                        execution_time_seconds=execution_time
                    )
                else:
                    self._stats["failed"] += 1
                    
                    return AgentResult(
                        success=False,
                        agent_type=AgentType.PRD,
                        output=result.get("analysis_output", {}),
                        errors=errors or ["Analysis produced no valid output"],
                        execution_time_seconds=execution_time
                    )
                
            except AgentException as e:
                self._stats["failed"] += 1
                execution_time = (datetime.utcnow() - start_time).total_seconds()
                
                logger.error("PRD analysis failed", error=str(e))
                
                return AgentResult(
                    success=False,
                    agent_type=AgentType.PRD,
                    output={},
                    errors=[str(e)],
                    execution_time_seconds=execution_time
                )
            except Exception as e:
                self._stats["failed"] += 1
                execution_time = (datetime.utcnow() - start_time).total_seconds()
                
                logger.error(
                    "PRD analysis failed with unexpected error",
                    error=str(e),
                    traceback=traceback.format_exc()
                )
                
                return AgentResult(
                    success=False,
                    agent_type=AgentType.PRD,
                    output={},
                    errors=[f"Unexpected error: {str(e)}"],
                    execution_time_seconds=execution_time
                )
    
    async def process_qa(
        self,
        prd_text: str,
        design_blueprint: Dict[str, Any],
        test_focus: str = "all",
        test_categories: List[str] = None
    ) -> AgentResult:
        """
        Generate comprehensive QA test strategy.
        
        Args:
            prd_text: The PRD text for context
            design_blueprint: The architectural blueprint from PRD analysis
            test_focus: Focus area for tests (all, unit, integration, e2e)
            test_categories: Specific categories to generate tests for
        
        Returns:
            AgentResult with the test strategy
        """
        start_time = datetime.utcnow()
        
        async with self.semaphore:
            self._stats["total_executions"] += 1
            self._stats["by_agent_type"][AgentType.QA.value]["total"] += 1
            
            try:
                logger.info(
                    "Starting QA test generation",
                    test_focus=test_focus,
                    blueprint_keys=list(design_blueprint.keys()) if design_blueprint else []
                )
                
                # Prepare initial state
                initial_state = {
                    "prd_text": prd_text,
                    "human_prompt": "",
                    "design_blueprint": design_blueprint,
                    "retrieved_context": {},
                    "analysis_output": {},
                    "test_focus": test_focus,
                    "test_categories": test_categories or ["unit", "integration", "e2e"],
                    "validation_results": [],
                    "errors": []
                }
                
                # Execute workflow with timeout
                try:
                    result = await asyncio.wait_for(
                        qa_workflow.ainvoke(initial_state),
                        timeout=settings.AGENT_TIMEOUT_SECONDS
                    )
                except asyncio.TimeoutError:
                    raise AgentException("QA generation timed out")
                
                execution_time = (datetime.utcnow() - start_time).total_seconds()
                
                # Check for errors
                errors = result.get("errors", [])
                has_output = bool(result.get("analysis_output"))
                
                if has_output and "error" not in str(result.get("analysis_output", {})).lower():
                    self._stats["successful"] += 1
                    self._stats["by_agent_type"][AgentType.QA.value]["success"] += 1
                    
                    test_strategy = result.get("analysis_output", {})
                    
                    logger.info(
                        "QA test generation completed",
                        execution_time=execution_time,
                        total_tests=test_strategy.get("summary", {}).get("total_test_cases", 0)
                    )
                    
                    return AgentResult(
                        success=True,
                        agent_type=AgentType.QA,
                        output={
                            "test_strategy": test_strategy,
                            "validation_results": result.get("validation_results", [])
                        },
                        errors=errors,
                        metadata={
                            "test_focus": test_focus,
                            "categories": test_categories
                        },
                        execution_time_seconds=execution_time
                    )
                else:
                    self._stats["failed"] += 1
                    
                    return AgentResult(
                        success=False,
                        agent_type=AgentType.QA,
                        output=result.get("analysis_output", {}),
                        errors=errors or ["QA generation produced no valid output"],
                        execution_time_seconds=execution_time
                    )
                
            except AgentException as e:
                self._stats["failed"] += 1
                execution_time = (datetime.utcnow() - start_time).total_seconds()
                
                logger.error("QA generation failed", error=str(e))
                
                return AgentResult(
                    success=False,
                    agent_type=AgentType.QA,
                    output={},
                    errors=[str(e)],
                    execution_time_seconds=execution_time
                )
            except Exception as e:
                self._stats["failed"] += 1
                execution_time = (datetime.utcnow() - start_time).total_seconds()
                
                logger.error(
                    "QA generation failed with unexpected error",
                    error=str(e),
                    traceback=traceback.format_exc()
                )
                
                return AgentResult(
                    success=False,
                    agent_type=AgentType.QA,
                    output={},
                    errors=[f"Unexpected error: {str(e)}"],
                    execution_time_seconds=execution_time
                )
    
    async def process_developer_question(
        self,
        question: str,
        chat_history: List[Dict[str, str]] = None,
        context_files: List[str] = None
    ) -> AgentResult:
        """
        Process a developer question with tool-assisted reasoning.
        
        Args:
            question: The developer's question
            chat_history: Previous messages in the conversation
            context_files: Files to use as context
        
        Returns:
            AgentResult with the response
        """
        start_time = datetime.utcnow()
        
        async with self.semaphore:
            self._stats["total_executions"] += 1
            self._stats["by_agent_type"][AgentType.DEVELOPER.value]["total"] += 1
            
            try:
                logger.info(
                    "Processing developer question",
                    question_length=len(question),
                    history_length=len(chat_history) if chat_history else 0
                )
                
                # Build messages list
                messages = []
                
                # Add chat history
                if chat_history:
                    for msg in chat_history:
                        role = msg.get("role", "user")
                        content = msg.get("content", "")
                        
                        if role == "user":
                            messages.append(HumanMessage(content=content))
                        elif role == "assistant":
                            from langchain_core.messages import AIMessage
                            messages.append(AIMessage(content=content))
                
                # Add current question
                messages.append(HumanMessage(content=question))
                
                # Prepare initial state
                initial_state = {
                    "messages": messages,
                    "retrieved_context": {},
                    "tool_results": {},
                    "final_answer": "",
                    "current_task": question,
                    "context_files": context_files or [],
                    "errors": []
                }
                
                # Execute workflow with timeout and increased recursion limit
                try:
                    result = await asyncio.wait_for(
                        developer_app.ainvoke(
                            initial_state,
                            config={"recursion_limit": 50}
                        ),
                        timeout=settings.AGENT_TIMEOUT_SECONDS
                    )
                except asyncio.TimeoutError:
                    raise AgentException("Developer assistant timed out")
                
                execution_time = (datetime.utcnow() - start_time).total_seconds()
                
                # Extract response from messages
                final_messages = result.get("messages", [])
                response_content = ""
                
                if final_messages:
                    last_message = final_messages[-1]
                    if hasattr(last_message, 'content'):
                        response_content = last_message.content
                
                if response_content:
                    self._stats["successful"] += 1
                    self._stats["by_agent_type"][AgentType.DEVELOPER.value]["success"] += 1
                    
                    logger.info(
                        "Developer question processed",
                        execution_time=execution_time,
                        response_length=len(response_content)
                    )
                    
                    return AgentResult(
                        success=True,
                        agent_type=AgentType.DEVELOPER,
                        output={
                            "answer": response_content,
                            "messages_count": len(final_messages),
                            "tools_used": self._extract_tools_used(final_messages)
                        },
                        errors=[],
                        metadata={
                            "question_length": len(question),
                            "context_files": context_files
                        },
                        execution_time_seconds=execution_time
                    )
                else:
                    self._stats["failed"] += 1
                    
                    return AgentResult(
                        success=False,
                        agent_type=AgentType.DEVELOPER,
                        output={"answer": "I couldn't generate a response. Please try rephrasing your question."},
                        errors=["No response generated"],
                        execution_time_seconds=execution_time
                    )
                
            except AgentException as e:
                self._stats["failed"] += 1
                execution_time = (datetime.utcnow() - start_time).total_seconds()
                
                logger.error("Developer assistant failed", error=str(e))
                
                return AgentResult(
                    success=False,
                    agent_type=AgentType.DEVELOPER,
                    output={"answer": f"Error: {str(e)}"},
                    errors=[str(e)],
                    execution_time_seconds=execution_time
                )
            except Exception as e:
                self._stats["failed"] += 1
                execution_time = (datetime.utcnow() - start_time).total_seconds()
                
                logger.error(
                    "Developer assistant failed with unexpected error",
                    error=str(e),
                    traceback=traceback.format_exc()
                )
                
                return AgentResult(
                    success=False,
                    agent_type=AgentType.DEVELOPER,
                    output={"answer": "An unexpected error occurred. Please try again."},
                    errors=[f"Unexpected error: {str(e)}"],
                    execution_time_seconds=execution_time
                )
    
    def _extract_tools_used(self, messages: List) -> List[str]:
        """Extract list of tools used from messages"""
        tools_used = []
        
        for msg in messages:
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_name = tc.get("name", "") if isinstance(tc, dict) else getattr(tc, 'name', '')
                    if tool_name and tool_name not in tools_used:
                        tools_used.append(tool_name)
        
        return tools_used
    
    async def process_with_policy(
        self,
        question: str,
        user_role: str = "developer",
        chat_history: List[Dict[str, str]] = None,
        context_files: List[str] = None,
    ) -> Dict[str, Any]:
        """
        Two-layer processing: Reasoning → Policy.
        
        This is the recommended entry point for all agent interactions.
        It applies RBAC and output shaping based on user role.
        
        Args:
            question: User's question.
            user_role: User's role (admin, architect, developer, qa, sales, viewer).
            chat_history: Previous messages.
            context_files: Context files.
            
        Returns:
            PolicyOutput as dict — role-appropriate, redacted, confidence-scored.
        """
        # Step 1: Reasoning Layer (full graph access)
        agent_result = await self.process_developer_question(
            question=question,
            chat_history=chat_history,
            context_files=context_files,
        )

        raw_answer = agent_result.output.get("answer", "")
        tools_used = agent_result.output.get("tools_used", [])

        # Step 2: Build ReasoningResult from agent output with REAL confidence
        if agent_result.success and raw_answer:
            # Compute confidence from evidence quality, not a hardcoded value
            tool_count = len(tools_used)
            has_graph_tools = any(t in tools_used for t in [
                "get_repository_architecture", "find_component", "get_api_flow",
                "get_behavioral_classification", "get_security_surface",
                "get_domain_context", "trace_cross_repo_flow", "get_file_dependencies",
                "get_qa_risk_assessment", "get_codebase_metrics",
            ])
            has_code_tools = any(t in tools_used for t in [
                "read_file_content", "search_code_pattern", "list_api_routes",
            ])
            has_search = "search_codebase" in tools_used

            # Confidence scoring: tools that read actual code/graph = higher trust
            if has_code_tools and has_graph_tools:
                base_confidence = 0.90  # Both code AND graph evidence
            elif has_code_tools:
                base_confidence = 0.85  # Read actual code
            elif has_graph_tools:
                base_confidence = 0.80  # Graph-backed evidence
            elif has_search:
                base_confidence = 0.70  # Search-backed
            elif tool_count > 0:
                base_confidence = 0.65  # Some tool used
            else:
                base_confidence = 0.45  # No tools = LLM-only (low trust)

            # Penalize very short answers (likely incomplete)
            if len(raw_answer) < 100:
                base_confidence *= 0.8

            # Build evidence list from tools
            evidence_items = []
            for t in tools_used:
                source = "graph_traversal" if t in (
                    "get_repository_architecture", "find_component", "get_codebase_metrics",
                    "get_behavioral_classification", "get_security_surface",
                    "get_domain_context", "trace_cross_repo_flow", "get_file_dependencies",
                    "get_qa_risk_assessment",
                ) else "code_analysis" if t in (
                    "read_file_content", "search_code_pattern", "list_api_routes",
                ) else "vector_search" if t == "search_codebase" else "tool_call"

                evidence_items.append(ReasoningEvidence(
                    source=source,
                    description=f"Data from {t}",
                    confidence=base_confidence,
                ))

            if not evidence_items:
                evidence_items.append(ReasoningEvidence(
                    source="llm_generation",
                    description="Generated without tool evidence (lower trust)",
                    confidence=base_confidence,
                ))

            # Extract affected files from the answer (simple heuristic)
            import re as _re
            file_mentions = _re.findall(r"[\w/]+\.(?:js|ts|py|java|jsx|tsx)\b", raw_answer)

            # Domain resolution on the question
            from src.intelligence.domain_model import domain_engine as _de
            domain_matches = _de.resolve_query_to_domains(question)
            affected_domains = [d[0] for d in domain_matches[:3]]

            conclusions = [
                ReasoningConclusion(
                    statement=raw_answer[:300],
                    confidence=base_confidence,
                    evidence=evidence_items,
                    domain_context=affected_domains,
                )
            ]
            reasoning_result = reasoning_layer.build_result(
                conclusions=conclusions,
                tools_used=tools_used,
                affected_files=list(set(file_mentions))[:20],
                affected_domains=affected_domains,
                execution_time_seconds=agent_result.execution_time_seconds,
            )
        else:
            reasoning_result = reasoning_layer.create_silence_result(
                reason=agent_result.errors[0] if agent_result.errors else "No response generated"
            )

        # Step 3: Policy Layer (RBAC, redaction, shaping)
        policy_output = policy_layer.apply_policy(
            reasoning=reasoning_result,
            role=user_role,
            raw_answer=raw_answer,
        )

        return {
            "answer": policy_output.answer,
            "confidence": policy_output.confidence_label,
            "risk_level": policy_output.risk_level,
            "affected_domains": policy_output.affected_domains,
            "impact_summary": policy_output.impact_summary,
            "affected_files": policy_output.affected_files,
            "evidence": policy_output.evidence_summary,
            "uncertainty": policy_output.uncertainty_notes,
            "is_silence": policy_output.is_silence,
            "role_applied": policy_output.role_applied,
            "tools_used": tools_used,
            "execution_time_seconds": agent_result.execution_time_seconds,
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get orchestrator statistics"""
        return {
            **self._stats,
            "success_rate": (
                self._stats["successful"] / self._stats["total_executions"] * 100
                if self._stats["total_executions"] > 0 else 0
            ),
            "reasoning_layer": reasoning_layer.stats,
            "policy_layer": policy_layer.stats,
        }


# Global orchestrator instance
agent_orchestrator = AgentOrchestrator()