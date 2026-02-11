"""
Enterprise Agent Orchestrator for Code Intelligence Platform
Manages PRD, QA, Developer, and Sales agent workflows with proper error handling and monitoring

Enhanced with:
    - Two-layer architecture (Reasoning Layer → Policy Layer) for ALL agent types
    - Role-based routing (sales → sales_workflow, qa → qa_workflow, etc.)
    - Trust-preserving silence
    - Observable reasoning traces
"""
import asyncio
from typing import Dict, Any, List, Optional
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
import json
import re
import traceback

from langchain_core.messages import HumanMessage, AIMessage

from src.core.logger import get_logger
from src.core.exceptions import AgentException, LLMServiceException
from src.agents.workflows.prd_flow import prd_workflow
from src.agents.workflows.qa_flow import qa_workflow
from src.agents.workflows.developer_flow import developer_app
from src.agents.workflows.sales_flow import sales_workflow
from src.agents.policy_layer import policy_layer, PolicyOutput, UserRole
from src.agents.reasoning_layer import reasoning_layer, ReasoningResult, ReasoningConclusion, ReasoningEvidence
from config.settings import settings

logger = get_logger(__name__)


class AgentType(Enum):
    """Supported agent types"""
    PRD = "prd"
    QA = "qa"
    DEVELOPER = "developer"
    SALES = "sales"


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
    Handles PRD analysis, QA generation, developer assistance, and sales support.
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
                AgentType.DEVELOPER.value: {"total": 0, "success": 0},
                AgentType.SALES.value: {"total": 0, "success": 0},
            }
        }
    
    async def initialize(self):
        """Initialize the orchestrator and verify workflows are compiled"""
        if self._initialized:
            return
        
        try:
            assert prd_workflow is not None, "PRD workflow not compiled"
            assert qa_workflow is not None, "QA workflow not compiled"
            assert developer_app is not None, "Developer workflow not compiled"
            assert sales_workflow is not None, "Sales workflow not compiled"
            
            self._initialized = True
            logger.info(
                "Agent orchestrator initialized",
                max_concurrent=settings.MAX_CONCURRENT_AGENTS,
                workflows=["prd", "qa", "developer", "sales"],
            )
        except Exception as e:
            logger.error(f"Failed to initialize orchestrator: {e}")
            raise
    
    # ------------------------------------------------------------------
    # Internal helper: run a message-based workflow
    # ------------------------------------------------------------------

    async def _run_message_workflow(
        self,
        workflow,
        agent_type: AgentType,
        question: str,
        chat_history: List[Dict[str, str]] = None,
        context_files: List[str] = None,
        recursion_limit: int = 50,
        extra_state: Dict[str, Any] = None,
    ) -> AgentResult:
        """
        Shared runner for any message-based workflow.
        Supports DeveloperState, PRDState, QAState — builds initial state
        based on agent_type.
        """
        start_time = datetime.utcnow()

        async with self.semaphore:
            self._stats["total_executions"] += 1
            self._stats["by_agent_type"][agent_type.value]["total"] += 1

            try:
                # Build messages list
                messages = []
                if chat_history:
                    for msg in chat_history:
                        role = msg.get("role", "user")
                        content = msg.get("content", "")
                        if role == "user":
                            messages.append(HumanMessage(content=content))
                        elif role == "assistant":
                            messages.append(AIMessage(content=content))

                messages.append(HumanMessage(content=question))

                # Build initial state based on agent type
                if agent_type == AgentType.PRD:
                    initial_state = {
                        "messages": messages,
                        "client_requirements": (extra_state or {}).get("client_requirements", question),
                        "business_context": (extra_state or {}).get("business_context", ""),
                        "constraints": (extra_state or {}).get("constraints", ""),
                        "retrieved_context": {},
                        "prd_output": {},
                        "repository_filter": [],
                        "errors": [],
                    }
                elif agent_type == AgentType.QA:
                    initial_state = {
                        "messages": messages,
                        "prd_data": (extra_state or {}).get("prd_data", {}),
                        "retrieved_context": {},
                        "qa_output": {},
                        "test_focus": "",
                        "test_categories": ["unit", "integration", "system", "regression", "performance"],
                        "errors": [],
                    }
                else:
                    # DeveloperState / SalesState
                    initial_state = {
                        "messages": messages,
                        "retrieved_context": {},
                        "tool_results": {},
                        "final_answer": "",
                        "current_task": question,
                        "context_files": context_files or [],
                        "errors": [],
                    }

                try:
                    result = await asyncio.wait_for(
                        workflow.ainvoke(
                            initial_state,
                            config={"recursion_limit": recursion_limit}
                        ),
                        timeout=settings.AGENT_TIMEOUT_SECONDS,
                    )
                except asyncio.TimeoutError:
                    raise AgentException(f"{agent_type.value} agent timed out")

                execution_time = (datetime.utcnow() - start_time).total_seconds()

                # Extract response
                final_messages = result.get("messages", [])
                response_content = ""
                if final_messages:
                    last_message = final_messages[-1]
                    if hasattr(last_message, 'content'):
                        response_content = last_message.content

                # Fallback: if last message has no content, search backwards
                # through AI messages for the last one with content
                if not response_content and final_messages:
                    for msg in reversed(final_messages):
                        if isinstance(msg, AIMessage) and msg.content and msg.content.strip():
                            response_content = msg.content
                            logger.info(
                                f"Using fallback: found content in earlier AI message",
                                content_length=len(response_content),
                            )
                            break

                logger.info(
                    f"{agent_type.value} workflow completed",
                    response_length=len(response_content),
                    total_messages=len(final_messages),
                )

                if response_content:
                    self._stats["successful"] += 1
                    self._stats["by_agent_type"][agent_type.value]["success"] += 1

                    return AgentResult(
                        success=True,
                        agent_type=agent_type,
                        output={
                            "answer": response_content,
                            "messages_count": len(final_messages),
                            "tools_used": self._extract_tools_used(final_messages),
                        },
                        errors=[],
                        metadata={"question_length": len(question)},
                        execution_time_seconds=execution_time,
                    )
                else:
                    self._stats["failed"] += 1
                    logger.warning(
                        f"{agent_type.value} returned empty response",
                        message_types=[type(m).__name__ for m in final_messages[-5:]],
                    )
                    return AgentResult(
                        success=False,
                        agent_type=agent_type,
                        output={"answer": "No response generated."},
                        errors=["No response generated"],
                        execution_time_seconds=execution_time,
                    )

            except (AgentException, LLMServiceException) as e:
                self._stats["failed"] += 1
                execution_time = (datetime.utcnow() - start_time).total_seconds()
                logger.error(f"{agent_type.value} agent failed", error=str(e))
                return AgentResult(
                    success=False,
                    agent_type=agent_type,
                    output={"answer": f"Error: {str(e)}"},
                    errors=[str(e)],
                    execution_time_seconds=execution_time,
                )
            except Exception as e:
                self._stats["failed"] += 1
                execution_time = (datetime.utcnow() - start_time).total_seconds()
                logger.error(
                    f"{agent_type.value} agent failed unexpectedly",
                    error=str(e),
                    traceback=traceback.format_exc(),
                )
                return AgentResult(
                    success=False,
                    agent_type=agent_type,
                    output={"answer": "An unexpected error occurred."},
                    errors=[f"Unexpected error: {str(e)}"],
                    execution_time_seconds=execution_time,
                )

    # ------------------------------------------------------------------
    # Public APIs: process_prd, process_qa, process_developer_question
    # ------------------------------------------------------------------

    async def process_prd(
        self,
        prd_text: str,
        human_prompt: str = "",
        repository_filter: List[str] = None,
        architectural_constraints: Dict[str, Any] = None,
        priority: str = "medium",
    ) -> AgentResult:
        """
        Process a PRD and generate architectural blueprint.
        Now uses a tool-calling agent that queries the graph.
        """
        # Build the user prompt that includes the PRD
        user_prompt_parts = [f"## PRD Document:\n{prd_text}"]
        if human_prompt:
            user_prompt_parts.append(f"\n## Additional Instructions:\n{human_prompt}")
        if repository_filter:
            user_prompt_parts.append(f"\n## Focus Repositories: {', '.join(repository_filter)}")
        if architectural_constraints:
            user_prompt_parts.append(
                f"\n## Architectural Constraints:\n{json.dumps(architectural_constraints, indent=2)}"
            )
        user_prompt_parts.append(f"\n## Priority: {priority}")
        user_prompt_parts.append(
            "\n\nAnalyze this PRD against the codebase and produce a comprehensive "
            "implementation blueprint. Use your tools to investigate the codebase."
        )

        full_prompt = "\n".join(user_prompt_parts)

        agent_result = await self._run_message_workflow(
            workflow=prd_workflow,
            agent_type=AgentType.PRD,
            question=full_prompt,
            recursion_limit=50,
        )

        # Try to parse JSON from the agent's answer for backward compatibility
        if agent_result.success:
            raw_answer = agent_result.output.get("answer", "")
            blueprint = self._extract_json_from_answer(raw_answer)
            if blueprint:
                agent_result.output["blueprint"] = blueprint
                agent_result.output["validation_results"] = []
                agent_result.output["context_sources"] = len(
                    agent_result.output.get("tools_used", [])
                )
            else:
                # If no JSON, wrap the text as a blueprint
                agent_result.output["blueprint"] = {
                    "summary": {"architecture_approach": raw_answer[:500]},
                    "raw_analysis": raw_answer,
                }
                agent_result.output["validation_results"] = []
                agent_result.output["context_sources"] = 0

        return agent_result

    async def process_qa(
        self,
        prd_text: str,
        design_blueprint: Dict[str, Any],
        test_focus: str = "all",
        test_categories: List[str] = None,
    ) -> AgentResult:
        """
        Generate comprehensive QA test strategy.
        Now uses a tool-calling agent that queries the graph.
        """
        user_prompt_parts = []
        if prd_text:
            user_prompt_parts.append(f"## PRD Summary:\n{prd_text[:3000]}")
        if design_blueprint:
            user_prompt_parts.append(
                f"## Architectural Blueprint:\n{json.dumps(design_blueprint, indent=2, default=str)[:4000]}"
            )
        user_prompt_parts.append(f"\n## Test Focus: {test_focus}")
        if test_categories:
            user_prompt_parts.append(f"## Test Categories: {', '.join(test_categories)}")
        user_prompt_parts.append(
            "\n\nGenerate a comprehensive testing strategy. "
            "Use your tools to investigate the codebase first."
        )

        full_prompt = "\n".join(user_prompt_parts)

        agent_result = await self._run_message_workflow(
            workflow=qa_workflow,
            agent_type=AgentType.QA,
            question=full_prompt,
            recursion_limit=50,
        )

        # Extract JSON test strategy for backward compatibility
        if agent_result.success:
            raw_answer = agent_result.output.get("answer", "")
            test_strategy = self._extract_json_from_answer(raw_answer)
            if test_strategy:
                agent_result.output["test_strategy"] = test_strategy
                agent_result.output["validation_results"] = []
            else:
                agent_result.output["test_strategy"] = {
                    "summary": {"raw_analysis": raw_answer[:500]},
                    "raw_analysis": raw_answer,
                }
                agent_result.output["validation_results"] = []

        return agent_result

    async def process_developer_question(
        self,
        question: str,
        chat_history: List[Dict[str, str]] = None,
        context_files: List[str] = None,
    ) -> AgentResult:
        """Process a developer question with tool-assisted reasoning."""
        return await self._run_message_workflow(
            workflow=developer_app,
            agent_type=AgentType.DEVELOPER,
            question=question,
            chat_history=chat_history,
            context_files=context_files,
            recursion_limit=50,
        )

    async def process_sales_question(
        self,
        question: str,
        chat_history: List[Dict[str, str]] = None,
    ) -> AgentResult:
        """Process a sales question with business-friendly tools."""
        return await self._run_message_workflow(
            workflow=sales_workflow,
            agent_type=AgentType.SALES,
            question=question,
            chat_history=chat_history,
            recursion_limit=30,
        )

    # ------------------------------------------------------------------
    # Two-layer processing: Reasoning → Policy (works for ALL roles)
    # ------------------------------------------------------------------

    async def process_with_policy(
        self,
        question: str,
        user_role: str = "developer",
        chat_history: List[Dict[str, str]] = None,
        context_files: List[str] = None,
    ) -> Dict[str, Any]:
        """
        Two-layer processing: Reasoning → Policy.
        
        Routes to the correct workflow based on user_role:
            - sales → sales_workflow
            - qa → qa_workflow (for test-related questions)
            - developer/architect/admin → developer_workflow
            - viewer → developer_workflow (read-only policy applied after)
        
        Then applies RBAC, redaction, and output shaping via Policy Layer.
        """
        # Step 1: Route to the correct workflow based on role
        role_lower = user_role.lower()

        if role_lower == "sales":
            agent_result = await self.process_sales_question(
                question=question,
                chat_history=chat_history,
            )
        else:
            # Developer, architect, admin, qa, viewer all use developer workflow
            # (QA workflow is for structured test generation, not chat Q&A)
            agent_result = await self.process_developer_question(
                question=question,
                chat_history=chat_history,
                context_files=context_files,
            )

        raw_answer = agent_result.output.get("answer", "")
        tools_used = agent_result.output.get("tools_used", [])

        # Step 2: Build ReasoningResult with evidence-based confidence
        if agent_result.success and raw_answer:
            tool_count = len(tools_used)
            has_graph_tools = any(t in tools_used for t in [
                "get_repository_architecture", "find_component", "get_api_flow",
                "get_behavioral_classification", "get_security_surface",
                "get_domain_context", "trace_cross_repo_flow", "get_file_dependencies",
                "get_qa_risk_assessment", "get_codebase_metrics",
                # Sales tools
                "get_product_overview", "get_feature_capabilities", "get_security_summary",
                "get_integration_info", "get_complexity_assessment",
            ])
            has_code_tools = any(t in tools_used for t in [
                "read_file_content", "search_code_pattern", "list_api_routes",
            ])
            has_search = "search_codebase" in tools_used

            # Confidence scoring
            if has_code_tools and has_graph_tools:
                base_confidence = 0.90
            elif has_code_tools:
                base_confidence = 0.85
            elif has_graph_tools:
                base_confidence = 0.80
            elif has_search:
                base_confidence = 0.70
            elif tool_count > 0:
                base_confidence = 0.65
            else:
                base_confidence = 0.45

            if len(raw_answer) < 100:
                base_confidence *= 0.8

            # Build evidence list
            evidence_items = []
            for t in tools_used:
                source = "graph_traversal" if t in (
                    "get_repository_architecture", "find_component", "get_codebase_metrics",
                    "get_behavioral_classification", "get_security_surface",
                    "get_domain_context", "trace_cross_repo_flow", "get_file_dependencies",
                    "get_qa_risk_assessment",
                    "get_product_overview", "get_feature_capabilities", "get_security_summary",
                    "get_integration_info", "get_complexity_assessment",
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

            # Extract affected files
            import re as _re
            file_mentions = _re.findall(r"[\w/]+\.(?:js|ts|py|java|jsx|tsx)\b", raw_answer)

            # Domain resolution
            try:
                from src.intelligence.domain_model import domain_engine as _de
                domain_matches = _de.resolve_query_to_domains(question)
                affected_domains = [d[0] for d in domain_matches[:3]]
            except Exception:
                affected_domains = []

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

        # Step 3: Policy Layer (RBAC, redaction, output shaping)
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

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

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

    def _extract_json_from_answer(self, answer: str) -> Optional[Dict]:
        """
        Try to extract a JSON object from the agent's answer text.
        Handles: markdown fences, truncated JSON, and nested structures.
        Uses section-by-section extraction as a last resort for truncated JSON.
        """
        if not answer:
            return None

        import re

        # Step 1: Strip markdown fences
        cleaned = answer.strip()
        cleaned = re.sub(r"^```(?:json)?\s*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```\s*$", "", cleaned)
        cleaned = cleaned.strip()

        # Step 2: Try direct parse
        try:
            return json.loads(cleaned)
        except (json.JSONDecodeError, ValueError):
            pass

        # Step 3: Extract from ```json ... ``` fences in the middle of text
        json_match = re.search(r"```(?:json)?\s*\n(.*?)(?:\n```|$)", answer, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except (json.JSONDecodeError, ValueError):
                # Try repairing the extracted text
                repaired = self._repair_truncated_json(json_match.group(1))
                if repaired:
                    return repaired

        # Step 4: Find the first { and try to parse the largest valid JSON block
        brace_start = cleaned.find("{")
        if brace_start >= 0:
            json_text = cleaned[brace_start:]
            # Direct try
            try:
                return json.loads(json_text)
            except (json.JSONDecodeError, ValueError) as e:
                logger.debug(f"JSON parse failed at brace_start: {e}",
                            text_start=json_text[:200])
            # Try repair
            repaired = self._repair_truncated_json(json_text)
            if repaired:
                return repaired

            # Step 5: Section-by-section extraction for truncated JSON
            # This extracts individual complete key-value pairs even when
            # the overall JSON is truncated (e.g., LLM hit max_tokens)
            try:
                from src.services.pdf_service import _extract_sections_from_truncated_json
                extracted = _extract_sections_from_truncated_json(json_text)
                if extracted and len(extracted) >= 2:
                    logger.info(
                        "Extracted sections from truncated JSON",
                        sections=list(extracted.keys()),
                        total_sections=len(extracted),
                    )
                    return extracted
            except ImportError:
                pass

        logger.warning(
            "All JSON extraction strategies failed",
            answer_length=len(answer),
            starts_with=answer[:100] if answer else "",
        )
        return None

    def _repair_truncated_json(self, text: str) -> Optional[Dict]:
        """
        Attempt to repair truncated JSON from LLM output.
        Closes unclosed strings, arrays, and objects.
        """
        if not text or not text.strip().startswith("{"):
            return None

        try:
            # First try: close unclosed string literals, then close brackets/braces
            repaired = text.rstrip()

            # Remove trailing comma
            repaired = re.sub(r",\s*$", "", repaired)

            # If we're in the middle of a string value, close it
            # Count unescaped quotes
            in_string = False
            escaped = False
            for ch in repaired:
                if escaped:
                    escaped = False
                    continue
                if ch == "\\":
                    escaped = True
                    continue
                if ch == '"':
                    in_string = not in_string

            if in_string:
                repaired += '"'

            # Count open braces and brackets
            open_braces = 0
            open_brackets = 0
            in_str = False
            esc = False
            for ch in repaired:
                if esc:
                    esc = False
                    continue
                if ch == "\\":
                    esc = True
                    continue
                if ch == '"':
                    in_str = not in_str
                    continue
                if in_str:
                    continue
                if ch == "{":
                    open_braces += 1
                elif ch == "}":
                    open_braces -= 1
                elif ch == "[":
                    open_brackets += 1
                elif ch == "]":
                    open_brackets -= 1

            # Close unclosed brackets and braces
            # Remove any trailing partial key-value pairs first
            repaired = re.sub(r',\s*"[^"]*"?\s*:?\s*$', "", repaired)

            for _ in range(max(0, open_brackets)):
                repaired += "]"
            for _ in range(max(0, open_braces)):
                repaired += "}"

            result = json.loads(repaired)
            logger.info("Successfully repaired truncated JSON",
                       original_len=len(text), repaired_len=len(repaired))
            return result

        except (json.JSONDecodeError, ValueError) as e:
            logger.debug(f"JSON repair failed: {e}")
            return None

    # ------------------------------------------------------------------
    # PRD → QA → PDF Pipeline (the main PRD generation flow)
    # ------------------------------------------------------------------

    async def generate_prd_with_qa(
        self,
        client_requirements: str,
        business_context: str = "",
        constraints: str = "",
        prd_format_template: str = "",
    ) -> Dict[str, Any]:
        """
        Full PRD → QA → PDF pipeline.

        1. PRD Agent searches codebase, generates structured PRD JSON
        2. QA Agent auto-generates QA Test Plan from PRD
        3. Both are converted to PDFs

        Args:
            client_requirements: The client's requirements text
            business_context: Business context and background
            constraints: Performance, security, timeline, scale constraints
            prd_format_template: Optional custom PRD format

        Returns:
            Dict with prd_data, qa_data, prd_pdf_path, qa_pdf_path
        """
        start_time = datetime.utcnow()

        try:
            # Build the user prompt for the PRD agent
            prompt_parts = ["## Client Requirements:\n" + client_requirements]
            if business_context:
                prompt_parts.append(f"\n## Business Context:\n{business_context}")
            if constraints:
                prompt_parts.append(f"\n## Constraints:\n{constraints}")
            if prd_format_template:
                prompt_parts.append(f"\n## PRD Format Template:\n{prd_format_template}")
            prompt_parts.append(
                "\n\nAnalyze these requirements against the codebase and generate a "
                "comprehensive enterprise-grade PRD. Search the codebase thoroughly "
                "using all available tools before producing the PRD JSON."
            )

            full_prompt = "\n".join(prompt_parts)

            # Step 1: Generate PRD
            logger.info("Starting PRD generation pipeline")
            prd_result = await self._run_message_workflow(
                workflow=prd_workflow,
                agent_type=AgentType.PRD,
                question=full_prompt,
                recursion_limit=60,
                extra_state={
                    "client_requirements": client_requirements,
                    "business_context": business_context,
                    "constraints": constraints,
                },
            )

            if not prd_result.success:
                logger.error(
                    "PRD agent returned failure",
                    errors=prd_result.errors,
                    output_answer_length=len(prd_result.output.get("answer", "")),
                )
                return {
                    "success": False,
                    "error": f"PRD generation failed: {'; '.join(prd_result.errors) if prd_result.errors else 'No response generated'}",
                    "errors": prd_result.errors,
                }

            # Extract PRD JSON from agent response
            prd_raw = prd_result.output.get("answer", "")
            logger.info(
                "PRD agent response received",
                response_length=len(prd_raw),
                tools_used=prd_result.output.get("tools_used", []),
                messages_count=prd_result.output.get("messages_count", 0),
            )
            prd_data = self._extract_json_from_answer(prd_raw)

            if not prd_data:
                # If no JSON found, create a rich fallback from the raw text
                logger.warning(
                    "Could not parse PRD JSON — using raw text fallback",
                    raw_length=len(prd_raw),
                    first_500=prd_raw[:500],
                    last_200=prd_raw[-200:] if len(prd_raw) > 200 else "",
                )
                prd_data = {
                    "feature_name": "Generated PRD",
                    "executive_summary": {
                        "problem_statement": client_requirements[:1000],
                        "objective": "Detailed analysis generated from codebase investigation.",
                    },
                    "raw_analysis": prd_raw,
                }
            else:
                logger.info("PRD JSON parsed successfully",
                            sections=list(prd_data.keys()),
                            feature_name=prd_data.get("feature_name", ""))

            # Integrate Structured PRD Graph (prd_graph intelligence layer)
            try:
                from src.intelligence.prd_graph import build_structured_prd_from_llm_output
                import uuid
                structured_prd = build_structured_prd_from_llm_output(
                    prd_id=str(uuid.uuid4()),
                    raw_text=client_requirements,
                    llm_json=prd_data,
                )
                prd_data["structured_prd"] = structured_prd.to_dict()
                gap_analysis = structured_prd.get_capability_gap_analysis()
                if gap_analysis:
                    prd_data["capability_gap_analysis"] = gap_analysis
                logger.info(
                    "Structured PRD graph created",
                    capabilities=len(structured_prd.capabilities),
                    constraints=len(structured_prd.constraints),
                )
            except Exception as prd_graph_err:
                logger.warning(f"Could not build structured PRD graph: {prd_graph_err}")

            # Step 2: Auto-generate QA Test Plan from PRD
            logger.info("Starting QA auto-generation from PRD")
            qa_prompt = (
                f"## PRD Document (auto-generated):\n"
                f"{json.dumps(prd_data, indent=2, default=str)[:8000]}\n\n"
                f"Generate a comprehensive QA Test Plan covering ALL 5 testing types "
                f"(Unit, Integration, System, Regression, Performance/Load). "
                f"Every test case must map to a requirement from the PRD above. "
                f"Reference real code artifacts from the codebase."
            )

            qa_result = await self._run_message_workflow(
                workflow=qa_workflow,
                agent_type=AgentType.QA,
                question=qa_prompt,
                recursion_limit=60,
                extra_state={
                    "prd_data": prd_data,
                },
            )

            qa_data = None
            qa_raw = ""
            if qa_result.success:
                qa_raw = qa_result.output.get("answer", "")
                logger.info(
                    "QA agent response received",
                    response_length=len(qa_raw),
                    tools_used=qa_result.output.get("tools_used", []),
                )
                qa_data = self._extract_json_from_answer(qa_raw)
            else:
                logger.warning("QA agent returned failure", errors=qa_result.errors)

            if not qa_data:
                logger.warning(
                    "Could not parse QA JSON — using raw text fallback",
                    raw_length=len(qa_raw if qa_result.success else ""),
                    first_500=qa_raw[:500] if qa_raw else "",
                    last_200=qa_raw[-200:] if len(qa_raw) > 200 else "",
                )
                qa_data = {
                    "feature_name": prd_data.get("feature_name", "Generated QA"),
                    "summary": {"total_test_cases": 0, "note": "QA generation produced unstructured output"},
                    "raw_analysis": qa_raw if qa_result.success else "",
                }
            else:
                logger.info("QA JSON parsed successfully", sections=list(qa_data.keys()))

            # Ensure feature_name consistency
            if "feature_name" not in qa_data:
                qa_data["feature_name"] = prd_data.get("feature_name", "Untitled")

            # Step 3: Generate PDFs
            logger.info("Generating PDF documents")
            from src.services.pdf_service import generate_prd_pdf, generate_qa_pdf

            prd_pdf_path = generate_prd_pdf(prd_data)
            qa_pdf_path = generate_qa_pdf(qa_data)

            execution_time = (datetime.utcnow() - start_time).total_seconds()

            logger.info(
                "PRD→QA→PDF pipeline completed",
                execution_time=execution_time,
                prd_pdf=prd_pdf_path,
                qa_pdf=qa_pdf_path,
            )

            return {
                "success": True,
                "feature_name": prd_data.get("feature_name", "Untitled"),
                "prd_data": prd_data,
                "qa_data": qa_data,
                "prd_pdf_path": prd_pdf_path,
                "qa_pdf_path": qa_pdf_path,
                "prd_tools_used": prd_result.output.get("tools_used", []),
                "qa_tools_used": qa_result.output.get("tools_used", []),
                "execution_time_seconds": execution_time,
            }

        except Exception as e:
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            logger.error(f"PRD→QA pipeline failed: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "execution_time_seconds": execution_time,
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
