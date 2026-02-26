"""
QA Test Plan Generator Workflow using LangGraph (Tool-Calling Agent)
Auto-generates comprehensive QA Test Plans from PRD output.

Key principles:
    - AUTO-GENERATED from PRD (no user interaction needed)
    - Covers ALL 5 testing types: Unit, Integration, System, Regression, Performance
    - Every test case maps to a real PRD requirement
    - Uses Gemini 2.5 Pro for maximum quality
    - References real code artifacts from the codebase
"""
import json
import yaml
import asyncio
from typing import Dict, Any, List, Literal
from pathlib import Path
from datetime import datetime

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from litellm import acompletion

from config.settings import settings
from src.agents.state_schema import QAState
from src.core.logger import get_logger
from src.core.exceptions import AgentException, LLMServiceException

logger = get_logger(__name__)

# Load ALL QA prompts from qa_prompt.yaml ONLY (fully separated from prd_prompt.yaml)
QA_PROMPTS_PATH = Path(__file__).parent.parent.parent.parent / "config" / "qa_prompt.yaml"

qa_prompts = {}
try:
    with open(QA_PROMPTS_PATH, 'r') as f:
        qa_prompts = yaml.safe_load(f) or {}
except FileNotFoundError:
    logger.warning(f"QA prompts file not found: {QA_PROMPTS_PATH}")

QA_SYSTEM_PROMPT = qa_prompts.get('qa_engineer_system', "You are a Senior QA Engineer.")
QA_JSON_FORMAT = qa_prompts.get('qa_json_output_format', "")


# ---------------------------------------------------------------------------
# QA-specific tools (graph-backed, intelligence-powered)
# ---------------------------------------------------------------------------

@tool
async def find_existing_test_patterns() -> Dict[str, Any]:
    """
    Search for existing test files, test frameworks, and testing utilities in the codebase.
    USE THIS FIRST to understand the current testing approach and adopt the same patterns.

    Returns:
        Existing test files, frameworks used, and test patterns.
    """
    try:
        from src.services.search_service import SearchService
        search_service = SearchService()
        await search_service.initialize()

        results = await search_service.search_codebase(
            query="test describe it expect mock jest mocha beforeEach afterEach pytest unittest",
            search_type="semantic",
            top_k=15,
        )
        combined = results.get("results", {}).get("combined", [])

        return {
            "test_files": [
                {
                    "file": r.get("file_path", r.get("metadata", {}).get("file_path", "")),
                    "preview": str(r.get("content", ""))[:300],
                    "score": round(r.get("score", 0), 3),
                }
                for r in combined[:8]
            ],
            "total_test_files": len(combined),
        }
    except Exception as e:
        return {"error": str(e)}


@tool
async def get_all_api_routes() -> Dict[str, Any]:
    """
    List all API routes with methods and handlers. Essential for integration
    and system testing — every route should have test coverage.

    Returns:
        Complete API route inventory.
    """
    try:
        from src.database.neo4j_client import neo4j_client
        await neo4j_client.initialize()

        query = """
        MATCH (r:VertxRoute)
        RETURN r.path as path, r.method as method, r.handler_method as handler,
               r.file as file
        ORDER BY r.path
        """
        routes = await neo4j_client.execute_query(query)

        return {
            "routes": [
                {"path": r["path"], "method": r.get("method", ""), "handler": r.get("handler", ""), "file": r.get("file", "")}
                for r in (routes or [])
            ],
            "total_routes": len(routes or []),
        }
    except Exception as e:
        return {"error": str(e)}


@tool
async def get_behavioral_hotspots() -> Dict[str, Any]:
    """
    Identify behaviorally risky areas needing extra testing: AuthCritical,
    TransactionBoundary, SideEffecting, AsyncFanOut nodes.

    Returns:
        Risky routes with behavioral labels and specific test recommendations.
    """
    try:
        from src.database.neo4j_client import neo4j_client
        from src.intelligence.behavioral_layer import behavioral_classifier
        await neo4j_client.initialize()

        routes_query = """
        MATCH (r:VertxRoute)
        RETURN r.name as name, r.method as http_method, r.handler_method as handler,
               r.path as path, r.file as file, labels(r) as labels,
               r.behavioral_labels as behavioral_labels,
               r.behavioral_confidence as behavioral_confidence
        """
        routes = await neo4j_client.execute_query(routes_query)

        hotspots = []
        for node in (routes or []):
            stored_labels = node.get("behavioral_labels")

            if stored_labels:
                if isinstance(stored_labels, str):
                    label_values = [l.strip() for l in stored_labels.split(",") if l.strip()]
                elif isinstance(stored_labels, list):
                    label_values = stored_labels
                else:
                    label_values = []
            else:
                cls = behavioral_classifier.classify_node(
                    node_id=node.get("name", ""),
                    name=node.get("handler", "") or node.get("name", ""),
                    node_labels=node.get("labels", []),
                    http_method=node.get("http_method"),
                    is_handler=True,
                )
                label_values = [l.value for l in cls.labels] if cls.labels else []

            if label_values:
                test_recs = []
                if "AuthCritical" in label_values:
                    test_recs.extend(["Test valid/invalid/expired tokens", "Test RBAC", "Test CSRF"])
                if "TransactionBoundary" in label_values:
                    test_recs.extend(["Test rollback on failure", "Test concurrent modifications"])
                if "SideEffecting" in label_values:
                    test_recs.extend(["Test idempotency", "Test external service failure"])
                if "AsyncFanOut" in label_values:
                    test_recs.extend(["Test message ordering", "Test race conditions"])

                hotspots.append({
                    "route": node.get("path", ""),
                    "method": node.get("http_method", ""),
                    "labels": label_values,
                    "test_recommendations": test_recs,
                })

        return {"hotspots": hotspots, "total_risky": len(hotspots)}
    except Exception as e:
        return {"error": str(e)}


@tool
async def get_security_test_targets() -> Dict[str, Any]:
    """
    Find mutation routes (POST/PUT/DELETE/PATCH) lacking auth middleware.
    These are CRITICAL priority for security testing.

    Returns:
        Unprotected mutation routes with specific test cases needed.
    """
    try:
        from src.database.neo4j_client import neo4j_client
        await neo4j_client.initialize()

        query = """
        MATCH (r:VertxRoute)
        WHERE r.method IN ['POST', 'PUT', 'DELETE', 'PATCH']
        RETURN r.path as path, r.method as method, r.handler_method as handler, r.file as file
        """
        routes = await neo4j_client.execute_query(query)

        protected = 0
        unprotected = []
        for r in (routes or []):
            handler = (r.get("handler") or "").lower()
            if any(kw in handler for kw in ["auth", "middleware", "protect", "guard"]):
                protected += 1
            else:
                unprotected.append({
                    "path": r["path"], "method": r["method"],
                    "handler": r.get("handler", ""), "file": r.get("file", ""),
                })

        return {
            "total_mutation_routes": len(routes or []),
            "protected_count": protected,
            "unprotected_routes": unprotected[:20],
        }
    except Exception as e:
        return {"error": str(e)}


@tool
async def find_cross_repo_flows() -> Dict[str, Any]:
    """
    Find frontend-to-backend flows (React components calling API routes).
    Essential for E2E and system testing.

    Returns:
        Cross-repository integration flows.
    """
    try:
        from src.database.neo4j_client import neo4j_client
        await neo4j_client.initialize()

        query = """
        MATCH (c:ReactComponent)-[:MAKES]->(a:ApiCall)
        OPTIONAL MATCH (a)-[:CALLS_ROUTE]->(r:VertxRoute)
        RETURN c.name as component, c.file as component_file,
               a.url as api_url, a.method as api_method,
               r.path as route_path, r.handler_method as handler
        LIMIT 30
        """
        flows = await neo4j_client.execute_query(query)

        return {
            "flows": [
                {
                    "component": f.get("component", ""),
                    "api_url": f.get("api_url", ""),
                    "api_method": f.get("api_method", ""),
                    "backend_route": f.get("route_path", ""),
                }
                for f in (flows or [])
            ],
            "total": len(flows or []),
        }
    except Exception as e:
        return {"error": str(e)}


@tool
async def search_code_patterns(pattern: str) -> Dict[str, Any]:
    """
    Search for specific code patterns relevant to testing (error handling,
    validation, middleware, etc.).

    Args:
        pattern: Code pattern or keyword (e.g., 'validation', 'error handler', 'middleware')

    Returns:
        Matching code locations with context.
    """
    try:
        from src.services.search_service import SearchService
        search_service = SearchService()
        await search_service.initialize()

        results = await search_service.search_codebase(
            query=pattern,
            search_type="hybrid",
            top_k=10,
        )
        combined = results.get("results", {}).get("combined", [])

        return {
            "matches": [
                {
                    "file": r.get("file_path", r.get("metadata", {}).get("file_path", "")),
                    "preview": str(r.get("content", ""))[:300],
                }
                for r in combined[:8]
            ],
            "total": len(combined),
        }
    except Exception as e:
        return {"error": str(e)}


@tool
async def get_qa_risk_projection(area: str) -> Dict[str, Any]:
    """
    Use the QA Projection Engine to generate risk-ranked testing scenarios.
    Identifies async chains, failure surfaces, race windows, and untested paths.

    Args:
        area: Area to analyze for QA risks (e.g., 'auth', 'payment', 'order')

    Returns:
        QA risk scenarios with severity, test intents, and failure modes.
    """
    try:
        from src.intelligence.qa_projection import qa_projection_engine, QARiskCategory, QASeverity
        from src.database.neo4j_client import neo4j_client
        await neo4j_client.initialize()

        query = """
        MATCH (r:VertxRoute)
        WHERE toLower(r.path) CONTAINS toLower($area)
           OR toLower(r.handler_method) CONTAINS toLower($area)
        RETURN r.path as path, r.method as method, r.handler_method as handler, r.file as file
        LIMIT 20
        """
        routes = await neo4j_client.execute_query(query, {"area": area})

        scenarios = []
        for route in (routes or []):
            handler = (route.get("handler") or "").lower()
            method = (route.get("method") or "GET").upper()

            risks = []
            if method in ("POST", "PUT", "DELETE", "PATCH"):
                risks.append({
                    "category": QARiskCategory.FAILURE_SURFACE.value,
                    "severity": QASeverity.HIGH.value,
                    "description": f"Mutation endpoint {route['path']} could fail with invalid data",
                })
            if "auth" in handler or "login" in handler or "token" in handler:
                risks.append({
                    "category": QARiskCategory.AUTH_BOUNDARY.value,
                    "severity": QASeverity.CRITICAL.value,
                    "description": f"Auth endpoint {route['path']} — test bypass, expired tokens, RBAC",
                })
            if "async" in handler or "event" in handler or "queue" in handler:
                risks.append({
                    "category": QARiskCategory.ASYNC_CHAIN.value,
                    "severity": QASeverity.HIGH.value,
                    "description": f"Async handler {route['path']} — test message loss, duplication",
                })

            if risks:
                scenarios.append({
                    "route": route.get("path", ""),
                    "handler": route.get("handler", ""),
                    "risks": risks,
                })

        return {
            "area": area,
            "total_scenarios": len(scenarios),
            "risk_scenarios": scenarios,
        }
    except Exception as e:
        return {"error": str(e)}


@tool
async def get_temporal_regression_targets() -> Dict[str, Any]:
    """
    Identify high-volatility code that should be priority targets for regression testing.
    Uses Temporal Intelligence to find frequently-changing files.

    Returns:
        Files ranked by volatility score — most volatile files need most regression tests.
    """
    try:
        from src.database.neo4j_client import neo4j_client
        from src.intelligence.temporal_metadata import compute_volatility_score, classify_temporal_risk
        await neo4j_client.initialize()

        query = """
        MATCH (f:File)
        WHERE f.change_count IS NOT NULL AND f.change_count > 0
        RETURN f.name as name, f.file as file,
               f.change_count as change_count, f.total_lines as total_lines,
               f.introduced_at as introduced_at, f.last_modified_at as last_modified_at
        ORDER BY f.change_count DESC
        LIMIT 30
        """
        files = await neo4j_client.execute_query(query)

        results = []
        for f in (files or []):
            from datetime import datetime as dt, timezone as tz
            last_mod = f.get("last_modified_at")
            if isinstance(last_mod, str):
                try:
                    last_mod = dt.fromisoformat(last_mod)
                except Exception:
                    last_mod = None
            now = dt.now(tz.utc)
            if last_mod:
                if last_mod.tzinfo is None:
                    last_mod = last_mod.replace(tzinfo=tz.utc)
                days_ago = max(0, (now - last_mod).days)
            else:
                days_ago = 30

            vol_score, _ = compute_volatility_score(
                last_modified_at=last_mod,
                change_count=f.get("change_count") or 0,
                total_lines=f.get("total_lines") or 0,
            )
            risk_level, risk_desc = classify_temporal_risk(vol_score, days_ago)
            results.append({
                "file": f.get("file", f.get("name", "")),
                "volatility_score": round(vol_score, 3),
                "change_count": f.get("change_count", 0),
                "risk_level": risk_level,
                "regression_priority": "CRITICAL" if risk_level in ("high", "critical") else "MEDIUM" if risk_level == "medium" else "LOW",
            })

        results.sort(key=lambda x: x["volatility_score"], reverse=True)
        return {
            "total_files": len(results),
            "critical_regression_targets": [r for r in results if r["regression_priority"] == "CRITICAL"],
            "all_files": results[:20],
        }
    except Exception as e:
        return {"error": str(e)}


@tool
async def get_domain_test_mapping(area: str) -> Dict[str, Any]:
    """
    Map an area to its business domain and associated code artifacts for targeted testing.
    Uses Domain Intelligence to provide domain-aware test strategies.

    Args:
        area: Feature area or keyword to map to a domain

    Returns:
        Domain mapping with capabilities, related files, and test focus areas.
    """
    try:
        from src.intelligence.domain_model import domain_engine
        from src.database.neo4j_client import neo4j_client
        await neo4j_client.initialize()

        domain_matches = domain_engine.resolve_query_to_domains(area)

        # Also get related files in the graph
        query = """
        MATCH (f:File)
        WHERE toLower(f.name) CONTAINS toLower($area) OR toLower(f.file) CONTAINS toLower($area)
        RETURN f.name as name, f.file as file
        LIMIT 15
        """
        files = await neo4j_client.execute_query(query, {"area": area})

        return {
            "matched_domains": [
                {"domain": d, "relevance": round(r, 3)} for d, r in domain_matches[:5]
            ],
            "primary_domain": domain_matches[0][0] if domain_matches else "unknown",
            "related_files": [
                {"name": f.get("name", ""), "file": f.get("file", "")}
                for f in (files or [])
            ],
            "test_focus": f"Focus testing on {domain_matches[0][0] if domain_matches else area} domain boundaries, contracts, and integration points.",
        }
    except Exception as e:
        return {"error": str(e)}


@tool
async def get_error_handling_coverage() -> Dict[str, Any]:
    """
    Find error handling patterns and catch blocks across the codebase.
    Identifies areas with weak error handling that need edge case testing.

    Returns:
        Error handling patterns found and areas lacking proper error handling.
    """
    try:
        from src.services.search_service import SearchService
        search_service = SearchService()
        await search_service.initialize()

        results = await search_service.search_codebase(
            query="try catch error throw exception handle reject",
            search_type="semantic",
            top_k=15,
        )
        combined = results.get("results", {}).get("combined", [])

        return {
            "error_patterns": [
                {
                    "file": r.get("file_path", r.get("metadata", {}).get("file_path", "")),
                    "preview": str(r.get("content", ""))[:300],
                }
                for r in combined[:10]
            ],
            "total_found": len(combined),
            "recommendation": "Test error propagation for each pattern. Verify: correct HTTP status, proper error messages, no stack trace leakage, rollback on transaction failures.",
        }
    except Exception as e:
        return {"error": str(e)}


# All QA tools — 10 tools for comprehensive test planning
qa_tools = [
    find_existing_test_patterns,
    get_all_api_routes,
    get_behavioral_hotspots,
    get_security_test_targets,
    find_cross_repo_flows,
    search_code_patterns,
    # Intelligence-backed tools
    get_qa_risk_projection,
    get_temporal_regression_targets,
    get_domain_test_mapping,
    get_error_handling_coverage,
]


# ---------------------------------------------------------------------------
# QA Agent (LangGraph tool-calling agent — auto-generates from PRD)
# ---------------------------------------------------------------------------

def should_continue(state: QAState) -> Literal["tools", "synthesize", "end"]:
    """Determine if we should continue to tools, synthesize, or end."""
    messages = state.get("messages", [])
    if not messages:
        return "end"

    tool_call_count = sum(1 for m in messages if hasattr(m, 'tool_calls') and m.tool_calls)
    last_message = messages[-1]
    has_tool_calls = hasattr(last_message, 'tool_calls') and last_message.tool_calls

    # Max 8 tool iterations for QA
    if tool_call_count >= 12:
        if has_tool_calls or not (last_message.content or "").strip():
            logger.info("QA agent hit tool iteration limit — routing to synthesize")
            return "synthesize"
        return "end"

    if has_tool_calls:
        return "tools"

    return "end"


def _build_qa_llm_messages(messages, system_prompt: str) -> list:
    """Convert LangGraph messages to LiteLLM chat format."""
    llm_messages = [{"role": "system", "content": system_prompt}]
    for msg in messages:
        if isinstance(msg, HumanMessage):
            llm_messages.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            msg_dict = {"role": "assistant", "content": msg.content or ""}
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                msg_dict["tool_calls"] = [
                    {
                        "id": tc.get("id", f"call_{i}"),
                        "type": "function",
                        "function": {
                            "name": tc.get("name", ""),
                            "arguments": json.dumps(tc.get("args", {}))
                        }
                    }
                    for i, tc in enumerate(msg.tool_calls)
                ]
            llm_messages.append(msg_dict)
        elif isinstance(msg, ToolMessage):
            llm_messages.append({
                "role": "tool",
                "tool_call_id": msg.tool_call_id,
                "content": msg.content
            })
    return llm_messages


def _build_qa_tools() -> list:
    """Convert qa_tools to LiteLLM tool schema."""
    llm_tools = []
    for t in qa_tools:
        params = {}
        if hasattr(t, 'args_schema') and t.args_schema:
            try:
                params = t.args_schema.schema()
            except Exception:
                params = {"type": "object", "properties": {}, "required": []}
        llm_tools.append({
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": params
            }
        })
    return llm_tools


async def qa_agent(state: QAState) -> Dict[str, Any]:
    """Main QA agent — auto-generates comprehensive test plan from PRD."""
    try:
        messages = list(state.get('messages', []))
        tool_descriptions = "\n".join([f"- {t.name}: {t.description}" for t in qa_tools])

        system_prompt = f"""{QA_SYSTEM_PROMPT}

{QA_JSON_FORMAT}

## TOOLS AVAILABLE:
{tool_descriptions}

## MANDATORY WORKFLOW:
1. Call `find_existing_test_patterns` to understand current test structure
2. Call `get_all_api_routes` to discover all endpoints that need testing
3. Call `get_behavioral_hotspots` to find risky high-impact areas
4. Call `get_security_test_targets` for security-critical targets
5. Call `find_cross_repo_flows` for end-to-end integration scenarios
6. Call `search_code_patterns` to find error handling, validation patterns
7. THEN produce the COMPLETE QA Test Plan JSON — every section filled, every test case detailed

CRITICAL RULES:
- Produce AT LEAST 5 test cases per testing type (Unit, Integration, System, Regression, Performance).
- Every test case MUST map to a real PRD requirement.
- Keep descriptions CONCISE (under 100 characters) to prevent JSON truncation.
- Output MUST be a single valid JSON object — no markdown fences, no text before/after.
"""

        llm_messages = _build_qa_llm_messages(messages, system_prompt)
        llm_tools = _build_qa_tools()

        # Use Gemini 2.5 Pro for QA (configured in settings)
        model_name = settings.get_litellm_model("qa")
        if "gemini" in settings.QA_MODEL.lower():
            model_name = f"vertex_ai/{settings.QA_MODEL}"

        response = await acompletion(
            model=model_name,
            messages=llm_messages,
            tools=llm_tools if llm_tools else None,
            tool_choice="auto",
            max_tokens=max(settings.LLM_MAX_TOKENS, 16000),
            temperature=0.2,
            timeout=settings.LLM_TIMEOUT,
            api_key=settings.get_api_key_for_model(settings.QA_MODEL) or "vertex-ai"
        )

        response_message = response.choices[0].message
        ai_message_kwargs = {"content": response_message.content or ""}

        usage = getattr(response, "usage", None)
        if usage is not None:
            pt = getattr(usage, "prompt_tokens", None) or (usage.get("prompt_tokens", 0) if isinstance(usage, dict) else 0)
            ct = getattr(usage, "completion_tokens", None) or (usage.get("completion_tokens", 0) if isinstance(usage, dict) else 0)
            tt = getattr(usage, "total_tokens", None) or (usage.get("total_tokens", 0) if isinstance(usage, dict) else (pt + ct))
            ai_message_kwargs["response_metadata"] = {"usage": {"input_tokens": pt, "output_tokens": ct, "total_tokens": tt}}

        if hasattr(response_message, 'tool_calls') and response_message.tool_calls:
            ai_message_kwargs["tool_calls"] = [
                {
                    "id": tc.id,
                    "name": tc.function.name,
                    "args": json.loads(tc.function.arguments) if tc.function.arguments else {}
                }
                for tc in response_message.tool_calls
            ]

        return {"messages": [AIMessage(**ai_message_kwargs)]}

    except Exception as e:
        logger.error(f"QA agent failed: {e}", exc_info=True)
        # Return error as message content so LangGraph can handle it gracefully
        # The orchestrator will detect the error in the output
        return {"messages": [AIMessage(
            content=json.dumps({
                "error": f"QA test plan generation failed: {str(e)}",
                "error_type": "agent_error",
                "agent": "qa"
            })
        )]}


async def execute_qa_tools(state: QAState) -> Dict[str, Any]:
    """Execute tools called by the QA agent."""
    messages = state.get("messages", [])
    last_message = messages[-1] if messages else None

    if not last_message or not hasattr(last_message, 'tool_calls') or not last_message.tool_calls:
        return {"messages": []}

    tool_messages = []
    for tool_call in last_message.tool_calls:
        tool_name = tool_call.get("name", "")
        tool_args = tool_call.get("args", {})
        tool_id = tool_call.get("id", f"call_{tool_name}")

        logger.info(f"QA tool executing: {tool_name}", args=tool_args)

        tool_result = {"error": f"Tool {tool_name} not found"}
        for t in qa_tools:
            if t.name == tool_name:
                try:
                    tool_result = await asyncio.wait_for(
                        t.ainvoke(tool_args),
                        timeout=30.0,
                    )
                except asyncio.TimeoutError:
                    logger.error(f"QA tool {tool_name} timed out")
                    tool_result = {"error": f"Tool {tool_name} timed out after 30s"}
                except Exception as e:
                    logger.error(f"QA tool {tool_name} failed: {e}")
                    tool_result = {"error": str(e)}
                break

        tool_messages.append(ToolMessage(
            content=json.dumps(tool_result, default=str),
            tool_call_id=tool_id
        ))

    return {"messages": tool_messages}


async def synthesize_qa(state: QAState) -> Dict[str, Any]:
    """
    Synthesize the final QA Test Plan when tool iteration limit is reached.
    Uses a COMPACT summary of tool results to avoid empty LLM responses.
    """
    try:
        messages = list(state.get('messages', []))
        logger.info("Synthesizing final QA Test Plan after tool limit", message_count=len(messages))

        # Extract original question (PRD content)
        original_question = ""
        for msg in messages:
            if isinstance(msg, HumanMessage):
                original_question = msg.content
                break

        # Collect and summarize tool results — keep concise to leave room for output
        tool_findings = []
        for msg in messages:
            if isinstance(msg, ToolMessage):
                try:
                    summary = str(msg.content)
                    if len(summary) > 2000:
                        summary = summary[:2000] + "...(truncated)"
                    tool_findings.append(summary)
                except Exception:
                    pass

        codebase_context = "\n---\n".join(tool_findings[-8:])

        synthesis_prompt = f"""{QA_SYSTEM_PROMPT}

{QA_JSON_FORMAT}

## CODEBASE INVESTIGATION RESULTS
The following data was gathered by searching the codebase knowledge graph:

{codebase_context}

## YOUR TASK
Using ONLY the codebase data above and the PRD requirements below, produce a COMPLETE
enterprise-grade QA Test Plan as a single valid JSON document.

CRITICAL FORMATTING RULES:
1. Include AT LEAST 5 test cases per testing type (Unit, Integration, System, Regression, Performance, Security).
2. Every test case must have: id, description, preconditions, expected_result, priority.
3. Keep ALL descriptions CONCISE — under 100 characters each.
4. Keep preconditions and expected_result under 80 characters.
5. Do NOT write verbose paragraphs in test case fields.
6. Do NOT call any tools. Do NOT ask questions.
7. Your response must be ONLY the JSON object — no markdown fences, no explanation text.

CONCISE EXAMPLE:
{{"id": "UT-001", "description": "Validate schedule creation with daily recurrence", "preconditions": "Valid transform ID exists", "expected_result": "Schedule stored with daily pattern", "priority": "high"}}
"""

        llm_messages = [
            {"role": "system", "content": synthesis_prompt},
            {"role": "user", "content": original_question or "Generate the QA Test Plan."},
        ]

        model_name = settings.get_litellm_model("qa")
        if "gemini" in settings.QA_MODEL.lower():
            model_name = f"vertex_ai/{settings.QA_MODEL}"

        # Use higher token limit for synthesis — QA JSON with 30+ test cases is large
        synthesis_max_tokens = max(settings.LLM_MAX_TOKENS, 16000)

        response = await acompletion(
            model=model_name,
            messages=llm_messages,
            tools=None,
            max_tokens=synthesis_max_tokens,
            temperature=0.2,
            timeout=settings.LLM_TIMEOUT,
            api_key=settings.get_api_key_for_model(settings.QA_MODEL) or "vertex-ai"
        )

        content = response.choices[0].message.content or ""
        logger.info("QA synthesis complete", response_length=len(content))

        usage = getattr(response, "usage", None)
        usage_meta = {}
        if usage is not None:
            pt = getattr(usage, "prompt_tokens", None) or (usage.get("prompt_tokens", 0) if isinstance(usage, dict) else 0)
            ct = getattr(usage, "completion_tokens", None) or (usage.get("completion_tokens", 0) if isinstance(usage, dict) else 0)
            tt = getattr(usage, "total_tokens", None) or (usage.get("total_tokens", 0) if isinstance(usage, dict) else (pt + ct))
            usage_meta = {"usage": {"input_tokens": pt, "output_tokens": ct, "total_tokens": tt}}

        if not content.strip():
            logger.warning("QA synthesis returned empty — using fallback")
            content = json.dumps({
                "feature_name": "Generated QA Plan",
                "summary": {"total_test_cases": 0, "note": "QA synthesis produced empty output"},
                "raw_analysis": "\n\n".join(tool_findings[-6:]),
            })

        return {"messages": [AIMessage(content=content, response_metadata=usage_meta)]}

    except Exception as e:
        logger.error(f"QA synthesis failed: {e}", exc_info=True)
        # Return error as JSON so the orchestrator can detect and handle it
        return {"messages": [AIMessage(
            content=json.dumps({
                "error": f"QA synthesis failed: {str(e)}",
                "error_type": "llm_service_error",
                "model": settings.QA_MODEL
            })
        )]}


def create_qa_workflow() -> StateGraph:
    """Create and compile the QA workflow."""
    graph = StateGraph(QAState)

    graph.add_node("agent", qa_agent)
    graph.add_node("tools", execute_qa_tools)
    graph.add_node("synthesize", synthesize_qa)

    graph.set_entry_point("agent")
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", "synthesize": "synthesize", "end": END}
    )
    graph.add_edge("tools", "agent")
    graph.add_edge("synthesize", END)

    return graph.compile()


# Compiled workflow instance
qa_workflow = create_qa_workflow()
