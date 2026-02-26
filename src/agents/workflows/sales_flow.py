"""
Sales Advisor Workflow using LangGraph
Provides business-friendly answers about the codebase for sales teams.

Key principles:
    - ELI5 language (explain like I'm 5)
    - ZERO code snippets, file names, or class names
    - Business-oriented framing (capabilities, scale, security, integrations)
    - Policy Layer strips any remaining technical artifacts
"""
import json
import yaml
import asyncio
from typing import Dict, Any, List, Literal
from pathlib import Path
from datetime import datetime

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from litellm import acompletion

from config.settings import settings
from src.agents.state_schema import DeveloperState  # Reuse message-based state
from src.core.logger import get_logger

logger = get_logger(__name__)

# Load prompts
PROMPTS_PATH = Path(__file__).parent.parent.parent.parent / "config" / "sales_prompt.yaml"
try:
    with open(PROMPTS_PATH, 'r') as f:
        prompts = yaml.safe_load(f) or {}
except FileNotFoundError:
    logger.warning(f"Sales prompts file not found: {PROMPTS_PATH}")
    prompts = {}

DEFAULT_SALES_PROMPT = """
You are a friendly Sales Assistant. Explain everything in simple, non-technical language.
NEVER use code snippets, file names, or developer jargon.
"""

SALES_SYSTEM_PROMPT = prompts.get('sales_system', DEFAULT_SALES_PROMPT)


# ---------------------------------------------------------------------------
# Sales-specific tools (graph-backed, no code exposure)
# ---------------------------------------------------------------------------

@tool
async def get_product_overview() -> Dict[str, Any]:
    """
    Get a high-level overview of what this product does, how big it is,
    and what technologies power it — described in business terms.
    USE THIS FIRST for any general product questions.
    
    Returns:
        Business-friendly product overview with capabilities and scale
    """
    try:
        from src.database.neo4j_client import neo4j_client
        await neo4j_client.initialize()

        # Get repository info
        repos_query = """
        MATCH (r:Repository)
        RETURN r.name as name, r.language as language, r.path as path
        """
        repos = await neo4j_client.execute_query(repos_query)

        # Get scale metrics
        metrics_query = """
        MATCH (n)
        WHERE n:File OR n:Function OR n:VertxRoute OR n:ReactComponent OR n:ApiCall
        RETURN labels(n)[0] as type, count(*) as count
        """
        metrics = await neo4j_client.execute_query(metrics_query)
        metric_map = {m["type"]: m["count"] for m in (metrics or [])}

        # Get route count for "services"
        route_count = metric_map.get("VertxRoute", 0)
        component_count = metric_map.get("ReactComponent", 0)
        file_count = metric_map.get("File", 0)

        # Determine product type
        has_frontend = component_count > 0
        has_backend = route_count > 0

        product_type = "full-stack application"
        if has_frontend and not has_backend:
            product_type = "frontend application"
        elif has_backend and not has_frontend:
            product_type = "backend service"

        return {
            "product_type": product_type,
            "applications": [{"name": r["name"], "technology": r.get("language", "unknown")} for r in (repos or [])],
            "scale": {
                "total_files": file_count,
                "backend_services": route_count,
                "frontend_screens": component_count,
                "total_functions": metric_map.get("Function", 0),
            },
            "summary": f"This is a {product_type} with {len(repos or [])} applications, "
                       f"{route_count} backend services, and {component_count} frontend screens.",
        }
    except Exception as e:
        return {"error": str(e)}


@tool
async def get_feature_capabilities() -> Dict[str, Any]:
    """
    List the main features and capabilities of the product.
    Groups backend services by business area and frontend by user-facing screens.
    USE THIS for 'what can this product do?' questions.
    
    Returns:
        Grouped list of features and capabilities in business language
    """
    try:
        from src.database.neo4j_client import neo4j_client
        from src.intelligence.domain_model import domain_engine
        await neo4j_client.initialize()

        # Get all routes grouped by path prefix
        routes_query = """
        MATCH (r:VertxRoute)
        RETURN r.path as path, r.method as method
        ORDER BY r.path
        """
        routes = await neo4j_client.execute_query(routes_query)

        # Group by business area
        areas: Dict[str, List[str]] = {}
        for r in (routes or []):
            path = r.get("path", "")
            # Extract first path segment as area
            parts = [p for p in path.split("/") if p]
            area = parts[0] if parts else "general"
            # Translate to business name
            area_name = area.replace("-", " ").replace("_", " ").title()
            if area_name not in areas:
                areas[area_name] = []
            areas[area_name].append(f"{r['method']} {path}")

        # Get frontend components
        comps_query = """
        MATCH (c:ReactComponent)
        RETURN c.name as name, c.file as file
        LIMIT 30
        """
        comps = await neo4j_client.execute_query(comps_query)

        capabilities = []
        for area, routes_list in areas.items():
            capabilities.append({
                "area": area,
                "service_count": len(routes_list),
                "description": f"{area} area with {len(routes_list)} services",
            })

        return {
            "backend_capabilities": capabilities,
            "frontend_screens": len(comps or []),
            "total_business_areas": len(areas),
            "summary": f"The product has {len(areas)} main business areas with {len(routes or [])} total services.",
        }
    except Exception as e:
        return {"error": str(e)}


@tool
async def get_security_summary() -> Dict[str, Any]:
    """
    Get a business-friendly security summary: how is the product protected?
    USE THIS for security, compliance, or trust-related questions.
    
    Returns:
        Security posture in non-technical language
    """
    try:
        from src.database.neo4j_client import neo4j_client
        await neo4j_client.initialize()

        # Count routes by protection status
        all_routes_query = """
        MATCH (r:VertxRoute)
        RETURN r.path as path, r.method as method, r.handler_method as handler
        """
        all_routes = await neo4j_client.execute_query(all_routes_query)

        total = len(all_routes or [])
        protected = 0
        unprotected_mutations = 0

        for r in (all_routes or []):
            handler = (r.get("handler") or "").lower()
            if any(kw in handler for kw in ["auth", "middleware", "protect", "guard"]):
                protected += 1
            elif r.get("method", "").upper() in ("POST", "PUT", "DELETE", "PATCH"):
                unprotected_mutations += 1

        # Check for auth-related functions
        auth_query = """
        MATCH (f:Function)
        WHERE f.name =~ '(?i).*(auth|login|token|session|password).*'
        RETURN count(*) as count
        """
        auth_result = await neo4j_client.execute_query(auth_query)
        auth_function_count = auth_result[0]["count"] if auth_result else 0

        protection_percentage = round((protected / max(total, 1)) * 100)

        return {
            "total_services": total,
            "protected_services": protected,
            "protection_percentage": protection_percentage,
            "has_authentication": auth_function_count > 0,
            "auth_components": auth_function_count,
            "security_grade": "Strong" if protection_percentage > 80 else "Moderate" if protection_percentage > 50 else "Needs improvement",
            "summary": f"The product has {total} services. {protection_percentage}% are protected by security checks. "
                       f"There are {auth_function_count} dedicated security components.",
        }
    except Exception as e:
        return {"error": str(e)}


@tool
async def get_integration_info() -> Dict[str, Any]:
    """
    Find external integrations, third-party services, and API connections.
    USE THIS for questions about integrations, partners, or external services.
    
    Returns:
        List of external connections and integrations in business terms
    """
    try:
        from src.database.neo4j_client import neo4j_client
        await neo4j_client.initialize()

        # Find API calls (frontend → external)
        api_calls_query = """
        MATCH (a:ApiCall)
        RETURN a.url as url, a.method as method, a.file as file
        LIMIT 30
        """
        api_calls = await neo4j_client.execute_query(api_calls_query)

        # Categorize external vs internal
        external = []
        internal = []
        for a in (api_calls or []):
            url = a.get("url", "")
            if url.startswith("http") and "localhost" not in url:
                external.append({"url": url, "method": a.get("method", "")})
            else:
                internal.append({"url": url, "method": a.get("method", "")})

        return {
            "total_connections": len(api_calls or []),
            "external_integrations": len(external),
            "internal_connections": len(internal),
            "external_services": external[:10],
            "summary": f"The product connects to {len(external)} external services and has {len(internal)} internal connections.",
        }
    except Exception as e:
        return {"error": str(e)}


@tool
async def get_complexity_assessment(area: str = None) -> Dict[str, Any]:
    """
    Assess the complexity and risk of a specific area or the whole product.
    USE THIS for questions about risk, complexity, or change impact.
    
    Args:
        area: Optional business area to focus on (e.g., 'authentication', 'payments')
    
    Returns:
        Complexity and risk assessment in business terms
    """
    try:
        from src.database.neo4j_client import neo4j_client
        from src.intelligence.domain_model import domain_engine
        await neo4j_client.initialize()

        # Overall complexity metrics
        complexity_query = """
        MATCH (n)
        WHERE n:File OR n:Function OR n:VertxRoute OR n:ReactComponent
        WITH labels(n)[0] as type, count(*) as cnt
        RETURN type, cnt ORDER BY cnt DESC
        """
        metrics = await neo4j_client.execute_query(complexity_query)

        # Relationship density (more = more complex)
        rels_query = """
        MATCH ()-[r]->()
        RETURN type(r) as rel_type, count(*) as cnt
        ORDER BY cnt DESC
        LIMIT 10
        """
        rels = await neo4j_client.execute_query(rels_query)

        total_nodes = sum(m["cnt"] for m in (metrics or []))
        total_rels = sum(r["cnt"] for r in (rels or []))

        # Simple complexity score
        if total_nodes > 2000:
            complexity_level = "High"
        elif total_nodes > 500:
            complexity_level = "Medium"
        else:
            complexity_level = "Low"

        result = {
            "complexity_level": complexity_level,
            "total_components": total_nodes,
            "total_connections": total_rels,
            "summary": f"Overall system complexity is {complexity_level} with {total_nodes} components and {total_rels} connections.",
        }

        # If area is specified, resolve it through domain engine
        if area:
            domain_matches = domain_engine.resolve_query_to_domains(area)
            if domain_matches:
                result["focus_area"] = area
                result["relevant_domains"] = [d[0] for d in domain_matches[:3]]

        return result
    except Exception as e:
        return {"error": str(e)}


@tool
async def find_product_capability(capability_description: str) -> Dict[str, Any]:
    """
    Search the codebase for evidence of a specific product capability.
    Returns a business-language answer: whether the capability exists,
    how it is implemented at a high level, and confidence level.
    Use for any client question about whether a feature is supported.

    Args:
        capability_description: Description of the capability to search for (e.g., 'multi-currency payments')

    Returns:
        Whether the capability exists with confidence and summary.
    """
    try:
        from src.services.search_service import SearchService
        search = SearchService()
        await search.initialize()
        results = await search.search_codebase(
            query=capability_description,
            search_type="hybrid",
            top_k=8,
        )
        combined = results.get("results", {}).get("combined", [])
        if not combined:
            return {
                "capability_found": False,
                "confidence": "low",
                "summary": f"No implementation found for: {capability_description}",
                "evidence_count": 0,
            }
        top_score = combined[0].get("score", 0)
        return {
            "capability_found": top_score > 0.5,
            "confidence": "high" if top_score > 0.7 else "medium" if top_score > 0.5 else "low",
            "evidence_count": len(combined),
            "implementation_areas": list({
                r.get("metadata", {}).get("file_path", "").split("/")[0]
                for r in combined[:5]
                if r.get("metadata", {}).get("file_path")
            }),
            "summary": f"Found {len(combined)} relevant code areas for '{capability_description}'",
        }
    except Exception as e:
        return {"error": str(e)}


@tool
async def get_system_health_summary() -> Dict[str, Any]:
    """
    Get a plain-English system health summary.
    Returns counts of stable vs. at-risk code areas without any
    technical details, file names, or code. Safe for client conversations.

    Returns:
        Overall health assessment in business language.
    """
    try:
        from src.database.neo4j_client import neo4j_client
        await neo4j_client.initialize()

        volatile_query = """
        MATCH (f:File)
        WHERE f.change_count IS NOT NULL AND f.change_count > 5
        RETURN count(f) as high_change_files
        """
        dead_query = """
        MATCH (fn:Function)
        WHERE fn.dead_code = true
        RETURN count(fn) as dead_functions
        """
        weak_query = """
        MATCH ()-[r:CALLS_FUNCTION]->()
        WHERE r.confidence_score < 0.5
        RETURN count(r) as weak_links
        """
        volatile = await neo4j_client.execute_query(volatile_query)
        dead = await neo4j_client.execute_query(dead_query)
        weak = await neo4j_client.execute_query(weak_query)

        high_change = (volatile or [{}])[0].get("high_change_files", 0)
        dead_count = (dead or [{}])[0].get("dead_functions", 0)
        weak_count = (weak or [{}])[0].get("weak_links", 0)

        health = "excellent" if weak_count < 10 else "good" if weak_count < 50 else "fair"
        return {
            "overall_health": health,
            "stable_areas": "Most core functionality is stable and well-tested.",
            "active_development_areas": high_change,
            "unused_code_areas": dead_count,
            "system_message": (
                "The system is operating normally with no critical issues."
                if health in ("excellent", "good")
                else "Some areas are under active development and may have recent changes."
            ),
        }
    except Exception as e:
        return {"error": str(e)}


# Tools list
sales_tools = [
    get_product_overview,
    get_feature_capabilities,
    get_security_summary,
    get_integration_info,
    get_complexity_assessment,
    find_product_capability,
    get_system_health_summary,
]


# ---------------------------------------------------------------------------
# Sales Agent (LangGraph tool-calling agent)
# ---------------------------------------------------------------------------

def should_continue(state: DeveloperState) -> Literal["tools", "end"]:
    """Determine if we should continue to tools or end"""
    messages = state.get("messages", [])
    if not messages:
        return "end"

    # Max 6 tool iterations for sales (enough for compound client questions)
    tool_call_count = sum(1 for m in messages if hasattr(m, 'tool_calls') and m.tool_calls)
    if tool_call_count >= 6:
        return "end"

    last_message = messages[-1]
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"

    return "end"


async def sales_agent(state: DeveloperState) -> Dict[str, Any]:
    """Main sales agent node"""
    try:
        messages = list(state.get('messages', []))

        tool_descriptions = "\n".join([
            f"- {t.name}: {t.description}" for t in sales_tools
        ])

        system_prompt = f"""{SALES_SYSTEM_PROMPT}

You have access to these tools to look up information about the product:
{tool_descriptions}

TOOL SELECTION GUIDE:
- "What does this product do?" → get_product_overview + get_feature_capabilities
- "Is it secure?" → get_security_summary
- "What integrations?" → get_integration_info
- "How complex/risky?" → get_complexity_assessment

ALWAYS use at least one tool before answering. NEVER make up information.
After getting tool results, translate everything into simple business language.
"""
        # Convert messages to LiteLLM format
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

        # Define tools for LLM
        llm_tools = []
        for t in sales_tools:
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

        model_name = settings.get_litellm_model("developer")
        if "gemini" in settings.DEVELOPER_MODEL.lower():
            model_name = f"vertex_ai/{settings.DEVELOPER_MODEL}"

        response = await acompletion(
            model=model_name,
            messages=llm_messages,
            tools=llm_tools if llm_tools else None,
            tool_choice="auto",
            max_tokens=settings.LLM_MAX_TOKENS,
            temperature=0.4,  # Slightly creative for sales language
            timeout=settings.LLM_TIMEOUT,
            api_key=settings.get_api_key_for_model(settings.DEVELOPER_MODEL) or "vertex-ai"
        )

        response_message = response.choices[0].message
        ai_message_kwargs = {"content": response_message.content or ""}

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
        logger.error(f"Sales agent failed: {e}", exc_info=True)
        return {"messages": [AIMessage(
            content="I'm having trouble looking that up right now. Let me connect you with the technical team for a detailed answer."
        )]}


async def execute_sales_tools(state: DeveloperState) -> Dict[str, Any]:
    """Execute tools called by the sales agent"""
    messages = state.get("messages", [])
    last_message = messages[-1] if messages else None

    if not last_message or not hasattr(last_message, 'tool_calls') or not last_message.tool_calls:
        return {"messages": []}

    tool_messages = []
    for tool_call in last_message.tool_calls:
        tool_name = tool_call.get("name", "")
        tool_args = tool_call.get("args", {})
        tool_id = tool_call.get("id", f"call_{tool_name}")

        tool_result = {"error": f"Tool {tool_name} not found"}
        for t in sales_tools:
            if t.name == tool_name:
                try:
                    tool_result = await asyncio.wait_for(
                        t.ainvoke(tool_args),
                        timeout=30.0,
                    )
                except asyncio.TimeoutError:
                    tool_result = {"error": f"Tool {tool_name} timed out after 30s"}
                except Exception as e:
                    tool_result = {"error": str(e)}
                break

        tool_messages.append(ToolMessage(
            content=json.dumps(tool_result, default=str),
            tool_call_id=tool_id
        ))

    return {"messages": tool_messages}


def create_sales_workflow() -> StateGraph:
    """Create and compile the sales advisor workflow"""
    graph = StateGraph(DeveloperState)

    graph.add_node("agent", sales_agent)
    graph.add_node("tools", execute_sales_tools)

    graph.set_entry_point("agent")
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", "end": END}
    )
    graph.add_edge("tools", "agent")

    return graph.compile()


# Compiled workflow instance
sales_workflow = create_sales_workflow()