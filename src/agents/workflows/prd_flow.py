"""
PRD Agent Workflow using LangGraph (Tool-Calling Agent)
Generates enterprise-grade, codebase-grounded PRDs from client requirements.

Key principles:
    - ZERO hallucination: every artifact must come from the knowledge graph
    - Codebase-first: searches before proposing
    - Structured JSON output matching the PDF template
    - Rejects out-of-scope questions
"""
import json
import yaml
from typing import Dict, Any, List, Literal
from pathlib import Path
from datetime import datetime

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from litellm import acompletion

from config.settings import settings
from src.agents.state_schema import PRDState
from src.core.logger import get_logger
from src.core.exceptions import AgentException, LLMServiceException

logger = get_logger(__name__)

# Load prompts
PROMPTS_PATH = Path(__file__).parent.parent.parent.parent / "config" / "prd_prompt.yaml"
try:
    with open(PROMPTS_PATH, 'r') as f:
        prompts = yaml.safe_load(f) or {}
except FileNotFoundError:
    logger.warning(f"Prompts file not found: {PROMPTS_PATH}")
    prompts = {}

PRD_SYSTEM_PROMPT = prompts.get('prd_agent_system', "You are a codebase-aware PRD Agent.")


# ---------------------------------------------------------------------------
# PRD Agent Tools (graph-backed, codebase-grounded)
# ---------------------------------------------------------------------------

@tool
async def get_repository_overview() -> Dict[str, Any]:
    """
    Get a complete overview of all indexed repositories: languages, file counts,
    route counts, component counts. USE THIS FIRST before any PRD analysis.

    Returns:
        Repository list with metrics and technology stack.
    """
    try:
        from src.database.neo4j_client import neo4j_client
        await neo4j_client.initialize()

        repos_query = """
        MATCH (r:Repository)
        OPTIONAL MATCH (f:File)-[:BELONGS_TO]->(r)
        WITH r, count(DISTINCT f) AS file_count
        OPTIONAL MATCH (f2:File)-[:BELONGS_TO]->(r)
        OPTIONAL MATCH (f2)-[:CONTAINS]->(rt:VertxRoute)
        WITH r, file_count, count(DISTINCT rt) AS route_count
        OPTIONAL MATCH (f3:File)-[:BELONGS_TO]->(r)
        OPTIONAL MATCH (f3)-[:CONTAINS]->(c:ReactComponent)
        RETURN r.name AS name, r.language AS language, r.path AS path,
               file_count, route_count, count(DISTINCT c) AS component_count
        """
        repos = await neo4j_client.execute_query(repos_query)

        # Also get total function count
        func_query = "MATCH (f:Function) RETURN count(f) as count"
        func_result = await neo4j_client.execute_query(func_query)
        func_count = func_result[0]["count"] if func_result else 0

        return {
            "repositories": [
                {
                    "name": r["name"],
                    "language": r.get("language", "unknown"),
                    "files": r["file_count"],
                    "api_routes": r["route_count"],
                    "ui_components": r["component_count"],
                }
                for r in (repos or [])
            ],
            "total_repositories": len(repos or []),
            "total_functions": func_count,
        }
    except Exception as e:
        return {"error": str(e)}


@tool
async def search_existing_implementations(query: str) -> Dict[str, Any]:
    """
    Search the codebase for existing implementations matching a requirement.
    Uses semantic vector search AND graph traversal for comprehensive results.

    Args:
        query: Natural language description of the requirement to search for

    Returns:
        Matching files, functions, routes with relevance scores and content previews.
    """
    try:
        from src.services.search_service import SearchService
        search_service = SearchService()
        await search_service.initialize()

        results = await search_service.search_codebase(
            query=query,
            search_type="hybrid",
            top_k=15,
        )
        combined = results.get("results", {}).get("combined", [])

        return {
            "matches": [
                {
                    "file": r.get("file_path", r.get("metadata", {}).get("file_path", "")),
                    "content_preview": str(r.get("content", ""))[:400],
                    "score": round(r.get("score", 0), 3),
                    "type": r.get("type", "unknown"),
                }
                for r in combined[:10]
            ],
            "total_found": len(combined),
        }
    except Exception as e:
        return {"error": str(e)}


@tool
async def find_api_routes_by_pattern(pattern: str) -> Dict[str, Any]:
    """
    Find existing API routes matching a URL pattern or keyword.
    Essential for discovering if an endpoint already exists.

    Args:
        pattern: URL pattern or keyword (e.g., 'auth', 'user', 'invoice', '/api/v1')

    Returns:
        Matching routes with methods, handlers, and file locations.
    """
    try:
        from src.database.neo4j_client import neo4j_client
        await neo4j_client.initialize()

        query = """
        MATCH (r:VertxRoute)
        WHERE toLower(r.path) CONTAINS toLower($pattern)
           OR toLower(r.name) CONTAINS toLower($pattern)
           OR toLower(r.handler_method) CONTAINS toLower($pattern)
        RETURN r.path as path, r.method as method, r.handler_method as handler,
               r.file as file, r.name as name
        ORDER BY r.path
        LIMIT 30
        """
        routes = await neo4j_client.execute_query(query, {"pattern": pattern})

        return {
            "routes": [
                {"path": r["path"], "method": r.get("method", ""), "handler": r.get("handler", ""), "file": r.get("file", "")}
                for r in (routes or [])
            ],
            "count": len(routes or []),
        }
    except Exception as e:
        return {"error": str(e)}


@tool
async def find_components_by_name(name: str) -> Dict[str, Any]:
    """
    Find existing React components, Functions, DTOs, or Files by name.
    Use to check if a UI component, service, or utility already exists.

    Args:
        name: Component/function name or keyword to search for

    Returns:
        Matching nodes with type, name, and file location.
    """
    try:
        from src.database.neo4j_client import neo4j_client
        await neo4j_client.initialize()

        query = """
        MATCH (n)
        WHERE (n:ReactComponent OR n:Function OR n:File OR n:DTO)
          AND (toLower(n.name) CONTAINS toLower($name) OR toLower(n.file) CONTAINS toLower($name))
        RETURN labels(n)[0] as type, n.name as name, n.file as file
        LIMIT 25
        """
        results = await neo4j_client.execute_query(query, {"name": name})

        return {
            "matches": [
                {"type": r["type"], "name": r.get("name", ""), "file": r.get("file", "")}
                for r in (results or [])
            ],
            "count": len(results or []),
        }
    except Exception as e:
        return {"error": str(e)}


@tool
async def get_dependency_graph(file_or_component: str) -> Dict[str, Any]:
    """
    Trace the dependency graph for a file or component — what it calls and what calls it.
    Critical for understanding impact of changes.

    Args:
        file_or_component: File path or component name to trace

    Returns:
        Outgoing dependencies (what this calls) and incoming (what depends on this).
    """
    try:
        from src.database.neo4j_client import neo4j_client
        await neo4j_client.initialize()

        out_query = """
        MATCH (source)-[r]->(target)
        WHERE source.name CONTAINS $name OR source.file CONTAINS $name
        RETURN source.name as source, type(r) as relationship,
               target.name as target, labels(target)[0] as target_type
        LIMIT 25
        """
        outgoing = await neo4j_client.execute_query(out_query, {"name": file_or_component})

        in_query = """
        MATCH (source)-[r]->(target)
        WHERE target.name CONTAINS $name OR target.file CONTAINS $name
        RETURN source.name as source, labels(source)[0] as source_type,
               type(r) as relationship, target.name as target
        LIMIT 25
        """
        incoming = await neo4j_client.execute_query(in_query, {"name": file_or_component})

        return {
            "outgoing": [
                {"from": r.get("source", ""), "relationship": r.get("relationship", ""), "to": r.get("target", ""), "to_type": r.get("target_type", "")}
                for r in (outgoing or [])
            ],
            "incoming": [
                {"from": r.get("source", ""), "from_type": r.get("source_type", ""), "relationship": r.get("relationship", ""), "to": r.get("target", "")}
                for r in (incoming or [])
            ],
        }
    except Exception as e:
        return {"error": str(e)}


@tool
async def get_behavioral_impact(area: str) -> Dict[str, Any]:
    """
    Assess behavioral risk labels (AuthCritical, TransactionBoundary, SideEffecting)
    for code in a specific area. Essential for PRD risk assessment.

    Args:
        area: Business area or keyword (e.g., 'authentication', 'payment', 'order')

    Returns:
        Behavioral classifications with risk flags and confidence scores.
    """
    try:
        from src.database.neo4j_client import neo4j_client
        from src.intelligence.behavioral_layer import behavioral_classifier
        await neo4j_client.initialize()

        routes_query = """
        MATCH (r:VertxRoute)
        WHERE toLower(r.path) CONTAINS toLower($area)
           OR toLower(r.handler_method) CONTAINS toLower($area)
           OR toLower(r.name) CONTAINS toLower($area)
        RETURN r.name as name, r.method as http_method, r.handler_method as handler,
               r.path as path, r.file as file, labels(r) as labels
        LIMIT 20
        """
        routes = await neo4j_client.execute_query(routes_query, {"area": area})

        classifications = []
        for node in (routes or []):
            cls = behavioral_classifier.classify_node(
                node_id=node.get("name", ""),
                name=node.get("handler", "") or node.get("name", ""),
                node_labels=node.get("labels", []),
                http_method=node.get("http_method"),
                is_handler=True,
            )
            if cls.labels:
                classifications.append({
                    "route": node.get("path", ""),
                    "method": node.get("http_method", ""),
                    "handler": node.get("handler", ""),
                    "behavioral_labels": [l.value for l in cls.labels],
                    "confidence": round(cls.confidence, 3),
                    "risk_flags": [l.value for l in cls.labels if l.value in ("AuthCritical", "TransactionBoundary", "SideEffecting")],
                })

        return {
            "area": area,
            "routes_analyzed": len(routes or []),
            "classifications": classifications,
            "high_risk_count": sum(1 for c in classifications if c["risk_flags"]),
        }
    except Exception as e:
        return {"error": str(e)}


@tool
async def get_domain_mapping(description: str) -> Dict[str, Any]:
    """
    Resolve a requirement description to business domains in the codebase.
    Maps requirements to domain → capability → code artifact hierarchy.

    Args:
        description: Requirement or feature description to resolve

    Returns:
        Matched domains with relevance scores.
    """
    try:
        from src.intelligence.domain_model import domain_engine
        domain_matches = domain_engine.resolve_query_to_domains(description)
        return {
            "matched_domains": [
                {"domain": d, "relevance": round(r, 3)} for d, r in domain_matches[:5]
            ],
            "primary_domain": domain_matches[0][0] if domain_matches else "unknown",
        }
    except Exception as e:
        return {"error": str(e)}


@tool
async def get_security_surface_for_area(area: str) -> Dict[str, Any]:
    """
    Analyze security posture of an area: protected vs unprotected mutation routes,
    auth patterns, RBAC structures.

    Args:
        area: Area to analyze (e.g., 'admin', 'auth', 'payment')

    Returns:
        Security analysis with protected/unprotected counts and recommendations.
    """
    try:
        from src.database.neo4j_client import neo4j_client
        await neo4j_client.initialize()

        query = """
        MATCH (r:VertxRoute)
        WHERE toLower(r.path) CONTAINS toLower($area)
           OR toLower(r.handler_method) CONTAINS toLower($area)
        RETURN r.path as path, r.method as method, r.handler_method as handler, r.file as file
        """
        routes = await neo4j_client.execute_query(query, {"area": area})

        protected = []
        unprotected_mutations = []
        for r in (routes or []):
            handler = (r.get("handler") or "").lower()
            is_protected = any(kw in handler for kw in ["auth", "middleware", "protect", "guard"])
            is_mutation = r.get("method", "").upper() in ("POST", "PUT", "DELETE", "PATCH")

            if is_protected:
                protected.append(r["path"])
            elif is_mutation:
                unprotected_mutations.append({"path": r["path"], "method": r["method"], "handler": r.get("handler", "")})

        return {
            "area": area,
            "total_routes": len(routes or []),
            "protected_count": len(protected),
            "unprotected_mutations": unprotected_mutations[:15],
            "security_recommendation": (
                "All mutation routes are protected." if not unprotected_mutations
                else f"WARNING: {len(unprotected_mutations)} mutation routes lack auth middleware."
            ),
        }
    except Exception as e:
        return {"error": str(e)}


@tool
async def read_file_content(file_path: str) -> Dict[str, Any]:
    """
    Read actual source file content from the codebase.
    Use when you need to see exact implementation details of a specific file.

    Args:
        file_path: Path to the file (relative or absolute)

    Returns:
        File content (first 200 lines) and metadata.
    """
    try:
        from src.services.search_service import SearchService
        search_service = SearchService()
        await search_service.initialize()

        results = await search_service.search_codebase(
            query=file_path,
            search_type="hybrid",
            top_k=5,
            filters={"file_path": file_path},
        )
        combined = results.get("results", {}).get("combined", [])

        if combined:
            return {
                "file": file_path,
                "content": str(combined[0].get("content", ""))[:5000],
                "found": True,
            }
        return {"file": file_path, "content": "", "found": False}
    except Exception as e:
        return {"error": str(e)}


@tool
async def get_database_schemas(keyword: str) -> Dict[str, Any]:
    """
    Search for database models, schemas, DTOs, and data structures in the codebase.
    Use to understand existing data models before proposing new ones.

    Args:
        keyword: Keyword to search for (e.g., 'user', 'order', 'payment')

    Returns:
        Matching DTOs, models, and schema definitions.
    """
    try:
        from src.database.neo4j_client import neo4j_client
        await neo4j_client.initialize()

        # Search DTOs
        dto_query = """
        MATCH (d:DTO)
        WHERE toLower(d.name) CONTAINS toLower($keyword)
           OR toLower(d.file) CONTAINS toLower($keyword)
        RETURN d.name as name, d.file as file, labels(d) as labels
        LIMIT 15
        """
        dtos = await neo4j_client.execute_query(dto_query, {"keyword": keyword})

        # Search for schema/model files
        file_query = """
        MATCH (f:File)
        WHERE toLower(f.name) CONTAINS toLower($keyword)
          AND (toLower(f.name) CONTAINS 'model' OR toLower(f.name) CONTAINS 'schema'
               OR toLower(f.name) CONTAINS 'dto' OR toLower(f.name) CONTAINS 'entity')
        RETURN f.name as name, f.file as file
        LIMIT 10
        """
        files = await neo4j_client.execute_query(file_query, {"keyword": keyword})

        return {
            "dtos": [{"name": d.get("name", ""), "file": d.get("file", "")} for d in (dtos or [])],
            "schema_files": [{"name": f.get("name", ""), "file": f.get("file", "")} for f in (files or [])],
            "total_found": len(dtos or []) + len(files or []),
        }
    except Exception as e:
        return {"error": str(e)}


@tool
async def get_temporal_volatility(area: str) -> Dict[str, Any]:
    """
    Analyze code change volatility and recency for a given area using Temporal Intelligence.
    Identifies unstable code with high regression risk — critical for PRD risk assessment.

    Args:
        area: Area keyword to analyze (e.g., 'auth', 'payment', 'order')

    Returns:
        Volatility scores, change frequency, and risk classification for matching files.
    """
    try:
        from src.database.neo4j_client import neo4j_client
        from src.intelligence.temporal_metadata import compute_volatility_score, classify_temporal_risk
        await neo4j_client.initialize()

        query = """
        MATCH (f:File)
        WHERE toLower(f.name) CONTAINS toLower($area) OR toLower(f.file) CONTAINS toLower($area)
        RETURN f.name as name, f.file as file,
               f.introduced_at as introduced_at, f.last_modified_at as last_modified_at,
               f.change_count as change_count, f.total_lines as total_lines
        LIMIT 20
        """
        files = await neo4j_client.execute_query(query, {"area": area})

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

            vol_score, breakdown = compute_volatility_score(
                last_modified_at=last_mod,
                change_count=f.get("change_count") or 0,
                total_lines=f.get("total_lines") or 0,
            )
            risk_level, risk_desc = classify_temporal_risk(vol_score, days_ago)
            results.append({
                "file": f.get("file", f.get("name", "")),
                "volatility_score": round(vol_score, 3),
                "risk_level": risk_level,
                "risk_description": risk_desc,
            })

        results.sort(key=lambda x: x["volatility_score"], reverse=True)
        return {
            "area": area,
            "files_analyzed": len(results),
            "high_risk_files": [r for r in results if r["risk_level"] in ("high", "critical")],
            "all_files": results[:15],
        }
    except Exception as e:
        return {"error": str(e)}


@tool
async def get_evidence_confidence_for_links(source_name: str) -> Dict[str, Any]:
    """
    Assess confidence of cross-language and cross-repository links using Evidence-Weighted Links.
    Essential for PRD recommendations — only recommend patterns backed by high-confidence links.

    Args:
        source_name: Source component/file name to check link confidence for

    Returns:
        Links with confidence scores, evidence types, and reliability assessment.
    """
    try:
        from src.database.neo4j_client import neo4j_client
        from src.intelligence.evidence_links import format_confidence_for_agent, CONFIDENCE_RELIABLE
        await neo4j_client.initialize()

        query = """
        MATCH (source)-[r]->(target)
        WHERE source.name CONTAINS $name OR source.file CONTAINS $name
        RETURN source.name as source, type(r) as relationship, target.name as target,
               r.confidence_score as confidence, r.evidence_types as evidence,
               r.link_origin as origin, labels(target)[0] as target_type
        LIMIT 30
        """
        links = await neo4j_client.execute_query(query, {"name": source_name})

        results = []
        for link in (links or []):
            confidence = link.get("confidence") or 0.5
            results.append({
                "source": link.get("source", ""),
                "relationship": link.get("relationship", ""),
                "target": link.get("target", ""),
                "target_type": link.get("target_type", ""),
                "confidence": round(confidence, 3),
                "is_reliable": confidence >= CONFIDENCE_RELIABLE,
                "assessment": format_confidence_for_agent(confidence),
                "evidence": link.get("evidence", ""),
            })

        reliable = [r for r in results if r["is_reliable"]]
        uncertain = [r for r in results if not r["is_reliable"]]

        return {
            "source": source_name,
            "total_links": len(results),
            "reliable_links": reliable[:15],
            "uncertain_links": uncertain[:10],
            "reliability_ratio": f"{len(reliable)}/{len(results)}" if results else "0/0",
        }
    except Exception as e:
        return {"error": str(e)}


@tool
async def get_cross_repository_relationships(repo_name: str = "") -> Dict[str, Any]:
    """
    Map cross-repository relationships and API call chains between frontend and backend.
    Critical for understanding full-stack data flow and API contracts.

    Args:
        repo_name: Optional repository name filter. If empty, shows all cross-repo links.

    Returns:
        Cross-repository links showing frontend→backend API calls and shared contracts.
    """
    try:
        from src.database.neo4j_client import neo4j_client
        await neo4j_client.initialize()

        # Find cross-repo links (ApiCall nodes linking FE components to BE routes)
        cross_query = """
        MATCH (a:ApiCall)-[:CALLS_ROUTE]->(r:VertxRoute)
        OPTIONAL MATCH (comp:ReactComponent)-[:MAKES_API_CALL]->(a)
        RETURN comp.name as frontend_component, a.endpoint as api_endpoint,
               a.method as http_method, r.path as backend_route,
               r.handler_method as handler, r.file as handler_file
        LIMIT 30
        """
        links = await neo4j_client.execute_query(cross_query)

        # Find EventBus patterns
        eventbus_query = """
        MATCH (e:EventBusNode)
        OPTIONAL MATCH (pub)-[:PUBLISHES_TO]->(e)
        OPTIONAL MATCH (sub)-[:CONSUMES_FROM]->(e)
        RETURN e.address as address, pub.name as publisher, sub.name as subscriber,
               e.file as file
        LIMIT 20
        """
        events = await neo4j_client.execute_query(eventbus_query)

        return {
            "api_links": [
                {
                    "frontend_component": l.get("frontend_component", ""),
                    "api_endpoint": l.get("api_endpoint", ""),
                    "http_method": l.get("http_method", ""),
                    "backend_route": l.get("backend_route", ""),
                    "handler": l.get("handler", ""),
                    "handler_file": l.get("handler_file", ""),
                }
                for l in (links or [])
            ],
            "event_bus_flows": [
                {
                    "address": e.get("address", ""),
                    "publisher": e.get("publisher", ""),
                    "subscriber": e.get("subscriber", ""),
                }
                for e in (events or [])
            ],
            "total_api_links": len(links or []),
            "total_event_flows": len(events or []),
        }
    except Exception as e:
        return {"error": str(e)}


@tool
async def get_middleware_chain(route_pattern: str) -> Dict[str, Any]:
    """
    Trace the middleware chain for specific routes — auth, validation, logging, CORS.
    Essential for understanding security layers and request lifecycle.

    Args:
        route_pattern: Route pattern to analyze (e.g., '/api', '/auth', 'admin')

    Returns:
        Routes with their middleware chains and handler functions.
    """
    try:
        from src.database.neo4j_client import neo4j_client
        await neo4j_client.initialize()

        query = """
        MATCH (r:VertxRoute)
        WHERE toLower(r.path) CONTAINS toLower($pattern)
        OPTIONAL MATCH (mw:Function)-[:MIDDLEWARE_FOR]->(r)
        RETURN r.path as path, r.method as method, r.handler_method as handler,
               r.file as file, collect(DISTINCT mw.name) as middleware_names
        ORDER BY r.path
        LIMIT 25
        """
        routes = await neo4j_client.execute_query(query, {"pattern": route_pattern})

        # Also find functions that look like middleware
        mw_query = """
        MATCH (f:Function)
        WHERE toLower(f.name) CONTAINS 'middleware'
           OR toLower(f.name) CONTAINS 'guard'
           OR toLower(f.name) CONTAINS 'validate'
           OR toLower(f.name) CONTAINS 'authenticate'
        RETURN f.name as name, f.file as file
        LIMIT 15
        """
        middleware_funcs = await neo4j_client.execute_query(mw_query)

        return {
            "routes": [
                {
                    "path": r.get("path", ""),
                    "method": r.get("method", ""),
                    "handler": r.get("handler", ""),
                    "file": r.get("file", ""),
                    "middleware": r.get("middleware_names", []),
                }
                for r in (routes or [])
            ],
            "available_middleware": [
                {"name": m.get("name", ""), "file": m.get("file", "")}
                for m in (middleware_funcs or [])
            ],
            "total_routes": len(routes or []),
        }
    except Exception as e:
        return {"error": str(e)}


@tool
async def get_function_signatures(keyword: str) -> Dict[str, Any]:
    """
    Search for function signatures with parameters, return types, and call counts.
    Use to find reusable utility functions, service methods, or controllers.

    Args:
        keyword: Function name or keyword to search for

    Returns:
        Function signatures with parameters, file locations, and usage counts.
    """
    try:
        from src.database.neo4j_client import neo4j_client
        await neo4j_client.initialize()

        query = """
        MATCH (f:Function)
        WHERE toLower(f.name) CONTAINS toLower($keyword)
        OPTIONAL MATCH (caller)-[:CALLS]->(f)
        WITH f, count(DISTINCT caller) as call_count
        RETURN f.name as name, f.file as file, f.params as params,
               f.return_type as return_type, f.line_start as line_start,
               f.line_end as line_end, call_count
        ORDER BY call_count DESC
        LIMIT 20
        """
        functions = await neo4j_client.execute_query(query, {"keyword": keyword})

        return {
            "functions": [
                {
                    "name": fn.get("name", ""),
                    "file": fn.get("file", ""),
                    "params": fn.get("params", ""),
                    "return_type": fn.get("return_type", ""),
                    "lines": f"{fn.get('line_start', '?')}-{fn.get('line_end', '?')}",
                    "call_count": fn.get("call_count", 0),
                }
                for fn in (functions or [])
            ],
            "total_found": len(functions or []),
        }
    except Exception as e:
        return {"error": str(e)}


# All PRD tools — 15 tools for thorough codebase investigation
prd_tools = [
    get_repository_overview,
    search_existing_implementations,
    find_api_routes_by_pattern,
    find_components_by_name,
    get_dependency_graph,
    get_behavioral_impact,
    get_domain_mapping,
    get_security_surface_for_area,
    read_file_content,
    get_database_schemas,
    # Intelligence-backed tools
    get_temporal_volatility,
    get_evidence_confidence_for_links,
    get_cross_repository_relationships,
    get_middleware_chain,
    get_function_signatures,
]


# ---------------------------------------------------------------------------
# PRD Agent (LangGraph tool-calling agent)
# ---------------------------------------------------------------------------

def should_continue(state: PRDState) -> Literal["tools", "synthesize", "end"]:
    """Determine if we should continue to tools, synthesize final output, or end."""
    messages = state.get("messages", [])
    if not messages:
        return "end"

    tool_call_count = sum(1 for m in messages if hasattr(m, 'tool_calls') and m.tool_calls)
    last_message = messages[-1]
    has_tool_calls = hasattr(last_message, 'tool_calls') and last_message.tool_calls

    # Max 15 tool iterations for PRD — needs thorough investigation with 15 tools
    if tool_call_count >= 15:
        if has_tool_calls or not (last_message.content or "").strip():
            logger.info("PRD agent hit tool iteration limit — routing to synthesize")
            return "synthesize"
        return "end"

    if has_tool_calls:
        return "tools"

    return "end"


def _build_llm_messages(messages, system_prompt: str) -> list:
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


def _build_llm_tools() -> list:
    """Convert prd_tools to LiteLLM tool schema."""
    llm_tools = []
    for t in prd_tools:
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


async def prd_agent(state: PRDState) -> Dict[str, Any]:
    """Main PRD agent node — codebase-grounded, zero hallucination."""
    try:
        messages = list(state.get('messages', []))

        tool_descriptions = "\n".join([f"- {t.name}: {t.description}" for t in prd_tools])

        system_prompt = f"""{PRD_SYSTEM_PROMPT}

## TOOLS AVAILABLE:
{tool_descriptions}

IMPORTANT: You MUST call tools to investigate the codebase BEFORE producing the PRD.
Start with get_repository_overview, then search_existing_implementations for EACH key requirement.
Use at least 5 tools before producing output.
Your final response MUST be a complete, valid JSON matching the PRD template structure.
Do NOT summarize or skip sections — fill every section with real codebase data.
"""

        llm_messages = _build_llm_messages(messages, system_prompt)
        llm_tools = _build_llm_tools()

        model_name = settings.get_litellm_model("architect")
        if "gemini" in settings.ARCHITECT_MODEL.lower():
            model_name = f"vertex_ai/{settings.ARCHITECT_MODEL}"

        response = await acompletion(
            model=model_name,
            messages=llm_messages,
            tools=llm_tools if llm_tools else None,
            tool_choice="auto",
            max_tokens=max(settings.LLM_MAX_TOKENS, 16000),
            temperature=0.1,
            timeout=settings.LLM_TIMEOUT,
            api_key=settings.get_api_key_for_model(settings.ARCHITECT_MODEL) or "vertex-ai"
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
        logger.error(f"PRD agent failed: {e}", exc_info=True)
        # Return error as message content so LangGraph can handle it gracefully
        return {"messages": [AIMessage(
            content=json.dumps({
                "error": f"PRD analysis failed: {str(e)}",
                "error_type": "agent_error",
                "agent": "prd"
            })
        )]}


async def execute_prd_tools(state: PRDState) -> Dict[str, Any]:
    """Execute tools called by the PRD agent."""
    messages = state.get("messages", [])
    last_message = messages[-1] if messages else None

    if not last_message or not hasattr(last_message, 'tool_calls') or not last_message.tool_calls:
        return {"messages": []}

    tool_messages = []
    for tool_call in last_message.tool_calls:
        tool_name = tool_call.get("name", "")
        tool_args = tool_call.get("args", {})
        tool_id = tool_call.get("id", f"call_{tool_name}")

        logger.info(f"PRD tool executing: {tool_name}", args=tool_args)

        tool_result = {"error": f"Tool {tool_name} not found"}
        for t in prd_tools:
            if t.name == tool_name:
                try:
                    tool_result = await t.ainvoke(tool_args)
                except Exception as e:
                    logger.error(f"PRD tool {tool_name} failed: {e}")
                    tool_result = {"error": str(e)}
                break

        tool_messages.append(ToolMessage(
            content=json.dumps(tool_result, default=str),
            tool_call_id=tool_id
        ))

    return {"messages": tool_messages}


async def synthesize_prd(state: PRDState) -> Dict[str, Any]:
    """
    Synthesize the final PRD when the tool iteration limit is reached.
    Uses a COMPACT summary of tool results (NOT the full conversation) to avoid
    overwhelming the LLM with a huge context that causes empty responses.
    """
    try:
        messages = list(state.get('messages', []))
        logger.info("Synthesizing final PRD after tool limit", message_count=len(messages))

        # Extract the original user question
        original_question = ""
        for msg in messages:
            if isinstance(msg, HumanMessage):
                original_question = msg.content
                break

        # Collect and summarize tool results into a compact context
        tool_findings = []
        for msg in messages:
            if isinstance(msg, ToolMessage):
                try:
                    data = json.loads(msg.content) if msg.content else {}
                    # Truncate each tool result to keep context manageable
                    summary = json.dumps(data, default=str)
                    if len(summary) > 3000:
                        summary = summary[:3000] + "...(truncated)"
                    tool_findings.append(summary)
                except (json.JSONDecodeError, TypeError):
                    tool_findings.append(str(msg.content)[:2000])

        # Build a compact codebase context from tool results
        codebase_context = "\n---\n".join(tool_findings[-12:])  # Last 12 results max

        synthesis_prompt = f"""{PRD_SYSTEM_PROMPT}

## CODEBASE INVESTIGATION RESULTS
The following data was gathered by searching the codebase knowledge graph:

{codebase_context}

## YOUR TASK
Using ONLY the codebase data above and the client requirements below, produce a COMPLETE
enterprise-grade PRD as a single valid JSON document.

CRITICAL FORMATTING RULES:
1. Fill EVERY section. If data is missing, write "No relevant artifacts found in codebase".
2. Do NOT invent APIs, files, or services not present in the tool results above.
3. Keep descriptions CONCISE — each field value under 200 characters where possible.
4. For arrays of items (requirements, stories, scenarios), keep each item focused and brief.
5. Do NOT call any tools. Do NOT ask questions.
6. Your response must be ONLY the JSON object — no markdown fences, no explanation text.
"""

        llm_messages = [
            {"role": "system", "content": synthesis_prompt},
            {"role": "user", "content": original_question or "Generate the PRD based on the codebase analysis."},
        ]

        model_name = settings.get_litellm_model("architect")
        if "gemini" in settings.ARCHITECT_MODEL.lower():
            model_name = f"vertex_ai/{settings.ARCHITECT_MODEL}"

        # Use higher token limit for synthesis — PRD JSON can be 8000+ tokens
        synthesis_max_tokens = max(settings.LLM_MAX_TOKENS, 16000)

        response = await acompletion(
            model=model_name,
            messages=llm_messages,
            tools=None,
            max_tokens=synthesis_max_tokens,
            temperature=0.1,
            timeout=settings.LLM_TIMEOUT,
            api_key=settings.get_api_key_for_model(settings.ARCHITECT_MODEL) or "vertex-ai"
        )

        content = response.choices[0].message.content or ""
        logger.info("PRD synthesis complete", response_length=len(content))

        if not content.strip():
            logger.warning("PRD synthesis returned empty — using fallback")
            content = json.dumps({
                "feature_name": "Generated PRD",
                "executive_summary": {
                    "problem_statement": original_question[:1000],
                    "objective": "See codebase analysis results below.",
                },
                "raw_analysis": "\n\n".join(tool_findings[-6:]),
            })

        return {"messages": [AIMessage(content=content)]}

    except Exception as e:
        logger.error(f"PRD synthesis failed: {e}", exc_info=True)
        # Return error as JSON so the orchestrator can detect and handle it
        return {"messages": [AIMessage(
            content=json.dumps({
                "error": f"PRD synthesis failed: {str(e)}",
                "error_type": "llm_service_error",
                "model": settings.ARCHITECT_MODEL
            })
        )]}


def create_prd_workflow() -> StateGraph:
    """Create and compile the PRD agent workflow."""
    graph = StateGraph(PRDState)

    graph.add_node("agent", prd_agent)
    graph.add_node("tools", execute_prd_tools)
    graph.add_node("synthesize", synthesize_prd)

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
prd_workflow = create_prd_workflow()
