"""
Developer Assistant Workflow using LangGraph
Provides interactive assistance for developers with code understanding and debugging

Enhanced with:
    - Domain-first query resolution
    - Confidence-aware tool responses
    - Behavioral context in search results
    - Trust-preserving silence when evidence is insufficient
"""
import logging
import yaml
import json
import asyncio
from typing import Dict, Any, List, Literal
from pathlib import Path
from datetime import datetime

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from litellm import acompletion

from config.settings import settings
from src.services.search_service import SearchService
from src.services.graph_service import GraphService
from src.agents.state_schema import DeveloperState
from src.intelligence.domain_model import domain_engine
from src.intelligence.evidence_links import format_confidence_for_agent
from src.core.logger import get_logger
from src.core.exceptions import AgentException

logger = get_logger(__name__)

# Load prompts
PROMPTS_PATH = Path(__file__).parent.parent.parent.parent / "config" / "dev_prompt.yaml"
try:
    with open(PROMPTS_PATH, 'r') as f:
        prompts = yaml.safe_load(f) or {}
except FileNotFoundError:
    logger.warning(f"Developer prompts file not found: {PROMPTS_PATH}")
    prompts = {}

# Default developer system prompt - ENTERPRISE-GRADE, DETERMINISTIC
DEFAULT_DEVELOPER_PROMPT = """
You are an expert software engineer assistant with deep knowledge of the INGESTED codebase only.

CRITICAL RULES:
1. You ONLY answer questions about the ingested codebase(s)
2. You NEVER hallucinate code, files, or components  
3. If data is not found, explicitly say "This information is not available in the ingested repositories"
4. ALWAYS include actual code snippets when the user asks for code

TOOL SELECTION GUIDE (PICK THE RIGHT TOOL):
- For API routes/endpoints questions → Use `list_api_routes` (BEST for finding routes!)
- For specific code patterns (router.get, function, etc.) → Use `search_code_pattern`
- For reading a specific file's content → Use `read_file_content`
- For general codebase search → Use `search_codebase`
- For architecture overview → Use `get_repository_architecture`
- For impact analysis → Use `analyze_impact`
- For security/auth questions (which routes have auth?) → Use `get_security_surface`
- For behavioral analysis (side-effects, auth-critical, async) → Use `get_behavioral_classification`
- For domain-level questions (how does auth/payments work?) → Use `get_domain_context`
- For cross-repo tracing (frontend→backend flow) → Use `trace_cross_repo_flow`
- For file imports/dependencies → Use `get_file_dependencies`
- For QA/testing/risk assessment → Use `get_qa_risk_assessment`

TOOL USAGE RULES:
- Maximum 8 tool calls per question
- Pick the MOST SPECIFIC tool for the question
- After getting results with code, IMMEDIATELY provide your answer
- If first tool doesn't find results, try a more specific tool

RESPONSE WORKFLOW:
1. Analyze what the user is asking for
2. Pick the RIGHT tool (see guide above)
3. Call 1-2 tools to get data
4. PROVIDE FINAL ANSWER with actual code snippets

You have access to these tools:
{tool_descriptions}

RESPONSE FORMAT:
📁 **File(s)**: [list the relevant files found]
🏷️ **Repository**: [repository name]
💡 **Answer**: [your detailed response with CODE SNIPPETS]
📊 **Confidence**: [high/medium/low]

Current context files: {context_files}
"""


# Define tools for the developer agent
@tool
async def search_codebase(query: str, search_type: str = "graph_first") -> Dict[str, Any]:
    """
    Search the codebase using GRAPH-FIRST resolution order.
    
    Resolution order:
    1. Neo4j graph (routes, components, functions, files)
    2. Qdrant vectors (semantic search)
    3. Explicit error if nothing found
    
    Args:
        query: The search query describing what you're looking for
        search_type: "graph_first" (default), "semantic", or "graph_only"
    
    Returns:
        Search results with relevant code snippets and file locations
    """
    try:
        search_service = SearchService()
        await search_service.initialize()
        
        # STEP 1: Graph search first (Neo4j)
        graph_results = []
        if search_type in ["graph_first", "graph_only", "hybrid"]:
            graph_response = await search_service.search_codebase(
                query=query,
                search_type="graph",
                top_k=10
            )
            graph_results = graph_response.get("results", {}).get("graph", [])[:5]
        
        # STEP 2: Vector search second (Qdrant) - run if graph didn't fully answer
        vector_results = []
        if search_type != "graph_only" and len(graph_results) < 5:
            vector_response = await search_service.search_codebase(
                query=query,
                search_type="semantic",
                top_k=10
            )
            vector_results = vector_response.get("results", {}).get("semantic", [])[:5]
        
        # STEP 3: Format results with REQUIRED metadata
        formatted = []
        for result in graph_results:
            formatted.append({
                "name": result.get("name", ""),
                "type": result.get("type", ""),
                "file": result.get("file", ""),
                "relative_path": result.get("relative_path", result.get("file", "")),
                "repository_name": result.get("repository_name", ""),
                "repo_type": result.get("repo_type", ""),
                "score": result.get("score", 1.0),
                "snippet": result.get("text", "")[:500],
                "source": "graph"
            })
        
        for result in vector_results:
            formatted.append({
                "name": result.get("name", ""),
                "type": result.get("type", ""),
                "file": result.get("file", ""),
                "relative_path": result.get("relative_path", result.get("file", "")),
                "repository_name": result.get("repository_name", ""),
                "repo_type": result.get("repo_type", ""),
                "score": result.get("score", 0),
                "snippet": result.get("text", "")[:500],
                "source": "vector"
            })
        
        # STEP 4: Explicit error if nothing found
        if not formatted:
            return {
                "query": query,
                "results_count": 0,
                "error": "NO_DATA_FOUND",
                "message": "No matching code found in the ingested repositories. This information may not be available in the codebase.",
                "results": []
            }
        
        return {
            "query": query,
            "results_count": len(formatted),
            "graph_results": len(graph_results),
            "vector_results": len(vector_results),
            "results": formatted
        }
    except Exception as e:
        return {"error": str(e), "query": query}


@tool
async def analyze_impact(file_path: str, change_type: str = "modify") -> Dict[str, Any]:
    """
    Analyze the impact of changing a specific file.
    
    Args:
        file_path: Path to the file being changed
        change_type: Type of change - "modify", "delete", or "add"
    
    Returns:
        Impact analysis including affected components and risk assessment
    """
    try:
        graph_service = GraphService()
        
        results = await graph_service.analyze_impact(
            file_path=file_path,
            change_type=change_type
        )
        
        return {
            "file": file_path,
            "change_type": change_type,
            "impact_score": results.get("impact_score", 0),
            "risk_level": results.get("risk_level", "unknown"),
            "affected_components": results.get("breakdown", {}),
            "details_summary": {
                "direct_dependencies": len(results.get("details", {}).get("direct_dependencies", [])),
                "api_calls": len(results.get("details", {}).get("api_calls", [])),
                "eventbus_flows": len(results.get("details", {}).get("eventbus_flows", []))
            }
        }
    except Exception as e:
        return {"error": str(e), "file_path": file_path}


@tool
async def find_component(component_name: str, component_type: str = None) -> Dict[str, Any]:
    """
    Find a specific component and its relationships in the codebase.
    
    Args:
        component_name: Name of the component to find
        component_type: Optional type filter (VertxRoute, ReactComponent, DTO, Service)
    
    Returns:
        Component details and its relationships
    """
    try:
        search_service = SearchService()
        await search_service.initialize()
        
        results = await search_service.search_by_component(
            component_name=component_name,
            component_type=component_type,
            include_related=True
        )
        
        return {
            "component_name": component_name,
            "found": results.get("component") is not None,
            "component": results.get("component"),
            "relationships": {
                "incoming": len(results.get("relationships", {}).get("incoming", [])),
                "outgoing": len(results.get("relationships", {}).get("outgoing", []))
            },
            "similar_components": results.get("similar_components", [])[:3]
        }
    except Exception as e:
        return {"error": str(e), "component_name": component_name}


@tool
async def get_api_flow(api_endpoint: str) -> Dict[str, Any]:
    """
    Trace the complete flow of an API endpoint from frontend to backend.
    
    Args:
        api_endpoint: The API endpoint path (e.g., "/api/users")
    
    Returns:
        Complete flow showing frontend component → API call → backend route → services
    """
    try:
        from src.data_ingestion.graph_builder.edge_linker import EdgeLinker
        
        edge_linker = EdgeLinker()
        flow = await edge_linker.get_api_flow(api_endpoint)
        
        return {
            "endpoint": api_endpoint,
            "flow": flow,
            "has_frontend": flow.get("frontend_component") is not None,
            "has_backend": flow.get("backend_route") is not None,
            "dtos_used": len(flow.get("dtos_used", [])),
            "eventbus_flows": len(flow.get("eventbus_flows", []))
        }
    except Exception as e:
        return {"error": str(e), "endpoint": api_endpoint}


@tool
async def get_codebase_metrics() -> Dict[str, Any]:
    """
    Get overall codebase metrics and statistics.
    
    Returns:
        Metrics including component counts, relationship counts, and repository info
    """
    try:
        graph_service = GraphService()
        metrics = await graph_service.get_codebase_metrics()
        
        return {
            "timestamp": metrics.get("timestamp"),
            "totals": metrics.get("totals", {}),
            "component_breakdown": metrics.get("breakdown", {}).get("component_counts", []),
            "top_connected": metrics.get("breakdown", {}).get("top_connected_components", [])[:5]
        }
    except Exception as e:
        return {"error": str(e)}


@tool
async def get_repository_architecture(repo_name: str = None) -> Dict[str, Any]:
    """
    Get detailed architecture information about a repository including technology stack,
    main files, routes, and folder structure. USE THIS FIRST to understand the codebase!
    
    Args:
        repo_name: Optional repository name filter (e.g., 'intergration-studio-be')
    
    Returns:
        Architecture details including technology stack, entry points, routes, and structure
    """
    try:
        from src.database.neo4j_client import neo4j_client
        await neo4j_client.initialize()
        
        # Get repository info
        repo_query = """
        MATCH (r:Repository)
        WHERE $repo_name IS NULL OR r.name CONTAINS $repo_name
        RETURN r.name as name, r.language as language, r.path as path
        """
        repos = await neo4j_client.execute_query(repo_query, {"repo_name": repo_name})
        
        result = {"repositories": []}
        
        for repo in repos:
            repo_info = {
                "name": repo["name"],
                "detected_language": repo["language"],
                "path": repo["path"],
                "structure": {},
                "entry_points": [],
                "routes": [],
                "key_files": []
            }
            
            # Get file structure overview
            files_query = """
            MATCH (f:File)-[:BELONGS_TO]->(r:Repository {name: $name})
            RETURN f.path as path, f.language as language, f.framework as framework
            ORDER BY f.path
            LIMIT 50
            """
            files = await neo4j_client.execute_query(files_query, {"name": repo["name"]})
            
            # Analyze structure
            folders = set()
            languages = {}
            frameworks = set()
            for f in files:
                path = f["path"]
                parts = path.split("/")
                if len(parts) > 1:
                    folders.add(parts[0])
                lang = f["language"] or "unknown"
                languages[lang] = languages.get(lang, 0) + 1
                if f["framework"]:
                    frameworks.add(f["framework"])
                # Identify entry points
                if path in ["index.js", "app.js", "server.js", "main.py", "Application.java"]:
                    repo_info["entry_points"].append(path)
            
            repo_info["structure"] = {
                "folders": list(folders)[:20],
                "languages": languages,
                "frameworks": list(frameworks),
                "file_count": len(files)
            }
            
            # Get routes — query VertxRoute nodes (backend) using correct relationship
            routes_query = """
            MATCH (f:File)-[:BELONGS_TO]->(r:Repository {name: $name})
            MATCH (f)-[:CONTAINS]->(route:VertxRoute)
            RETURN route.method as method, route.path as path, route.handler_method as handler, route.file as file
            ORDER BY route.path
            LIMIT 40
            """
            routes = await neo4j_client.execute_query(routes_query, {"name": repo["name"]})
            repo_info["routes"] = [
                {"method": r["method"], "path": r["path"], "handler": r.get("handler", ""), "file": r.get("file", "")}
                for r in routes
            ]
            
            # Get React components (frontend)
            components_query = """
            MATCH (f:File)-[:BELONGS_TO]->(r:Repository {name: $name})
            MATCH (f)-[:CONTAINS]->(c:ReactComponent)
            RETURN c.name as name, c.file as file
            ORDER BY c.name
            LIMIT 30
            """
            components = await neo4j_client.execute_query(components_query, {"name": repo["name"]})
            repo_info["react_components"] = [
                {"name": c["name"], "file": c.get("file", "")} for c in components
            ]
            
            # Get API calls (frontend → backend connections)
            api_calls_query = """
            MATCH (f:File)-[:BELONGS_TO]->(r:Repository {name: $name})
            MATCH (f)-[:CONTAINS]->(a:ApiCall)
            RETURN a.url as url, a.method as method, a.file as file
            LIMIT 20
            """
            api_calls = await neo4j_client.execute_query(api_calls_query, {"name": repo["name"]})
            repo_info["api_calls"] = [
                {"url": a.get("url", ""), "method": a.get("method", ""), "file": a.get("file", "")}
                for a in api_calls
            ]
            
            # Get relationship summary
            rels_query = """
            MATCH (f:File)-[:BELONGS_TO]->(r:Repository {name: $name})
            MATCH (f)-[rel]-()
            RETURN type(rel) as rel_type, count(*) as cnt
            ORDER BY cnt DESC
            LIMIT 10
            """
            rels = await neo4j_client.execute_query(rels_query, {"name": repo["name"]})
            repo_info["relationship_types"] = {r["rel_type"]: r["cnt"] for r in rels}
            
            # Get key files (controllers, services, models, routes)
            key_files_query = """
            MATCH (f:File)-[:BELONGS_TO]->(r:Repository {name: $name})
            WHERE f.path CONTAINS 'controller' OR f.path CONTAINS 'service' 
                  OR f.path CONTAINS 'model' OR f.path CONTAINS 'route'
                  OR f.path CONTAINS 'middleware' OR f.path CONTAINS 'auth'
            RETURN f.path as path
            LIMIT 30
            """
            key_files = await neo4j_client.execute_query(key_files_query, {"name": repo["name"]})
            repo_info["key_files"] = [f["path"] for f in key_files]
            
            result["repositories"].append(repo_info)
        
        return result
        
    except Exception as e:
        return {"error": str(e)}


@tool
async def read_file_content(file_path: str, max_lines: int = 200) -> Dict[str, Any]:
    """
    Read the actual source code content of a file from the ingested repositories.
    Use this to get the ACTUAL CODE when you need to show code snippets.
    
    Args:
        file_path: The file path (can be partial, will search for matching files)
        max_lines: Maximum lines to return (default 200)
    
    Returns:
        The actual source code content of the file
    """
    try:
        from src.database.neo4j_client import neo4j_client
        import os
        
        await neo4j_client.initialize()
        
        # Find the file in the graph
        file_query = """
        MATCH (f:File)
        WHERE f.path CONTAINS $file_path OR f.name = $file_path
        RETURN f.path as path, f.name as name, f.repository_name as repo, f.full_path as full_path
        LIMIT 5
        """
        files = await neo4j_client.execute_query(file_query, {"file_path": file_path})
        
        if not files:
            return {"error": f"File not found: {file_path}", "suggestion": "Try using a partial filename like 'integrationRoutes.js'"}
        
        results = []
        for file_info in files:
            full_path = file_info.get("full_path") or file_info.get("path")
            
            # Try to find the file in the repos directory
            base_dir = str(settings.REPOS_BASE_DIR)
            
            # Search for the file
            actual_path = None
            for root, dirs, filenames in os.walk(base_dir):
                for filename in filenames:
                    if filename == file_info.get("name") or file_path in os.path.join(root, filename):
                        potential_path = os.path.join(root, filename)
                        if os.path.exists(potential_path):
                            actual_path = potential_path
                            break
                if actual_path:
                    break
            
            if actual_path and os.path.exists(actual_path):
                with open(actual_path, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()[:max_lines]
                    content = ''.join(lines)
                    results.append({
                        "file": file_info.get("name"),
                        "path": file_info.get("path"),
                        "repository": file_info.get("repo"),
                        "content": content,
                        "lines_read": len(lines),
                        "total_lines": len(open(actual_path).readlines()) if os.path.exists(actual_path) else 0
                    })
        
        if not results:
            return {"error": f"Could not read file content for: {file_path}"}
        
        return {"files": results, "count": len(results)}
        
    except Exception as e:
        return {"error": str(e), "file_path": file_path}


@tool
async def search_code_pattern(pattern: str, file_extension: str = None, repo_name: str = None) -> Dict[str, Any]:
    """
    Search for a specific code pattern (like 'router.get', 'router.post', 'function', etc.)
    across all files in the ingested repositories. Returns matching code snippets.
    
    Args:
        pattern: The code pattern to search for (e.g., "router.get", "router.post", "async function")
        file_extension: Optional file extension filter (e.g., "js", "py", "ts")
        repo_name: Optional repository name filter
    
    Returns:
        List of files and code snippets containing the pattern
    """
    try:
        import os
        import re
        
        base_dir = str(settings.REPOS_BASE_DIR)
        matches = []
        
        for root, dirs, files in os.walk(base_dir):
            # Skip node_modules and common non-source directories
            dirs[:] = [d for d in dirs if d not in ['node_modules', '.git', '__pycache__', 'dist', 'build']]
            
            for filename in files:
                # Filter by extension if specified
                if file_extension and not filename.endswith(f".{file_extension}"):
                    continue
                
                # Filter by repo name if specified
                if repo_name and repo_name not in root:
                    continue
                
                # Only search code files
                if not filename.endswith(('.js', '.ts', '.jsx', '.tsx', '.py', '.java', '.go', '.rs', '.vue', '.svelte')):
                    continue
                
                filepath = os.path.join(root, filename)
                
                try:
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        lines = content.split('\n')
                        
                        for i, line in enumerate(lines):
                            if pattern.lower() in line.lower():
                                # Get context (2 lines before and after)
                                start = max(0, i - 2)
                                end = min(len(lines), i + 3)
                                context = '\n'.join(lines[start:end])
                                
                                # Get relative path
                                rel_path = filepath.replace(base_dir + '/', '')
                                
                                matches.append({
                                    "file": filename,
                                    "path": rel_path,
                                    "line_number": i + 1,
                                    "matching_line": line.strip(),
                                    "context": context
                                })
                                
                                # Limit matches per file
                                if len([m for m in matches if m['file'] == filename]) >= 10:
                                    break
                except Exception:
                    continue
                
                # Limit total matches
                if len(matches) >= 50:
                    break
            
            if len(matches) >= 50:
                break
        
        return {
            "pattern": pattern,
            "matches_found": len(matches),
            "matches": matches[:30],  # Return top 30 matches
            "file_extension_filter": file_extension,
            "repo_filter": repo_name
        }
        
    except Exception as e:
        return {"error": str(e), "pattern": pattern}


@tool
async def list_api_routes(repo_name: str = None, include_code: bool = True) -> Dict[str, Any]:
    """
    List ALL API routes (GET, POST, PUT, DELETE, PATCH) in the ingested repositories
    with their actual code definitions. BEST TOOL for finding API endpoints!
    
    Args:
        repo_name: Optional repository name filter (e.g., 'intergration-studio-be')
        include_code: Whether to include the actual route code (default True)
    
    Returns:
        Complete list of API routes with code snippets
    """
    try:
        import os
        import re
        
        base_dir = str(settings.REPOS_BASE_DIR)
        routes = []
        
        # Patterns for different frameworks
        route_patterns = [
            # Express.js patterns
            (r"router\.(get|post|put|delete|patch)\s*\(['\"]([^'\"]+)['\"]", "express"),
            (r"app\.(get|post|put|delete|patch)\s*\(['\"]([^'\"]+)['\"]", "express"),
            # FastAPI patterns
            (r"@(app|router)\.(get|post|put|delete|patch)\s*\(['\"]([^'\"]+)['\"]", "fastapi"),
            # Flask patterns
            (r"@(app|blueprint)\.route\s*\(['\"]([^'\"]+)['\"].*methods=\[([^\]]+)\]", "flask"),
            # Spring/Java patterns
            (r"@(GetMapping|PostMapping|PutMapping|DeleteMapping|PatchMapping)\s*\(['\"]?([^'\")\s]+)", "spring"),
        ]
        
        for root, dirs, files in os.walk(base_dir):
            dirs[:] = [d for d in dirs if d not in ['node_modules', '.git', '__pycache__', 'dist', 'build']]
            
            # Filter by repo if specified
            if repo_name and repo_name not in root:
                continue
            
            for filename in files:
                # Focus on route files
                if not filename.endswith(('.js', '.ts', '.py', '.java')):
                    continue
                
                filepath = os.path.join(root, filename)
                
                try:
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        lines = content.split('\n')
                        
                        for i, line in enumerate(lines):
                            for pattern, framework in route_patterns:
                                match = re.search(pattern, line, re.IGNORECASE)
                                if match:
                                    groups = match.groups()
                                    
                                    if framework == "express":
                                        method = groups[0].upper()
                                        path = groups[1]
                                    elif framework == "fastapi":
                                        method = groups[1].upper()
                                        path = groups[2] if len(groups) > 2 else groups[1]
                                    elif framework == "spring":
                                        method_map = {"GetMapping": "GET", "PostMapping": "POST", 
                                                     "PutMapping": "PUT", "DeleteMapping": "DELETE", "PatchMapping": "PATCH"}
                                        method = method_map.get(groups[0], groups[0])
                                        path = groups[1]
                                    else:
                                        method = "ROUTE"
                                        path = groups[1] if len(groups) > 1 else groups[0]
                                    
                                    rel_path = filepath.replace(base_dir + '/', '')
                                    
                                    route_info = {
                                        "method": method,
                                        "path": path,
                                        "file": filename,
                                        "file_path": rel_path,
                                        "line_number": i + 1,
                                        "framework": framework
                                    }
                                    
                                    if include_code:
                                        # Get the route handler code (next 10 lines or until next route)
                                        start = max(0, i)
                                        end = min(len(lines), i + 15)
                                        route_info["code_snippet"] = '\n'.join(lines[start:end])
                                    
                                    routes.append(route_info)
                except Exception:
                    continue
        
        # Group by method
        by_method = {}
        for route in routes:
            method = route["method"]
            if method not in by_method:
                by_method[method] = []
            by_method[method].append(route)
        
        return {
            "total_routes": len(routes),
            "by_method": {k: len(v) for k, v in by_method.items()},
            "routes": routes[:50],  # Return top 50 routes
            "repo_filter": repo_name
        }
        
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Intelligence-powered tools (behavioral, temporal, domain, security, QA)
# ---------------------------------------------------------------------------

@tool
async def get_behavioral_classification(file_path: str = None, function_name: str = None) -> Dict[str, Any]:
    """
    Get behavioral classification of code nodes — AuthCritical, SideEffecting,
    AsyncFanOut, TransactionBoundary, Idempotent, ExternallyVisible.
    USE THIS for security, auth, side-effect, and risk questions.
    
    Args:
        file_path: Optional file path filter
        function_name: Optional function/handler name filter
    
    Returns:
        Behavioral labels, confidence, and evidence for matching nodes
    """
    try:
        from src.database.neo4j_client import neo4j_client
        from src.intelligence.behavioral_layer import behavioral_classifier
        await neo4j_client.initialize()

        # Build query based on filters
        conditions = []
        params: Dict[str, Any] = {}
        if file_path:
            conditions.append("n.file CONTAINS $file_path")
            params["file_path"] = file_path
        if function_name:
            conditions.append("(n.name CONTAINS $func_name OR n.handler_method CONTAINS $func_name)")
            params["func_name"] = function_name

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        query = f"""
        MATCH (n)
        WHERE (n:Function OR n:VertxRoute OR n:ReactComponent)
        {f'AND {" AND ".join(conditions)}' if conditions else ''}
        OPTIONAL MATCH (f:File)-[:CONTAINS]->(n)
        OPTIONAL MATCH (f)-[:BELONGS_TO]->(r:Repository)
        RETURN n.name as name, labels(n) as labels, n.file as file,
               n.method as http_method, n.handler_method as handler,
               n.code as code, r.name as repo
        LIMIT 30
        """
        nodes = await neo4j_client.execute_query(query, params)

        classifications = []
        for node in (nodes or []):
            cls = behavioral_classifier.classify_node(
                node_id=node.get("name", ""),
                name=node.get("name", "") or node.get("handler", ""),
                code_body=node.get("code", "") or "",
                node_labels=node.get("labels", []),
                http_method=node.get("http_method"),
                is_handler=bool(node.get("handler")),
            )
            if cls.labels:
                classifications.append({
                    "name": node.get("name", ""),
                    "file": node.get("file", ""),
                    "repo": node.get("repo", ""),
                    "behavioral_labels": [l.value for l in cls.labels],
                    "confidence": round(cls.confidence, 3),
                    "evidence": cls.evidence[:5],
                })

        # Summary
        label_counts: Dict[str, int] = {}
        for c in classifications:
            for lbl in c["behavioral_labels"]:
                label_counts[lbl] = label_counts.get(lbl, 0) + 1

        return {
            "total_classified": len(classifications),
            "label_summary": label_counts,
            "nodes": classifications[:20],
            "filter_applied": {"file_path": file_path, "function_name": function_name},
        }
    except Exception as e:
        return {"error": str(e)}


@tool
async def get_security_surface() -> Dict[str, Any]:
    """
    Analyze the security surface of the codebase: which routes have auth middleware,
    which don't, and which code nodes are AuthCritical.
    USE THIS for any security, authentication, or authorization questions.
    
    Returns:
        Routes grouped by auth status, AuthCritical nodes, and security gaps
    """
    try:
        from src.database.neo4j_client import neo4j_client
        await neo4j_client.initialize()

        # Get all routes with their handler info
        routes_query = """
        MATCH (route:VertxRoute)
        RETURN route.path as path, route.method as method, 
               route.handler_method as handler, route.file as file
        ORDER BY route.path
        """
        routes = await neo4j_client.execute_query(routes_query)

        protected = []
        unprotected = []
        for r in (routes or []):
            handler = (r.get("handler") or "").lower()
            file_name = (r.get("file") or "").lower()
            route_info = {
                "method": r.get("method", ""),
                "path": r.get("path", ""),
                "handler": r.get("handler", ""),
                "file": r.get("file", ""),
            }
            # Heuristic: check if auth middleware is in the handler chain
            if any(kw in handler for kw in ["auth", "middleware", "protect", "guard"]):
                protected.append(route_info)
            elif any(kw in file_name for kw in ["auth"]):
                protected.append(route_info)
            elif r.get("method", "").upper() in ("GET",) and r.get("path", "") in ("/health", "/health/details"):
                unprotected.append({**route_info, "reason": "Public health endpoint"})
            else:
                unprotected.append(route_info)

        # Get nodes with auth-related patterns
        auth_query = """
        MATCH (n:Function)
        WHERE n.name =~ '(?i).*(auth|login|logout|token|session|password|credential).*'
        RETURN n.name as name, n.file as file
        LIMIT 20
        """
        auth_nodes = await neo4j_client.execute_query(auth_query)

        return {
            "total_routes": len(routes or []),
            "protected_routes": len(protected),
            "unprotected_routes": len(unprotected),
            "protected": protected[:20],
            "unprotected": unprotected[:20],
            "auth_related_functions": [{"name": n["name"], "file": n.get("file", "")} for n in (auth_nodes or [])],
            "security_gaps": [r for r in unprotected if r.get("method", "").upper() in ("POST", "PUT", "DELETE", "PATCH")],
        }
    except Exception as e:
        return {"error": str(e)}


@tool
async def get_domain_context(query: str) -> Dict[str, Any]:
    """
    Resolve a user query to business domains (authentication, payments, etc.)
    and find all code nodes in those domains across ALL repositories.
    USE THIS for high-level questions like 'how does auth work' or 'show me payment flow'.
    
    Args:
        query: Natural language question about a business capability
    
    Returns:
        Matched domains, confidence, and relevant code nodes in each domain
    """
    try:
        from src.intelligence.domain_model import domain_engine
        from src.database.neo4j_client import neo4j_client
        await neo4j_client.initialize()

        # Step 1: Resolve query to domains
        domain_matches = domain_engine.resolve_query_to_domains(query)

        if not domain_matches:
            return {
                "query": query,
                "domains_found": 0,
                "message": "No business domain could be inferred from this query.",
            }

        result_domains = []
        for domain_name, relevance in domain_matches[:3]:
            # Step 2: Search for nodes in this domain using domain keywords
            keywords = domain_engine._domain_keywords.get(domain_name, [])
            keyword_pattern = "|".join(kw.replace(".", "\\.?") for kw in keywords[:10])

            # Search routes matching domain
            routes_query = """
            MATCH (r:VertxRoute)
            WHERE r.path =~ $pattern OR r.handler_method =~ $pattern
            RETURN r.path as path, r.method as method, r.handler_method as handler, r.file as file
            LIMIT 15
            """
            routes = await neo4j_client.execute_query(
                routes_query, {"pattern": f"(?i).*({keyword_pattern}).*"}
            )

            # Search functions matching domain
            funcs_query = """
            MATCH (f:Function)
            WHERE f.name =~ $pattern OR f.file =~ $pattern
            RETURN f.name as name, f.file as file
            LIMIT 15
            """
            funcs = await neo4j_client.execute_query(
                funcs_query, {"pattern": f"(?i).*({keyword_pattern}).*"}
            )

            # Search components
            comps_query = """
            MATCH (c:ReactComponent)
            WHERE c.name =~ $pattern OR c.file =~ $pattern
            RETURN c.name as name, c.file as file
            LIMIT 10
            """
            comps = await neo4j_client.execute_query(
                comps_query, {"pattern": f"(?i).*({keyword_pattern}).*"}
            )

            result_domains.append({
                "domain": domain_name,
                "relevance": round(relevance, 3),
                "routes": [{"method": r["method"], "path": r["path"], "handler": r.get("handler", ""), "file": r.get("file", "")} for r in (routes or [])],
                "functions": [{"name": f["name"], "file": f.get("file", "")} for f in (funcs or [])],
                "components": [{"name": c["name"], "file": c.get("file", "")} for c in (comps or [])],
            })

        return {
            "query": query,
            "domains_found": len(result_domains),
            "domains": result_domains,
        }
    except Exception as e:
        return {"error": str(e)}


@tool
async def trace_cross_repo_flow(endpoint_or_component: str) -> Dict[str, Any]:
    """
    Trace the complete cross-repository flow for an API endpoint or component.
    Finds: Frontend component → API call → Backend route → Handler → Service layer.
    USE THIS for cross-repo questions about how frontend connects to backend.
    
    Args:
        endpoint_or_component: API path (e.g. '/login') or component name (e.g. 'Login')
    
    Returns:
        Complete flow trace across repositories with confidence
    """
    try:
        from src.database.neo4j_client import neo4j_client
        await neo4j_client.initialize()

        search_term = endpoint_or_component
        trace: Dict[str, Any] = {
            "search_term": search_term,
            "frontend": {"components": [], "api_calls": []},
            "backend": {"routes": [], "handlers": [], "services": []},
        }

        # 1. Find backend routes matching the endpoint
        routes_query = """
        MATCH (r:VertxRoute)
        WHERE r.path CONTAINS $term OR r.handler_method CONTAINS $term
        RETURN r.path as path, r.method as method, r.handler_method as handler, r.file as file
        LIMIT 10
        """
        routes = await neo4j_client.execute_query(routes_query, {"term": search_term})
        trace["backend"]["routes"] = [
            {"method": r["method"], "path": r["path"], "handler": r.get("handler", ""), "file": r.get("file", "")}
            for r in (routes or [])
        ]

        # 2. Find frontend API calls matching the endpoint
        api_calls_query = """
        MATCH (a:ApiCall)
        WHERE a.url CONTAINS $term OR a.file CONTAINS $term
        RETURN a.url as url, a.method as method, a.file as file
        LIMIT 10
        """
        api_calls = await neo4j_client.execute_query(api_calls_query, {"term": search_term})
        trace["frontend"]["api_calls"] = [
            {"url": a.get("url", ""), "method": a.get("method", ""), "file": a.get("file", "")}
            for a in (api_calls or [])
        ]

        # 3. Find React components matching the name
        comps_query = """
        MATCH (c:ReactComponent)
        WHERE c.name CONTAINS $term OR c.file CONTAINS $term
        RETURN c.name as name, c.file as file
        LIMIT 10
        """
        comps = await neo4j_client.execute_query(comps_query, {"term": search_term})
        trace["frontend"]["components"] = [
            {"name": c["name"], "file": c.get("file", "")}
            for c in (comps or [])
        ]

        # 4. Find related functions (controllers, services)
        funcs_query = """
        MATCH (f:Function)
        WHERE f.name CONTAINS $term OR f.file CONTAINS $term
        RETURN f.name as name, f.file as file
        LIMIT 10
        """
        funcs = await neo4j_client.execute_query(funcs_query, {"term": search_term})
        for f in (funcs or []):
            fname = (f.get("file") or "").lower()
            info = {"name": f["name"], "file": f.get("file", "")}
            if "service" in fname:
                trace["backend"]["services"].append(info)
            else:
                trace["backend"]["handlers"].append(info)

        # 5. Follow CALLS_ROUTE edges from frontend
        if trace["frontend"]["api_calls"]:
            cross_query = """
            MATCH (a:ApiCall)-[:CALLS_ROUTE]->(r:VertxRoute)
            WHERE a.url CONTAINS $term OR a.file CONTAINS $term
            RETURN a.url as frontend_url, a.file as frontend_file,
                   r.path as backend_path, r.method as backend_method, r.file as backend_file
            LIMIT 10
            """
            cross = await neo4j_client.execute_query(cross_query, {"term": search_term})
            trace["cross_repo_links"] = [
                {
                    "frontend_call": c.get("frontend_url", ""),
                    "frontend_file": c.get("frontend_file", ""),
                    "backend_route": c.get("backend_path", ""),
                    "backend_file": c.get("backend_file", ""),
                }
                for c in (cross or [])
            ]

        total_hits = (
            len(trace["backend"]["routes"]) + len(trace["frontend"]["api_calls"]) +
            len(trace["frontend"]["components"]) + len(trace["backend"]["handlers"])
        )
        trace["total_hits"] = total_hits
        trace["has_cross_repo_evidence"] = bool(trace.get("cross_repo_links"))

        return trace
    except Exception as e:
        return {"error": str(e)}


@tool
async def get_file_dependencies(file_path: str) -> Dict[str, Any]:
    """
    Show all dependencies for a file: what it imports, what imports it,
    what functions it contains, and what routes/components it defines.
    USE THIS for understanding code structure, imports, and dependencies.
    
    Args:
        file_path: Path to the file (partial match supported)
    
    Returns:
        Complete dependency map for the file
    """
    try:
        from src.database.neo4j_client import neo4j_client
        await neo4j_client.initialize()

        # Find the file
        file_query = """
        MATCH (f:File)
        WHERE f.path CONTAINS $path
        RETURN f.path as path, f.name as name, f.language as language
        LIMIT 3
        """
        files = await neo4j_client.execute_query(file_query, {"path": file_path})

        if not files:
            return {"error": f"File not found: {file_path}"}

        results = []
        for file_info in (files or []):
            fpath = file_info["path"]
            dep_info: Dict[str, Any] = {
                "file": fpath,
                "language": file_info.get("language"),
                "contains": [],
                "imports_from": [],
                "imported_by": [],
                "calls_routes": [],
                "renders_components": [],
            }

            # What this file contains (functions, components, routes)
            contains_query = """
            MATCH (f:File {path: $path})-[:CONTAINS]->(child)
            RETURN child.name as name, labels(child) as labels
            """
            contained = await neo4j_client.execute_query(contains_query, {"path": fpath})
            dep_info["contains"] = [
                {"name": c["name"], "type": c["labels"][0] if c.get("labels") else "Unknown"}
                for c in (contained or [])
            ]

            # What this file calls / depends on
            outgoing_query = """
            MATCH (f:File {path: $path})-[rel]->(target)
            WHERE type(rel) <> 'BELONGS_TO' AND type(rel) <> 'CONTAINS'
            RETURN type(rel) as rel_type, target.name as name, target.path as path, labels(target) as labels
            LIMIT 20
            """
            outgoing = await neo4j_client.execute_query(outgoing_query, {"path": fpath})
            dep_info["depends_on"] = [
                {"rel": o["rel_type"], "name": o.get("name", ""), "path": o.get("path", ""),
                 "type": o["labels"][0] if o.get("labels") else "Unknown"}
                for o in (outgoing or [])
            ]

            # What depends on this file
            incoming_query = """
            MATCH (source)-[rel]->(f:File {path: $path})
            WHERE type(rel) <> 'BELONGS_TO'
            RETURN type(rel) as rel_type, source.name as name, source.path as path, labels(source) as labels
            LIMIT 20
            """
            incoming = await neo4j_client.execute_query(incoming_query, {"path": fpath})
            dep_info["depended_on_by"] = [
                {"rel": i["rel_type"], "name": i.get("name", ""), "path": i.get("path", ""),
                 "type": i["labels"][0] if i.get("labels") else "Unknown"}
                for i in (incoming or [])
            ]

            results.append(dep_info)

        return {"files": results, "count": len(results)}
    except Exception as e:
        return {"error": str(e)}


@tool
async def get_qa_risk_assessment(focus_area: str = None) -> Dict[str, Any]:
    """
    Generate QA-oriented risk assessment: identify high-risk code areas
    that need testing, including async chains, auth boundaries, side effects,
    and unprotected mutation routes.
    USE THIS for QA, testing, and risk assessment questions.
    
    Args:
        focus_area: Optional focus (e.g. 'authentication', 'file-upload', 'workflow')
    
    Returns:
        Risk-ranked test scenarios with severity and rationale
    """
    try:
        from src.database.neo4j_client import neo4j_client
        from src.intelligence.behavioral_layer import behavioral_classifier
        await neo4j_client.initialize()

        risks = []

        # 1. Find mutation routes (POST/PUT/DELETE) without obvious auth
        mutation_query = """
        MATCH (r:VertxRoute)
        WHERE r.method IN ['POST', 'PUT', 'DELETE', 'PATCH']
        RETURN r.path as path, r.method as method, r.handler_method as handler, r.file as file
        """
        if focus_area:
            mutation_query = """
            MATCH (r:VertxRoute)
            WHERE r.method IN ['POST', 'PUT', 'DELETE', 'PATCH']
            AND (r.path CONTAINS $focus OR r.handler_method CONTAINS $focus OR r.file CONTAINS $focus)
            RETURN r.path as path, r.method as method, r.handler_method as handler, r.file as file
            """
        params = {"focus": focus_area} if focus_area else {}
        mutations = await neo4j_client.execute_query(mutation_query, params)

        for m in (mutations or []):
            handler = (m.get("handler") or "").lower()
            has_auth = any(kw in handler for kw in ["auth", "middleware", "protect", "guard", "validate"])
            severity = "high" if not has_auth else "medium"
            risks.append({
                "category": "auth_boundary",
                "severity": severity,
                "title": f"{m['method']} {m['path']} — {'UNPROTECTED' if not has_auth else 'protected'}",
                "rationale": f"Mutation endpoint {m['method']} {m['path']} handled by {m.get('handler', 'unknown')}" +
                             (" has NO visible auth middleware" if not has_auth else " has auth protection"),
                "file": m.get("file", ""),
                "test_intent": f"Verify {'authentication is enforced' if not has_auth else 'auth + authorization'} for {m['method']} {m['path']}",
            })

        # 2. Find side-effecting functions (HTTP calls, file I/O, email)
        side_effect_query = """
        MATCH (f:Function)
        WHERE f.code IS NOT NULL
        AND (f.code =~ '(?i).*(fetch|axios|http|request|sendEmail|sendSms|fs\\.write|upload).*')
        RETURN f.name as name, f.file as file
        LIMIT 15
        """
        side_effects = await neo4j_client.execute_query(side_effect_query)
        for se in (side_effects or []):
            risks.append({
                "category": "external_dependency",
                "severity": "medium",
                "title": f"Side-effecting function: {se['name']}",
                "rationale": f"{se['name']} in {se.get('file', '')} makes external calls (HTTP, file I/O, etc.)",
                "file": se.get("file", ""),
                "test_intent": f"Test {se['name']} with external service down, slow, or returning errors",
            })

        # 3. Find concurrent execution risks (run-transform, run-workflow)
        concurrent_query = """
        MATCH (r:VertxRoute)
        WHERE r.path CONTAINS 'run-' OR r.path CONTAINS 'execute' OR r.path CONTAINS 'trigger'
        RETURN r.path as path, r.method as method, r.handler_method as handler, r.file as file
        """
        concurrent_routes = await neo4j_client.execute_query(concurrent_query)
        for cr in (concurrent_routes or []):
            risks.append({
                "category": "race_window",
                "severity": "high",
                "title": f"Concurrent execution risk: {cr['method']} {cr['path']}",
                "rationale": f"Long-running operation {cr['path']} could be called concurrently. Check for race conditions on shared state.",
                "file": cr.get("file", ""),
                "test_intent": f"Load test: 10 concurrent calls to {cr['method']} {cr['path']} with same resource ID",
            })

        # Sort by severity
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        risks.sort(key=lambda r: severity_order.get(r["severity"], 99))

        return {
            "total_risks": len(risks),
            "by_severity": {
                "critical": len([r for r in risks if r["severity"] == "critical"]),
                "high": len([r for r in risks if r["severity"] == "high"]),
                "medium": len([r for r in risks if r["severity"] == "medium"]),
                "low": len([r for r in risks if r["severity"] == "low"]),
            },
            "by_category": {
                cat: len([r for r in risks if r["category"] == cat])
                for cat in set(r["category"] for r in risks)
            },
            "risks": risks[:25],
            "focus_area": focus_area,
        }
    except Exception as e:
        return {"error": str(e)}


# Create tools list
tools = [
    search_codebase,
    analyze_impact,
    find_component,
    get_api_flow,
    get_codebase_metrics,
    get_repository_architecture,
    read_file_content,
    search_code_pattern,
    list_api_routes,
 
    get_behavioral_classification,
    get_security_surface,
    get_domain_context,
    trace_cross_repo_flow,
    get_file_dependencies,
    get_qa_risk_assessment,
]


def should_continue(state: DeveloperState) -> Literal["tools", "final_answer", "end"]:
    """Determine if we should continue to tools, generate final answer, or end"""
    messages = state.get("messages", [])
    
    if not messages:
        return "end"
    
    # Count tool call iterations to prevent infinite loops
    tool_call_count = 0
    tool_result_count = 0
    for msg in messages:
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            tool_call_count += 1
        if isinstance(msg, ToolMessage):
            tool_result_count += 1
    
    # Max 8 tool call iterations - then force final answer generation
    MAX_TOOL_ITERATIONS = 8
    if tool_call_count >= MAX_TOOL_ITERATIONS:
        logger.warning(f"Max tool iterations ({MAX_TOOL_ITERATIONS}) reached, forcing final answer")
        return "final_answer"
    
    last_message = messages[-1]
    
    # Check if the last message has tool calls
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    
    # If we have content, we're done
    if hasattr(last_message, 'content') and last_message.content:
        return "end"
    
    return "end"


def _is_codebase_question(query: str) -> bool:
    """
    Check if the query is about the codebase (domain restriction).
    
    Returns True if the question is likely about:
    - Code, files, functions, classes, components
    - API endpoints, routes, services
    - Architecture, structure, dependencies
    - How something works in the code
    
    Returns False for general knowledge questions.
    """
    query_lower = query.lower()
    
    # Codebase-related keywords
    codebase_keywords = [
        'code', 'file', 'function', 'class', 'method', 'component',
        'api', 'endpoint', 'route', 'service', 'controller', 'model',
        'import', 'export', 'module', 'package', 'dependency',
        'architecture', 'structure', 'folder', 'directory',
        'how does', 'where is', 'find', 'search', 'show me',
        'what is', 'explain', 'implementation', 'implements',
        'react', 'express', 'node', 'vert.x', 'frontend', 'backend',
        'repository', 'repo', 'codebase', 'project',
        'dto', 'entity', 'schema', 'database', 'query',
        'hook', 'state', 'props', 'jsx', 'tsx',
        'handler', 'middleware', 'router', 'routes'
    ]
    
    # Check if any codebase keyword is present
    for keyword in codebase_keywords:
        if keyword in query_lower:
            return True
    
    # Non-codebase patterns (general knowledge)
    non_code_patterns = [
        'what is the capital',
        'who is the president',
        'tell me a joke',
        'write a poem',
        'translate this',
        'calculate',
        'weather',
        'news',
        'recipe'
    ]
    
    for pattern in non_code_patterns:
        if pattern in query_lower:
            return False
    
    # Default: assume it's about codebase
    return True


async def developer_agent(state: DeveloperState) -> Dict[str, Any]:
    """Main developer agent node that processes messages and decides actions"""
    try:
        messages = list(state.get('messages', []))
        
        # DOMAIN RESTRICTION: Check if the question is about the codebase
        last_human_message = None
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                last_human_message = msg.content
                break
        
        if last_human_message and not _is_codebase_question(last_human_message):
            domain_restriction_response = AIMessage(
                content="""⚠️ **Domain Restriction**

I can only answer questions about the **ingested codebase(s)**. 

Your question appears to be outside my scope. I'm designed to help with:
- Code analysis and understanding
- Finding files, functions, and components
- API endpoints and routes
- Architecture and dependencies
- Code flow and relationships

Please rephrase your question to focus on the codebase, or ask about specific code elements."""
            )
            return {"messages": [domain_restriction_response]}
        
        # Add tool results to context if available
        tool_results = state.get('tool_results', {})
        if tool_results:
            # Tool results are already in messages as ToolMessage
            pass
        
        # Build system prompt from YAML or default
        system_prompt = prompts.get('developer_system', DEFAULT_DEVELOPER_PROMPT)
        
        tool_descriptions = "\n".join([
            f"- {t.name}: {t.description}" 
            for t in tools
        ])
        
        # Use safer string formatting to avoid KeyErrors from JSON in prompts
        try:
            formatted_system = system_prompt.format(
                tool_descriptions=tool_descriptions,
                context_files=", ".join(state.get('context_files', []))
            )
        except KeyError:
            # If format fails (e.g., due to { } in the prompt), use as-is
            formatted_system = system_prompt
        
        # Convert messages to LiteLLM format
        llm_messages = [{"role": "system", "content": formatted_system}]
        
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
            elif isinstance(msg, SystemMessage):
                # Skip additional system messages
                pass
        
        # Define tools for LiteLLM
        llm_tools = []
        for t in tools:
            try:
                # Try to get schema from args_schema
                params = {}
                if hasattr(t, 'args_schema') and t.args_schema:
                    try:
                        params = t.args_schema.schema()
                    except (KeyError, AttributeError, ValueError):
                        # Fallback: try to extract parameters from docstring or use empty
                        params = {
                            "type": "object",
                            "properties": {},
                            "required": []
                        }
                
                llm_tools.append({
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": params
                    }
                })
            except Exception as e:
                logger.warning(f"Failed to add tool {t.name} to LLM tools: {e}")
                # Still add the tool but with minimal parameters
                llm_tools.append({
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": {"type": "object", "properties": {}}
                    }
                })
        
        # Call LLM with proper authentication
        # For Gemini models, use Vertex AI API with service account credentials
        model_name = settings.get_litellm_model("developer")
        
        if "gemini" in settings.DEVELOPER_MODEL.lower():
            # Use Vertex AI endpoint which supports service account auth
            model_name = f"vertex_ai/{settings.DEVELOPER_MODEL}"
        
        response = await acompletion(
            model=model_name,
            messages=llm_messages,
            tools=llm_tools if llm_tools else None,
            tool_choice="auto",
            max_tokens=settings.LLM_MAX_TOKENS,
            temperature=0.2,
            timeout=settings.LLM_TIMEOUT,
            api_key=settings.get_api_key_for_model(settings.DEVELOPER_MODEL) or "vertex-ai"
        )
        
        response_message = response.choices[0].message
        
        # Create AIMessage from response
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
        
        ai_message = AIMessage(**ai_message_kwargs)
        
        return {"messages": [ai_message]}
        
    except Exception as e:
        logger.error(f"Developer agent failed: {e}", exc_info=True)
        error_message = AIMessage(
            content=f"I encountered an error while processing your request. Please try rephrasing your question."
        )
        return {"messages": [error_message]}


async def execute_tools(state: DeveloperState) -> Dict[str, Any]:
    """Execute tools called by the agent"""
    messages = state.get("messages", [])
    last_message = messages[-1] if messages else None
    
    if not last_message or not hasattr(last_message, 'tool_calls') or not last_message.tool_calls:
        return {"messages": []}
    
    tool_messages = []
    
    for tool_call in last_message.tool_calls:
        tool_name = tool_call.get("name", "")
        tool_args = tool_call.get("args", {})
        tool_id = tool_call.get("id", f"call_{tool_name}")
        
        # Find and execute the tool
        tool_result = {"error": f"Tool {tool_name} not found"}
        
        for t in tools:
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
        
        # Create ToolMessage
        tool_messages.append(ToolMessage(
            content=json.dumps(tool_result, default=str),
            tool_call_id=tool_id
        ))
    
    return {"messages": tool_messages}


async def generate_final_answer(state: DeveloperState) -> Dict[str, Any]:
    """Force generate a final answer from accumulated tool results when max iterations reached"""
    try:
        messages = state.get("messages", [])
        
        # Extract all tool results
        tool_results = []
        original_question = ""
        
        for msg in messages:
            if isinstance(msg, HumanMessage):
                original_question = msg.content
            elif isinstance(msg, ToolMessage):
                try:
                    result = json.loads(msg.content)
                    tool_results.append(result)
                except:
                    tool_results.append({"raw": msg.content})
        
        # Build a summary prompt
        summary_prompt = f"""Based on the tool results below, provide a FINAL ANSWER to the user's question.

USER QUESTION: {original_question}

TOOL RESULTS:
{json.dumps(tool_results, indent=2, default=str)[:8000]}

INSTRUCTIONS:
- Synthesize the information from the tool results above
- Provide a clear, structured answer
- If the data shows API routes, list them in a table format
- If information was not found, say so explicitly
- DO NOT call any more tools - just provide your final answer

FORMAT YOUR RESPONSE:
📁 **File(s)**: [relevant files found]
🏷️ **Repository**: [repository name]
💡 **Answer**: [your synthesized answer]
📊 **Confidence**: [high/medium/low]
"""
        
        model_name = settings.get_litellm_model("developer")
        if "gemini" in settings.DEVELOPER_MODEL.lower():
            model_name = f"vertex_ai/{settings.DEVELOPER_MODEL}"
        
        response = await acompletion(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a code assistant. Synthesize the tool results and provide a final answer. DO NOT call any tools."},
                {"role": "user", "content": summary_prompt}
            ],
            max_tokens=settings.LLM_MAX_TOKENS,
            temperature=0.2,
            timeout=30,
            api_key=settings.get_api_key_for_model(settings.DEVELOPER_MODEL) or "vertex-ai"
        )
        
        final_content = response.choices[0].message.content or "I was unable to generate a response. Please try rephrasing your question."
        
        return {"messages": [AIMessage(content=final_content)]}
        
    except Exception as e:
        logger.error(f"Failed to generate final answer: {e}")
        return {"messages": [AIMessage(content=f"I encountered an error while processing. Please try again with a simpler question.")]}


def create_developer_workflow() -> StateGraph:
    """Create and compile the developer assistant workflow"""
    
    # Build the workflow graph
    graph = StateGraph(DeveloperState)
    
    # Add nodes
    graph.add_node("agent", developer_agent)
    graph.add_node("tools", execute_tools)
    graph.add_node("final_answer", generate_final_answer)
    
    # Set entry point
    graph.set_entry_point("agent")
    
    # Add conditional edges
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "final_answer": "final_answer",
            "end": END
        }
    )
    
    # Tools always return to agent
    graph.add_edge("tools", "agent")
    
    # Final answer goes to end
    graph.add_edge("final_answer", END)
    
    return graph.compile()


# Compiled workflow instance
developer_app = create_developer_workflow()