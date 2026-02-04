"""
Developer Assistant Workflow using LangGraph
Provides interactive assistance for developers with code understanding and debugging
"""
import logging
import yaml
import json
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

# Default developer system prompt
DEFAULT_DEVELOPER_PROMPT = """
You are an expert software engineer assistant with deep knowledge of the codebase.

You have access to the following tools:
{tool_descriptions}

When answering questions:
1. First search the codebase for relevant context
2. Analyze the code structure and relationships
3. Provide clear, actionable answers
4. Include code examples when helpful
5. Explain potential impacts of changes

Always ground your answers in the actual codebase - don't hallucinate code that doesn't exist.
If you're unsure, say so and suggest how to find the information.

Current context files: {context_files}
"""


# Define tools for the developer agent
@tool
async def search_codebase(query: str, search_type: str = "hybrid") -> Dict[str, Any]:
    """
    Search the codebase for relevant code snippets and components.
    
    Args:
        query: The search query describing what you're looking for
        search_type: Type of search - "semantic", "graph", or "hybrid"
    
    Returns:
        Search results with relevant code snippets
    """
    try:
        search_service = SearchService()
        await search_service.initialize()
        
        results = await search_service.search_codebase(
            query=query,
            search_type=search_type,
            top_k=10
        )
        
        # Format results for LLM consumption
        formatted = []
        for result in results.get("results", {}).get("combined", [])[:5]:
            formatted.append({
                "name": result.get("name", ""),
                "type": result.get("type", ""),
                "file": result.get("file", ""),
                "score": result.get("score", 0),
                "snippet": result.get("text", "")[:500]
            })
        
        return {
            "query": query,
            "results_count": len(formatted),
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
            
            # Get routes
            routes_query = """
            MATCH (route:Route)-[:EXPOSES]-(f:File)-[:BELONGS_TO]->(r:Repository {name: $name})
            RETURN route.method as method, route.path as path, f.path as file
            ORDER BY route.path
            LIMIT 30
            """
            routes = await neo4j_client.execute_query(routes_query, {"name": repo["name"]})
            repo_info["routes"] = [{"method": r["method"], "path": r["path"], "file": r["file"]} for r in routes]
            
            # Get key files (controllers, services, models)
            key_files_query = """
            MATCH (f:File)-[:BELONGS_TO]->(r:Repository {name: $name})
            WHERE f.path CONTAINS 'controller' OR f.path CONTAINS 'service' 
                  OR f.path CONTAINS 'model' OR f.path CONTAINS 'route'
            RETURN f.path as path
            LIMIT 20
            """
            key_files = await neo4j_client.execute_query(key_files_query, {"name": repo["name"]})
            repo_info["key_files"] = [f["path"] for f in key_files]
            
            result["repositories"].append(repo_info)
        
        return result
        
    except Exception as e:
        return {"error": str(e)}


# Create tools list
tools = [
    search_codebase,
    analyze_impact,
    find_component,
    get_api_flow,
    get_codebase_metrics,
    get_repository_architecture
]


def should_continue(state: DeveloperState) -> Literal["tools", "end"]:
    """Determine if we should continue to tools or end"""
    messages = state.get("messages", [])
    
    if not messages:
        return "end"
    
    last_message = messages[-1]
    
    # Check if the last message has tool calls
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    
    return "end"


async def developer_agent(state: DeveloperState) -> Dict[str, Any]:
    """Main developer agent node that processes messages and decides actions"""
    try:
        messages = list(state.get('messages', []))
        
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
                    # Execute the async tool
                    tool_result = await t.ainvoke(tool_args)
                except Exception as e:
                    tool_result = {"error": str(e)}
                break
        
        # Create ToolMessage
        tool_messages.append(ToolMessage(
            content=json.dumps(tool_result, default=str),
            tool_call_id=tool_id
        ))
    
    return {"messages": tool_messages}


def create_developer_workflow() -> StateGraph:
    """Create and compile the developer assistant workflow"""
    
    # Build the workflow graph
    graph = StateGraph(DeveloperState)
    
    # Add nodes
    graph.add_node("agent", developer_agent)
    graph.add_node("tools", execute_tools)
    
    # Set entry point
    graph.set_entry_point("agent")
    
    # Add conditional edges
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "end": END
        }
    )
    
    # Tools always return to agent
    graph.add_edge("tools", "agent")
    
    return graph.compile()


# Compiled workflow instance
developer_app = create_developer_workflow()