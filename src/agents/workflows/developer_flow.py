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


# Create tools list
tools = [
    search_codebase,
    analyze_impact,
    find_component,
    get_api_flow,
    get_codebase_metrics
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
        
        formatted_system = system_prompt.format(
            tool_descriptions=tool_descriptions,
            context_files=", ".join(state.get('context_files', []))
        )
        
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
        llm_tools = [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.args_schema.schema() if hasattr(t, 'args_schema') else {}
                }
            }
            for t in tools
        ]
        
        # Call LLM
        response = await acompletion(
            model=settings.get_litellm_model("developer"),
            messages=llm_messages,
            tools=llm_tools if llm_tools else None,
            tool_choice="auto",
            api_key=settings.get_api_key_for_model(settings.DEVELOPER_MODEL),
            max_tokens=settings.LLM_MAX_TOKENS,
            temperature=0.2,
            timeout=settings.LLM_TIMEOUT
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
        logger.error(f"Developer agent failed: {e}")
        error_message = AIMessage(
            content=f"I encountered an error while processing your request: {str(e)}. Please try rephrasing your question."
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