from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, AIMessage
from litellm import completion
from config.settings import settings
from src.agents.tools import (CodeRetriever, ImpactAnalyzer, ModuleScanner, PRAnalyzer)

tools = [CodeRetriever(), ImpactAnalyzer(), ModuleScanner(), PRAnalyzer()]
tool_node = ToolNode(tools)

async def developer_agent(state: DeveloperState):
    messages = state['messages']
    if state.get('tool_results'):
        messages.append(AIMessage(content=str(state['tool_results'])))

    prompt = prompts['developer_system'].format(
        tool_descriptions="\n".join([f"- {t.__class__.__name__}: {t.__doc__}" for t in tools])
    )

    response = completion(
        model=settings.LITELLM_MODEL,
        messages=[{"role": "system", "content": prompt}] + messages,
        tools=[t.as_tool() for t in tools],  
        tool_choice="auto"
    )

    return {"messages": messages + [response.choices[0].message]}

graph = StateGraph(DeveloperState)
graph.add_node("agent", developer_agent)
graph.add_node("tools", tool_node)
graph.add_edge("tools", "agent")
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", lambda x: "tools" if x['messages'][-1].tool_calls else END)

developer_app = graph.compile()