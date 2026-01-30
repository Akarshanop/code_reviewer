import logging
import yaml
from langgraph.graph import StateGraph, END
from langchain.prompts import PromptTemplate
from litellm import completion
from src.services.search_service import SearchService
from src.agents.state_schema import AgentState  # Reuse, but add design_blueprint
from config.settings import settings
from pathlib import Path

logger = logging.getLogger(__name__)

# Extend state for QA
class QAState(AgentState):
    design_blueprint: dict  

# Load prompts
PROMPTS_PATH = Path(__file__).parent.parent.parent.parent / "config" / "qa_prompt.yaml"
with open(PROMPTS_PATH, 'r') as f:
    prompts = yaml.safe_load(f)

async def retrieve_additional_context(state: QAState) -> QAState:
    try:
        search_service = SearchService()
        context = await search_service.hybrid_search(state['prd_text'], top_k=5) 
        state['retrieved_context'].update(context)
        logger.info("Retrieved additional context for QA")
        return state
    except Exception as e:
        logger.error(f"QA retrieval failed: {str(e)}")
        raise

async def generate_tests(state: QAState) -> QAState:
    try:
        prompt = PromptTemplate.from_template(prompts['qa_system'])
        formatted_prompt = prompt.format(
            design_blueprint=state['design_blueprint'],
            prd_text=state['prd_text'],
            context=state['retrieved_context']
        )
        
        api_key = settings.GEMINI_API_KEY if "gemini" in settings.LITELLM_MODEL else settings.OPENAI_API_KEY
        response = completion(
            model=settings.LITELLM_MODEL,
            messages=[{"role": "system", "content": formatted_prompt}],
            api_key=api_key,
            max_tokens=settings.MAX_TOKENS,
            temperature=0.6,  
            response_format={"type": "json_object"}
        )
        
        state['analysis_output'] = response.choices[0].message.content 
        logger.info("QA test generation completed")
        return state
    except Exception as e:
        logger.error(f"QA LLM generation failed: {str(e)}")
        raise

graph = StateGraph(QAState)
graph.add_node("retrieve_additional", retrieve_additional_context)
graph.add_node("generate", generate_tests)
graph.add_edge("retrieve_additional", "generate")
graph.add_edge("generate", END)
qa_workflow = graph.compile()