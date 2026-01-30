# prd_flow.py
import logging
import yaml
import json
from typing import Dict, Any, List
from pathlib import Path

from langgraph.graph import StateGraph, END
from langchain.prompts import PromptTemplate
from litellm import completion
from pydantic import BaseModel

from src.services.search_service import SearchService
from src.services.graph_service import GraphService
from src.agents.state_schema import AgentState
from src.core.config import settings
from src.core.logger import get_logger

logger = get_logger(__name__)

class PRDState(AgentState):
    """State for PRD analysis workflow"""
    repository_filter: List[str] = []
    priority: str = "medium"
    architectural_constraints: Dict[str, Any] = {}
    validation_results: List[Dict] = []

# Load prompts
PROMPTS_PATH = Path(__file__).parent.parent.parent / "config" / "prd_prompt.yaml"
with open(PROMPTS_PATH, 'r') as f:
    prompts = yaml.safe_load(f)

async def validate_prd(state: PRDState) -> PRDState:
    """Validate PRD for completeness and clarity"""
    try:
        validation_prompt = PromptTemplate.from_template("""
        Analyze this PRD for completeness:
        
        PRD: {prd_text}
        
        Questions:
        1. Are acceptance criteria clearly defined?
        2. Are non-functional requirements specified?
        3. Are success metrics defined?
        4. Are constraints mentioned?
        
        Return JSON: {{
            "is_complete": boolean,
            "missing_elements": ["list"],
            "clarity_score": 1-10,
            "recommendations": ["list"]
        }}
        """)
        
        response = await completion(
            model=settings.LITELLM_MODEL,
            messages=[{
                "role": "user",
                "content": validation_prompt.format(prd_text=state['prd_text'])
            }],
            response_format={"type": "json_object"}
        )
        
        validation_result = json.loads(response.choices[0].message.content)
        state['validation_results'].append({
            "step": "prd_validation",
            "result": validation_result
        })
        
        if not validation_result.get("is_complete", True):
            logger.warning(f"PRD may be incomplete: {validation_result.get('missing_elements', [])}")
        
        return state
        
    except Exception as e:
        logger.error(f"PRD validation failed: {e}")
        raise

async def retrieve_codebase_context(state: PRDState) -> PRDState:
    """Retrieve relevant codebase context for the PRD"""
    try:
        search_service = SearchService()
        graph_service = GraphService()
        
        # Extract key concepts from PRD
        extraction_prompt = """
        Extract key technical concepts from this PRD:
        {prd_text}
        
        Return: {{
            "technologies": ["list"],
            "domains": ["list"],
            "key_entities": ["list"],
            "operations": ["list"]
        }}
        """
        
        extraction_response = await completion(
            model=settings.LITELLM_MODEL,
            messages=[{"role": "user", "content": extraction_prompt.format(prd_text=state['prd_text'])}],
            response_format={"type": "json_object"}
        )
        
        concepts = json.loads(extraction_response.choices[0].message.content)
        
        # Search for relevant code using multiple strategies
        contexts = {}
        
        # 1. Vector search for semantic similarity
        contexts["semantic"] = await search_service.hybrid_search(
            query=state['prd_text'],
            top_k=10,
            filter_repos=state.get('repository_filter', [])
        )
        
        # 2. Graph search for architectural patterns
        for entity in concepts.get("key_entities", []):
            graph_context = await graph_service.find_related_entities(
                entity_name=entity,
                relationship_types=["CALLS", "USES", "EXTENDS"]
            )
            contexts[f"graph_{entity}"] = graph_context
        
        # 3. Search for similar implementations
        if concepts.get("operations"):
            for operation in concepts["operations"][:3]:  # Top 3 operations
                similar = await search_service.search_by_operation(
                    operation=operation,
                    repo_filter=state.get('repository_filter', [])
                )
                contexts[f"similar_{operation}"] = similar
        
        state['retrieved_context'] = contexts
        logger.info(f"Retrieved context with {len(contexts)} sources")
        return state
        
    except Exception as e:
        logger.error(f"Context retrieval failed: {e}")
        raise

async def analyze_architecture(state: PRDState) -> PRDState:
    """Generate architectural blueprint"""
    try:
        # Load the architect system prompt
        architect_prompt = prompts['architect_system']
        
        # Format the prompt with PRD and context
        formatted_prompt = architect_prompt.format(
            prd_text=state['prd_text'],
            context=json.dumps(state['retrieved_context'], indent=2),
            human_prompt=state.get('human_prompt', ''),
            constraints=json.dumps(state.get('architectural_constraints', {}))
        )
        
        # Get LLM response
        response = await completion(
            model=settings.LITELLM_MODEL,
            messages=[{"role": "system", "content": formatted_prompt}],
            api_key=settings.get_api_key_for_model(settings.LITELLM_MODEL),
            max_tokens=settings.MAX_TOKENS,
            temperature=0.1,  # Low temperature for architectural decisions
            response_format={"type": "json_object"}
        )
        
        # Parse and validate response
        blueprint = json.loads(response.choices[0].message.content)
        
        # Add metadata
        blueprint["metadata"] = {
            "prd_hash": hash(state['prd_text']),
            "context_sources": list(state['retrieved_context'].keys()),
            "generated_at": datetime.utcnow().isoformat(),
            "model": settings.LITELLM_MODEL
        }
        
        state['analysis_output'] = blueprint
        logger.info(f"Architectural blueprint generated: {blueprint.get('summary', {}).get('architecture_approach', 'N/A')}")
        return state
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse LLM response as JSON: {e}")
        raise
    except Exception as e:
        logger.error(f"Architecture analysis failed: {e}")
        raise

async def validate_blueprint(state: PRDState) -> PRDState:
    """Validate the architectural blueprint against constraints"""
    try:
        # Check for single points of failure
        blueprint = state['analysis_output']
        
        validation_checks = []
        
        # 1. Check for SPOFs
        if "dependency_graph" in blueprint:
            for flow in blueprint["dependency_graph"].get("eventbus_flows", []):
                if len(flow.get("consumers", [])) == 1:
                    validation_checks.append({
                        "type": "warning",
                        "message": f"Single consumer for EventBus flow: {flow.get('producer')}",
                        "suggestion": "Consider adding backup consumer or circuit breaker"
                    })
        
        # 2. Check for breaking changes
        breaking_changes = blueprint.get("risk_mitigation", {}).get("breaking_changes", [])
        for change in breaking_changes:
            if not change.get("backward_compatible", True):
                validation_checks.append({
                    "type": "critical",
                    "message": f"Breaking change: {change.get('change')}",
                    "impact": f"Affects: {change.get('affected_clients', [])}"
                })
        
        # 3. Check complexity
        complexity = blueprint.get("complexity_assessment", {}).get("overall_complexity", "unknown")
        if complexity == "high":
            validation_checks.append({
                "type": "warning",
                "message": "High complexity implementation",
                "suggestion": "Consider phased rollout or feature flags"
            })
        
        # Add validation results to state
        state['validation_results'].append({
            "step": "blueprint_validation",
            "checks": validation_checks,
            "passed": len([c for c in validation_checks if c["type"] == "critical"]) == 0
        })
        
        logger.info(f"Blueprint validation completed with {len(validation_checks)} findings")
        return state
        
    except Exception as e:
        logger.error(f"Blueprint validation failed: {e}")
        raise

# Build the workflow graph
graph = StateGraph(PRDState)

graph.add_node("validate_prd", validate_prd)
graph.add_node("retrieve_context", retrieve_codebase_context)
graph.add_node("analyze_architecture", analyze_architecture)
graph.add_node("validate_blueprint", validate_blueprint)

# Define workflow flow
graph.set_entry_point("validate_prd")
graph.add_edge("validate_prd", "retrieve_context")
graph.add_edge("retrieve_context", "analyze_architecture")
graph.add_edge("analyze_architecture", "validate_blueprint")
graph.add_edge("validate_blueprint", END)

prd_workflow = graph.compile()