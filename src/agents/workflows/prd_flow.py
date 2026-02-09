"""
PRD Analysis Workflow using LangGraph
Analyzes Product Requirements Documents against the codebase knowledge graph

Enhanced with:
    - Structured PRD extraction (capabilities, constraints, assumptions)
    - Domain-first resolution (Domain → Capability → Code)
    - Behavioral and temporal impact assessment
"""
import logging
import yaml
import json
from typing import Dict, Any, List
from pathlib import Path
from datetime import datetime

from langgraph.graph import StateGraph, END
from langchain.prompts import PromptTemplate
from litellm import acompletion

from src.services.search_service import SearchService
from src.services.graph_service import GraphService
from src.agents.state_schema import PRDState
from src.intelligence.prd_graph import (
    PRD_EXTRACTION_SYSTEM_PROMPT,
    build_structured_prd_from_llm_output,
)
from src.intelligence.domain_model import domain_engine
from config.settings import settings
from src.core.logger import get_logger
from src.core.exceptions import AgentException

logger = get_logger(__name__)

# Load prompts
PROMPTS_PATH = Path(__file__).parent.parent.parent.parent / "config" / "prd_prompt.yaml"
try:
    with open(PROMPTS_PATH, 'r') as f:
        prompts = yaml.safe_load(f) or {}
except FileNotFoundError:
    logger.warning(f"Prompts file not found: {PROMPTS_PATH}")
    prompts = {}

# Default prompts if file not found
DEFAULT_ARCHITECT_PROMPT = """
You are a Senior Software Architect analyzing a PRD against an existing enterprise codebase.

## PRD Content:
{prd_text}

## Additional Instructions from User:
{human_prompt}

## Retrieved Codebase Context:
{context}

## Architectural Constraints:
{constraints}

Analyze the PRD and provide a comprehensive implementation blueprint.

Return a JSON response with this structure:
{{
    "summary": {{
        "architecture_approach": "string",
        "estimated_complexity": "low|medium|high",
        "estimated_effort_days": number,
        "risk_level": "low|medium|high|critical"
    }},
    "affected_components": [
        {{
            "name": "string",
            "type": "route|component|service|dto|eventbus",
            "change_type": "modify|create|delete",
            "impact_description": "string"
        }}
    ],
    "dependency_graph": {{
        "frontend_changes": [],
        "backend_changes": [],
        "database_changes": [],
        "eventbus_flows": []
    }},
    "implementation_plan": [
        {{
            "phase": number,
            "description": "string",
            "tasks": [],
            "dependencies": []
        }}
    ],
    "risk_mitigation": {{
        "breaking_changes": [],
        "backward_compatibility_notes": "string",
        "rollback_strategy": "string"
    }},
    "existing_reusable_components": [],
    "recommended_testing_strategy": {{
        "unit_tests": [],
        "integration_tests": [],
        "e2e_tests": []
    }}
}}
"""


async def validate_prd(state: PRDState) -> PRDState:
    """Validate PRD for completeness and clarity"""
    try:
        validation_prompt = """
        Analyze this PRD for completeness:
        
        PRD: {prd_text}
        
        Return JSON with this structure:
        {{
            "is_complete": boolean,
            "missing_elements": ["list of missing elements"],
            "clarity_score": 1-10,
            "recommendations": ["list of recommendations"],
            "key_requirements": ["list of extracted requirements"]
        }}
        """
        
        # Use Vertex AI API for Gemini models (service account auth)
        model_name = settings.get_litellm_model("architect")
        if "gemini" in settings.ARCHITECT_MODEL.lower():
            model_name = f"vertex_ai/{settings.ARCHITECT_MODEL}"
        
        llm_kwargs = {
            "model": model_name,
            "messages": [{
                "role": "user",
                "content": validation_prompt.format(prd_text=state.get('prd_text', ''))
            }],
            "response_format": {"type": "json_object"},
            "timeout": settings.LLM_TIMEOUT,
            "max_tokens": 1000,
            "api_key": settings.get_api_key_for_model(settings.ARCHITECT_MODEL) or "vertex-ai"
        }
        
        response = await acompletion(**llm_kwargs)
        
        validation_result = json.loads(response.choices[0].message.content)
        
        # Initialize validation_results if not present
        if 'validation_results' not in state or state['validation_results'] is None:
            state['validation_results'] = []
        
        state['validation_results'].append({
            "step": "prd_validation",
            "result": validation_result,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        if not validation_result.get("is_complete", True):
            logger.warning(
                "PRD may be incomplete",
                missing_elements=validation_result.get('missing_elements', [])
            )
        
        # Initialize errors list if not present
        if 'errors' not in state:
            state['errors'] = []
        
        return state
        
    except Exception as e:
        logger.error(f"PRD validation failed: {e}")
        if 'errors' not in state:
            state['errors'] = []
        state['errors'].append(f"PRD validation failed: {str(e)}")
        return state


async def retrieve_codebase_context(state: PRDState) -> PRDState:
    """Retrieve relevant codebase context for the PRD"""
    try:
        search_service = SearchService()
        graph_service = GraphService()
        
        await search_service.initialize()
        
        # Extract key concepts from PRD using LLM
        extraction_prompt = """
        Extract key technical concepts from this PRD:
        {prd_text}
        
        Return JSON:
        {{
            "technologies": ["list of technologies mentioned"],
            "domains": ["business domains"],
            "key_entities": ["entity names like User, Order, Payment"],
            "operations": ["CRUD operations, workflows"],
            "api_endpoints": ["potential API paths mentioned"]
        }}
        """
        
        # Use Vertex AI API for Gemini models
        model_name = settings.get_litellm_model("architect")
        if "gemini" in settings.ARCHITECT_MODEL.lower():
            model_name = f"vertex_ai/{settings.ARCHITECT_MODEL}"
        
        extract_kwargs = {
            "model": model_name,
            "messages": [{"role": "user", "content": extraction_prompt.format(prd_text=state.get('prd_text', ''))}],
            "response_format": {"type": "json_object"},
            "timeout": settings.LLM_TIMEOUT,
            "max_tokens": 1000,
            "api_key": settings.get_api_key_for_model(settings.ARCHITECT_MODEL) or "vertex-ai"
        }
        
        extraction_response = await acompletion(**extract_kwargs)
        
        concepts = json.loads(extraction_response.choices[0].message.content)
        
        # Initialize retrieved_context
        contexts = {
            "extracted_concepts": concepts,
            "semantic_results": [],
            "graph_results": {},
            "similar_features": []
        }
        
        # 1. Semantic search for relevant code
        try:
            search_results = await search_service.search_codebase(
                query=state.get('prd_text', ''),
                search_type="hybrid",
                top_k=15,
                filters={
                    "repository": state.get('repository_filter', [])
                }
            )
            contexts["semantic_results"] = search_results.get("results", {}).get("combined", [])
        except Exception as e:
            logger.warning(f"Semantic search failed: {e}")
        
        # 2. Graph search for architectural patterns
        for entity in concepts.get("key_entities", [])[:5]:  # Limit to 5 entities
            try:
                graph_context = await graph_service.find_related_entities(
                    entity_name=entity,
                    relationship_types=["CALLS_ROUTE", "USES_DTO", "PUBLISHES_TO"],
                    max_depth=2
                )
                if graph_context.get("relationships"):
                    contexts["graph_results"][entity] = graph_context
            except Exception as e:
                logger.debug(f"Graph search for {entity} failed: {e}")
        
        # 3. Find similar existing features
        for operation in concepts.get("operations", [])[:3]:
            try:
                similar = await search_service.search_codebase(
                    query=operation,
                    search_type="semantic",
                    top_k=5
                )
                if similar.get("results", {}).get("combined"):
                    contexts["similar_features"].extend(
                        similar["results"]["combined"][:2]
                    )
            except Exception as e:
                logger.debug(f"Similar feature search failed: {e}")
        
        state['retrieved_context'] = contexts
        
        logger.info(
            "Retrieved codebase context",
            semantic_results=len(contexts.get("semantic_results", [])),
            graph_entities=len(contexts.get("graph_results", {})),
            similar_features=len(contexts.get("similar_features", []))
        )
        
        return state
        
    except Exception as e:
        logger.error(f"Context retrieval failed: {e}")
        if 'errors' not in state:
            state['errors'] = []
        state['errors'].append(f"Context retrieval failed: {str(e)}")
        state['retrieved_context'] = {}
        return state


async def analyze_architecture(state: PRDState) -> PRDState:
    """Generate architectural blueprint based on PRD and context"""
    try:
        # Use custom prompt from YAML or default (key: prd_architect_system)
        architect_prompt = prompts.get('prd_architect_system', prompts.get('architect_system', DEFAULT_ARCHITECT_PROMPT))
        
        # Prepare context summary (limit size to avoid token limits)
        context_summary = {
            "semantic_results": state.get('retrieved_context', {}).get('semantic_results', [])[:10],
            "graph_results": state.get('retrieved_context', {}).get('graph_results', {}),
            "similar_features": state.get('retrieved_context', {}).get('similar_features', [])[:5],
            "extracted_concepts": state.get('retrieved_context', {}).get('extracted_concepts', {})
        }
        
        # Format the prompt
        formatted_prompt = architect_prompt.format(
            prd_text=state.get('prd_text', ''),
            context=json.dumps(context_summary, indent=2, default=str)[:8000],  # Limit context size
            human_prompt=state.get('human_prompt', ''),
            constraints=json.dumps(state.get('architectural_constraints', {}))
        )
        
        # Get LLM response
        model_name = settings.get_litellm_model("architect")
        if "gemini" in settings.ARCHITECT_MODEL.lower():
            model_name = f"vertex_ai/{settings.ARCHITECT_MODEL}"
        
        arch_kwargs = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": "You are a Senior Software Architect. Always respond with valid JSON."},
                {"role": "user", "content": formatted_prompt}
            ],
            "max_tokens": settings.LLM_MAX_TOKENS,
            "temperature": 0.1,  # Low temperature for architectural decisions
            "response_format": {"type": "json_object"},
            "timeout": settings.LLM_TIMEOUT,
            "api_key": settings.get_api_key_for_model(settings.ARCHITECT_MODEL) or "vertex-ai"
        }
        
        response = await acompletion(**arch_kwargs)
        
        # Parse and validate response
        blueprint = json.loads(response.choices[0].message.content)
        
        # Add metadata
        blueprint["metadata"] = {
            "prd_length": len(state.get('prd_text', '')),
            "context_sources": len(state.get('retrieved_context', {}).get('semantic_results', [])),
            "generated_at": datetime.utcnow().isoformat(),
            "model": settings.ARCHITECT_MODEL
        }
        
        state['analysis_output'] = blueprint
        
        logger.info(
            "Architectural blueprint generated",
            complexity=blueprint.get('summary', {}).get('estimated_complexity', 'unknown'),
            affected_components=len(blueprint.get('affected_components', []))
        )
        
        return state
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse LLM response as JSON: {e}")
        if 'errors' not in state:
            state['errors'] = []
        state['errors'].append(f"JSON parsing failed: {str(e)}")
        state['analysis_output'] = {"error": "Failed to generate valid blueprint"}
        return state
    except Exception as e:
        logger.error(f"Architecture analysis failed: {e}")
        if 'errors' not in state:
            state['errors'] = []
        state['errors'].append(f"Architecture analysis failed: {str(e)}")
        state['analysis_output'] = {"error": str(e)}
        return state


async def validate_blueprint(state: PRDState) -> PRDState:
    """Validate the architectural blueprint against constraints"""
    try:
        blueprint = state.get('analysis_output', {})
        
        if not blueprint or "error" in blueprint:
            return state
        
        validation_checks = []
        
        # 1. Check for single points of failure
        if "dependency_graph" in blueprint:
            eventbus_flows = blueprint["dependency_graph"].get("eventbus_flows", [])
            for flow in eventbus_flows:
                if isinstance(flow, dict) and len(flow.get("consumers", [])) == 1:
                    validation_checks.append({
                        "type": "warning",
                        "category": "reliability",
                        "message": f"Single consumer for EventBus flow: {flow.get('producer', 'unknown')}",
                        "suggestion": "Consider adding backup consumer or circuit breaker"
                    })
        
        # 2. Check for breaking changes
        breaking_changes = blueprint.get("risk_mitigation", {}).get("breaking_changes", [])
        for change in breaking_changes:
            if isinstance(change, dict) and not change.get("backward_compatible", True):
                validation_checks.append({
                    "type": "critical",
                    "category": "compatibility",
                    "message": f"Breaking change: {change.get('change', 'unknown')}",
                    "impact": f"Affects: {change.get('affected_clients', [])}"
                })
        
        # 3. Check complexity
        complexity = blueprint.get("summary", {}).get("estimated_complexity", "unknown")
        if complexity == "high":
            validation_checks.append({
                "type": "warning",
                "category": "complexity",
                "message": "High complexity implementation detected",
                "suggestion": "Consider phased rollout or feature flags"
            })
        
        # 4. Check risk level
        risk_level = blueprint.get("summary", {}).get("risk_level", "unknown")
        if risk_level in ["high", "critical"]:
            validation_checks.append({
                "type": "critical" if risk_level == "critical" else "warning",
                "category": "risk",
                "message": f"Implementation has {risk_level} risk level",
                "suggestion": "Review risk mitigation strategies before proceeding"
            })
        
        # Initialize validation_results if not present
        if 'validation_results' not in state or state['validation_results'] is None:
            state['validation_results'] = []
        
        # Add validation results to state
        state['validation_results'].append({
            "step": "blueprint_validation",
            "checks": validation_checks,
            "passed": len([c for c in validation_checks if c["type"] == "critical"]) == 0,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        logger.info(
            "Blueprint validation completed",
            total_checks=len(validation_checks),
            critical_issues=len([c for c in validation_checks if c["type"] == "critical"]),
            warnings=len([c for c in validation_checks if c["type"] == "warning"])
        )
        
        return state
        
    except Exception as e:
        logger.error(f"Blueprint validation failed: {e}")
        if 'errors' not in state:
            state['errors'] = []
        state['errors'].append(f"Blueprint validation failed: {str(e)}")
        return state


def create_prd_workflow() -> StateGraph:
    """Create and compile the PRD analysis workflow"""
    
    # Build the workflow graph
    graph = StateGraph(PRDState)
    
    # Add nodes
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
    
    return graph.compile()


# Compiled workflow instance
prd_workflow = create_prd_workflow()