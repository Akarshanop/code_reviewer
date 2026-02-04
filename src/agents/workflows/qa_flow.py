"""
QA Test Generation Workflow using LangGraph
Generates comprehensive testing strategies based on PRD and architectural blueprints
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
from src.agents.state_schema import QAState
from config.settings import settings
from src.core.logger import get_logger
from src.core.exceptions import AgentException

logger = get_logger(__name__)

# Load prompts
PROMPTS_PATH = Path(__file__).parent.parent.parent.parent / "config" / "qa_prompt.yaml"
try:
    with open(PROMPTS_PATH, 'r') as f:
        prompts = yaml.safe_load(f) or {}
except FileNotFoundError:
    logger.warning(f"QA prompts file not found: {PROMPTS_PATH}")
    prompts = {}

# Default QA prompt if file not found
DEFAULT_QA_PROMPT = """
You are a Senior QA Engineer analyzing a feature implementation to generate comprehensive test strategies.

## PRD Summary:
{prd_text}

## Architectural Blueprint:
{design_blueprint}

## Codebase Context (existing tests and patterns):
{context}

## Test Focus Areas:
{test_focus}

Generate a comprehensive testing strategy covering:
1. Unit Tests - For individual components and functions
2. Integration Tests - For API endpoints and service interactions
3. E2E Tests - For complete user workflows
4. Edge Case Tests - For boundary conditions and error handling
5. Performance Tests - For load and stress scenarios
6. Security Tests - For authentication, authorization, and data validation

Return a JSON response with this structure:
{{
    "summary": {{
        "total_test_cases": number,
        "estimated_coverage": "percentage",
        "priority_tests": ["list of critical tests"],
        "risk_areas": ["areas needing extra testing"]
    }},
    "unit_tests": [
        {{
            "test_name": "string",
            "component": "string",
            "description": "string",
            "test_cases": [
                {{
                    "name": "string",
                    "input": "description",
                    "expected_output": "description",
                    "priority": "high|medium|low"
                }}
            ],
            "mocking_requirements": ["list"]
        }}
    ],
    "integration_tests": [
        {{
            "test_name": "string",
            "endpoints_covered": ["list"],
            "description": "string",
            "test_scenarios": [],
            "dependencies": ["services/databases needed"]
        }}
    ],
    "e2e_tests": [
        {{
            "test_name": "string",
            "user_journey": "string",
            "steps": [],
            "assertions": [],
            "setup_requirements": []
        }}
    ],
    "edge_case_tests": [
        {{
            "category": "string",
            "test_cases": [],
            "risk_level": "high|medium|low"
        }}
    ],
    "performance_tests": [
        {{
            "test_name": "string",
            "type": "load|stress|endurance|spike",
            "metrics": [],
            "thresholds": {{}}
        }}
    ],
    "race_condition_tests": [
        {{
            "scenario": "string",
            "eventbus_address": "string if applicable",
            "test_approach": "string",
            "potential_issues": ["list"]
        }}
    ],
    "test_data_requirements": {{
        "fixtures": [],
        "seed_data": [],
        "mock_services": []
    }}
}}
"""


async def analyze_blueprint_for_testing(state: QAState) -> QAState:
    """Analyze the design blueprint to identify testing requirements"""
    try:
        blueprint = state.get('design_blueprint', {})
        
        # Extract key testing targets from blueprint
        testing_targets = {
            "components_to_test": [],
            "apis_to_test": [],
            "eventbus_flows": [],
            "dtos_to_validate": [],
            "breaking_changes": [],
            "high_risk_areas": []
        }
        
        # Extract affected components
        for component in blueprint.get('affected_components', []):
            if isinstance(component, dict):
                testing_targets["components_to_test"].append({
                    "name": component.get("name"),
                    "type": component.get("type"),
                    "change_type": component.get("change_type")
                })
        
        # Extract API changes
        dependency_graph = blueprint.get('dependency_graph', {})
        for backend_change in dependency_graph.get('backend_changes', []):
            if isinstance(backend_change, dict) and 'endpoint' in str(backend_change):
                testing_targets["apis_to_test"].append(backend_change)
        
        # Extract EventBus flows (high priority for race condition testing)
        for flow in dependency_graph.get('eventbus_flows', []):
            testing_targets["eventbus_flows"].append(flow)
        
        # Identify breaking changes
        risk_mitigation = blueprint.get('risk_mitigation', {})
        testing_targets["breaking_changes"] = risk_mitigation.get('breaking_changes', [])
        
        # Store analysis results
        if 'retrieved_context' not in state or state['retrieved_context'] is None:
            state['retrieved_context'] = {}
        
        state['retrieved_context']['testing_targets'] = testing_targets
        
        logger.info(
            "Blueprint analyzed for testing",
            components=len(testing_targets["components_to_test"]),
            apis=len(testing_targets["apis_to_test"]),
            eventbus_flows=len(testing_targets["eventbus_flows"])
        )
        
        return state
        
    except Exception as e:
        logger.error(f"Blueprint analysis failed: {e}")
        if 'errors' not in state:
            state['errors'] = []
        state['errors'].append(f"Blueprint analysis failed: {str(e)}")
        return state


async def retrieve_existing_tests(state: QAState) -> QAState:
    """Retrieve existing test patterns from the codebase"""
    try:
        search_service = SearchService()
        await search_service.initialize()
        
        existing_tests = {
            "unit_test_patterns": [],
            "integration_test_patterns": [],
            "test_utilities": []
        }
        
        # Search for existing test patterns
        try:
            # Search for unit tests
            unit_test_results = await search_service.search_codebase(
                query="unit test describe it expect mock",
                search_type="semantic",
                top_k=10,
                filters={"type": "test"}
            )
            existing_tests["unit_test_patterns"] = unit_test_results.get("results", {}).get("combined", [])[:5]
            
            # Search for integration tests
            integration_results = await search_service.search_codebase(
                query="integration test API endpoint request response",
                search_type="semantic",
                top_k=10
            )
            existing_tests["integration_test_patterns"] = integration_results.get("results", {}).get("combined", [])[:5]
            
        except Exception as e:
            logger.warning(f"Test pattern search failed: {e}")
        
        # Merge with existing context
        if 'retrieved_context' not in state or state['retrieved_context'] is None:
            state['retrieved_context'] = {}
        
        state['retrieved_context']['existing_tests'] = existing_tests
        
        logger.info(
            "Retrieved existing test patterns",
            unit_patterns=len(existing_tests["unit_test_patterns"]),
            integration_patterns=len(existing_tests["integration_test_patterns"])
        )
        
        return state
        
    except Exception as e:
        logger.error(f"Test retrieval failed: {e}")
        if 'errors' not in state:
            state['errors'] = []
        state['errors'].append(f"Test retrieval failed: {str(e)}")
        return state


async def generate_test_strategy(state: QAState) -> QAState:
    """Generate comprehensive test strategy using LLM"""
    try:
        # Use custom prompt from YAML or default (key: qa_engineer_system)
        qa_prompt = prompts.get('qa_engineer_system', prompts.get('qa_system', DEFAULT_QA_PROMPT))
        
        # Prepare context
        context = {
            "testing_targets": state.get('retrieved_context', {}).get('testing_targets', {}),
            "existing_tests": state.get('retrieved_context', {}).get('existing_tests', {}),
            "test_categories": state.get('test_categories', [])
        }
        
        # Format the prompt
        formatted_prompt = qa_prompt.format(
            design_blueprint=json.dumps(state.get('design_blueprint', {}), indent=2, default=str)[:4000],
            prd_text=state.get('prd_text', '')[:2000],
            context=json.dumps(context, indent=2, default=str)[:3000],
            test_focus=state.get('test_focus', 'all')
        )
        
        # Get LLM response using Vertex AI for Gemini models
        model_name = settings.get_litellm_model("qa")
        if "gemini" in settings.QA_MODEL.lower():
            model_name = f"vertex_ai/{settings.QA_MODEL}"
        
        qa_kwargs = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": "You are a Senior QA Engineer. Generate comprehensive test strategies. Always respond with valid JSON."},
                {"role": "user", "content": formatted_prompt}
            ],
            "max_tokens": settings.LLM_MAX_TOKENS,
            "temperature": 0.3,  # Slightly higher for creative test case generation
            "response_format": {"type": "json_object"},
            "timeout": settings.LLM_TIMEOUT,
            "api_key": settings.get_api_key_for_model(settings.QA_MODEL) or "vertex-ai"
        }
        
        response = await acompletion(**qa_kwargs)
        
        # Parse response
        test_strategy = json.loads(response.choices[0].message.content)
        
        # Add metadata
        test_strategy["metadata"] = {
            "generated_at": datetime.utcnow().isoformat(),
            "model": settings.QA_MODEL,
            "test_focus": state.get('test_focus', 'all'),
            "blueprint_summary": state.get('design_blueprint', {}).get('summary', {})
        }
        
        state['analysis_output'] = test_strategy
        
        logger.info(
            "Test strategy generated",
            total_tests=test_strategy.get('summary', {}).get('total_test_cases', 0),
            unit_tests=len(test_strategy.get('unit_tests', [])),
            integration_tests=len(test_strategy.get('integration_tests', [])),
            e2e_tests=len(test_strategy.get('e2e_tests', []))
        )
        
        return state
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse QA response as JSON: {e}")
        if 'errors' not in state:
            state['errors'] = []
        state['errors'].append(f"JSON parsing failed: {str(e)}")
        state['analysis_output'] = {"error": "Failed to generate valid test strategy"}
        return state
    except Exception as e:
        logger.error(f"Test strategy generation failed: {e}")
        if 'errors' not in state:
            state['errors'] = []
        state['errors'].append(f"Test strategy generation failed: {str(e)}")
        state['analysis_output'] = {"error": str(e)}
        return state


async def identify_race_conditions(state: QAState) -> QAState:
    """Specifically analyze for race conditions in async/EventBus flows"""
    try:
        blueprint = state.get('design_blueprint', {})
        test_strategy = state.get('analysis_output', {})
        
        race_condition_analysis = []
        
        # Analyze EventBus flows
        eventbus_flows = state.get('retrieved_context', {}).get('testing_targets', {}).get('eventbus_flows', [])
        
        for flow in eventbus_flows:
            if isinstance(flow, dict):
                race_condition_analysis.append({
                    "flow": flow,
                    "potential_issues": [
                        "Message ordering not guaranteed",
                        "Consumer processing overlap",
                        "Database write conflicts"
                    ],
                    "recommended_tests": [
                        "Concurrent message publishing test",
                        "Consumer idempotency test",
                        "Message replay handling test"
                    ]
                })
        
        # Add race condition tests to strategy if not present
        if 'race_condition_tests' not in test_strategy:
            test_strategy['race_condition_tests'] = []
        
        for analysis in race_condition_analysis:
            test_strategy['race_condition_tests'].append({
                "scenario": f"EventBus flow: {analysis.get('flow', {})}",
                "potential_issues": analysis.get('potential_issues', []),
                "test_approach": "Use concurrent test runners with shared state verification"
            })
        
        # Initialize validation_results if not present
        if 'validation_results' not in state or state['validation_results'] is None:
            state['validation_results'] = []
        
        state['validation_results'].append({
            "step": "race_condition_analysis",
            "flows_analyzed": len(eventbus_flows),
            "issues_identified": len(race_condition_analysis),
            "timestamp": datetime.utcnow().isoformat()
        })
        
        state['analysis_output'] = test_strategy
        
        logger.info(
            "Race condition analysis completed",
            flows_analyzed=len(eventbus_flows),
            issues_found=len(race_condition_analysis)
        )
        
        return state
        
    except Exception as e:
        logger.error(f"Race condition analysis failed: {e}")
        if 'errors' not in state:
            state['errors'] = []
        state['errors'].append(f"Race condition analysis failed: {str(e)}")
        return state


def create_qa_workflow() -> StateGraph:
    """Create and compile the QA workflow"""
    
    # Build the workflow graph
    graph = StateGraph(QAState)
    
    # Add nodes
    graph.add_node("analyze_blueprint", analyze_blueprint_for_testing)
    graph.add_node("retrieve_tests", retrieve_existing_tests)
    graph.add_node("generate_strategy", generate_test_strategy)
    graph.add_node("analyze_race_conditions", identify_race_conditions)
    
    # Define workflow flow
    graph.set_entry_point("analyze_blueprint")
    graph.add_edge("analyze_blueprint", "retrieve_tests")
    graph.add_edge("retrieve_tests", "generate_strategy")
    graph.add_edge("generate_strategy", "analyze_race_conditions")
    graph.add_edge("analyze_race_conditions", END)
    
    return graph.compile()


# Compiled workflow instance
qa_workflow = create_qa_workflow()