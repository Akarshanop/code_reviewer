"""
Intelligence Layer for Enterprise Codebase Understanding & Risk Prediction System.

This package provides the foundational abstractions that transform structural
code knowledge into behavioral, temporal, and domain-aware intelligence.

Shared utilities:
    utils                - Canonical generate_node_id() and batch_cypher() helper

Core intelligence:
    behavioral_layer     - Behavioral classification of code nodes and edges
    temporal_metadata    - Temporal intelligence (volatility, recency, change frequency)
    evidence_links       - Evidence-weighted cross-language link model
    confidence_engine    - Confidence-aware query wrapper for graph and vector results
    domain_model         - Domain-centric reasoning (Domain → Capability → Repo → Code)
    prd_graph            - Structured PRD → Graph object extraction
    qa_projection        - QA-specific projection layer for test reasoning

Post-ingestion passes:
    cross_file_resolver  - Cross-file call resolution + type flow / data flow analysis
    test_coverage_mapper - Parse test files and link test functions to production code
    config_scanner       - Configuration & secret scanning (.env, YAML, docker-compose)
    dead_code_detector   - Identify unreachable functions/classes using call graph
    performance_profiler - Flag N+1 queries, blocking I/O, missing async patterns
"""
