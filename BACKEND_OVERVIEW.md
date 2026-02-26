# Backend Overview — Code Intelligence Platform

High-level map of the backend and cleanups applied. Use this to navigate and continue readability/cleanup work.

---

## 1. Entry: `src/api/main.py`

- **FastAPI app**: lifespan initializes `agent_orchestrator`; CORS, GZip, validation handler, request-logging middleware.
- **Routers**: `admin`, `ingestion`, `developer`, `chat`, `prd_qa`, `intelligence`, `auth` under `/api/v1/*`.
- **System routes**: `/`, `/health`, `/ready`.

**Cleanups applied:**
- `/ready` now returns a proper JSON response with `ready` and per-service `checks` (was missing return).
- Removed large commented-out CORS preflight middleware block.
- Fixed `v1/endpoints/__init__.py` to export `intelligence` and `auth` (was out of sync with main imports).
- Normalized import spacing (`intelligence , auth` → `intelligence, auth`).

---

## 2. API Layer: `src/api/`

| File | Purpose |
|------|--------|
| `dependencies.py` | FastAPI Depends for Neo4j, Qdrant, Supabase, SearchService, EmbeddingService, Orchestrator. |
| `v1/endpoints/admin.py` | Health, stats, graph connectivity, config, cache invalidation, reindex, DB reset, audit, metrics. |
| `v1/endpoints/ingestion.py` | Trigger ingestion, status, repositories, refresh, stats, cache, incremental ingestion. |
| `v1/endpoints/developer.py` | `/ask`, `/scan/{module}`, `/analyze-pr`, `/search`, `/component/`, `/api-flow/`. |
| `v1/endpoints/chat.py` | Chat `/ask`, role-aware chat, PRD, QA, generate-prd+QA, jobs, sessions, PDF download. |
| `v1/endpoints/prd_qa.py` | Combined PRD+QA analyze, job status, validate-blueprint, estimate-effort. |
| `v1/endpoints/intelligence.py` | Domains resolve, behavioral summary, temporal hotspots, evidence links, stats. |
| `v1/endpoints/auth.py` | Signup, signin, whitelist CRUD. |

**Cleanups applied:**
- **admin.py**: Replaced bare `except: pass` in `/metrics` with `except Exception as e` and `logger.debug(...)`.
- **ingestion.py**: Removed duplicate imports of `neo4j_client` and `qdrant_client` inside `get_ingestion_stats()` (use module-level imports).

---

## 3. Core: `src/core/`

| File | Purpose |
|------|--------|
| `logger.py` | Structlog setup, trace_id context var, `get_logger`, `setup_logging`. |
| `exceptions.py` | BaseAPIException, ValidationException, Auth, ResourceNotFound, AgentException, etc. |
| `security.py` | JWT create/decode, password hash/verify, TokenBlacklist (Redis), RBAC, JWTBearer. |
| `id_generator.py` | Deterministic node/repo IDs (SHA-256). |

**Cleanups applied:**
- **security.py**: `decode_access_token` is sync; blacklist check is async. Moved blacklist check into async callers (`JWTBearer.__call__`, `require_role` wrapper) with `await TokenBlacklist.is_blacklisted(token)` so revocation works correctly.

---

## 4. Data Ingestion: `src/data_ingestion/`

- **git_service.py**: Clone/pull repos, repo info.
- **parsers/**: `parser_coordinator`, `language_detector`, `java_parser`, `react_parser`, `python_parser`, `generic_parser`.
- **graph_builder/**: `node_factory`, `edge_linker`.
- **vector_builder/**: `vector_embedding_ingestion_builder`.
- **entry_point_detector.py**, **import_resolver.py**, **enhanced_graph_linker.py**.

Ingestion flow: clone → parse (by language) → build graph (Neo4j) → embeddings (Qdrant). Optional intelligence passes (cross-file, test coverage, config, dead code, perf, temporal, type registry).

---

## 5. Services: `src/services/`

| Service | Role |
|---------|------|
| `data_ingestion_service.py` | Orchestrates parse → graph → embeddings; uses parsers, graph_builder, vector builder, intelligence modules. |
| `search_service.py` | Hybrid search: intent detection → semantic/graph/structural/temporal/impact. |
| `graph_service.py` | Neo4j queries: related entities, API flow tracing. |
| `vector_service.py` | Qdrant semantic/hybrid search. |
| `embedding_service.py` | Text → embeddings (caching). |
| `incremental_ingestion.py` | Diff-based incremental ingestion (CI-triggered). |
| `auth_service.py` | Signup/signin, whitelist. |
| `pdf_service.py` | PRD/QA PDF generation. |

---

## 6. Agents: `src/agents/`

- **orchestrator.py**: Entry for PRD, QA, developer, sales; `process_with_policy` (reasoning → policy); `generate_prd_with_qa`.
- **state_schema.py**: TypedDict state for PRD, QA, Developer, Impact, FeatureEstimation.
- **reasoning_layer.py**, **policy_layer.py**: Two-layer architecture and role-based output.
- **workflows/**: `prd_flow`, `qa_flow`, `developer_flow`, `sales_flow` (LangGraph).
- **tools/**: `module_scanner`, `pr_analyzer`.

---

## 7. Intelligence: `src/intelligence/`

Post-ingestion / query-time: `behavioral_layer`, `temporal_metadata`, `domain_model`, `evidence_links`, `qa_projection`, `cross_file_resolver`, `test_coverage_mapper`, `config_scanner`, `dead_code_detector`, `performance_profiler`, `type_registry`, `subgraph_compression`, etc. Used by ingestion (enrichment) and by search/orchestrator (domain resolution, confidence).

---

## 8. Schemas: `src/schemas/`

- **admin_schema.py**: Health, stats, graph connectivity, reindex, cache, DB reset.
- **developer_schema.py**: DeveloperQuestion, DeveloperResponse, ModuleScanRequest, PRAnalysisRequest.
- **chat_schema.py**: ChatRequest, ChatResponse, PRDRequest, QARequest, JobStatus.
- **endpoints/ingestion_schema.py**: IngestionRequest, IngestionStatus, RepositoryStatus.

---

## Suggested Next Cleanups (by area)

- **data_ingestion**: Extract constants (e.g. file_patterns), trim redundant comments, ensure single place for repo config handling.
- **services**: Ensure `SearchService` / `GraphService` initialization is consistent (e.g. lazy init or explicit in endpoints).
- **agents**: Split large workflow files (e.g. `developer_flow.py`) into smaller modules by concern; document state transitions.
- **intelligence**: Group by “ingestion vs query-time” and document which modules are required for minimal runs.
- **schemas**: Move request/response models used only in one endpoint next to that endpoint or keep in schemas with clear naming (e.g. prefix with route name).

---

## How to Run

- From repo root: `uvicorn src.api.main:app --reload` (or run `main.py` with `python -m` from `backend`).
- Docs: `/docs`, `/redoc`. Health: `/health`, readiness: `/ready`.
