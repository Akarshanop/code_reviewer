# 🧠 Enterprise Code Intelligence Platform

A **production-ready GraphRAG system** for enterprise codebase understanding, PRD analysis, and AI-powered developer assistance. Built to ingest and understand multi-repo systems (Java/Vert.x, React/TS, and modern JavaScript/Node/Express codebases).

## 🎯 Key Features

### 1. PRD Analysis & Architecture Blueprint Generation
- Transform Product Requirements Documents into executable technical blueprints
- Analyze PRD against existing codebase to identify reuse opportunities
- Generate implementation plans with file-level specificity
- Cross-repository dependency mapping

### 2. QA Strategy Generation
- Comprehensive test strategy from architectural blueprints
- Unit, integration, E2E, performance, and security test plans
- Automated test case generation with edge cases
- Coverage targets and quality gates

### 3. Developer Assistant
- Natural language codebase exploration
- Cross-repository semantic search
- API flow tracing and impact analysis
- PR risk assessment and security scanning

### 4. Multi-Language Ingestion (Tree-sitter)
- Parses Java, TypeScript/TSX, JavaScript/JSX (incl. Express routes), Python, and common config/build files
- Builds a Neo4j knowledge graph (repositories, files, routes, API calls, components)
- Generates embeddings into Qdrant for semantic search

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      FastAPI Gateway                            │
│              (Auth, Rate Limiting, Routing)                     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Agent Orchestrator                           │
│           (LangGraph StateGraph Workflows)                      │
├─────────────────┬─────────────────┬─────────────────────────────┤
│  PRD Architect  │   QA Engineer   │   Developer Assistant       │
│     Agent       │      Agent      │        Agent                │
└─────────────────┴─────────────────┴─────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────────────┐
│     Neo4j       │ │     Qdrant      │ │       Supabase          │
│  (Knowledge     │ │  (Vector        │ │  (Auth, Chat History,   │
│   Graph)        │ │   Search)       │ │   Audit Logs)           │
└─────────────────┘ └─────────────────┘ └─────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- Docker & Docker Compose
- Node.js 18+ (for frontend)

### 1. Clone and Setup

```bash
cd backend
cp .env.example .env
# Edit .env with your configuration
```

### 2. Start Infrastructure

```bash
docker-compose up -d
```

This starts:
- **Neo4j** (bolt://localhost:7687) - Knowledge Graph
- **Qdrant** (http://localhost:6333) - Vector Database

### 3. Install Dependencies

```bash
# Using Poetry (recommended)
poetry install

# Or using pip
pip install -e .
```

### 4. Configure Repositories

Edit `.env` or `config/settings.py`:

```python
# In .env
GIT_REPOS_CONFIG='[
  {"url": "https://github.com/your-org/vertx-api-gateway.git", "branch": "main", "language": "java"},
  {"url": "https://github.com/your-org/vertx-user-service.git", "branch": "main", "language": "java"},
  {"url": "https://github.com/your-org/react-dashboard.git", "branch": "main", "language": "typescript"}
]'
```

### 5. Ingest Codebase

```bash
# Start the server
uvicorn src.api.main:app --reload

# Trigger ingestion (in another terminal)
curl -X POST http://localhost:8000/api/v1/trigger/trigger \
  -H "Content-Type: application/json" \
  -d '{"force_refresh": false, "skip_embeddings": false}'

# Check job status
# (use the job_id returned by the trigger endpoint)
curl http://localhost:8000/api/v1/trigger/status/{job_id}
```

### 6. Start Using

```bash
# PRD Analysis
curl -X POST http://localhost:8000/api/v1/prd-qa/analyze \
  -H "Content-Type: application/json" \
  -d '{"prd_content": "Your PRD content here..."}'

# Developer Question
curl -X POST http://localhost:8000/api/v1/developer/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How does the user authentication flow work across services?"}'
```

## 📁 Project Structure

```
backend/
├── config/
│   ├── settings.py          # Enterprise configuration
│   ├── prompts.yaml          # Agent system prompts
│   ├── dev_prompt.yaml       # Developer agent prompts
│   ├── prd_promt.yaml        # PRD architect prompts
│   └── qa_prompt.yaml        # QA engineer prompts
├── src/
│   ├── api/
│   │   ├── main.py           # FastAPI application
│   │   └── v1/endpoints/
│   │       ├── chat.py       # Chat interface
│   │       ├── developer.py  # Developer assistant
│   │       ├── ingestion.py  # Data ingestion
│   │       ├── prd_qa.py     # PRD & QA workflows
│   │       └── admin.py      # Health & admin
│   ├── agents/
│   │   ├── orchestrator.py   # Agent coordination
│   │   ├── state_schema.py   # LangGraph state definitions
│   │   ├── tools/
│   │   │   ├── pr_analyzer.py    # PR impact analysis
│   │   │   └── module_scanner.py # Module quality scanning
│   │   └── workflows/
│   │       ├── prd_flow.py       # PRD analysis workflow
│   │       ├── qa_flow.py        # QA generation workflow
│   │       └── developer_flow.py # Developer assistant workflow
│   ├── data_ingestion/
│   │   ├── git_service.py        # Multi-repo git operations
│   │   ├── parsers/
│   │   │   ├── java_parser.py    # Vert.x/Java AST parsing
│   │   │   └── react_parser.py   # React/TypeScript parsing
│   │   └── graph_builder/
│   │       ├── node_factory.py   # Graph node creation
│   │       └── edge_linker.py    # Relationship mapping
│   ├── database/
│   │   ├── neo4j_client.py       # Knowledge graph client
│   │   ├── qdrant_client.py      # Vector search client
│   │   └── supabase_session.py   # Auth & audit client
│   └── services/
│       ├── data_ingestion_service.py  # Ingestion orchestration
│       ├── embedding_service.py       # Vertex AI embeddings (text-embedding-004)
│       ├── graph_service.py           # Graph queries
│       ├── search_service.py          # Hybrid search
│       └── vector_service.py          # Vector operations
├── docker/
│   └── Dockerfile
├── docker-compose.yml
├── pyproject.toml
└── .env.example
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `NEO4J_URI` | Neo4j connection URI | bolt://localhost:7687 |
| `NEO4J_PASSWORD` | Neo4j password | password |
| `QDRANT_URL` | Qdrant server URL | http://localhost:6333 |
| `QDRANT_COLLECTION_NAME` | Qdrant collection name | code_embeddings |
| `QDRANT_VECTOR_SIZE` | Embedding vector size (must match model) | 768 |
| `SUPABASE_URL` | Supabase project URL | - |
| `SUPABASE_KEY` | Supabase anon key | - |
| `SUPABASE_SERVICE_KEY` | Supabase service key (admin ops) | - |
| `LLM_PROVIDER` | LLM provider (openai/azure/anthropic/gemini) | gemini |
| `GEMINI_API_KEY` | Gemini API key (if using Gemini) | - |
| `OPENAI_API_KEY` | OpenAI API key (if using OpenAI) | - |
| `ARCHITECT_MODEL` | Model for PRD analysis | gemini-2.5-flash |
| `DEVELOPER_MODEL` | Model for dev assistant | gemini-2.5-flash |
| `QA_MODEL` | Model for QA generation | gemini-2.5-flash |
| `EMBEDDING_MODEL` | Embedding model (Vertex AI) | text-embedding-004 |
| `GOOGLE_CLOUD_PROJECT_ID` | GCP project id for Vertex AI | - |
| `GOOGLE_CLOUD_REGION` | Vertex AI region | us-central1 |
| `GOOGLE_CLOUD_SERVICE_ACCOUNT_FILE` | Service account JSON path | ./service.json |

### Multi-Repository Configuration

```json
// GIT_REPOS_CONFIG format
[
  {
    "url": "https://github.com/org/repo.git",
    "branch": "main",
    "language": "java",
    "include_patterns": ["src/**/*.java"],
    "exclude_patterns": ["**/test/**"]
  }
]
```

## 📊 API Endpoints

### Ingestion
- `POST /api/v1/trigger/trigger` - Start codebase ingestion
- `GET /api/v1/trigger/status/{job_id}` - Check ingestion status
- `GET /api/v1/trigger/repositories` - List configured repositories + ingestion state
- `POST /api/v1/trigger/repositories/{repo_name}/refresh` - Force refresh a single repo
- `GET /api/v1/trigger/stats` - Neo4j/Qdrant ingestion statistics
- `DELETE /api/v1/trigger/cache` - Clear ingestion job cache

### PRD & QA
- `POST /api/v1/prd-qa/analyze` - Analyze PRD and generate blueprint
- `GET /api/v1/prd-qa/jobs/{job_id}` - Get analysis job status
- `POST /api/v1/prd-qa/validate-blueprint` - Validate implementation plan
- `POST /api/v1/prd-qa/estimate-effort` - Estimate implementation effort

### Developer Assistant
- `POST /api/v1/developer/ask` - Ask development questions
- `POST /api/v1/developer/scan/{module}` - Scan module quality
- `POST /api/v1/developer/analyze-pr` - Analyze PR impact
- `GET /api/v1/developer/search` - Semantic code search
- `GET /api/v1/developer/component/{name}` - Get component details
- `GET /api/v1/developer/api-flow/{path}` - Trace API flow

### Chat
- `POST /api/v1/chat/ask` - General chat interface
- `POST /api/v1/chat/prd` - PRD-focused chat
- `POST /api/v1/chat/qa` - QA-focused chat
- `GET /api/v1/chat/conversations/{id}` - Get conversation history

### Admin
- `GET /api/v1/admin/health` - Comprehensive health check
- `GET /api/v1/admin/ready` - Kubernetes readiness probe
- `GET /api/v1/admin/live` - Kubernetes liveness probe
- `GET /api/v1/admin/stats` - System statistics
- `GET /api/v1/admin/config` - Effective configuration (sanitized)
- `POST /api/v1/admin/cache/invalidate` - Clear caches
- `POST /api/v1/admin/reindex` - Reindex repositories
- `GET /api/v1/admin/audit/recent` - Recent audit events
- `GET /api/v1/admin/metrics` - Prometheus metrics

## 🔒 Security

- JWT-based authentication via Supabase
- Role-based access control
- Audit logging for all operations
- Sensitive data sanitization in logs
- API rate limiting

## 🧪 Testing

```bash
# Run unit tests
pytest tests/unit

# Run integration tests
pytest tests/integration

# Run with coverage
pytest --cov=src tests/
```

## 🐳 Docker Deployment

```bash
# Build image
docker build -f docker/Dockerfile -t code-intel-platform .

# Run with compose
docker-compose up -d
```

## 📈 Monitoring

- Health endpoints for Kubernetes probes
- Prometheus metrics at `/api/v1/admin/metrics`
- Structured JSON logging
- Sentry integration (optional)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.
