"""
Enterprise Configuration Management for Code Intelligence Platform
Supports 10+ repositories with multi-language codebase analysis
Auto-detection of languages based on file extensions (no need to specify repo type)
Supports GitLab, GitHub, Bitbucket, and Azure DevOps
"""
from pathlib import Path
from typing import List, Optional, Dict, Any
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import field_validator, computed_field, Field, model_validator
from functools import lru_cache
from enum import Enum
import logging
import json
import re
import os

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"


class GitProvider(str, Enum):
    """Supported Git providers"""
    GITLAB = "gitlab"
    GITHUB = "github"
    BITBUCKET = "bitbucket"
    AZURE_DEVOPS = "azure_devops"
    UNKNOWN = "unknown"


class RepositoryConfig:
    """
    Configuration for a single repository.
    
    Language/type is automatically detected - no need to specify.
    Just provide the URL and optionally the branch.
    """
    def __init__(self, url: str, branch: str = "main", name: str = None,
                 include_patterns: List[str] = None, exclude_patterns: List[str] = None,
                 repo_type: str = None):
        self.url = url.strip()
        self.branch = branch
        self._name = name
        self.repo_type = repo_type  # 'backend' or 'frontend' - from .env classification
        self.include_patterns = include_patterns or ["**/*"]
        self.exclude_patterns = exclude_patterns or [
            "**/node_modules/**", "**/target/**", "**/build/**", 
            "**/.git/**", "**/dist/**", "**/__pycache__/**",
            "**/vendor/**", "**/.gradle/**", "**/bin/**", "**/obj/**",
            "**/.venv/**", "**/venv/**", "**/.env/**", "**/env/**",
        ]
        # Auto-detected fields (populated during parsing)
        self.detected_language: Optional[str] = None
        self.detected_framework: Optional[str] = None
        self.file_count: int = 0
        self.entry_file: Optional[str] = None  # Detected entry point
    
    @property
    def name(self) -> str:
        """Extract repository name from URL"""
        if self._name:
            return self._name
        # Handle different URL formats
        url = self.url.rstrip("/")
        if url.endswith(".git"):
            url = url[:-4]
        return url.split("/")[-1]
    
    @property
    def provider(self) -> GitProvider:
        """Detect Git provider from URL"""
        url_lower = self.url.lower()
        if "gitlab" in url_lower:
            return GitProvider.GITLAB
        elif "github" in url_lower:
            return GitProvider.GITHUB
        elif "bitbucket" in url_lower:
            return GitProvider.BITBUCKET
        elif "dev.azure.com" in url_lower or "visualstudio.com" in url_lower:
            return GitProvider.AZURE_DEVOPS
        return GitProvider.UNKNOWN
    
    @property
    def is_gitlab(self) -> bool:
        return self.provider == GitProvider.GITLAB
    
    @property
    def project_path(self) -> str:
        """Extract project path (owner/repo) from URL"""
        # Remove protocol and domain
        url = self.url.rstrip("/")
        if url.endswith(".git"):
            url = url[:-4]
        
        # Match various URL patterns
        patterns = [
            r"(?:https?://)?(?:[\w.]+)/(.+)",  # https://gitlab.com/group/project
            r"git@[\w.]+:(.+)",  # git@gitlab.com:group/project
        ]
        
        for pattern in patterns:
            match = re.match(pattern, url)
            if match:
                return match.group(1)
        
        return url.split("/")[-1]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "url": self.url,
            "branch": self.branch,
            "name": self.name,
            "provider": self.provider.value,
            "project_path": self.project_path,
            "repo_type": self.repo_type,
            "detected_language": self.detected_language,
            "detected_framework": self.detected_framework,
            "entry_file": self.entry_file,
            "include_patterns": self.include_patterns,
            "exclude_patterns": self.exclude_patterns
        }


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE), 
        env_ignore_empty=True, 
        extra="ignore",
        case_sensitive=False
    )
    
    # ======================
    # Environment Configuration
    # ======================
    ENVIRONMENT: str = Field(default="development")
    DEBUG: bool = Field(default=False)
    LOG_LEVEL: str = Field(default="INFO")
    LOG_FORMAT: str = Field(default="json")  # json or text
    
    # ======================
    # Neo4j Graph Database
    # ======================
    NEO4J_URI: str = Field(default="bolt://localhost:7687")
    NEO4J_USER: str = Field(default="neo4j")
    NEO4J_PASSWORD: str = Field(default="password")
    NEO4J_DATABASE: str = Field(default="neo4j")
    NEO4J_MAX_CONNECTION_POOL_SIZE: int = Field(default=50)
    NEO4J_CONNECTION_TIMEOUT: int = Field(default=30)
    NEO4J_MAX_TRANSACTION_RETRY_TIME: int = Field(default=30)
    
    # ======================
    # Qdrant Vector Database
    # ======================
    QDRANT_URL: str = Field(default="http://localhost:6333")
    QDRANT_API_KEY: Optional[str] = Field(default=None)
    QDRANT_COLLECTION_NAME: str = Field(default="code_embeddings")
    # Gemini `gemini-embedding-001` outputs 3072-dim vectors.
    # Keep this aligned with the embedding model (otherwise Qdrant upserts will fail).
    QDRANT_VECTOR_SIZE: int = Field(default=3072)
    QDRANT_DISTANCE_METRIC: str = Field(default="Cosine")  # Cosine, Euclid, Dot
    QDRANT_ON_DISK: bool = Field(default=True)
    
    # ======================
    # Supabase Configuration
    # ======================
    SUPABASE_URL: str = Field(default="")
    SUPABASE_KEY: str = Field(default="")
    SUPABASE_SERVICE_KEY: Optional[str] = Field(default=None)  # For admin operations
    
    # ======================
    # Google Cloud Configuration
    # ======================
    GOOGLE_CLOUD_PROJECT_ID: Optional[str] = Field(default=None)
    GOOGLE_CLOUD_REGION: str = Field(default="us-central1")
    GOOGLE_CLOUD_SERVICE_ACCOUNT_FILE: Optional[str]
    
    # ======================
    # LLM Configuration
    # ======================
    LLM_PROVIDER: str = Field(default="gemini")  # openai, azure, anthropic, gemini
    OPENAI_API_KEY: Optional[str] = Field(default=None)
    AZURE_OPENAI_API_KEY: Optional[str] = Field(default=None)
    AZURE_OPENAI_ENDPOINT: Optional[str] = Field(default=None)
    AZURE_OPENAI_API_VERSION: str = Field(default="2024-02-15-preview")
    AZURE_OPENAI_DEPLOYMENT: Optional[str] = Field(default=None)
    ANTHROPIC_API_KEY: Optional[str] = Field(default=None)
    GEMINI_API_KEY: Optional[str] = Field(default=None)
    
    # Model Selection per Agent Type
    ARCHITECT_MODEL: str = Field(default="gemini-2.5-flash")
    DEVELOPER_MODEL: str = Field(default="gemini-2.5-flash")
    QA_MODEL: str = Field(default="gemini-2.5-flash")
    EMBEDDING_MODEL: str = Field(default="gemini-embedding-001")  # Google Vertex AI embedding model
    HF_TOKEN: Optional[str] = Field(default=None)  # HuggingFace API token (deprecated)
    
    # LLM Parameters
    LLM_TEMPERATURE: float = Field(default=0.1)
    LLM_MAX_TOKENS: int = Field(default=8192)
    LLM_TIMEOUT: int = Field(default=120)
    LLM_MAX_RETRIES: int = Field(default=3)
    
    # ======================
    # Git Configuration (Multi-Repository Support)
    # GitLab is the primary provider - just provide repo URLs
    # ======================
    GIT_REPOS: str = Field(default="[]")  # JSON array of repository URLs (simple list)
    GIT_REPOS_CONFIG: str = Field(default="[]")  
    GIT_PROVIDER: str = Field(default="gitlab")  # gitlab, github, bitbucket, azure_devops
    GIT_TOKEN: Optional[str] = Field(default=None)  # GitLab: Personal Access Token or OAuth token
    GIT_USERNAME: Optional[str] = Field(default=None)  # Optional username
    GIT_DEFAULT_BRANCH: str = Field(default="main")
    GIT_WORK_DIR: Path = Field(default=PROJECT_ROOT / "data" / "repos")
    REPOS_BASE_DIR: Path = Field(default=PROJECT_ROOT / "data" / "repos")
    GIT_CLONE_DEPTH: int = Field(default=1)  # Shallow clone for faster ingestion
    GIT_TIMEOUT: int = Field(default=300)  # Timeout for git operations in seconds
    # GitLab-specific settings
    GITLAB_URL: str = Field(default="https://gitlab.com")  # Self-hosted GitLab URL
    GITLAB_API_VERSION: str = Field(default="v4")  # GitLab API version
    
    # ======================
    # API Configuration
    # ======================
    API_HOST: str = Field(default="0.0.0.0")
    API_PORT: int = Field(default=8000)
    API_WORKERS: int = Field(default=4)
    API_PREFIX: str = Field(default="/api/v1")
    SECRET_KEY: str = Field(default="your-secret-key-change-in-production")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=30)
    REFRESH_TOKEN_EXPIRE_DAYS: int = Field(default=7)
    CORS_ORIGINS: str = Field(default='["http://localhost:3000","http://localhost:5173"]')
    
    # ======================
    # Agent Configuration
    # ======================
    MAX_CONCURRENT_AGENTS: int = Field(default=10)
    AGENT_TIMEOUT_SECONDS: int = Field(default=300)
    MAX_TOKENS: int = Field(default=4096)
    AGENT_TEMPERATURE: float = Field(default=0.1)
    AGENT_MAX_ITERATIONS: int = Field(default=10)
    
    # ======================
    # Search Configuration
    # ======================
    HYBRID_SEARCH_ALPHA: float = Field(default=0.7)  # Weight for vector vs graph
    VECTOR_SEARCH_TOP_K: int = Field(default=20)
    GRAPH_SEARCH_DEPTH: int = Field(default=3)
    SEARCH_SCORE_THRESHOLD: float = Field(default=0.5)
    
    # ======================
    # Ingestion Configuration
    # ======================
    INGESTION_BATCH_SIZE: int = Field(default=100)
    INGESTION_MAX_FILE_SIZE_MB: int = Field(default=10)
    INGESTION_PARALLEL_FILES: int = Field(default=10)
    INGESTION_RETRY_COUNT: int = Field(default=3)
    
    # ======================
    # Cache Configuration
    # ======================
    REDIS_URL: Optional[str] = Field(default=None)
    REDIS_TTL: int = Field(default=7200)  # 2 hours
    ENABLE_CACHE: bool = Field(default=True)
    CACHE_TTL: int = Field(default=300)  # 5 minutes
    
    # ======================
    # Feature Flags
    # ======================
    ENABLE_PRD_ANALYSIS: bool = Field(default=True)
    ENABLE_QA_GENERATION: bool = Field(default=True)
    ENABLE_DEVELOPER_ASSISTANT: bool = Field(default=True)
    ENABLE_PR_ANALYSIS: bool = Field(default=True)
    
    # ======================
    # Monitoring & Logging
    # ======================
    ENABLE_METRICS: bool = Field(default=True)
    METRICS_PORT: int = Field(default=9090)
    SENTRY_DSN: Optional[str] = Field(default=None)
    
    @field_validator('GIT_WORK_DIR', mode='before')
    @classmethod
    def ensure_git_work_dir_exists(cls, v) -> Path:
        path = Path(v) if isinstance(v, str) else v
        path.mkdir(parents=True, exist_ok=True)
        return path

    @field_validator('REPOS_BASE_DIR', mode='before')
    @classmethod
    def ensure_repos_base_dir_exists(cls, v) -> Path:
        path = Path(v) if isinstance(v, str) else v
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    @computed_field
    @property
    def database_url(self) -> str:
        return f"{self.SUPABASE_URL}"
    
    @computed_field
    @property
    def redis_enabled(self) -> bool:
        return self.REDIS_URL is not None
    
    @computed_field
    @property
    def cors_origins_list(self) -> List[str]:
        try:
            return json.loads(self.CORS_ORIGINS)
        except json.JSONDecodeError:
            return ["http://localhost:3000"]
    
    @computed_field
    @property
    def LLM_MODEL(self) -> str:
        """Get the primary LLM model for general use"""
        return self.ARCHITECT_MODEL
    
    @computed_field
    @property
    def REPOSITORIES(self) -> List["RepositoryConfig"]:
        """Get list of configured repositories"""
        return self.get_repositories()
    
    def get_repositories(self) -> List[RepositoryConfig]:
    
        repos = []
        
        # First try detailed config
        try:
            repo_configs = json.loads(self.GIT_REPOS_CONFIG)
            for config in repo_configs:
                if isinstance(config, dict) and config.get("url"):
                    repos.append(RepositoryConfig(
                        url=config.get("url", ""),
                        branch=config.get("branch", self.GIT_DEFAULT_BRANCH),
                
                        repo_type=config.get("repo_type")
                    ))
        except (json.JSONDecodeError, TypeError) as e:
            logger.debug(f"Could not parse GIT_REPOS_CONFIG: {e}")
        
        # Try typed format: [{"backend": [...]}, {"frontend": [...]}]
       
        return repos
    
    def get_api_key_for_model(self, model_name: str) -> Optional[str]:
        """Get appropriate API key based on model name
        
        Note: For Gemini/Vertex AI models, this returns None as they use
        service account authentication via GOOGLE_CLOUD_SERVICE_ACCOUNT_FILE
        """
        model_lower = model_name.lower()
        
        if "gpt" in model_lower or model_lower.startswith("text-embedding"):
            return self.OPENAI_API_KEY
        elif "claude" in model_lower:
            return self.ANTHROPIC_API_KEY
        elif "gemini" in model_lower or "vertex" in model_lower:
            # Gemini/Vertex AI uses service account authentication, not API key
            return None
        elif self.LLM_PROVIDER == "azure":
            return self.AZURE_OPENAI_API_KEY
        
        # Default to OpenAI
        return self.OPENAI_API_KEY
    
    def get_llm_api_key(self, model_name: str = None) -> Optional[str]:
        """Alias for get_api_key_for_model for backward compatibility"""
        return self.get_api_key_for_model(model_name or self.ARCHITECT_MODEL)
    
    def get_litellm_model(self, agent_type: str = "architect") -> str:
        """Get LiteLLM model string based on agent type and provider"""
        model_map = {
            "architect": self.ARCHITECT_MODEL,
            "developer": self.DEVELOPER_MODEL,
            "qa": self.QA_MODEL
        }
        
        base_model = model_map.get(agent_type, self.ARCHITECT_MODEL)
        
        # Add provider prefix for LiteLLM if needed
        if self.LLM_PROVIDER == "azure" and not base_model.startswith("azure/"):
            return f"azure/{self.AZURE_OPENAI_DEPLOYMENT or base_model}"
        elif self.LLM_PROVIDER == "anthropic" and not base_model.startswith("anthropic/"):
            return f"anthropic/{base_model}"
        elif self.LLM_PROVIDER == "gemini" and not base_model.startswith("gemini/"):
            return f"gemini/{base_model}"
        
        return base_model
    
    def validate_settings(self) -> bool:
        """Validate all required settings"""
        errors = []
        warnings = []
        
        # Check repository configuration
        repos = self.get_repositories()
        if not repos:
            warnings.append("No repositories configured. Set GIT_REPOS or GIT_REPOS_CONFIG")
        
        # Check LLM configuration
        if self.LLM_PROVIDER == "openai" and not self.OPENAI_API_KEY:
            errors.append("OPENAI_API_KEY is required when using OpenAI models")
        
        if self.LLM_PROVIDER == "azure":
            if not self.AZURE_OPENAI_API_KEY:
                errors.append("AZURE_OPENAI_API_KEY is required for Azure")
            if not self.AZURE_OPENAI_ENDPOINT:
                errors.append("AZURE_OPENAI_ENDPOINT is required for Azure")
        
        if self.LLM_PROVIDER == "anthropic" and not self.ANTHROPIC_API_KEY:
            errors.append("ANTHROPIC_API_KEY is required for Anthropic models")
        
        if self.LLM_PROVIDER == "gemini":
            if not self.GOOGLE_CLOUD_SERVICE_ACCOUNT_FILE or not os.path.exists(self.GOOGLE_CLOUD_SERVICE_ACCOUNT_FILE):
                errors.append(f"GOOGLE_CLOUD_SERVICE_ACCOUNT_FILE is required for Gemini models: {self.GOOGLE_CLOUD_SERVICE_ACCOUNT_FILE}")
        
        # Check database connections
        if not self.SUPABASE_URL:
            warnings.append("SUPABASE_URL not configured. Some features may be unavailable")
        
        # Log warnings
        for warning in warnings:
            logger.warning(f"Configuration warning: {warning}")
        
        # Raise errors
        if errors:
            raise ValueError(f"Configuration errors: {'; '.join(errors)}")
        
        logger.info(
            "Settings validated successfully",
            extra={
                "environment": self.ENVIRONMENT,
                "repositories": len(repos),
                "llm_provider": self.LLM_PROVIDER
            }
        )
        
        return True


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    _settings = Settings()
    try:
        _settings.validate_settings()
    except ValueError as e:
        logger.error(f"Settings validation failed: {e}")
        # In development, continue with warnings
        if _settings.ENVIRONMENT != "production":
            logger.warning("Continuing with invalid settings in non-production environment")
        else:
            raise
    return _settings


# Global settings instance
settings = get_settings()