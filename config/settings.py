from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_ignore_empty=True, extra="ignore")

    # Database URLs
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str
    QDRANT_URL: str = "http://localhost:6333"
    POSTGRES_DSN: str = "postgresql://user:password@localhost:5432/code_intel"

    # LLM
    LITELLM_MODEL: str = "gpt-4o-mini"
    OPENAI_API_KEY: str

    # Git
    GIT_REPO_VERTX: str = "https://github.com/your-org/vertx-backend-repo.git"
    GIT_REPO_REACT: str = "https://github.com/your-org/react-frontend-repo.git"
    GIT_TOKEN: str  # For private repos

    # Security
    SECRET_KEY: str = "your-random-secret-key"  # Generate with secrets.token_hex(32)
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

settings = Settings()