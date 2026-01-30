from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

# Get the project root directory (parent of config/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=str(ENV_FILE), env_ignore_empty=True, extra="ignore")

    NEO4J_URI: str 
    NEO4J_USER: str
    NEO4J_PASSWORD: str
    NEO4J_DATABASE: str = "neo4j"
    QDRANT_URL: str 
    QDRANT_API_KEY: str 
    SUPABASE_URL: str
    SUPABASE_KEY: str

    GEMINI_API_KEY: str = ""
    LITELLM_MODEL: str = "gemini-3-pro"

    GIT_REPO_VERTX: str = "https://github.com/vert-x3/vertx-examples.git"
    GIT_REPO_REACT: str = "https://github.com/your-org/react-frontend-repo.git"
    GIT_TOKEN: str  

    SECRET_KEY: str 
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

settings = Settings()