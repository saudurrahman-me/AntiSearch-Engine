import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

# Compute the root directory of the project dynamically (assuming config.py is in execution/)
ROOT_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    # Search Engine Database Paths
    DB_PATH: str = str(ROOT_DIR / "data" / "search_engine.db")
    FAISS_PATH: str = str(ROOT_DIR / "data" / "vectors.faiss")
    
    # Crawler Configuration
    CRAWL_START_URL: str = "https://docs.python.org/3.14/"
    CRAWL_LIMIT: int = 1500
    
    # API Server Configuration
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    
    # Model Configuration
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384
    
    model_config = SettingsConfigDict(
        env_file=str(ROOT_DIR / ".env"),
        env_file_encoding='utf-8',
        extra="ignore"
    )

settings = Settings()
