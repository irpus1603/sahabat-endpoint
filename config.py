"""
Configuration management for Sahabat-9B API
"""
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings"""

    # API Settings
    APP_NAME: str = "Sahabat-9B API"
    APP_VERSION: str = "1.0.0"
    API_PREFIX: str = "/api/v1"
    DEBUG: bool = False

    # Server Settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 1

    # Model Settings
    MODEL_NAME: str = "Sahabat-AI/gemma2-9b-cpt-sahabatai-v1-instruct"
    MODEL_MAX_LENGTH: int = 8192  # Gemma2 supports up to 8k context
    DEVICE: str = "cuda"  # or "cpu" or "mps" for Mac
    LOAD_IN_8BIT: bool = False
    LOAD_IN_4BIT: bool = False
    HUGGINGFACE_TOKEN: str = ""  # HuggingFace API token for model downloads

    # Generation Settings
    DEFAULT_MAX_NEW_TOKENS: int = 512
    DEFAULT_TEMPERATURE: float = 0.7
    DEFAULT_TOP_P: float = 0.9
    DEFAULT_TOP_K: int = 50

    # RAG Settings
    EMBEDDING_MODEL: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50

    # Rate Limiting
    RATE_LIMIT_ENABLED: bool = True
    MAX_REQUESTS_PER_MINUTE: int = 60

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"

    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()
