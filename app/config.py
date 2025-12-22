"""Application configuration settings."""

from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings , SettingsConfigDict

class Settings(BaseSettings):
    """Application config setting loaded from environment variables."""
    
    model_config = SettingsConfigDict(env_file=".env", 
                                      env_file_encoding="utf-8",
                                      extra="ignore")
    
    #Open AI Config 
    OPENAI_API_KEY: str
    
    #qdrant config cloud
    QDRANT_URL: str
    QDRANT_API_KEY: str
    
    #collection settings   
    COLLECTION_NAME: str = "rag_documents"
    
    #document processing settings
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    
    #model configuration
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.0
    embedding_model: str = "text-embedding-3-small"
    
    #retrieval settings
    top_k_retrieval: int = 5
    
    #ragas evaluation settings
    enable_ragas_evaluation: bool = True
    ragas_timeout_seconds: float = 30.0
    ragas_log_results: bool = True
    ragas_llm_model: str | None = None  # Defaults to llm_model if not set
    ragas_llm_temperature: float | None = None  # Defaults to llm_temperature if not set
    ragas_embedding_model: str | None = None  # Defaults to embedding_model if not set
    
    # #adding langsmith config
    # LANGSMITH_API_KEY: str
    # LANGSMITH_PROJECT: str 
    # LANGSMITH_TRACING: str 
    # LANGSMITH_ENDPOINT: str
    
    
    # API settings
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    
    # Application info
    APP_NAME: str = "RAG Q&A System"
    APP_VERSION: str = "0.1.0"
    
    #logging settings
    LOG_LEVEL: str = Field("INFO", description="Logging level")
    
@lru_cache
def get_settings() -> Settings:
    """Get the application settings with caching."""
    return Settings()