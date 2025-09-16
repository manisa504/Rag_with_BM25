"""
Settings configuration using Pydantic BaseSettings for environment variable management.
"""

from pydantic_settings import BaseSettings
from typing import Literal
import os
from pathlib import Path

# Load environment variables from .env file
from dotenv import load_dotenv

# Load .env file from the project root
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Google API Configuration
    google_api_key: str
    gemini_model_text: str = "gemini-2.5-flash"
    google_embed_model: str = "text-embedding-004"
    
    # Embedding Backend
    embedding_backend: Literal["google", "gemma"] = "google"
    
    # Qdrant Configuration
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    qdrant_collection: str = "airline_rag"
    
    # Chunking Configuration
    chunk_tokens: int = 400
    chunk_overlap: int = 50
    
    # Retrieval Configuration
    top_k: int = 8
    rerank_k: int = 5
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()
