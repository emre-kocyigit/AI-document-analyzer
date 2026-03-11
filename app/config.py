from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    app_name: str = "AI Document Analyzer"
    app_version: str = "1.0.0"
    debug: bool = False

    # Ollama
    ollama_base_url: str = "http://ollama:11434"
    ollama_model: str = "mistral"

    # ChromaDB
    chroma_persist_dir: str = "./chroma_db"
    chroma_collection_name: str = "documents"

    # Document Processing
    upload_dir: str = "./uploads"
    max_file_size_mb: int = 20
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # Embedding Model
    embedding_model: str = "all-MiniLM-L6-v2"

    # ONNX
    onnx_ner_model: str = "dslim/bert-base-NER"
    onnx_classification_model: str = "typeform/distilbert-base-uncased-mnli"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache
def get_settings() -> Settings:
    return Settings()
