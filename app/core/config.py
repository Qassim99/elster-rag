import os

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()


class Settings(BaseSettings):
    # Qdrant Settings
    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key: str = os.getenv("QDRANT_API_KEY", "")
    collection_name: str = os.getenv("QDRANT_COLLECTION", "elster_help")

    # Models
    dense_local_embedding_model: str = os.getenv(
        "DENSE_LOCAL_EMBEDDING_MODEL",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    dense_embedding_model: str = os.getenv(
        "DENSE_EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-8B"
    )
    sparse_model: str = os.getenv("SPARSE_MODEL_NAME", "Qdrant/bm25")

    # LLM Settings
    llm_api_key: str = os.getenv("OPENROUTER_API_KEY", "")
    llm_base_url: str = os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1")
    llm_model: str = os.getenv("LLM_MODEL", "google/gemini-3-flash-preview")
    top_k: int = int(os.getenv("RAG_TOP_K", "5"))
    api_port: int = int(os.getenv("RETRIEVAL_API_PORT", "8100"))

    # Reranker Settings
    reranker_model: str = os.getenv("RERANKER_MODEL", "jinaai/jina-reranker-v3")


settings = Settings()
# print(f"Qdrant URL: {settings.qdrant_url}")
# print(f"LLM Base URL: {settings.llm_api_key}")
