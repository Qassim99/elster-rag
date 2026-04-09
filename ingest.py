import json

from langchain_core.documents import Document

from app.core.config import settings
from app.infrastructure.vector_store import QdrantRepository

DATA_PATH = "data/chunksnew.json"


def load_chunks(path: str) -> list[Document]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return [
        Document(page_content=chunk["page_content"], metadata=chunk.get("metadata", {}))
        for chunk in raw
    ]


def main():
    docs = load_chunks(DATA_PATH)
    print(f"Loaded {len(docs)} documents from {DATA_PATH}")

    repo = QdrantRepository(settings, mode="docker")
    repo.ingest_documents(docs, use_local_dense=True)
    print(f"Ingested {len(docs)} documents into collection '{settings.collection_name}'")


if __name__ == "__main__":
    main()
