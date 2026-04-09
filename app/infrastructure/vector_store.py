from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient


class QdrantRepository:
    def __init__(self, settings, mode="url", path=None):
        self.settings = settings
        self.vectorstore = None

        self.connect_kwargs = {}
        if mode == "docker":
            self.connect_kwargs["url"] = "http://localhost:6333"
        elif mode in ["cloud", "url"]:
            self.connect_kwargs["url"] = self.settings.qdrant_url
            if self.settings.qdrant_api_key:
                self.connect_kwargs["api_key"] = self.settings.qdrant_api_key
        elif path:
            self.connect_kwargs["path"] = path
        else:
            self.connect_kwargs["location"] = ":memory:"

    def _get_sparse_embeddings(self):
        return FastEmbedSparse(model_name=self.settings.sparse_model, language="german")

    def _get_dense_embeddings(self, use_local: bool):
        if use_local:
            return HuggingFaceEmbeddings(
                model_name=self.settings.dense_local_embedding_model,
                model_kwargs={
                    "device": "cpu"
                },  # Default to CPU for API, CLI can override to CUDA
                encode_kwargs={"normalize_embeddings": True},
            )
        return OpenAIEmbeddings(
            model=self.settings.dense_embedding_model,
            api_key=self.settings.llm_api_key,
            base_url=self.settings.llm_base_url,
        )

    def initialize_for_retrieval(self):
        self.vectorstore = QdrantVectorStore.from_existing_collection(
            embedding=self._get_dense_embeddings(
                use_local=True
            ),  # Defaulting API to dense model using API
            sparse_embedding=self._get_sparse_embeddings(),
            collection_name=self.settings.collection_name,
            retrieval_mode=RetrievalMode.HYBRID,
            **self.connect_kwargs,
        )

    def ingest_documents(self, docs: list, use_local_dense: bool):
        """Write documents into Qdrant with both dense and sparse embeddings."""
        self.vectorstore = QdrantVectorStore.from_documents(
            documents=docs,
            embedding=self._get_dense_embeddings(use_local=use_local_dense),
            sparse_embedding=self._get_sparse_embeddings(),
            collection_name=self.settings.collection_name,
            retrieval_mode=RetrievalMode.HYBRID,
            force_recreate=True,
            **self.connect_kwargs,
        )

    # using Reciprocal Rank Fusion (RRF) to compine dense and sparse results in hybrid search
    def hybrid_search(self, query: str, top_k: int = 3):
        if not self.vectorstore:
            raise RuntimeError("VectorStore not initialized")
        return self.vectorstore.similarity_search(query, k=top_k)

    def test_search_modes(self, query: str, top_k: int = 2):
        results_map = {}
        for mode in [RetrievalMode.HYBRID, RetrievalMode.DENSE, RetrievalMode.SPARSE]:
            self.vectorstore.retrieval_mode = mode
            results_map[mode.name] = self.vectorstore.similarity_search(query, k=top_k)
        self.vectorstore.retrieval_mode = RetrievalMode.HYBRID  # Reset
        return results_map
