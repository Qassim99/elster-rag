from app.core.config import Settings
from sentence_transformers import CrossEncoder
from transformers import AutoModel


class Reranker:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.model_name = self.settings.reranker_model
        self.model = CrossEncoder(self.model_name)

    def rerank_documents(self, query: str, documents: list, top_n: int = 3) -> str:
        """
        Takes a query and a list of LangChain Document objects, scores them,
        and returns the top N documents combined into a single context string.
        """
        if not documents:
            return "NO RELEVANT CONTEXT FOUND."

        # Create pairs of [query, document_text] for the CrossEncoder
        pairs = [[query, doc.page_content] for doc in documents]

        # Get relevance scores
        scores = self.model.predict(pairs)

        # Combine documents with their scores
        scored_docs = list(zip(documents, scores))

        # Sort by score descending (highest score first)
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        # Keep only the top N
        top_docs = [doc for doc, score in scored_docs[:top_n]]

        # Format the final context string exactly like we did before
        parts = []
        for i, doc in enumerate(top_docs, 1):
            source = doc.metadata.get("source", "Unknown")
            path = doc.metadata.get("context_path", "")
            parts.append(
                f"--- Quelle {i}: {source} ---\nPfad: {path}\n\n{doc.page_content}\n"
            )

        return "\n".join(parts)
