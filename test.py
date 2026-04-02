from app.core.config import Settings
from app.infrastructure.llm_provider import LLMProvider
from app.infrastructure.vector_store import QdrantRepository
from app.services.workflow_test import RAGWorkflowEngine
from dotenv import load_dotenv

load_dotenv()
# Example usage:
settings = Settings()
llm_provider = LLMProvider(settings)
if llm_provider.is_available():
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
    ]
    response = llm_provider.generate_chat_completion(messages)
    print(response.choices[0].message.content)
else:
    print("LLM Provider is not available. Please check your API key and configuration.")


# test qdrant connection
qdrant_repo = QdrantRepository(settings, mode="docker")
try:
    qdrant_repo.initialize_for_retrieval()
    print("Qdrant connection successful and vector store initialized.")
except Exception as e:
    print(f"Error initializing Qdrant vector store: {e}")

print("==" * 50)
print("==" * 50)
results_map = qdrant_repo.test_search_modes("Wie kann ich mein Benutzerkonto löschen?")
print(results_map["HYBRID"])
print("==" * 50)
print(results_map["DENSE"])
print("==" * 50)
print(results_map["SPARSE"])

qdrant_repo.hybrid_search("Wie kann ich mein Benutzerkonto löschen?", top_k=3)


print("==" * 50)
print("==" * 50)
rag_engin_test = RAGWorkflowEngine(qdrant_repo, llm_provider)
response = rag_engin_test.execute("Wie kann ich mein Benutzerkonto löschen?", [])
print("RAG Response:")
print(response)

print("==" * 50)
print("==" * 50)
print("Test Nodes")

from app.services.nodes import Node

node = Node(qdrant_repo, llm_provider, settings)
state = {
    "user_question": "Wie kann ich es löschen?",
    "chat_history": ["mein Benutzerkonto"],
}
updated_state = node.paraphraser(state)
print("Updated State after Paraphrasing:")
print(updated_state)
