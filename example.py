from dotenv import load_dotenv

from app.core.config import Settings
from app.infrastructure.llm_provider import LLMProvider
from app.infrastructure.vector_store import QdrantRepository
from app.services.workflow import RAGWorkflowEngine

load_dotenv()


settings = Settings()
llm_provider = LLMProvider(settings, model="qwen/qwen3-32b")

qdrant_repo = QdrantRepository(settings, mode="docker")
try:
    qdrant_repo.initialize_for_retrieval()
    print("Qdrant connection successful and vector store initialized.")
except Exception as e:
    print(f"Error initializing Qdrant vector store: {e}")

rag_engin_test = RAGWorkflowEngine(qdrant_repo, llm_provider, settings)

# Visualize the workflow graph and save as PNG
png = rag_engin_test.graph.get_graph().draw_mermaid_png()
with open("rag_workflow_graph.png", "wb") as f:
    f.write(png)
response = rag_engin_test.execute("Wie kann ich mein Benutzerkonto löschen?", [])
print("RAG Response:")
print(response)
