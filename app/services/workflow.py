import json

from app.infrastructure.llm_provider import LLMProvider
from app.infrastructure.vector_store import QdrantRepository
from app.services.agent_state import AgentState
from langgraph.graph import END, StateGraph


class RAGWorkflowEngine:
    def __init__(
        self,
        vector_repo: QdrantRepository,
        llm_provider: LLMProvider,
    ):
        self.vector_repo = vector_repo
        self.llm_provider = llm_provider
        self.graph = self._build_graph()

    def _build_graph(self):
        pass
