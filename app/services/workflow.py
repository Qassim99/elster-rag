from typing import TYPE_CHECKING

from app.core.config import Settings
from app.infrastructure.llm_provider import LLMProvider
from app.infrastructure.vector_store import QdrantRepository
from app.services.agent_state import AgentState
from app.services.nodes import Node
from langgraph.graph import END, StateGraph

if TYPE_CHECKING:
    from app.infrastructure.reranker import Reranker


class RAGWorkflowEngine:
    def __init__(
        self,
        vector_repo: QdrantRepository,
        llm_provider: LLMProvider,
        settings: Settings | None = None,
        reranker: "Reranker | None" = None,
    ):
        self.vector_repo = vector_repo
        self.llm_provider = llm_provider
        self.settings = settings or vector_repo.settings
        self.nodes = Node(
            vector_repo=self.vector_repo,
            llm_provider=self.llm_provider,
            settings=self.settings,
            reranker=reranker,
        )
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(AgentState)

        workflow.add_node("paraphraser", self.nodes.paraphraser)
        workflow.add_node("intent_detector", self.nodes.intent_detector)
        workflow.add_node("retriever", self.nodes.retriever)
        workflow.add_node("reranker", self.nodes.reranker)
        workflow.add_node("hallucination_detector", self.nodes.hallucination_detector)
        workflow.add_node("generate_answer", self.nodes.generate_answer)
        workflow.add_node("apology", self.nodes.apology)
        workflow.add_node("greeting", self.nodes.greeting)
        workflow.add_node("off_topic", self.nodes.off_topic)

        workflow.set_entry_point("paraphraser")
        workflow.add_edge("paraphraser", "intent_detector")
        workflow.add_conditional_edges(
            "intent_detector",
            self._route_intent,
            {
                "tax": "retriever",
                "off_topic": "off_topic",
                "greeting": "greeting",
            },
        )
        workflow.add_edge("retriever", "reranker")
        workflow.add_edge("reranker", "hallucination_detector")
        workflow.add_conditional_edges(
            "hallucination_detector",
            self._route_context,
            {
                "answer": "generate_answer",
                "apology": "apology",
            },
        )
        workflow.add_edge("generate_answer", END)
        workflow.add_edge("apology", END)
        workflow.add_edge("greeting", END)
        workflow.add_edge("off_topic", END)

        return workflow.compile()

    @staticmethod
    def _route_intent(state: AgentState) -> str:
        return state.get("intent", "tax")

    @staticmethod
    def _route_context(state: AgentState) -> str:
        if state.get("context_sufficient", False):
            return "answer"
        return "apology"

    def execute(self, user_question: str, history: list | None = None) -> str:
        initial_state: AgentState = {
            "user_question": user_question,
            "chat_history": history or [],
        }

        final_state = self.graph.invoke(initial_state)
        return final_state["final_answer"]
