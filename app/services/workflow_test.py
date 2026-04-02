import json
from typing import List, TypedDict

from app.infrastructure.llm_provider import LLMProvider
from app.infrastructure.vector_store import QdrantRepository
from langgraph.graph import END, StateGraph


class AgentState(TypedDict):
    user_question: str
    chat_history: List[dict]

    retrieved_context: str
    final_answer: str


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
        """Constructs a minimal test State Machine."""
        workflow = StateGraph(AgentState)

        # add Nodes
        workflow.add_node("retriever", self.node_retriever)
        workflow.add_node("generate_answer", self.node_generate_answer)

        # Add Edges
        workflow.set_entry_point("retriever")
        workflow.add_edge("retriever", "generate_answer")
        workflow.add_edge("generate_answer", END)

        # Compile graph into runnable application
        return workflow.compile()

    def execute(self, user_question: str, history: list) -> str:
        """Entry point for the API."""
        print(f"\n[Workflow Started] Question: '{user_question}'")

        initial_state = {
            "user_question": user_question,
            "chat_history": history,
            "retrieved_context": "",
            "final_answer": "",
        }

        # Invoke the compiled LangGraph
        final_state = self.graph.invoke(initial_state)

        print("[Workflow Completed]\n")
        return final_state["final_answer"]

    # NODE IMPLEMENTATIONS

    def node_retriever(self, state: AgentState) -> dict:
        print(" -> [Node: Retriever] Searching Qdrant database...")

        # Search Qdrant
        docs = self.vector_repo.hybrid_search(state["user_question"], top_k=5)

        # Format the retrieved documents
        context = "\n\n".join(
            [
                f"Source: {d.metadata.get('source', 'Unknown')}\n{d.page_content}"
                for d in docs
            ]
        )

        if not context:
            context = "Keine relevanten Informationen in der Datenbank gefunden."

        print(f" -> [Node: Retriever] Found {len(docs)} document chunks.")

        return {"retrieved_context": context}

    def node_generate_answer(self, state: AgentState) -> dict:
        print(" -> [Node: Generator] Sending context and question to LLM...")

        sys_prompt = """Du bist ein hilfreicher ELSTER-Assistent. Beantworte Fragen ausschließlich
            basierend auf den folgenden Informationen aus der ELSTER-Wissensdatenbank.

            Regeln:
            - Antworte immer auf Deutsch.
            - Wenn die bereitgestellten Informationen die Frage nicht beantworten, sage ehrlich:
              "Dazu habe ich leider keine Information in der Wissensdatenbank."
            - Zitiere die Quelle am Ende deiner Antwort: section > subsection > topic > question.
            - Sei präzise und hilfreich.
            Wissensdatenbank:
            {retrieved_context}
            """

        messages = [
            {"role": "system", "content": sys_prompt},
            {
                "role": "user",
                "content": f"Kontext:\n{state['retrieved_context']}\n\nFrage: {state['user_question']}",
            },
        ]

        res = self.llm_provider.generate_chat_completion(messages)

        return {"final_answer": res.choices[0].message.content}
