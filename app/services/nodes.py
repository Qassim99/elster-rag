import json

from app.core.config import Settings
from app.infrastructure.llm_provider import LLMProvider
from app.infrastructure.reranker import Reranker
from app.infrastructure.vector_store import QdrantRepository
from app.services.agent_state import AgentState


class Node:
    """Contians all nodes and routing logic for the langGraph."""

    def __init__(
        self, vector_repo: QdrantRepository, llm_provider: LLMProvider, settings
    ):
        self.vector_repo = vector_repo
        self.llm_provider = llm_provider
        self.settings: Settings = settings

    def paraphraser(self, state: AgentState) -> AgentState:
        sys_prompt = "You are Query Refiner + Language Detector. Return strict JSON with keys: 'rewritten_question', 'detected_language'."
        messages = [
            {"role": "system", "content": sys_prompt},
            {
                "role": "user",
                "content": f"History: {state['chat_history']}\nQuestion: {state['user_question']}",
            },
        ]

        res = self.llm_provider.generate_chat_completion(messages)
        # print(res)  # Debug: Print the raw LLM response
        content = res.choices[0].message.content

        try:
            json_str = content[content.find("{") : content.rfind("}") + 1]
            data = json.loads(json_str)

            # Mutate state
            state["paraphrased_question"] = data.get(
                "rewritten_question", state["user_question"]
            )
            state["language"] = data.get("detected_language", "German")
        except json.JSONDecodeError:
            # Fallback to defaults if JSON parsing fails
            state["paraphrased_question"] = state["user_question"]
            state["language"] = "German"

        return state

    def intent_detector(self, state: AgentState) -> AgentState:
        sys_prompt = "You are an intent router. Return ONLY '0' (Tax), '1' (Off-topic), or '2' (Greeting/Capability)."
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": state["paraphrased_question"]},
        ]

        res = self.llm_provider.generate_chat_completion(messages, temperature=0.0)
        scenario = res.choices[0].message.content.strip()

        if "2" in scenario:
            state["intent"] = "greeting"
        elif "1" in scenario:
            state["intent"] = "off_topic"
        else:
            state["intent"] = "tax"

        return state

    def decomppser(self, state: AgentState) -> AgentState:
        """Decomposes the user question into sub-questions."""
        pass

    def retriever(self, state: AgentState) -> AgentState:
        docs = self.vector_repo.hybrid_search(
            state["paraphrased_question"], top_k=self.settings.top_k
        )

        # Store RAW docs in state for the reranker to evaluate
        state["raw_documents"] = docs
        return state

    def reranker(self, state: AgentState) -> AgentState:
        pass
