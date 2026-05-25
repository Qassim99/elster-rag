import json
from typing import TYPE_CHECKING

from app.core.config import Settings
from app.infrastructure.llm_provider import LLMProvider
from app.infrastructure.vector_store import QdrantRepository
from app.services.agent_state import AgentState

if TYPE_CHECKING:
    from app.infrastructure.reranker import Reranker


class Node:
    """Contains all nodes and routing logic for the LangGraph."""

    def __init__(
        self,
        vector_repo: QdrantRepository,
        llm_provider: LLMProvider,
        settings: Settings,
        reranker: "Reranker | None" = None,
    ):
        self.vector_repo = vector_repo
        self.llm_provider = llm_provider
        self.settings = settings
        self.reranker_service = reranker

    @staticmethod
    def _extract_json_object(content: str) -> dict:
        start = content.find("{")
        end = content.rfind("}")
        if start == -1 or end == -1 or end < start:
            raise json.JSONDecodeError("No JSON object found", content, 0)
        return json.loads(content[start : end + 1])

    @staticmethod
    def _format_documents(documents: list, top_n: int = 3) -> str:
        if not documents:
            return "NO RELEVANT CONTEXT FOUND."

        parts = []
        for i, doc in enumerate(documents[:top_n], 1):
            source = doc.metadata.get("source", "Unknown")
            path = doc.metadata.get("context_path", "")
            parts.append(
                f"--- Quelle {i}: {source} ---\nPfad: {path}\n\n{doc.page_content}\n"
            )
        return "\n".join(parts)

    def paraphraser(self, state: AgentState) -> AgentState:
        sys_prompt = """
        You are a Query Refiner and Language Detector for the ELSTER tax portal assistant.

        Return a JSON object with exactly two keys: rewritten_question and detected_language.

        Do not wrap the response in quotes. Do not escape the JSON. Do not use markdown or code fences.

        Begin your response with { and end with }.

        Rules:

        - Rewrite the question only when it contains pronouns or references that require chat history to resolve (e.g. "What about that?" → full standalone question).

        - For greetings, small-talk, or capability questions, keep the original question text exactly unchanged.

        - detected_language must be exactly one of: German, English, Turkish, Arabic, French.

        Example output:

        {"rewritten_question": "what is elster?", "detected_language": "English"}

        /no_think
        """
        messages = [
            {"role": "system", "content": sys_prompt},
            {
                "role": "user",
                "content": f"History: {state['chat_history']}\nQuestion: {state['user_question']}",
            },
        ]

        res = self.llm_provider.generate_chat_completion(messages)
        content = res.choices[0].message.content or ""

        try:
            data = self._extract_json_object(content)
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
        question = state["paraphrased_question"]

        sys_prompt = """You are an intent router for ELSTER tax assistant.

        Select exactly one scenario.
            0 = Tax/ELSTER topic, forms, deadlines, certificate, registration, laws, deductions, or follow-up tax question.
            1 = Clearly off-topic (sports, weather, coding, jokes, unrelated chat).
            2 = Greeting, capabilities question, or rewrite-style request (shorter/simpler/clarify/summary of previous answer).

            If uncertain choose Scenario 0.
            Return scenario only, no explanation.
            /no_think"""
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": question},
        ]

        res = self.llm_provider.generate_chat_completion(messages, temperature=0.0)
        scenario = (res.choices[0].message.content or "").strip().strip("`\"'")
        route = scenario[0] if scenario and scenario[0] in {"0", "1", "2"} else "0"

        if route == "2":
            state["intent"] = "greeting"
        elif route == "1":
            state["intent"] = "off_topic"
        else:
            state["intent"] = "tax"

        return state

    def decomposer(self, state: AgentState) -> AgentState:
        """Decomposes the user question into sub-questions."""

        sys_prompt = """You are a question decomposition assistant. Decompose the question into smaller sub-questions if needed.
        Return a JSON array of sub-questions with the key 'sub_questions'. If no decomposition is needed return the original question as
        the only element in the array. Example output: {"sub_questions": ["What is the income tax rate for 2023?", "How does it differ for freelancers?"]}
        /no_think
        """

        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": state["paraphrased_question"]},
        ]

        res = self.llm_provider.generate_chat_completion(messages, temperature=0.0)
        content = res.choices[0].message.content or ""

        try:
            data = self._extract_json_object(content)
            state["sub_questions"] = data.get(
                "sub_questions", [state["paraphrased_question"]]
            )
        except json.JSONDecodeError:
            state["sub_questions"] = [state["paraphrased_question"]]

        return state

    def decompser(self, state: AgentState) -> AgentState:
        """Backward-compatible alias for the misspelled node name."""
        return self.decomposer(state)

    def retriever(self, state: AgentState) -> AgentState:
        query = state.get("paraphrased_question", state["user_question"])
        docs = self.vector_repo.hybrid_search(query, top_k=self.settings.top_k)

        state["raw_documents"] = docs
        return state

    def reranker(self, state: AgentState) -> AgentState:
        query = state.get("paraphrased_question", state["user_question"])
        documents = state.get("raw_documents", [])

        if self.reranker_service:
            state["retrieved_context"] = self.reranker_service.rerank_documents(
                query=query,
                documents=documents,
                top_n=min(3, self.settings.top_k),
            )
        else:
            state["retrieved_context"] = self._format_documents(
                documents,
                top_n=min(3, self.settings.top_k),
            )

        return state

    def hallucination_detector(self, state: AgentState) -> AgentState:
        context = state.get("retrieved_context", "")
        if not context or context == "NO RELEVANT CONTEXT FOUND.":
            state["context_sufficient"] = False
            return state

        sys_prompt = """You are a context sufficiency evaluator for an ELSTER RAG assistant.
        Return ONLY '0' if the context contains enough information to answer the question.
        Return ONLY '1' if the answer would require guessing or unsupported information.
        /no_think
        """
        messages = [
            {"role": "system", "content": sys_prompt},
            {
                "role": "user",
                "content": (
                    f"Question: {state.get('paraphrased_question', state['user_question'])}\n\n"
                    f"Context:\n{context}"
                ),
            },
        ]

        res = self.llm_provider.generate_chat_completion(messages, temperature=0.0)
        scenario = (res.choices[0].message.content or "").strip().strip("`\"'")
        state["context_sufficient"] = scenario.startswith("0")
        return state

    def generate_answer(self, state: AgentState) -> AgentState:
        language = state.get("language", "German")
        sys_prompt = f"""
        You are ELSTER grounded answer assistant.

        Use ONLY AVAILABLE_CONTEXT facts. Never use external memory.

        If required fact is missing, say you cannot verify and ask one clarifying question.

        No invented dates, amounts, legal claims, or office names.

        Respond in {language} language.

        Format: very short intro + max 3-5 bullet points + next action.

        If helpful, append optional UI block exactly:

        <ui>{{"cards":[{{"label":"...","prompt":"...","topic":"Registrierung|Zertifikat|Formulare|Pruefung|Abgabe"}}],"question":"..."}}</ui>
        /no_think
        """
        messages = [
            {"role": "system", "content": sys_prompt},
            {
                "role": "user",
                "content": (
                    f"Context:\n{state.get('retrieved_context', '')}\n\n"
                    f"Question: {state.get('paraphrased_question', state['user_question'])}"
                ),
            },
        ]

        res = self.llm_provider.generate_chat_completion(messages, temperature=0.1)
        answer = res.choices[0].message.content or ""
        state["candidate_answer"] = answer
        state["final_answer"] = answer
        state["is_grounded"] = True
        return state

    def apology(self, state: AgentState) -> AgentState:
        language = state.get("language", "German")

        sys_prompt = f"""
        You are ELSTER assistant safe fallback.

        Reply in {language}.

        State that information cannot be verified from trusted context.

        Ask one clarifying question and suggest official source www.elster.de.

        Add optional <ui>{{"cards":[{{"label":"...","prompt":"...","topic":"Registrierung|Zertifikat|Formulare|Pruefung|Abgabe"}}],"question":"..."}}</ui> block with 2-3 safe follow-up cards.

        /no_think
        """

        messages = [
            {"role": "system", "content": sys_prompt},
            {
                "role": "user",
                "content": f"History: {state['chat_history']}\nQuestion: {state.get('paraphrased_question', state['user_question'])}",
            },
        ]

        res = self.llm_provider.generate_chat_completion(messages, temperature=0.7)
        content = res.choices[0].message.content or ""
        state["final_answer"] = content
        return state

    def greeting(self, state: AgentState) -> AgentState:
        language = state.get("language", "German")
        sys_prompt = f"""
        You handle greeting/capability and rewrite-style requests.

        Language: {language}.

        If user asks summary/simplify/clarify, transform the previous assistant answer into short bullets.

        If user greets, respond briefly and ask one tax-focused follow-up question.

        Keep response concise.

        Optionally include an <ui>{{"cards":[{{"label":"...","prompt":"...","topic":"Registrierung|Zertifikat|Formulare|Pruefung|Abgabe"}}],"question":"..."}}</ui> block with 2-3 next-step cards.

        /no_think
        """

        messages = [
            {"role": "system", "content": sys_prompt},
            {
                "role": "user",
                "content": f"History: {state['chat_history']}\nQuestion: {state.get('paraphrased_question', state['user_question'])}",
            },
        ]

        res = self.llm_provider.generate_chat_completion(messages, temperature=0.7)
        content = res.choices[0].message.content or ""
        state["final_answer"] = content
        return state

    def off_topic(self, state: AgentState) -> AgentState:
        language = state.get("language", "German")

        sys_prompt = f"""
        You are ELSTER assistant. User request is off-topic.

        Reply briefly in {language} language.

        Politely refuse off-topic help and redirect to ELSTER tax help.

        Add this optional UI block for quick actions in {language} language:

        <ui>{{"cards":[{{"label":"ELSTER Registrierung","prompt":"Ich brauche Hilfe bei der ELSTER Registrierung","topic":"Registrierung"}},{{"label":"Welche Anlage?","prompt":"Welche Anlage brauche ich?","topic":"Formulare"}},{{"label":"Fristen","prompt":"Welche Fristen gelten fuer meine Steuererklaerung?","topic":"Pruefung"}}],"question":"Wobei genau kann ich dir bei ELSTER helfen?"}}<ui>

        /no_think
        """

        messages = [
            {"role": "system", "content": sys_prompt},
            {
                "role": "user",
                "content": f"History: {state['chat_history']}\nQuestion: {state.get('paraphrased_question', state['user_question'])}",
            },
        ]

        res = self.llm_provider.generate_chat_completion(messages, temperature=0.7)
        content = res.choices[0].message.content or ""
        state["final_answer"] = content
        return state
