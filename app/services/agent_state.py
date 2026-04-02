from typing import List, TypedDict


class AgentState(TypedDict):
    # Inputs
    user_question: str
    chat_history: List[dict]

    # Updated by Paraphraser
    paraphrased_question: str
    language: str

    # Updated by Routers/Evaluators
    intent: str  # 'tax', 'off_topic', or 'greeting'
    context_sufficient: bool  # True if local context is enough
    is_grounded: bool  # True if web answer doesn't hallucinate

    # Content Pipeline
    raw_documents: list
    retrieved_context: str
    candidate_answer: str
    final_answer: str
