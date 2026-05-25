from typing import TYPE_CHECKING, Any, Literal, NotRequired, TypedDict

if TYPE_CHECKING:
    from langchain_core.documents import Document
else:
    Document = Any


class AgentState(TypedDict):
    # Inputs
    user_question: str
    chat_history: list[dict[str, Any]]

    # Updated by Paraphraser
    paraphrased_question: NotRequired[str]
    language: NotRequired[str]

    # Updated by Routers/Evaluators
    intent: NotRequired[Literal["tax", "off_topic", "greeting"]]
    context_sufficient: NotRequired[bool]  # True if local context is enough
    is_grounded: NotRequired[bool]  # True if web answer doesn't hallucinate

    # Content Pipeline
    sub_questions: NotRequired[list[str]]
    raw_documents: NotRequired[list[Document]]
    retrieved_context: NotRequired[str]
    candidate_answer: NotRequired[str]
    final_answer: NotRequired[str]
