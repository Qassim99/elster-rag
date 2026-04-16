from openai import OpenAI


class LLMProvider:
    def __init__(self, settings, is_evaluation: bool = False):
        self.settings = settings
        self.is_evaluation = is_evaluation
        self.client = None
        if settings.llm_api_key:
            self.client = OpenAI(
                api_key=settings.llm_api_key, base_url=settings.llm_base_url
            )

    def is_available(self) -> bool:
        return self.client is not None

    def generate_chat_completion(
        self,
        messages: list,
        model: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ):
        if not self.client:
            raise RuntimeError("LLM Client is not initialized (missing API key).")

        if self.is_evaluation:
            temperature = 0.0  # Force deterministic output during evaluation
            model = model or self.settings.llm_evaluator_model
        else:
            model = model or self.settings.llm_model
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response
