import os

try:
    import google.generativeai as genai
except ImportError as e:
    msg = (
        "The Gemini controller requires the optional `gemini` dependency to be installed using "
        "`pip install tetris[gemini]`!"
    )
    raise ImportError(msg) from e


class Gemini:
    def __init__(
        self, api_key: str | None = None, model_name: str = "gemini-1.5-flash", requests_per_minute_limit: int = 15
    ) -> None:
        """Initialize the Gemini LLM.

        For setting up an API key, see https://ai.google.dev/gemini-api/docs/api-key.
        For a list of available models and their rate limits, see https://ai.google.dev/gemini-api/docs/models/gemini.

        Args:
            api_key: The API key to use for the Gemini API. If not provided, the `GEMINI_API_KEY` environment variable
                will be used.
            model_name: The name of the Gemini model to use. Defaults to "gemini-1.5-flash".
            requests_per_minute_limit: The maximum number of requests per minute to make to the Gemini API. Defaults to
                15.
        """
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            msg = "No API key provided and no `GEMINI_API_KEY` environment variable found!"
            raise ValueError(msg)

        genai.configure(api_key=api_key)
        self._chat: genai.ChatSession | None = None
        self._model = genai.GenerativeModel(model_name)
        self._requests_per_minute_limit = requests_per_minute_limit

    @property
    def requests_per_minute_limit(self) -> float:
        return self._requests_per_minute_limit

    def start_new_chat(self, system_prompt: str | None) -> None:
        self._chat = self._model.start_chat(history={"role": "user", "parts": system_prompt} if system_prompt else None)

    def send_message(self, message: str) -> str:
        if self._chat is None:
            msg = "Chat has not been started yet! Call `start_new_chat` before `send_message`."
            raise ValueError(msg)

        return self._chat.send_message(message)._result.candidates[0].content.parts[0].text.strip()  # noqa: SLF001
