from langchain_core.language_models.llms import LLM
from typing import Optional, List
from google import genai
from google.genai import types
from dotenv import load_dotenv
from pydantic import PrivateAttr
import os

class GeminiLLM(LLM):
    model: str = "gemini-2.5-pro"

    # Declare private attributes for Pydantic compatibility
    _client: genai.Client = PrivateAttr()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        load_dotenv()
        self._client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        contents = [
            types.Content(
                role="user",
                parts=[types.Part(text=prompt)]  # ✅ FIXED line
            )
        ]

        config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=-1),
            response_mime_type="text/plain"
        )

        response = self._client.models.generate_content_stream(
            model=self.model,
            contents=contents,
            config=config
        )

        return "".join([chunk.text for chunk in response])


    @property
    def _llm_type(self) -> str:
        return "custom-gemini"

if __name__ == "__main__":
    GeminiLLM()