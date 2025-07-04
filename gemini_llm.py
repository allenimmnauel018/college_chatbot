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
                parts=[types.Part(text=prompt)]
            )
        ]

        # Removed thinking_config and response_mime_type as they might be redundant
        # if the default behavior is sufficient for plain text responses.
        # If specific configurations are needed, they can be re-added.
        config = types.GenerateContentConfig() # Simpler config initialization

        response = self._client.models.generate_content_stream(
            model=self.model,
            contents=contents,
            config=config
        )

        return "".join([chunk.text for chunk in response])

    @property
    def _llm_type(self) -> str:
        return "custom-gemini"

# Removed the if __name__ == "__main__": GeminiLLM() block
# as this file is primarily meant to be imported as a module.