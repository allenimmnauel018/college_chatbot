import os
import logging
from typing import Optional, List
from pydantic import PrivateAttr
from dotenv import load_dotenv
from langchain_core.language_models.llms import LLM
from google import genai
from google.genai import types

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class GeminiLLM(LLM):
    """
    LangChain-compatible LLM wrapper for Google's Gemini 2.x models.
    """

    model: str = "gemini-2.5-pro"
    _client: genai.Client = PrivateAttr()

    def __init__(self, model: Optional[str] = None, **kwargs):
        """
        Initialize the GeminiLLM with environment API key and selected model.
        """
        super().__init__(**kwargs)
        load_dotenv()

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            logging.error("❌ Missing GEMINI_API_KEY in .env or environment.")
            raise ValueError("GEMINI_API_KEY not set in the environment.")

        self._client = genai.Client(api_key=api_key)

        if model:
            self.model = model

        logging.info(f"✅ GeminiLLM initialized with model: {self.model}")

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """
        Generates a response using Gemini model for a given prompt.
        """
        try:
            part = types.Part.from_text(text=prompt)
            content = types.Content(role="user", parts=[part])

            response = self._client.models.generate_content_stream(
                model=self.model,
                contents=[content],
                config=types.GenerateContentConfig(response_mime_type="text/plain")
            )

            result = "".join(chunk.text for chunk in response if hasattr(chunk, "text") and chunk.text)
            return result.strip()

        except Exception as e:
            logging.error(f"❌ GeminiLLM error while generating response: {e}", exc_info=True)
            return "⚠️ LLM failed to respond."

    @property
    def _llm_type(self) -> str:
        return "gemini-llm"
