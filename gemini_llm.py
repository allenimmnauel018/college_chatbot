# gemini_llm.py
from langchain_core.language_models.llms import LLM
from typing import Optional, List
from google import genai
from google.genai import types # Keep this for other type definitions if used
from dotenv import load_dotenv
from pydantic import PrivateAttr
import os
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GeminiLLM(LLM):
    model: str = "gemini-2.5-pro"

    _client: genai.Client = PrivateAttr()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            logger.error("GEMINI_API_KEY not found in .env file. Please set it.")
            raise ValueError("GEMINI_API_KEY environment variable is not set.")
        try:
            self._client = genai.Client(api_key=api_key)
            logger.info("Gemini API client initialized successfully.")
        except Exception as e:
            logger.error(f"Error initializing Gemini API client: {e}")
            raise


    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        time.sleep(0.5)
        logger.info(f"[_call] Received prompt (first 100 chars): {prompt[:100]}...")
        contents = [
            types.Content(
                role="user",
                parts=[types.Part(text=prompt)]
            )
        ]

        config = types.GenerateContentConfig()

        full_response_text = ""
        start_time = time.time()
        try:
            logger.info("[_call] Attempting to call generate_content_stream...")
            response_stream = self._client.models.generate_content_stream(
                model=self.model,
                contents=contents,
                config=config
            )

            for i, chunk in enumerate(response_stream):
                if hasattr(chunk, 'text') and chunk.text:
                    full_response_text += chunk.text
                else:
                    logger.warning(f"[_call] Received empty or non-text chunk {i}: {chunk}")

            end_time = time.time()
            logger.info(f"[_call] Stream completed. Total duration: {end_time - start_time:.2f} seconds.")
            logger.info(f"[_call] Full response text (first 200 chars): {full_response_text[:200]}...")

        # --- FIX STARTS HERE ---
        except genai.APIError as api_error: # Changed from genai.types.APIError
        # --- FIX ENDS HERE ---
            logger.error(f"[_call] Gemini API Error: {api_error.message} (Code: {api_error.code})")
            return f"Error from Gemini API: {api_error.message}"
        except Exception as e:
            logger.error(f"[_call] Unexpected error during Gemini API call: {e}")
            return f"An unexpected error occurred: {e}"

        return full_response_text

    @property
    def _llm_type(self) -> str:
        return "custom-gemini"

if __name__ == "__main__":
    # Test the LLM directly
    print("Testing GeminiLLM directly...")
    try:
        llm = GeminiLLM()
        test_prompt = "What is the capital of France?"
        print(f"Sending test prompt: '{test_prompt}'")
        response = llm._call(test_prompt)
        print(f"\nTest Response: {response}")
    except Exception as e:
        print(f"Test failed: {e}")