# gemini_llm.py
from langchain_core.language_models.llms import LLM
from typing import Optional, List
from google import genai
from google.genai import types
from dotenv import load_dotenv
from pydantic import PrivateAttr
import os
import time # Import time for measuring duration
import logging # Import logging for more structured output

# Configure logging for this module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__) # Get a logger for this module

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

            # Iterate through the stream to get chunks
            for i, chunk in enumerate(response_stream):
                # Using hasattr and getattr to safely check for text attribute
                if hasattr(chunk, 'text') and chunk.text:
                    full_response_text += chunk.text
                    # Log chunks to see if data is coming through
                    # logger.debug(f"[_call] Received chunk {i}: {chunk.text[:50]}...") # Use debug for less verbosity
                else:
                    logger.warning(f"[_call] Received empty or non-text chunk {i}: {chunk}")

            end_time = time.time()
            logger.info(f"[_call] Stream completed. Total duration: {end_time - start_time:.2f} seconds.")
            logger.info(f"[_call] Full response text (first 200 chars): {full_response_text[:200]}...")

        except genai.types.APIError as api_error:
            logger.error(f"[_call] Gemini API Error: {api_error.message} (Code: {api_error.code})")
            # You might want to re-raise or return a specific error message
            return f"Error from Gemini API: {api_error.message}"
        except Exception as e:
            logger.error(f"[_call] Unexpected error during Gemini API call: {e}")
            # You might want to re-raise or return a specific error message
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