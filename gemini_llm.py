# gemini_llm.py

import os
import logging
import time
from typing import Optional, List, Dict
from pydantic import PrivateAttr
from dotenv import load_dotenv
from langchain_core.language_models.llms import LLM
from google import genai
from google.genai import types
from typing import cast

# ---------------- Logging Setup ----------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ---------------- Load Environment ----------------
load_dotenv()

# ---------------- Multi-Key Setup ----------------
API_KEYS_RAW = [
    os.getenv("GEMINI_API_KEY_1"),
    os.getenv("GEMINI_API_KEY_2"),
    os.getenv("GEMINI_API_KEY_3"),
    os.getenv("GEMINI_API_KEY_4"),
]

API_KEYS: List[str] = cast(List[str], [key for key in API_KEYS_RAW if key is not None])

if not API_KEYS:
    logging.error("❌ No Gemini API keys found (GEMINI_API_KEY_1..4).")
    raise ValueError("At least one GEMINI_API_KEY is required.")

_key_index = 0
_key_usage: Dict[str, int] = {key: 0 for key in API_KEYS}  # Track usage per key


def get_next_api_key() -> str:
    global _key_index
    key = API_KEYS[_key_index]
    _key_usage[key] += 1
    logging.info(f"🔄 Switching to API key {_key_index} | Used {_key_usage[key]} times")
    _key_index = (_key_index + 1) % len(API_KEYS)
    return key



# ---------------- Gemini LLM Wrapper ----------------
class GeminiLLM(LLM):
    """
    LangChain-compatible wrapper for Google's Gemini models with:
    - API key rotation
    - model fallback (flash → pro)
    - call retry + timeout
    - key usage tracking
    """

    model: str = "gemini-2.5-pro"
    _client: genai.Client = PrivateAttr()
    _api_key: Optional[str] = PrivateAttr(default=None)

    def __init__(self, model: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)

        api_key = get_next_api_key()
        self._api_key = api_key
        self._client = genai.Client(api_key=api_key)

        if model:
            self.model = model

        logging.info(f"✅ GeminiLLM initialized using model: {self.model} and key index {(_key_index - 1) % len(API_KEYS)}")

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """
        Generate a response using Gemini. Retry on rate-limit. Fallback to Pro if Flash fails.
        """
        models_to_try = [self.model]
        if self.model.startswith("gemini-2.5-flash"):
            models_to_try.append("gemini-2.5-pro")

        max_attempts = len(API_KEYS)
        retry_delay = 5  # seconds
        timeout_limit = 30  # seconds max per streaming attempt

        for model_name in models_to_try:
            attempt = 0
            while attempt < max_attempts:
                try:
                    api_key = self._api_key or "UNKNOWN"
                    usage_count = _key_usage.get(api_key, 0)
                    logging.info(f"🧠 Calling {model_name} | API key index: {(_key_index - 1) % len(API_KEYS)} | Usage: {usage_count}")


                    part = types.Part.from_text(text=prompt)
                    content = types.Content(role="user", parts=[part])

                    start_time = time.time()
                    response = self._client.models.generate_content_stream(
                        model=model_name,
                        contents=[content],
                        config=types.GenerateContentConfig(response_mime_type="text/plain")
                    )

                    chunks = []
                    try:
                        for chunk in response:
                            if hasattr(chunk, "text") and chunk.text:
                                chunks.append(chunk.text)
                            if time.time() - start_time > timeout_limit:
                                raise TimeoutError("⚠️ Streaming timeout")
                    except Exception as stream_error:
                        logging.warning(f"⚠️ Stream error: {stream_error}")
                        break


                    result = "".join(chunks).strip()

                    if not result:
                        raise ValueError("⚠️ Gemini returned empty response.")

                    return result

                except Exception as e:
                    if "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e):
                        logging.warning(f"⚠️ Quota exceeded for key. Rotating key...")
                        attempt += 1
                        self._api_key = get_next_api_key()
                        self._client = genai.Client(api_key=self._api_key)
                    elif isinstance(e, TimeoutError):
                        logging.warning(str(e))
                        attempt += 1
                        self._api_key = get_next_api_key()
                        self._client = genai.Client(api_key=self._api_key)
                    else:
                        logging.warning(f"❌ Unexpected error on {model_name}: {e}")
                        break  # Try next model in fallback

        raise RuntimeError("❌ All Gemini API keys or models exhausted.")

    @property
    def _llm_type(self) -> str:
        return "gemini-llm"
