# src/llm_config/llm_manager.py
import os
import logging
from typing import List, Optional
import requests  # Import requests for HTTP calls (no explicit timeout here)

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OpenAIEmbeddings, SentenceTransformerEmbeddings
from langchain_core.embeddings import Embeddings  # Base class for custom embeddings
from src.config import (
    OPENAI_API_KEY, ANTHROPIC_API_KEY, OLLAMA_BASE_URL,
    EMBEDDING_MODEL_NAME, GATEWAY_EMBEDDING_URL, GATEWAY_EMBEDDING_API_KEY, LOCAL_EMBEDDING_MODEL_PATH
)
from src.utils.exceptions import ConfigurationError, LLMError, RAGError  # APIError removed

logger = logging.getLogger(__name__)

class LLMManager:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.llm = None

    def get_llm(self):
        if self.llm:
            return self.llm # Returns already initialized LLM

        try:
            if self.model_name.startswith("gpt"):
                if not OPENAI_API_KEY:
                    raise LLMError("OpenAI API Key not found. Please set OPENAI_API_KEY in your .env file.")
                self.llm = ChatOpenAI(model=self.model_name, api_key=OPENAI_API_KEY, temperature=0.1)
                logger.info(f"Initialized OpenAI model: {self.model_name}")
            elif self.model_name.startswith("claude"):
                if not ANTHROPIC_API_KEY:
                    raise LLMError("Anthropic API Key not found. Please set ANTHROPIC_API_KEY in your .env file.")
                self.llm = ChatAnthropic(model=self.model_name, api_key=ANTHROPIC_API_KEY, temperature=0.1)
                logger.info(f"Initialized Anthropic model: {self.model_name}")
            elif self.model_name.startswith("ollama-"): # ADD THIS ELIF BLOCK FOR OLLAMA
                # Extract the actual Ollama model name (e.g., "llama2" from "ollama-llama2")
                ollama_model_name = self.model_name.replace("ollama-", "")
                self.llm = ChatOllama(model=ollama_model_name, base_url=OLLAMA_BASE_URL, temperature=0.1)
                logger.info(f"Initialized Ollama model: {ollama_model_name} at {OLLAMA_BASE_URL}")
            else:
                raise LLMError(f"Unsupported LLM model: {self.model_name}")
            return self.llm
        except Exception as e:
            logger.exception(f"Failed to initialize LLM '{self.model_name}'.")
            raise LLMError(f"Failed to initialize LLM: {e}. Check your API keys, Ollama server status, or model name.")

# --- Custom Gateway Embedding Model ---
class CustomGatewayEmbeddings(Embeddings):
    """
    Custom Embedding class to interact with a company's internal LLM Gateway for embeddings.
    Assumes the gateway expects a JSON POST request with 'input' (list of strings)
    and returns a JSON with 'data' (list of embeddings).
    No explicit timeout/retry logic internally in this version, relying on system/requests defaults.
    """

    def __init__(self, url: str, api_key: Optional[str] = None):
        if not url:
            raise ConfigurationError("Gateway Embedding URL must be provided.")
        self.url = url
        self.api_key = api_key
        logger.info(f"Initialized CustomGatewayEmbeddings for URL: {self.url}")

    def _call_gateway_api(self, texts: List[str]) -> List[List[float]]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {"input": texts}  # Adjust payload format based on your gateway's API spec

        try:
            # No explicit timeout parameter in requests.post in this version, relies on global/system defaults
            response = requests.post(self.url, headers=headers, json=payload)
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)

            data = response.json()
            # Adjust extraction based on your gateway's response structure
            embeddings = data.get("data")
            if not isinstance(embeddings, list) or not all(isinstance(e, list) for e in embeddings):
                raise ValueError(f"Gateway response 'data' is not a list of embeddings: {data}")

            return embeddings
        except requests.exceptions.HTTPError as e:
            logger.error(f"Gateway embedding HTTP error {e.response.status_code}: {e.response.text}")
            raise LLMError(f"Gateway embedding HTTP error: {e}") from e  # Changed to LLMError
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Gateway embedding connection error: {e}")
            raise LLMError(f"Gateway embedding connection error: {e}") from e  # Changed to LLMError
        except Exception as e:
            logger.error(f"Unexpected error during gateway embedding call: {e}", exc_info=True)
            raise LLMError(f"Failed to get embeddings from gateway: {e}") from e

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        logger.info(f"Embedding {len(texts)} documents via custom gateway.")
        return self._call_gateway_api(texts)

    def embed_query(self, text: str) -> List[float]:
        """Embed a query text."""
        logger.info(f"Embedding query via custom gateway: {text[:50]}...")
        return self._call_gateway_api([text])[0]  # Gateway expects list, return single embedding


# --- Embedding Model Factory ---
class EmbeddingFactory:
    """
    Factory class to get different embedding model instances, supporting local loading and custom gateways.
    """
    _embedding_model_instance = None  # Cache the instance

    @staticmethod
    def get_embedding_model(model_name: str = EMBEDDING_MODEL_NAME):
        if EmbeddingFactory._embedding_model_instance:
            logger.debug(f"Returning cached embedding model: {model_name}")
            return EmbeddingFactory._embedding_model_instance

        logger.info(f"Attempting to load embedding model: {model_name}")
        try:
            if model_name == "text-embedding-ada-002":
                if not OPENAI_API_KEY:
                    raise ConfigurationError("OPENAI_API_KEY not set for OpenAIEmbeddings.")
                # Removed request_timeout and max_retries
                EmbeddingFactory._embedding_model_instance = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY,
                                                                              model=model_name, chunk_size=1600)
            elif model_name.startswith("local-"):
                st_model_name = model_name.replace("local-", "")
                local_model_path = os.path.join(LOCAL_EMBEDDING_MODEL_PATH, st_model_name)

                if not os.path.exists(local_model_path) or not os.path.isdir(local_model_path):
                    raise ConfigurationError(
                        f"Local embedding model '{st_model_name}' not found at '{local_model_path}'. "
                        f"Please ensure it's manually downloaded and placed there. "
                        f"Instructions: https://www.sbert.net/docs/usage/semantic_textual_similarity.html#download-models"
                    )
                EmbeddingFactory._embedding_model_instance = SentenceTransformerEmbeddings(model_name=local_model_path)
            elif model_name == "company-embedding-gateway":  # Custom Gateway
                if not GATEWAY_EMBEDDING_URL:
                    raise ConfigurationError("GATEWAY_EMBEDDING_URL not set for company embedding gateway.")
                EmbeddingFactory._embedding_model_instance = CustomGatewayEmbeddings(
                    url=GATEWAY_EMBEDDING_URL,
                    api_key=GATEWAY_EMBEDDING_API_KEY
                )
            else:
                raise ConfigurationError(f"Unsupported or misconfigured embedding model: {model_name}. "
                                         "Use 'text-embedding-ada-002', 'local-MODEL_NAME', or 'company-embedding-gateway'.")

            logger.info(f"Successfully loaded embedding model: {model_name}")
            return EmbeddingFactory._embedding_model_instance
        except Exception as e:
            logger.exception(f"Failed to load embedding model '{model_name}'.")
            raise RAGError(f"Embedding model loading failed: {e}. Check configuration and local path/gateway.")