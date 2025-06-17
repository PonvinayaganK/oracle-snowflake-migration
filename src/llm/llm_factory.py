# src/llm/llm_factory.py
import os
import logging
import random  # For simulating embeddings
from typing import List, Optional

# IMPORTANT: This import needs to be your company's actual SDK
# Example: from company_sdk import EmbeddingsClient as CompanyEmbeddingsSDK
# Since I don't have it, I'll simulate it.
import requests  # Used for simulating HTTP call to gateway if not using SDK directly

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OpenAIEmbeddings, SentenceTransformerEmbeddings
from langchain_core.embeddings import Embeddings  # Base class for custom embeddings

from src.config import (
    OPENAI_API_KEY, ANTHROPIC_API_KEY, OLLAMA_BASE_URL,
    EMBEDDING_MODEL_NAME,
    LOCAL_EMBEDDING_MODEL_PATH  # GATEWAY_EMBEDDING_URL and GATEWAY_EMBEDDING_API_KEY removed here as per request
)
from src.utils.exceptions import ConfigurationError, LLMError, RAGError

logger = logging.getLogger(__name__)


# --- Simulate Company's Embedding SDK (REPLACE THIS WITH YOUR ACTUAL SDK INTEGRATION) ---
class SimulatedCompanyEmbeddingsSDK:
    """
    This class simulates your company's SDK method for getting embeddings.
    **REPLACE THIS WITH YOUR ACTUAL SDK INTEGRATION.**

    Assumes your SDK provides a method `get_embeddings` that accepts `model_name` and `texts` (list of strings)
    and directly returns `List[List[float]]`.
    It also assumes your SDK handles its own authentication (e.g., API key) and endpoint configuration internally.
    """

    def __init__(self, api_key: Optional[str] = None):  # api_key might still be passed from config if SDK needs it
        self.api_key = api_key  # Store if your real SDK needs it
        # Initialize your actual company SDK client here if needed
        # Example: self.sdk_client = CompanyEmbeddingsSDK(api_key=api_key, other_config='internal')
        logger.info("SimulatedCompanyEmbeddingsSDK initialized. Remember to replace this with your actual SDK.")
        if self.api_key:
            logger.info("Simulated SDK received an API key. Your actual SDK might use this for auth.")

    def get_embeddings(self, model_name: str, texts: List[str]) -> List[List[float]]:
        """
        This method simulates the call to your company's SDK.
        **You would replace the content of this method with your actual SDK call.**
        """
        # --- REPLACE THE FOLLOWING MOCK LOGIC WITH YOUR ACTUAL SDK CALL ---
        logger.info(f"SIMULATING SDK call for embeddings with model '{model_name}' for {len(texts)} texts.")

        try:
            # Example: If your SDK is a simple wrapper around a REST endpoint it manages:
            # headers = {"Content-Type": "application/json"}
            # if self.api_key: # If SDK uses a passed API key for header
            #     headers["Authorization"] = f"Bearer {self.api_key}"
            # response = requests.post("https://your.company.gateway/embed", json={"model": model_name, "inputs": texts}, headers=headers, timeout=60) # SDK might handle its own timeout
            # response.raise_for_status()
            # return response.json().get("embeddings") # Adjust based on your gateway's response

            # Mocking embeddings with random data for demonstration
            # The dimension (e.g., 384) should match your actual embedding model's dimension.
            embedding_dim = 384  # Common for models like MiniLM
            random.seed(sum(ord(c) for c in "".join(texts)) + len(texts) + len(model_name))  # Simple repeatable seed
            embeddings = []
            for _ in texts:
                embeddings.append([random.random() for _ in range(embedding_dim)])
            return embeddings
        except Exception as e:  # Catch any errors from your SDK call
            logger.error(f"Error calling company SDK for embedding: {e}", exc_info=True)
            raise LLMError(f"Company SDK embedding failure: {e}. Check SDK configuration and network access.") from e
        # --- END OF SIMULATION LOGIC ---


# Global instance of the simulated SDK
_simulated_company_sdk_instance: Optional[SimulatedCompanyEmbeddingsSDK] = None


def get_company_sdk_instance(api_key: Optional[str] = None) -> SimulatedCompanyEmbeddingsSDK:
    """
    Retrieves or initializes the global simulated company SDK instance.
    """
    global _simulated_company_sdk_instance
    if _simulated_company_sdk_instance is None:
        _simulated_company_sdk_instance = SimulatedCompanyEmbeddingsSDK(api_key=api_key)
    return _simulated_company_sdk_instance


# --- Chat LLM Factory ---
class LLMFactory:
    """
    Factory class to get different chat LLM instances.
    """
    _chat_llm_instance = {}  # Cache for chat LLMs

    @staticmethod
    def get_llm(model_name: str):
        if model_name in LLMFactory._chat_llm_instance:
            logger.debug(f"Returning cached chat LLM: {model_name}")
            return LLMFactory._chat_llm_instance[model_name]

        logger.info(f"Attempting to load chat LLM: {model_name}")
        try:
            if model_name.startswith("gpt"):
                if not OPENAI_API_KEY:
                    raise ConfigurationError("OpenAI API Key not found. Please set OPENAI_API_KEY in your .env file.")
                # Removed explicit timeout from ChatOpenAI as per request, it will use requests' default.
                llm = ChatOpenAI(model=model_name, api_key=OPENAI_API_KEY, temperature=0.1)
                logger.info(f"Initialized OpenAI model: {model_name}")
            elif model_name.startswith("claude"):
                if not ANTHROPIC_API_KEY:
                    raise ConfigurationError(
                        "Anthropic API Key not found. Please set ANTHROPIC_API_KEY in your .env file.")
                # Removed explicit timeout from ChatAnthropic as per request, it will use requests' default.
                llm = ChatAnthropic(model=model_name, api_key=ANTHROPIC_API_KEY, temperature=0.1)
                logger.info(f"Initialized Anthropic model: {model_name}")
            elif model_name.startswith("ollama-"):
                ollama_model_name = model_name.replace("ollama-", "")
                llm = ChatOllama(model=ollama_model_name, base_url=OLLAMA_BASE_URL, temperature=0.1)
                logger.info(f"Initialized Ollama model: {ollama_model_name} at {OLLAMA_BASE_URL}")
            else:
                raise ConfigurationError(f"Unsupported LLM model: {model_name}")

            LLMFactory._chat_llm_instance[model_name] = llm
            return llm
        except Exception as e:
            logger.exception(f"Failed to initialize chat LLM '{model_name}'.")
            raise LLMError(f"Failed to initialize chat LLM: {e}. Check API keys, Ollama server status, or model name.")


# --- Wrapper for Company's SDK as LangChain Embeddings Interface ---
class CompanySDKEmbeddingsWrapper(Embeddings):
    """
    A wrapper class to make your company's SDK method (which returns embeddings directly)
    conform to LangChain's Embeddings interface.
    It calls the `SimulatedCompanyEmbeddingsSDK` (which you will replace with your real SDK).
    """

    def __init__(self, sdk_model_name: str, api_key: Optional[str] = None):
        self.sdk_model_name = sdk_model_name
        # The SDK instance is obtained here. This is where you would pass any internal configs.
        self.sdk_instance = get_company_sdk_instance(api_key=api_key)  # Pass API key if your SDK needs it
        logger.info(f"CompanySDKEmbeddingsWrapper initialized for model: {sdk_model_name}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents using the company SDK."""
        logger.info(f"Embedding {len(texts)} documents via company SDK: {self.sdk_model_name}")
        try:
            return self.sdk_instance.get_embeddings(self.sdk_model_name, texts)
        except Exception as e:
            logger.error(f"Error calling company SDK for document embeddings: {e}", exc_info=True)
            raise LLMError(f"Company SDK embedding failure for documents: {e}") from e

    def embed_query(self, text: str) -> List[float]:
        """Embed a query text using the company SDK."""
        logger.info(f"Embedding query via company SDK: {self.sdk_model_name} - {text[:50]}...")
        try:
            # Assuming get_embeddings can take a single item list and returns a list of one embedding
            embeddings_list = self.sdk_instance.get_embeddings(self.sdk_model_name, [text])
            if embeddings_list and isinstance(embeddings_list[0], list):
                return embeddings_list[0]
            else:
                raise ValueError("Company SDK did not return expected list of embedding for single query.")
        except Exception as e:
            logger.error(f"Error calling company SDK for query embedding: {e}", exc_info=True)
            raise LLMError(f"Company SDK embedding failure for query: {e}") from e


# --- Main Embedding Model Factory ---
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
            elif model_name == "company-embedding-gateway":  # Custom Gateway via SDK
                # Here, we pass the logical model name to the wrapper.
                # The wrapper's __init__ will then obtain the SDK instance.
                # It is assumed your SDK handles its own internal configuration (URL, API key)
                # or picks it up from global environment variables that are NOT passed via this UI.
                EmbeddingFactory._embedding_model_instance = CompanySDKEmbeddingsWrapper(
                    sdk_model_name=model_name
                )
            else:
                raise ConfigurationError(f"Unsupported or misconfigured embedding model: {model_name}. "
                                         "Use 'text-embedding-ada-002', 'local-MODEL_NAME', or 'company-embedding-gateway'.")

            logger.info(f"Successfully loaded embedding model: {model_name}")
            return EmbeddingFactory._embedding_model_instance
        except Exception as e:
            logger.exception(f"Failed to load embedding model '{model_name}'.")
            raise RAGError(
                f"Embedding model loading failed: {e}. Check configuration, local path, or company SDK setup.")
