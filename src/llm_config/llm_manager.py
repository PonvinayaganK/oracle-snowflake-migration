# src/llm_config/llm_manager.py
import logging
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_community.chat_models import ChatOllama # ADD THIS IMPORT
from src.config import OPENAI_API_KEY, ANTHROPIC_API_KEY, OLLAMA_BASE_URL
from src.utils.exceptions import LLMError

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