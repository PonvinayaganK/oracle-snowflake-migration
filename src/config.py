# src/config.py
import os
from dotenv import load_dotenv

load_dotenv() # Load environment variables from .env file

# --- LLM Configuration ---
# List of supported LLM models that can be selected in the UI.
# Ensure you have the necessary API keys configured for your chosen models,
# or have local Ollama models pulled and running.
SUPPORTED_LLM_MODELS = [
    "gpt-4o",
    "gpt-3.5-turbo",
    "claude-3-opus-20240229",
    "claude-3-sonnet-20240229",
    "ollama-llama2",     # ADD THESE FOR OLLAMA
    "ollama-mistral",    # You can add more specific Ollama models here
    "ollama-codellama",
    "ollama-qwen2.5",
    "ollama-codellama:13b"
]

# The default LLM model to pre-select in the UI.
DEFAULT_LLM_MODEL = "ollama-codellama" # You can change this to "ollama-llama2" if you prefer local by default

# --- View Decomposition Threshold (NEW) ---
# Oracle View DDL line count threshold above which decomposition is attempted.
VIEW_DECOMPOSITION_THRESHOLD = int(os.getenv("VIEW_DECOMPOSITION_THRESHOLD", 100)) # e.g., 100 lines for testing, 1000 for real

# --- Optimization Cycles Configuration ---
# Maximum number of times the LLM will attempt to optimize the generated code
# based on its reflection. Set to 1 for initial generation + one reflection pass.
MAX_OPTIMIZATION_CYCLES = int(os.getenv("MAX_OPTIMIZATION_CYCLES", 3))

# --- Object Type Configuration (NEW) ---
# Supported types of database objects for migration
SUPPORTED_OBJECT_TYPES = ["Procedure", "View"]
# Default object type to select in the UI
DEFAULT_OBJECT_TYPE = "Procedure"

# API Keys (get from environment variables)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Ollama Base URL
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434") # Default Ollama URL

# --- Database Connection Configuration (Optional) ---
# These are placeholders. Uncomment and fill in if you intend to use
# direct database connections for RAG or Oracle DDL extraction.
# Ensure these are also set in your .env file.

# Snowflake Connection Details
SNOWFLAKE_ACCOUNT = os.getenv("SNOWFLAKE_ACCOUNT")
SNOWFLAKE_USER = os.getenv("SNOWFLAKE_USER")
SNOWFLAKE_PASSWORD = os.getenv("SNOWFLAKE_PASSWORD")
SNOWFLAKE_ROLE = os.getenv("SNOWFLAKE_ROLE")
SNOWFLAKE_WAREHOUSE = os.getenv("SNOWFLAKE_WAREHOUSE")
SNOWFLAKE_DATABASE = os.getenv("SNOWFLAKE_DATABASE")
SNOWFLAKE_SCHEMA = os.getenv("SNOWFLAKE_SCHEMA")

# Oracle Connection Details
# ORACLE_USER = os.getenv("ORACLE_USER")
# ORACLE_PASSWORD = os.getenv("ORACLE_PASSWORD")
# ORACLE_HOST = os.getenv("ORACLE_HOST")
# ORACLE_PORT = os.getenv("ORACLE_PORT", "1521") # Default Oracle port is 1521
# ORACLE_SERVICE_NAME = os.getenv("ORACLE_SERVICE_NAME") # Use SERVICE_NAME or SID

# --- Logging Configuration ---
# Path for the log file. Logs will be stored in the 'logs' directory relative to the project root.
LOG_FILE_PATH = "logs/migration.log"
# Logging level (e.g., "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL").
# Can be controlled via .env for easier adjustment without code changes.
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# --- Proxy Configuration (Optional) ---
# If your environment requires proxy settings for outbound internet connections (e.g., to LLM APIs).
# These should also be set in your .env file.
PROXY_CONFIG = {
    "http_proxy": os.getenv("HTTP_PROXY"),
    "https_proxy": os.getenv("HTTPS_PROXY"),
}

# --- FastAPI API Configuration (NEW) ---
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8001))


# --- Embedding Model Configuration ---
# Supported embedding models for RAG. Use "local-" prefix for models loaded from disk.
SUPPORTED_EMBEDDING_MODELS = [
    "text-embedding-ada-002",         # OpenAI (requires API key)
    "local-all-MiniLM-L6-v2",         # Sentence Transformers (load from LOCAL_EMBEDDING_MODEL_PATH)
    "local-BAAI/bge-small-en-v1.5",   # Sentence Transformers (load from LOCAL_EMBEDDING_MODEL_PATH)
    "company-embedding-gateway"       # Your custom company LLM gateway embedding model
]
DEFAULT_EMBEDDING_MODEL = "local-all-MiniLM-L6-v2" # Recommended default for local execution
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", DEFAULT_EMBEDDING_MODEL)
LOCAL_EMBEDDING_MODEL_PATH = os.getenv("LOCAL_EMBEDDING_MODEL_PATH", "data/embedding_models") # Path to pre-downloaded models

# Company Gateway Embedding Model
GATEWAY_EMBEDDING_URL = os.getenv("GATEWAY_EMBEDDING_URL", "http://localhost:your_gateway_port/embed")
GATEWAY_EMBEDDING_API_KEY = os.getenv("GATEWAY_EMBEDDING_API_KEY") # Or other auth token


# --- ChromaDB Configuration ---
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "data/vector_store")
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "snowflake_migration_examples")

# --- RAG Retrieval Configuration ---
RAG_TOP_K_SEMANTIC = int(os.getenv("RAG_TOP_K_SEMANTIC", 5)) # Top K for semantic search
RAG_TOP_K_KEYWORD = int(os.getenv("RAG_TOP_K_KEYWORD", 5))   # Top K for keyword search






