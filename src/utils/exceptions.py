# src/utils/exceptions.py
class MigrationError(Exception):
    """Base exception for migration-related errors."""
    pass

class LLMError(MigrationError):
    """Exception for errors specific to LLM interactions."""
    pass

class DatabaseConnectionError(MigrationError):
    """Exception for database connection failures."""
    pass

class InvalidInputError(MigrationError):
    """Exception for invalid or missing user inputs."""
    pass

class RAGError(MigrationError):
    """Exception for issues during RAG retrieval."""
    pass

class ConfigurationError(Exception): # Moved from LLMError to be more general
    """Exception for invalid or missing configuration settings."""
    pass

# APIError is removed as per request for no explicit retry/timeout handling in custom code.
# If you want to re-add it for other purposes, keep it.
# class APIError(Exception):
#     """Base exception for API errors that might be transient (e.g., timeouts, 5xx errors)."""
#     pass