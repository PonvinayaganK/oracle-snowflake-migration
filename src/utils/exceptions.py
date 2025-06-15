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


class ConfigurationError:
    pass