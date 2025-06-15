# src/rag/document_loader.py
import os
import logging
from typing import List, Dict
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

def load_text_file(filepath: str, metadata: Dict = None) -> Document:
    """Loads content from a single text file into a Document."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        if metadata is None:
            metadata = {}
        metadata['source'] = filepath
        metadata['file_name'] = os.path.basename(filepath)
        logger.debug(f"Loaded document from {filepath}")
        return Document(page_content=content, metadata=metadata)
    except Exception as e:
        logger.error(f"Error loading file {filepath}: {e}", exc_info=True)
        raise

def load_documents_from_directory(directory_path: str, file_extension: str = '.txt') -> List[Document]:
    """Loads all files with a given extension from a directory into Documents."""
    documents = []
    if not os.path.exists(directory_path):
        logger.warning(f"Directory not found: {directory_path}")
        return []

    for filename in os.listdir(directory_path):
        if filename.endswith(file_extension):
            filepath = os.path.join(directory_path, filename)
            try:
                documents.append(load_text_file(filepath))
            except Exception as e:
                logger.warning(f"Skipping file {filepath} due to error: {e}")
    logger.info(f"Loaded {len(documents)} documents from directory: {directory_path}")
    return documents

def prepare_migration_pair_document(oracle_code: str, snowflake_code: str, object_type: str, source_name: str = "feedback_loop_validated") -> Document:
    """Prepares a document for a validated Oracle-Snowflake migration pair for feedback loop."""
    content = f"--- Oracle {object_type} ---\n{oracle_code}\n\n--- Snowflake {object_type} ---\n{snowflake_code}"
    metadata = {
        "source": source_name,
        "object_type": object_type,
        "type": "migration_example",
        "oracle_code_snippet_start": oracle_code[:200].replace('\n', ' '), # Store small snippets for context
        "snowflake_code_snippet_start": snowflake_code[:200].replace('\n', ' ')
    }
    return Document(page_content=content, metadata=metadata)