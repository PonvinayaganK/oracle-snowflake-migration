# src/data_processing/file_utils.py
import streamlit as st
import os
import tempfile
import logging
from src.utils.exceptions import InvalidInputError

logger = logging.getLogger(__name__)

def save_uploaded_file(uploaded_file, subdir: str = "temp") -> str:
    """Saves an uploaded Streamlit file to a temporary directory."""
    temp_dir = os.path.join(tempfile.gettempdir(), "snowflake_migration_uploads", subdir)
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, uploaded_file.name)
    try:
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        logger.info(f"Uploaded file '{uploaded_file.name}' saved to '{file_path}'")
        return file_path
    except Exception as e:
        logger.error(f"Failed to save uploaded file '{uploaded_file.name}': {e}", exc_info=True)
        raise InvalidInputError(f"Could not save uploaded file {uploaded_file.name}: {e}")

def read_file_content(file_path: str) -> str:
    """Reads the content of a text file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        logger.debug(f"Read content from file: {file_path}")
        return content
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}", exc_info=True)
        raise InvalidInputError(f"Required file not found: {file_path}")
    except Exception as e:
        logger.error(f"Failed to read file '{file_path}': {e}", exc_info=True)
        raise InvalidInputError(f"Could not read file {file_path}: {e}")