# src/utils/logger.py
import logging
import os
from src.config import LOG_FILE_PATH, LOG_LEVEL

def setup_logging():
    """Configures project-wide logging."""
    log_dir = os.path.dirname(LOG_FILE_PATH)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(LOG_FILE_PATH),
            logging.StreamHandler()
        ]
    )
    logging.getLogger(__name__).info(f"Logging initialized. Log level: {LOG_LEVEL}")