"""
utils.py
--------
Shared helper utilities: logging setup and small reusable functions.
"""
import logging
import sys
from src import config


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger that writes to console and a log file."""
    logger = logging.getLogger(name)
    if logger.handlers:
        # Logger already configured (avoid duplicate handlers on re-import)
        return logger

    logger.setLevel(config.LOG_LEVEL)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    try:
        file_handler = logging.FileHandler(config.LOG_FILE)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except OSError:
        # Filesystem may be read-only in some environments; console logging still works
        pass

    return logger


def safe_str(value) -> str:
    """Convert a value to a trimmed string, handling None gracefully."""
    if value is None:
        return ""
    return str(value).strip()
