"""
UTF-8 Logging Utility for Windows Compatibility

This module provides UTF-8 compatible logging handlers for Windows consoles.
Import this module at the top of any file that uses logging with emoji characters.

Usage:
    from utf8_logging import get_logger
    
    logger = get_logger("MyLogger")
"""

import logging
import sys


class UTF8StreamHandler(logging.StreamHandler):
    """
    A StreamHandler that forces UTF-8 encoding for console output.
    This prevents UnicodeEncodeError on Windows consoles with cp1252 encoding.
    """
    def emit(self, record):
        try:
            msg = self.format(record)
            # Encode to UTF-8 bytes, then decode with 'replace' for safety
            utf8_msg = msg.encode('utf-8', errors='replace').decode('utf-8', errors='replace')
            stream = self.stream
            stream.write(utf8_msg)
            stream.write(self.terminator)
            self.flush()
        except Exception:
            self.handleError(record)


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Get a logger with UTF-8 encoding support.
    
    Args:
        name: Logger name (usually __name__ or module name)
        level: Logging level (default: INFO)
    
    Returns:
        Logger instance with UTF-8 handler
    """
    logger = logging.getLogger(name)
    
    # Only add handler if none exists
    if not logger.handlers:
        ch = UTF8StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        logger.setLevel(level)
        logger.propagate = False
    
    return logger


def get_logger_compact(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Get a logger with UTF-8 encoding support and compact format.
    
    Args:
        name: Logger name
        level: Logging level (default: INFO)
    
    Returns:
        Logger instance with UTF-8 handler and compact format
    """
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        ch = UTF8StreamHandler()
        formatter = logging.Formatter('[%(asctime)s] [%(name)s] %(levelname)s: %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        logger.setLevel(level)
        logger.propagate = False
    
    return logger


def configure_root_logger(level: int = logging.INFO):
    """
    Configure the root logger with UTF-8 encoding.
    Call this once at application startup.
    
    Args:
        level: Logging level (default: INFO)
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add UTF-8 handler
    utf8_handler = UTF8StreamHandler(sys.stdout)
    utf8_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    root_logger.addHandler(utf8_handler)
