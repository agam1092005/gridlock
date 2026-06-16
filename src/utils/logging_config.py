"""
Structured Logging Configuration for Gridlock 2.0

Provides JSON-formatted structured logging with timestamps, severity levels,
component names, and contextual information for debugging and monitoring.
"""

import json
import logging
import logging.handlers
import os
import sys
from enum import Enum
from typing import Any, Dict, Optional

import structlog
from pythonjsonlogger import jsonlogger


class LogLevel(str, Enum):
    """Log level enumeration."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class CustomJsonFormatter(jsonlogger.JsonFormatter):
    """Custom JSON formatter that adds structured context."""

    def add_fields(self, log_record: Dict[str, Any], record: logging.LogRecord, message_dict: Dict[str, Any]) -> None:
        """Add custom fields to log record."""
        super().add_fields(log_record, record, message_dict)
        
        # Add timestamp in ISO format
        log_record['timestamp'] = self.formatTime(record, self.datefmt)
        
        # Add severity level
        log_record['severity'] = record.levelname.upper()
        
        # Add component name
        log_record['component'] = record.name
        
        # Add line number and function name for debugging
        log_record['location'] = {
            'file': record.filename,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        # Add process and thread information
        log_record['process_id'] = record.process
        log_record['thread_id'] = record.thread
        log_record['thread_name'] = record.threadName


# Global logger instance
_logger_instance: Optional[logging.Logger] = None


def configure_logging(
    level: LogLevel = LogLevel.INFO,
    log_dir: str = './logs',
    json_format: bool = True,
    console_output: bool = True,
    file_output: bool = True,
) -> logging.Logger:
    """
    Configure structured logging for the application.
    
    Args:
        level: Logging level
        log_dir: Directory for log files
        json_format: Use JSON formatted logs
        console_output: Enable console logging
        file_output: Enable file logging
    
    Returns:
        Configured logger instance
    """
    global _logger_instance
    
    # Create log directory if it doesn't exist
    if file_output:
        os.makedirs(log_dir, exist_ok=True)
    
    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.value))
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.value))
        
        if json_format:
            formatter = CustomJsonFormatter('%(message)s %(levelname)s %(name)s')
        else:
            formatter = logging.Formatter(
                fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # File handler
    if file_output:
        file_path = os.path.join(log_dir, 'gridlock.log')
        file_handler = logging.handlers.RotatingFileHandler(
            file_path,
            maxBytes=100 * 1024 * 1024,  # 100 MB
            backupCount=10
        )
        file_handler.setLevel(getattr(logging, level.value))
        
        if json_format:
            formatter = CustomJsonFormatter('%(message)s %(levelname)s %(name)s')
        else:
            formatter = logging.Formatter(
                fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt='iso'),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    _logger_instance = root_logger
    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a specific component.
    
    Args:
        name: Component name (e.g., 'data_pipeline', 'module_a', 'api')
    
    Returns:
        Logger instance for the component
    """
    if _logger_instance is None:
        configure_logging()
    
    return logging.getLogger(name)


class LogContext:
    """Context manager for structured logging with additional context."""
    
    def __init__(self, logger: logging.Logger, **context):
        """
        Initialize context manager.
        
        Args:
            logger: Logger instance
            **context: Additional context to include in logs
        """
        self.logger = logger
        self.context = context
    
    def __enter__(self):
        """Enter context."""
        if self.context:
            self.logger.info("Starting operation", extra={'context': self.context})
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context."""
        if exc_type is not None:
            self.logger.error(
                f"Operation failed: {exc_type.__name__}",
                exc_info=(exc_type, exc_val, exc_tb),
                extra={'context': self.context}
            )
        else:
            self.logger.info("Operation completed", extra={'context': self.context})


# Example usage:
if __name__ == '__main__':
    # Configure logging
    configure_logging(level=LogLevel.DEBUG, json_format=True)
    
    # Get loggers for different components
    data_logger = get_logger('data_pipeline')
    api_logger = get_logger('api')
    
    # Log messages
    data_logger.info("Data pipeline started", extra={'batch_size': 32})
    api_logger.warning("High latency detected", extra={'latency_ms': 450})
    
    try:
        raise ValueError("Example error")
    except ValueError:
        data_logger.exception("An error occurred during processing")
