"""Custom Error Types and Exception Handling for Gridlock 2.0."""

from enum import Enum
from typing import Any, Dict, Optional


class ErrorType(str, Enum):
    """Error type enumeration."""

    VALIDATION_ERROR = "validation_error"
    DATA_PIPELINE_ERROR = "data_pipeline_error"
    MODEL_ERROR = "model_error"
    DATABASE_ERROR = "database_error"
    CACHE_ERROR = "cache_error"
    EXTERNAL_SERVICE_ERROR = "external_service_error"
    TIMEOUT_ERROR = "timeout_error"
    CONFIGURATION_ERROR = "configuration_error"
    UNKNOWN_ERROR = "unknown_error"


class GridlockException(Exception):
    """Base exception for Gridlock 2.0."""

    def __init__(
        self,
        message: str,
        error_type: ErrorType = ErrorType.UNKNOWN_ERROR,
        context: Optional[Dict[str, Any]] = None,
        original_exception: Optional[Exception] = None,
    ):
        """
        Initialize exception.

        Args:
            message: Error message
            error_type: Type of error
            context: Additional context for debugging
            original_exception: Original exception if this is wrapping another
        """
        super().__init__(message)
        self.message = message
        self.error_type = error_type
        self.context = context or {}
        self.original_exception = original_exception

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "message": self.message,
            "error_type": self.error_type.value,
            "context": self.context,
            "original_error": str(self.original_exception) if self.original_exception else None,
        }


class ValidationError(GridlockException):
    """Raised when data validation fails."""

    def __init__(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        original_exception: Optional[Exception] = None,
    ):
        super().__init__(
            message=message,
            error_type=ErrorType.VALIDATION_ERROR,
            context=context,
            original_exception=original_exception,
        )


class DataPipelineError(GridlockException):
    """Raised when data pipeline processing fails."""

    def __init__(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        original_exception: Optional[Exception] = None,
    ):
        super().__init__(
            message=message,
            error_type=ErrorType.DATA_PIPELINE_ERROR,
            context=context,
            original_exception=original_exception,
        )


class ModelError(GridlockException):
    """Raised when model inference or training fails."""

    def __init__(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        original_exception: Optional[Exception] = None,
    ):
        super().__init__(
            message=message,
            error_type=ErrorType.MODEL_ERROR,
            context=context,
            original_exception=original_exception,
        )


class DatabaseError(GridlockException):
    """Raised when database operations fail."""

    def __init__(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        original_exception: Optional[Exception] = None,
    ):
        super().__init__(
            message=message,
            error_type=ErrorType.DATABASE_ERROR,
            context=context,
            original_exception=original_exception,
        )


class CacheError(GridlockException):
    """Raised when cache operations fail."""

    def __init__(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        original_exception: Optional[Exception] = None,
    ):
        super().__init__(
            message=message,
            error_type=ErrorType.CACHE_ERROR,
            context=context,
            original_exception=original_exception,
        )


class ExternalServiceError(GridlockException):
    """Raised when external service calls fail."""

    def __init__(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        original_exception: Optional[Exception] = None,
    ):
        super().__init__(
            message=message,
            error_type=ErrorType.EXTERNAL_SERVICE_ERROR,
            context=context,
            original_exception=original_exception,
        )


class TimeoutError(GridlockException):
    """Raised when operations exceed timeout."""

    def __init__(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        original_exception: Optional[Exception] = None,
    ):
        super().__init__(
            message=message,
            error_type=ErrorType.TIMEOUT_ERROR,
            context=context,
            original_exception=original_exception,
        )


class ConfigurationError(GridlockException):
    """Raised when configuration is invalid."""

    def __init__(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        original_exception: Optional[Exception] = None,
    ):
        super().__init__(
            message=message,
            error_type=ErrorType.CONFIGURATION_ERROR,
            context=context,
            original_exception=original_exception,
        )
