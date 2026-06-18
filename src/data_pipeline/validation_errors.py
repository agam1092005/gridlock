"""
Error aggregation and management for data validation.

Collects validation errors from Pydantic models and provides detailed
error messages for API responses and logging.
"""

from typing import Dict, List, Optional, Any
from pydantic import ValidationError as PydanticValidationError

from src.data_pipeline.models import ValidationError


class ErrorAggregator:
    """Aggregates and organizes validation errors."""

    def __init__(self):
        """Initialize error aggregator."""
        self.errors: List[ValidationError] = []

    def add_error(
        self, field: str, message: str, error_code: str, value: Optional[str] = None
    ) -> None:
        """
        Add a validation error.

        Args:
            field: Field that failed validation
            message: Human-readable error message
            error_code: Machine-readable error code
            value: The value that caused the error (sanitized)
        """
        error = ValidationError(field=field, message=message, error_code=error_code, value=value)
        self.errors.append(error)

    def add_errors(self, errors: List[ValidationError]) -> None:
        """
        Add multiple validation errors.

        Args:
            errors: List of ValidationError instances
        """
        self.errors.extend(errors)

    def add_pydantic_errors(self, pydantic_error: PydanticValidationError) -> None:
        """
        Convert and add Pydantic validation errors.

        Args:
            pydantic_error: Pydantic ValidationError instance
        """
        for error in pydantic_error.errors():
            field = ".".join(str(loc) for loc in error["loc"])
            message = error["msg"]
            error_type = error["type"]

            # Extract the value if available
            value = None
            if "ctx" in error and "error" in error["ctx"]:
                value = str(error["ctx"]["error"])[:50]  # Limit to 50 chars

            self.add_error(field=field, message=message, error_code=error_type, value=value)

    def has_errors(self) -> bool:
        """Check if any errors have been collected."""
        return len(self.errors) > 0

    def get_errors(self) -> List[ValidationError]:
        """Get all collected errors."""
        return self.errors.copy()

    def get_errors_by_field(self) -> Dict[str, List[ValidationError]]:
        """Get errors grouped by field."""
        errors_by_field: Dict[str, List[ValidationError]] = {}

        for error in self.errors:
            if error.field not in errors_by_field:
                errors_by_field[error.field] = []
            errors_by_field[error.field].append(error)

        return errors_by_field

    def get_error_codes(self) -> List[str]:
        """Get all unique error codes."""
        codes = set()
        for error in self.errors:
            codes.add(error.error_code)
        return sorted(list(codes))

    def get_field_names_with_errors(self) -> List[str]:
        """Get list of fields that have validation errors."""
        fields = set()
        for error in self.errors:
            fields.add(error.field)
        return sorted(list(fields))

    def get_detailed_message(self) -> str:
        """
        Get a formatted, detailed error message.

        Returns:
            Multi-line error message suitable for logging
        """
        if not self.has_errors():
            return "No errors"

        lines = [f"Validation failed with {len(self.errors)} error(s):"]

        for i, error in enumerate(self.errors, 1):
            line = f"  {i}. [{error.field}] {error.message}"
            if error.value:
                line += f" (value: {error.value})"
            lines.append(line)

        return "\n".join(lines)

    def get_summary_message(self) -> str:
        """
        Get a brief summary error message.

        Returns:
            Single-line error message suitable for API responses
        """
        if not self.has_errors():
            return "Validation passed"

        if len(self.errors) == 1:
            error = self.errors[0]
            return f"Validation failed: {error.field} - {error.message}"

        error_counts = {}
        for error in self.errors:
            code = error.error_code
            error_counts[code] = error_counts.get(code, 0) + 1

        codes_str = ", ".join(f"{code} ({count})" for code, count in error_counts.items())
        return f"Validation failed with {len(self.errors)} error(s): {codes_str}"

    def clear(self) -> None:
        """Clear all collected errors."""
        self.errors = []

    def to_dict(self) -> Dict[str, Any]:
        """Convert aggregator state to dictionary."""
        return {
            "has_errors": self.has_errors(),
            "error_count": len(self.errors),
            "errors": [
                {
                    "field": error.field,
                    "message": error.message,
                    "error_code": error.error_code,
                    "value": error.value,
                }
                for error in self.errors
            ],
            "fields_with_errors": self.get_field_names_with_errors(),
            "error_codes": self.get_error_codes(),
            "summary": self.get_summary_message(),
        }


class ValidationErrorContext:
    """Context manager for error aggregation during validation."""

    def __init__(self):
        """Initialize error context."""
        self.aggregator = ErrorAggregator()

    def __enter__(self) -> ErrorAggregator:
        """Enter context and return aggregator."""
        return self.aggregator

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context."""
        # Error handling is done by caller
        pass

    def has_errors(self) -> bool:
        """Check if any errors were collected."""
        return self.aggregator.has_errors()

    def get_errors(self) -> List[ValidationError]:
        """Get collected errors."""
        return self.aggregator.get_errors()

    def get_summary(self) -> str:
        """Get summary message."""
        return self.aggregator.get_summary_message()


class ValidationErrorFormatter:
    """Formats validation errors for different output contexts."""

    @staticmethod
    def format_for_api(aggregator: ErrorAggregator) -> Dict[str, Any]:
        """
        Format errors for API response.

        Args:
            aggregator: ErrorAggregator instance

        Returns:
            Dictionary suitable for JSON API response
        """
        return {
            "error": "validation_error",
            "message": aggregator.get_summary_message(),
            "details": [
                {"field": error.field, "message": error.message, "code": error.error_code}
                for error in aggregator.get_errors()
            ],
            "error_count": len(aggregator.get_errors()),
            "affected_fields": aggregator.get_field_names_with_errors(),
        }

    @staticmethod
    def format_for_logging(
        aggregator: ErrorAggregator, incident_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Format errors for logging.

        Args:
            aggregator: ErrorAggregator instance
            incident_id: Associated incident ID if available

        Returns:
            Dictionary suitable for JSON logging
        """
        return {
            "event": "validation_failed",
            "incident_id": incident_id,
            "error_count": len(aggregator.get_errors()),
            "errors": [
                {"field": error.field, "code": error.error_code, "message": error.message}
                for error in aggregator.get_errors()
            ],
            "affected_fields": aggregator.get_field_names_with_errors(),
            "error_codes": aggregator.get_error_codes(),
        }

    @staticmethod
    def format_for_console(aggregator: ErrorAggregator) -> str:
        """
        Format errors for console output.

        Args:
            aggregator: ErrorAggregator instance

        Returns:
            Human-readable error string
        """
        return aggregator.get_detailed_message()
