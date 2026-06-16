"""Tests for error handling."""

import pytest

from src.utils import (
    CacheError,
    ConfigurationError,
    DataPipelineError,
    DatabaseError,
    ErrorType,
    ExternalServiceError,
    GridlockException,
    ModelError,
    TimeoutError,
    ValidationError,
)


class TestErrorTypes:
    """Test custom error types."""
    
    def test_gridlock_exception_basic(self):
        """Test basic GridlockException."""
        exc = GridlockException("Test error")
        
        assert str(exc) == "Test error"
        assert exc.message == "Test error"
        assert exc.error_type == ErrorType.UNKNOWN_ERROR
        assert exc.context == {}
    
    def test_gridlock_exception_with_context(self):
        """Test GridlockException with context."""
        context = {'incident_id': 'uuid-123', 'component': 'data_pipeline'}
        exc = GridlockException("Pipeline error", context=context)
        
        assert exc.context == context
    
    def test_gridlock_exception_with_original_exception(self):
        """Test GridlockException wrapping another exception."""
        original = ValueError("Original error")
        exc = GridlockException(
            "Wrapped error",
            original_exception=original,
        )
        
        assert exc.original_exception is original
    
    def test_gridlock_exception_to_dict(self):
        """Test converting exception to dictionary."""
        context = {'incident_id': 'uuid-123'}
        original = ValueError("Original")
        exc = GridlockException(
            "Test error",
            error_type=ErrorType.DATABASE_ERROR,
            context=context,
            original_exception=original,
        )
        
        exc_dict = exc.to_dict()
        
        assert exc_dict['message'] == "Test error"
        assert exc_dict['error_type'] == 'database_error'
        assert exc_dict['context'] == context
        assert 'original_error' in exc_dict
    
    def test_validation_error(self):
        """Test ValidationError."""
        exc = ValidationError("Validation failed")
        
        assert exc.error_type == ErrorType.VALIDATION_ERROR
        assert exc.message == "Validation failed"
    
    def test_data_pipeline_error(self):
        """Test DataPipelineError."""
        exc = DataPipelineError("Pipeline processing failed")
        
        assert exc.error_type == ErrorType.DATA_PIPELINE_ERROR
    
    def test_model_error(self):
        """Test ModelError."""
        exc = ModelError("Model inference failed")
        
        assert exc.error_type == ErrorType.MODEL_ERROR
    
    def test_database_error(self):
        """Test DatabaseError."""
        exc = DatabaseError("Database connection failed")
        
        assert exc.error_type == ErrorType.DATABASE_ERROR
    
    def test_cache_error(self):
        """Test CacheError."""
        exc = CacheError("Cache operation failed")
        
        assert exc.error_type == ErrorType.CACHE_ERROR
    
    def test_external_service_error(self):
        """Test ExternalServiceError."""
        exc = ExternalServiceError("API call failed")
        
        assert exc.error_type == ErrorType.EXTERNAL_SERVICE_ERROR
    
    def test_timeout_error(self):
        """Test TimeoutError."""
        exc = TimeoutError("Operation timed out")
        
        assert exc.error_type == ErrorType.TIMEOUT_ERROR
    
    def test_configuration_error(self):
        """Test ConfigurationError."""
        exc = ConfigurationError("Config validation failed")
        
        assert exc.error_type == ErrorType.CONFIGURATION_ERROR
    
    def test_error_inheritance(self):
        """Test that specific errors inherit from GridlockException."""
        errors = [
            ValidationError("Test"),
            DataPipelineError("Test"),
            ModelError("Test"),
            DatabaseError("Test"),
            CacheError("Test"),
            ExternalServiceError("Test"),
            TimeoutError("Test"),
            ConfigurationError("Test"),
        ]
        
        for exc in errors:
            assert isinstance(exc, GridlockException)
    
    def test_error_can_be_raised_and_caught(self):
        """Test that custom errors can be raised and caught."""
        with pytest.raises(ValidationError) as exc_info:
            raise ValidationError("Test validation error")
        
        assert "Test validation error" in str(exc_info.value)
