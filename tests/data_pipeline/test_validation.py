"""
Comprehensive tests for data validation layer.

Tests cover:
- Pydantic model validation with required/optional fields
- Timestamp format and future date validation
- Location bounds checking
- Description length and quality validation
- Duplicate incident detection
- Batch validation
- Validation error aggregation
- Audit logging
"""

import pytest
from datetime import datetime, timedelta
from uuid import uuid4

from pydantic import ValidationError as PydanticValidationError

from src.data_pipeline.models import (
    IncidentInput,
    IncidentType,
    LocationData,
    WeatherData,
    ValidationResult,
    ValidationError,
    AuditLogEntry,
    ValidationStats,
)
from src.data_pipeline.validators import IncidentValidator, DuplicateDetector
from src.data_pipeline.validation_errors import (
    ErrorAggregator,
    ValidationErrorContext,
    ValidationErrorFormatter,
)
from src.data_pipeline.audit import AuditLogger, AuditContext, get_audit_logger


class TestLocationDataModel:
    """Test LocationData Pydantic model."""

    def test_valid_location(self):
        """Test valid location data."""
        location = LocationData(latitude=-33.8688, longitude=151.2093)
        assert location.latitude == -33.8688
        assert location.longitude == 151.2093

    def test_latitude_out_of_bounds_low(self):
        """Test latitude below minimum."""
        with pytest.raises(PydanticValidationError) as exc_info:
            LocationData(latitude=-91, longitude=0)
        assert "latitude" in str(exc_info.value).lower()

    def test_latitude_out_of_bounds_high(self):
        """Test latitude above maximum."""
        with pytest.raises(PydanticValidationError) as exc_info:
            LocationData(latitude=91, longitude=0)
        assert "latitude" in str(exc_info.value).lower()

    def test_longitude_out_of_bounds_low(self):
        """Test longitude below minimum."""
        with pytest.raises(PydanticValidationError) as exc_info:
            LocationData(latitude=0, longitude=-181)
        assert "longitude" in str(exc_info.value).lower()

    def test_longitude_out_of_bounds_high(self):
        """Test longitude above maximum."""
        with pytest.raises(PydanticValidationError) as exc_info:
            LocationData(latitude=0, longitude=181)
        assert "longitude" in str(exc_info.value).lower()

    def test_boundary_values(self):
        """Test boundary values for lat/lon."""
        location1 = LocationData(latitude=-90, longitude=-180)
        assert location1.latitude == -90
        assert location1.longitude == -180

        location2 = LocationData(latitude=90, longitude=180)
        assert location2.latitude == 90
        assert location2.longitude == 180


class TestWeatherDataModel:
    """Test WeatherData Pydantic model."""

    def test_valid_weather_data(self):
        """Test valid weather data."""
        weather = WeatherData(temperature=22.5, precipitation=0.0, wind_speed=5.0, humidity=65.0)
        assert weather.temperature == 22.5
        assert weather.humidity == 65.0

    def test_all_fields_optional(self):
        """Test that all weather fields are optional."""
        weather = WeatherData()
        assert weather.temperature is None
        assert weather.precipitation is None

    def test_precipitation_negative_fails(self):
        """Test that negative precipitation fails."""
        with pytest.raises(PydanticValidationError):
            WeatherData(precipitation=-1.0)

    def test_humidity_out_of_range(self):
        """Test humidity constraints."""
        with pytest.raises(PydanticValidationError):
            WeatherData(humidity=101)


class TestIncidentInputModel:
    """Test IncidentInput Pydantic model."""

    def test_valid_incident(self):
        """Test valid incident data."""
        incident_data = {
            "location": {"latitude": -33.8688, "longitude": 151.2093},
            "timestamp": datetime.utcnow() - timedelta(minutes=5),
            "description": "Multi-vehicle collision on M1 northbound",
            "incident_type": IncidentType.ACCIDENT,
        }
        incident = IncidentInput(**incident_data)
        assert incident.location.latitude == -33.8688
        assert incident.incident_type == IncidentType.ACCIDENT

    def test_missing_required_fields(self):
        """Test that required fields are enforced."""
        with pytest.raises(PydanticValidationError):
            IncidentInput(
                location={"latitude": -33.8688, "longitude": 151.2093},
                # Missing timestamp and description
            )

    def test_description_too_short(self):
        """Test description minimum length."""
        incident_data = {
            "location": {"latitude": -33.8688, "longitude": 151.2093},
            "timestamp": datetime.utcnow() - timedelta(minutes=5),
            "description": "short",  # Less than 10 characters
        }
        with pytest.raises(PydanticValidationError):
            IncidentInput(**incident_data)

    def test_description_too_long(self):
        """Test description maximum length."""
        incident_data = {
            "location": {"latitude": -33.8688, "longitude": 151.2093},
            "timestamp": datetime.utcnow() - timedelta(minutes=5),
            "description": "x" * 5001,  # More than 5000 characters
        }
        with pytest.raises(PydanticValidationError):
            IncidentInput(**incident_data)

    def test_timestamp_in_future_fails(self):
        """Test that future timestamps fail validation."""
        incident_data = {
            "location": {"latitude": -33.8688, "longitude": 151.2093},
            "timestamp": datetime.utcnow() + timedelta(hours=1),  # Future
            "description": "Valid incident description",
        }
        with pytest.raises(PydanticValidationError):
            IncidentInput(**incident_data)

    def test_end_datetime_before_timestamp_fails(self):
        """Test that end_datetime before timestamp fails."""
        now = datetime.utcnow()
        incident_data = {
            "location": {"latitude": -33.8688, "longitude": 151.2093},
            "timestamp": now - timedelta(minutes=10),
            "description": "Valid incident description",
            "end_datetime": now - timedelta(minutes=15),  # Before start
            "is_ongoing": False,
        }
        with pytest.raises(PydanticValidationError):
            IncidentInput(**incident_data)

    def test_end_datetime_in_future_fails(self):
        """Test that future end_datetime fails."""
        incident_data = {
            "location": {"latitude": -33.8688, "longitude": 151.2093},
            "timestamp": datetime.utcnow() - timedelta(minutes=10),
            "description": "Valid incident description",
            "end_datetime": datetime.utcnow() + timedelta(hours=1),  # Future
            "is_ongoing": False,
        }
        with pytest.raises(PydanticValidationError):
            IncidentInput(**incident_data)

    def test_is_ongoing_with_end_datetime_fails(self):
        """Test that is_ongoing=True with end_datetime fails."""
        now = datetime.utcnow()
        incident_data = {
            "location": {"latitude": -33.8688, "longitude": 151.2093},
            "timestamp": now - timedelta(minutes=10),
            "description": "Valid incident description",
            "end_datetime": now - timedelta(minutes=5),
            "is_ongoing": True,  # Inconsistent with end_datetime
        }
        with pytest.raises(PydanticValidationError):
            IncidentInput(**incident_data)

    def test_is_ongoing_false_without_end_datetime_fails(self):
        """Test that is_ongoing=False requires end_datetime."""
        incident_data = {
            "location": {"latitude": -33.8688, "longitude": 151.2093},
            "timestamp": datetime.utcnow() - timedelta(minutes=10),
            "description": "Valid incident description",
            "is_ongoing": False
            # Missing end_datetime
        }
        with pytest.raises(PydanticValidationError):
            IncidentInput(**incident_data)

    def test_severity_initial_bounds(self):
        """Test severity_initial constraints."""
        # Valid: 0-100
        incident_data = {
            "location": {"latitude": -33.8688, "longitude": 151.2093},
            "timestamp": datetime.utcnow() - timedelta(minutes=5),
            "description": "Valid incident description",
            "severity_initial": 75,
        }
        incident = IncidentInput(**incident_data)
        assert incident.severity_initial == 75

        # Invalid: negative
        with pytest.raises(PydanticValidationError):
            incident_data["severity_initial"] = -1
            IncidentInput(**incident_data)

        # Invalid: > 100
        with pytest.raises(PydanticValidationError):
            incident_data["severity_initial"] = 101
            IncidentInput(**incident_data)


class TestDuplicateDetector:
    """Test DuplicateDetector functionality."""

    def test_no_duplicates_different_type(self):
        """Test that incidents of different types are not duplicates."""
        detector = DuplicateDetector()
        now = datetime.utcnow()

        detector.add_incident(
            "incident_1",
            {"latitude": -33.8688, "longitude": 151.2093},
            now - timedelta(minutes=5),
            "accident",
        )

        duplicates = detector.find_duplicates(
            "incident_2",
            {"latitude": -33.8688, "longitude": 151.2093},
            now,
            "congestion",  # Different type
        )

        assert len(duplicates) == 0

    def test_no_duplicates_outside_time_window(self):
        """Test that incidents outside time window are not duplicates."""
        detector = DuplicateDetector(time_window_minutes=30)
        now = datetime.utcnow()

        detector.add_incident(
            "incident_1",
            {"latitude": -33.8688, "longitude": 151.2093},
            now - timedelta(hours=2),  # Outside 30-min window
            "accident",
        )

        duplicates = detector.find_duplicates(
            "incident_2", {"latitude": -33.8688, "longitude": 151.2093}, now, "accident"
        )

        assert len(duplicates) == 0

    def test_no_duplicates_outside_distance(self):
        """Test that incidents outside distance threshold are not duplicates."""
        detector = DuplicateDetector(distance_threshold_km=2.0)
        now = datetime.utcnow()

        detector.add_incident(
            "incident_1",
            {"latitude": -33.8688, "longitude": 151.2093},
            now - timedelta(minutes=5),
            "accident",
        )

        # Approximately 5 km away
        duplicates = detector.find_duplicates(
            "incident_2", {"latitude": -33.8688 + 0.05, "longitude": 151.2093}, now, "accident"
        )

        assert len(duplicates) == 0

    def test_detects_duplicate_same_location_time_type(self):
        """Test that duplicate is detected with same location, time, type."""
        detector = DuplicateDetector(time_window_minutes=30, distance_threshold_km=2.0)
        now = datetime.utcnow()

        detector.add_incident(
            "incident_1",
            {"latitude": -33.8688, "longitude": 151.2093},
            now - timedelta(minutes=5),
            "accident",
        )

        duplicates = detector.find_duplicates(
            "incident_2",
            {"latitude": -33.8688, "longitude": 151.2093},  # Same location
            now - timedelta(minutes=3),  # Within time window
            "accident",  # Same type
        )

        assert "incident_1" in duplicates

    def test_multiple_duplicates(self):
        """Test detection of multiple duplicates."""
        detector = DuplicateDetector()
        now = datetime.utcnow()

        # Add three related incidents
        for i in range(3):
            detector.add_incident(
                f"incident_{i}",
                {"latitude": -33.8688, "longitude": 151.2093},
                now - timedelta(minutes=10 - i),
                "accident",
            )

        duplicates = detector.find_duplicates(
            "incident_new", {"latitude": -33.8688, "longitude": 151.2093}, now, "accident"
        )

        assert len(duplicates) == 3


class TestIncidentValidator:
    """Test IncidentValidator functionality."""

    @pytest.fixture
    def validator(self):
        """Provide a fresh validator for each test."""
        return IncidentValidator()

    @pytest.fixture
    def valid_incident_data(self):
        """Provide valid incident data."""
        return {
            "incident_id": str(uuid4()),
            "location": {"latitude": -33.8688, "longitude": 151.2093},
            "timestamp": (datetime.utcnow() - timedelta(minutes=5)).isoformat(),
            "description": "Multi-vehicle collision on M1 northbound, 2 lanes blocked",
            "incident_type": "accident",
            "severity_initial": 75,
            "weather": {"temperature": 22.5, "precipitation": 0.0, "wind_speed": 5.0},
            "is_ongoing": True,
        }

    def test_validate_valid_incident(self, validator, valid_incident_data):
        """Test validation of completely valid incident."""
        result = validator.validate_single(valid_incident_data)
        assert result.valid is True
        assert len(result.errors) == 0

    def test_validate_missing_required_field(self, validator, valid_incident_data):
        """Test validation fails when required field missing."""
        del valid_incident_data["description"]
        result = validator.validate_single(valid_incident_data)
        assert result.valid is False
        assert len(result.errors) > 0

    def test_validate_invalid_type(self, validator, valid_incident_data):
        """Test validation fails for wrong type."""
        valid_incident_data["severity_initial"] = "not_a_number"
        result = validator.validate_single(valid_incident_data)
        assert result.valid is False

    def test_validate_timestamp_in_future(self, validator, valid_incident_data):
        """Test validation fails for future timestamp."""
        valid_incident_data["timestamp"] = (datetime.utcnow() + timedelta(hours=1)).isoformat()
        result = validator.validate_single(valid_incident_data)
        assert result.valid is False

    def test_validate_location_out_of_bounds(self, validator, valid_incident_data):
        """Test validation fails for out-of-bounds location."""
        valid_incident_data["location"] = {"latitude": 100, "longitude": 200}
        result = validator.validate_single(valid_incident_data)
        assert result.valid is False

    def test_validate_description_too_short(self, validator, valid_incident_data):
        """Test validation fails for short description."""
        valid_incident_data["description"] = "short"
        result = validator.validate_single(valid_incident_data)
        assert result.valid is False

    def test_validate_description_noise(self, validator, valid_incident_data):
        """Test validation fails for noisy description."""
        valid_incident_data["description"] = "aaaaaaaaaa"  # All same character
        result = validator.validate_single(valid_incident_data)
        assert result.valid is False

    def test_validate_batch(self, validator, valid_incident_data):
        """Test batch validation."""
        incidents = []
        for i in range(5):
            incident = valid_incident_data.copy()
            incident["incident_id"] = str(uuid4())
            incidents.append(incident)

        results = validator.validate_batch(incidents)
        assert len(results) == 5
        assert all(r.valid for r in results)

    def test_validate_batch_mixed_valid_invalid(self, validator, valid_incident_data):
        """Test batch validation with mix of valid and invalid."""
        incidents = []

        # Add valid incident
        incident1 = valid_incident_data.copy()
        incidents.append(incident1)

        # Add invalid incident (missing description)
        incident2 = valid_incident_data.copy()
        del incident2["description"]
        incidents.append(incident2)

        results = validator.validate_batch(incidents)
        assert len(results) == 2
        assert results[0].valid is True
        assert results[1].valid is False

    def test_validation_duration_tracked(self, validator, valid_incident_data):
        """Test that validation duration is tracked."""
        result = validator.validate_single(valid_incident_data)
        assert result.validation_duration_ms >= 0
        assert result.validation_duration_ms < 20  # Should be well under budget

    def test_duplicate_detection_logged(self, validator, valid_incident_data):
        """Test that duplicates are tracked during validation."""
        result1 = validator.validate_single(valid_incident_data)
        assert result1.valid is True

        # Same incident again should be detected as duplicate
        result2 = validator.validate_single(valid_incident_data)
        assert len(result2.duplicate_incident_ids) > 0


class TestErrorAggregator:
    """Test ErrorAggregator functionality."""

    def test_add_single_error(self):
        """Test adding a single error."""
        aggregator = ErrorAggregator()
        aggregator.add_error(
            field="location.latitude",
            message="Out of bounds",
            error_code="out_of_range",
            value="-95.0",
        )
        assert aggregator.has_errors() is True
        assert len(aggregator.get_errors()) == 1

    def test_add_multiple_errors(self):
        """Test adding multiple errors."""
        aggregator = ErrorAggregator()
        aggregator.add_error("field1", "error1", "code1")
        aggregator.add_error("field2", "error2", "code2")
        assert len(aggregator.get_errors()) == 2

    def test_get_errors_by_field(self):
        """Test grouping errors by field."""
        aggregator = ErrorAggregator()
        aggregator.add_error("field1", "error1", "code1")
        aggregator.add_error("field1", "error1b", "code1")
        aggregator.add_error("field2", "error2", "code2")

        by_field = aggregator.get_errors_by_field()
        assert len(by_field["field1"]) == 2
        assert len(by_field["field2"]) == 1

    def test_summary_message_single_error(self):
        """Test summary message for single error."""
        aggregator = ErrorAggregator()
        aggregator.add_error("field1", "error message", "code1")
        summary = aggregator.get_summary_message()
        assert "field1" in summary
        assert "error message" in summary

    def test_summary_message_multiple_errors(self):
        """Test summary message for multiple errors."""
        aggregator = ErrorAggregator()
        aggregator.add_error("field1", "error1", "code1")
        aggregator.add_error("field2", "error2", "code1")  # Same code
        aggregator.add_error("field3", "error3", "code2")
        summary = aggregator.get_summary_message()
        assert "code1" in summary
        assert "code2" in summary

    def test_clear_errors(self):
        """Test clearing all errors."""
        aggregator = ErrorAggregator()
        aggregator.add_error("field1", "error1", "code1")
        assert aggregator.has_errors()
        aggregator.clear()
        assert not aggregator.has_errors()


class TestValidationErrorFormatter:
    """Test ValidationErrorFormatter."""

    def test_format_for_api(self):
        """Test formatting errors for API response."""
        aggregator = ErrorAggregator()
        aggregator.add_error("field1", "error1", "code1")
        aggregator.add_error("field2", "error2", "code2")

        formatted = ValidationErrorFormatter.format_for_api(aggregator)
        assert formatted["error"] == "validation_error"
        assert formatted["error_count"] == 2
        assert len(formatted["details"]) == 2

    def test_format_for_logging(self):
        """Test formatting errors for logging."""
        aggregator = ErrorAggregator()
        aggregator.add_error("field1", "error1", "code1")

        formatted = ValidationErrorFormatter.format_for_logging(
            aggregator, incident_id="incident_123"
        )
        assert formatted["event"] == "validation_failed"
        assert formatted["incident_id"] == "incident_123"
        assert formatted["error_count"] == 1


class TestAuditLogger:
    """Test AuditLogger functionality."""

    @pytest.fixture
    def audit_logger(self):
        """Provide a fresh audit logger for each test."""
        logger = AuditLogger()
        yield logger
        logger.clear()

    def test_log_operation_success(self, audit_logger):
        """Test logging successful operation."""
        entry = audit_logger.log_operation(
            operation="validation",
            status="success",
            incident_id="incident_123",
            details={"records": 10},
        )
        assert entry.status == "success"
        assert entry.operation == "validation"

    def test_log_validation_batch(self, audit_logger):
        """Test logging batch validation operation."""
        entry = audit_logger.log_validation_batch(
            batch_size=100,
            records_valid=95,
            records_invalid=5,
            avg_validation_time_ms=2.5,
            duplicates_detected=2,
        )
        assert entry.details["batch_size"] == 100
        assert entry.details["pass_rate"] == 0.95

    def test_audit_stats_tracking(self, audit_logger):
        """Test that audit logger tracks statistics."""
        audit_logger.log_validation_batch(50, 45, 5, 2.0)
        audit_logger.log_validation_batch(50, 48, 2, 2.1)

        stats = audit_logger.get_validation_stats()
        assert stats.total_records == 100
        assert stats.valid_records == 93
        assert stats.invalid_records == 7

    def test_get_recent_entries(self, audit_logger):
        """Test retrieving recent entries."""
        for i in range(5):
            audit_logger.log_operation(operation="test_op", status="success")

        recent = audit_logger.get_recent_entries(3)
        assert len(recent) == 3

    def test_get_entries_by_operation(self, audit_logger):
        """Test filtering entries by operation."""
        audit_logger.log_operation("op1", "success")
        audit_logger.log_operation("op2", "success")
        audit_logger.log_operation("op1", "failure")

        op1_entries = audit_logger.get_entries_by_operation("op1")
        assert len(op1_entries) == 2

    def test_summary_report(self, audit_logger):
        """Test generating summary report."""
        audit_logger.log_operation("op1", "success")
        audit_logger.log_operation("op1", "success")
        audit_logger.log_operation("op1", "failure")

        report = audit_logger.get_summary_report()
        assert report["total_operations"] == 3
        assert report["successful"] == 2
        assert report["failed"] == 1
        assert report["success_rate"] == pytest.approx(2 / 3)


class TestValidationIntegration:
    """Integration tests for validation layer."""

    def test_end_to_end_incident_validation(self):
        """Test complete incident validation workflow."""
        validator = IncidentValidator()
        audit_logger = AuditLogger()

        incident_data = {
            "incident_id": str(uuid4()),
            "location": {"latitude": -33.8688, "longitude": 151.2093},
            "timestamp": (datetime.utcnow() - timedelta(minutes=5)).isoformat(),
            "description": "Multi-vehicle collision on M1 northbound",
            "incident_type": "accident",
        }

        # Validate
        result = validator.validate_single(incident_data)

        # Log
        audit_logger.log_validation_batch(
            batch_size=1,
            records_valid=1 if result.valid else 0,
            records_invalid=0 if result.valid else 1,
            avg_validation_time_ms=result.validation_duration_ms,
        )

        # Check results
        assert result.valid is True
        stats = audit_logger.get_validation_stats()
        assert stats.total_records == 1

    def test_batch_validation_with_audit(self):
        """Test batch validation with audit logging."""
        validator = IncidentValidator()
        audit_logger = AuditLogger()

        incidents = []
        for i in range(10):
            incidents.append(
                {
                    "incident_id": str(uuid4()),
                    "location": {"latitude": -33.8688 + (i * 0.01), "longitude": 151.2093},
                    "timestamp": (datetime.utcnow() - timedelta(minutes=5)).isoformat(),
                    "description": f"Incident {i}: Multi-vehicle collision on M1 northbound",
                    "incident_type": "accident",
                }
            )

        results = validator.validate_batch(incidents)
        valid_count = sum(1 for r in results if r.valid)

        audit_logger.log_validation_batch(
            batch_size=len(incidents),
            records_valid=valid_count,
            records_invalid=len(incidents) - valid_count,
            avg_validation_time_ms=sum(r.validation_duration_ms for r in results) / len(results),
        )

        stats = audit_logger.get_validation_stats()
        assert stats.total_records == 10
        assert stats.valid_records == valid_count
