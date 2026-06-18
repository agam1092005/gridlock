"""
Pydantic models for data validation in Gridlock 2.0 data pipeline.

Provides comprehensive validation for incident input data, weather information,
and validation results with strong type checking and business rule validation.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class IncidentType(str, Enum):
    """Enumeration of incident types."""

    ACCIDENT = "accident"
    CONGESTION = "congestion"
    ROADWORK = "roadwork"
    WEATHER = "weather"
    UNKNOWN = "unknown"


class LocationData(BaseModel):
    """Location information for an incident."""

    model_config = ConfigDict(
        json_schema_extra={"example": {"latitude": -33.8688, "longitude": 151.2093}}
    )

    latitude: float = Field(..., ge=-90, le=90, description="Latitude coordinate (-90 to 90)")
    longitude: float = Field(..., ge=-180, le=180, description="Longitude coordinate (-180 to 180)")


class WeatherData(BaseModel):
    """Weather conditions at incident location."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "temperature": 22.5,
                "precipitation": 0.0,
                "wind_speed": 5.0,
                "humidity": 65.0,
                "visibility": 10000.0,
            }
        }
    )

    temperature: Optional[float] = Field(None, description="Temperature in Celsius")
    precipitation: Optional[float] = Field(None, ge=0, description="Precipitation in mm")
    wind_speed: Optional[float] = Field(None, ge=0, description="Wind speed in km/h")
    humidity: Optional[float] = Field(None, ge=0, le=100, description="Humidity percentage")
    visibility: Optional[float] = Field(None, ge=0, description="Visibility in meters")


class IncidentInput(BaseModel):
    """Input model for incident report submission."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "incident_id": "550e8400-e29b-41d4-a716-446655440000",
                "location": {"latitude": -33.8688, "longitude": 151.2093},
                "timestamp": "2024-01-15T14:30:00Z",
                "description": "Multi-vehicle collision on M1 northbound, 2 lanes blocked",
                "incident_type": "accident",
                "severity_initial": 75,
                "is_ongoing": True,
                "weather": {"temperature": 22.5, "precipitation": 0.0, "wind_speed": 5.0},
            }
        }
    )

    incident_id: Optional[str] = Field(None, description="Unique incident identifier (UUID format)")
    location: LocationData = Field(..., description="Incident location coordinates")
    timestamp: datetime = Field(..., description="Incident timestamp in ISO 8601 format")
    description: str = Field(
        ..., min_length=10, max_length=5000, description="Incident description (10-5000 characters)"
    )
    incident_type: IncidentType = Field(IncidentType.UNKNOWN, description="Type of incident")
    severity_initial: Optional[float] = Field(
        None, ge=0, le=100, description="Initial severity estimate (0-100)"
    )
    end_datetime: Optional[datetime] = Field(
        None, description="Incident end time (typically null for ongoing incidents)"
    )
    weather: Optional[WeatherData] = Field(
        None, description="Weather conditions at incident location"
    )
    is_ongoing: Optional[bool] = Field(True, description="Whether incident is still ongoing")
    num_lanes_blocked: Optional[int] = Field(None, ge=0, description="Number of lanes blocked")
    num_vehicles: Optional[int] = Field(None, ge=0, description="Number of vehicles involved")

    @field_validator("timestamp")
    @classmethod
    def validate_timestamp_not_in_future(cls, v: datetime) -> datetime:
        """Ensure timestamp is not in the future."""
        if v > datetime.utcnow():
            raise ValueError("Timestamp cannot be in the future")
        return v

    @field_validator("end_datetime")
    @classmethod
    def validate_end_datetime_after_start(cls, v: Optional[datetime], info) -> Optional[datetime]:
        """Ensure end_datetime is after timestamp if provided."""
        if v is None:
            return v

        timestamp = info.data.get("timestamp")
        if timestamp and v <= timestamp:
            raise ValueError("end_datetime must be after timestamp")

        if v > datetime.utcnow():
            raise ValueError("end_datetime cannot be in the future")

        return v

    @model_validator(mode="after")
    def validate_incident_consistency(self) -> "IncidentInput":
        """Validate overall incident consistency."""
        # If end_datetime is provided, is_ongoing should be False
        if self.end_datetime is not None and self.is_ongoing:
            raise ValueError("is_ongoing must be False when end_datetime is provided")

        # If is_ongoing is False, end_datetime should be provided
        if not self.is_ongoing and self.end_datetime is None:
            raise ValueError("end_datetime must be provided when is_ongoing is False")

        return self


class ValidationError(BaseModel):
    """Single validation error."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "field": "location.latitude",
                "message": "Value -95.2 is outside valid range [-90, 90]",
                "error_code": "out_of_range",
                "value": "-95.2",
            }
        }
    )

    field: str = Field(..., description="Field that failed validation")
    message: str = Field(..., description="Error message")
    error_code: str = Field(..., description="Error code for categorization")
    value: Optional[str] = Field(None, description="The value that failed validation (sanitized)")


class ValidationResult(BaseModel):
    """Result of incident validation."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "incident_id": "550e8400-e29b-41d4-a716-446655440000",
                "valid": False,
                "errors": [
                    {
                        "field": "location.latitude",
                        "message": "Value -95.2 is outside valid range",
                        "error_code": "out_of_range",
                        "value": "-95.2",
                    }
                ],
                "warnings": ["Weather data is missing, using defaults"],
                "validation_timestamp": "2024-01-15T14:30:01Z",
                "validation_duration_ms": 3.5,
                "duplicate_incident_ids": [],
            }
        }
    )

    incident_id: str = Field(..., description="Incident ID being validated")
    valid: bool = Field(..., description="Whether validation passed")
    errors: List[ValidationError] = Field(
        default_factory=list, description="List of validation errors (empty if valid)"
    )
    warnings: List[str] = Field(default_factory=list, description="Non-blocking warnings")
    validation_timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="When validation was performed"
    )
    validation_duration_ms: float = Field(
        0.0, description="Time taken for validation in milliseconds"
    )
    duplicate_incident_ids: List[str] = Field(
        default_factory=list, description="IDs of duplicate incidents detected"
    )


class AuditLogEntry(BaseModel):
    """Single audit log entry."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "timestamp": "2024-01-15T14:30:01Z",
                "operation": "validation_batch",
                "incident_id": None,
                "status": "success",
                "details": {
                    "batch_size": 32,
                    "records_valid": 31,
                    "records_invalid": 1,
                    "pass_rate": 0.96875,
                },
            }
        }
    )

    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="When the operation occurred"
    )
    operation: str = Field(
        ..., description="Type of operation (validation, embedding, imputation, etc.)"
    )
    incident_id: Optional[str] = Field(None, description="Associated incident ID if applicable")
    status: str = Field(..., description="Operation status (success, failure, partial)")
    details: Dict[str, Any] = Field(
        default_factory=dict, description="Additional operation details"
    )
    error_message: Optional[str] = Field(None, description="Error message if operation failed")


class ValidationStats(BaseModel):
    """Statistics from validation operations."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "total_records": 1000,
                "valid_records": 960,
                "invalid_records": 40,
                "pass_rate": 0.96,
                "avg_validation_time_ms": 2.5,
                "duplicates_detected": 3,
                "timestamp": "2024-01-15T14:30:01Z",
            }
        }
    )

    total_records: int = Field(0, description="Total records processed")
    valid_records: int = Field(0, description="Records that passed validation")
    invalid_records: int = Field(0, description="Records that failed validation")
    pass_rate: float = Field(0.0, description="Percentage of records that passed")
    avg_validation_time_ms: float = Field(0.0, description="Average validation time per record")
    duplicates_detected: int = Field(0, description="Number of duplicate incidents detected")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="When these stats were calculated"
    )
