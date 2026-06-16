"""
Incident validation rules and validator class.

Implements comprehensive validation rules for incident data including
timestamp format validation, location bounds checking, description length,
and duplicate detection.
"""

import time
from datetime import datetime
from typing import Dict, List, Optional, Set, Any
from uuid import UUID

from pydantic import ValidationError as PydanticValidationError

from src.data_pipeline.models import IncidentInput, ValidationResult, ValidationError
from src.data_pipeline.validation_errors import ErrorAggregator, ValidationErrorFormatter
from src.utils.logging_config import get_logger


logger = get_logger(__name__)


class DuplicateDetector:
    """Detects duplicate incidents using location, timestamp, and type matching."""
    
    def __init__(self, time_window_minutes: int = 30, distance_threshold_km: float = 2.0):
        """
        Initialize duplicate detector.
        
        Args:
            time_window_minutes: Time window for duplicate detection
            distance_threshold_km: Distance threshold for spatial matching
        """
        self.time_window_minutes = time_window_minutes
        self.distance_threshold_km = distance_threshold_km
        self.incident_history: List[Dict[str, Any]] = []
    
    def add_incident(self, incident_id: str, location: Dict[str, float], 
                    timestamp: datetime, incident_type: str) -> None:
        """
        Add incident to detection history.
        
        Args:
            incident_id: Incident ID
            location: Location dict with latitude and longitude
            timestamp: Incident timestamp
            incident_type: Type of incident
        """
        self.incident_history.append({
            'incident_id': incident_id,
            'location': location,
            'timestamp': timestamp,
            'incident_type': incident_type
        })
    
    def find_duplicates(self, incident_id: str, location: Dict[str, float],
                       timestamp: datetime, incident_type: str) -> List[str]:
        """
        Find duplicate incidents for the given incident.
        
        Uses spatial and temporal proximity matching within configured thresholds.
        
        Args:
            incident_id: Incident ID to check
            location: Location dict with latitude and longitude
            timestamp: Incident timestamp
            incident_type: Type of incident
        
        Returns:
            List of incident IDs that are duplicates
        """
        duplicates: List[str] = []
        
        for historical in self.incident_history:
            # Check if same incident type
            if historical['incident_type'] != incident_type:
                continue
            
            # Check time window
            time_diff = abs((timestamp - historical['timestamp']).total_seconds() / 60)
            if time_diff > self.time_window_minutes:
                continue
            
            # Check spatial proximity (simplified calculation)
            lat_diff = abs(location['latitude'] - historical['location']['latitude'])
            lon_diff = abs(location['longitude'] - historical['location']['longitude'])
            
            # Very rough approximation: 1 degree ≈ 111 km
            distance_km = ((lat_diff ** 2 + lon_diff ** 2) ** 0.5) * 111
            
            if distance_km <= self.distance_threshold_km:
                duplicates.append(historical['incident_id'])
        
        return duplicates
    
    def clear(self) -> None:
        """Clear detection history."""
        self.incident_history.clear()


class IncidentValidator:
    """
    Validates incident data against business rules.
    
    Performs validation including:
    - Required field presence and non-null checks
    - Timestamp format and future-date validation
    - Location bounds validation
    - Description length constraints
    - Duplicate incident detection
    - Type constraints and consistency checks
    """
    
    def __init__(self):
        """Initialize validator."""
        self.duplicate_detector = DuplicateDetector()
    
    def validate_single(self, incident_data: Dict[str, Any], 
                       incident_id: Optional[str] = None) -> ValidationResult:
        """
        Validate a single incident record.
        
        Args:
            incident_data: Raw incident data dictionary
            incident_id: Optional incident ID (will be extracted from data if not provided)
        
        Returns:
            ValidationResult with detailed validation information
        """
        start_time = time.time()
        validation_duration_ms = 0.0
        aggregator = ErrorAggregator()
        
        try:
            # Use provided incident_id or extract from data
            effective_incident_id = incident_id or incident_data.get('incident_id', 'unknown')
            
            # First, try to parse with Pydantic (catches schema validation)
            try:
                incident = IncidentInput(**incident_data)
            except PydanticValidationError as e:
                aggregator.add_pydantic_errors(e)
                validation_duration_ms = (time.time() - start_time) * 1000
                
                return ValidationResult(
                    incident_id=effective_incident_id,
                    valid=False,
                    errors=aggregator.get_errors(),
                    validation_duration_ms=validation_duration_ms
                )
            
            # Additional business rule validations
            self._validate_location_service_area(incident, aggregator)
            self._validate_description_quality(incident, aggregator)
            
            # Check for duplicates
            duplicate_ids = self.duplicate_detector.find_duplicates(
                incident_id=incident.incident_id or effective_incident_id,
                location=incident.location.model_dump(),
                timestamp=incident.timestamp,
                incident_type=incident.incident_type.value
            )
            
            # Add incident to detection history
            self.duplicate_detector.add_incident(
                incident_id=incident.incident_id or effective_incident_id,
                location=incident.location.model_dump(),
                timestamp=incident.timestamp,
                incident_type=incident.incident_type.value
            )
            
            validation_duration_ms = (time.time() - start_time) * 1000
            
            # Validate latency budget
            if validation_duration_ms > 20:
                logger.warning(
                    "Validation exceeded latency budget",
                    extra={
                        'incident_id': effective_incident_id,
                        'duration_ms': validation_duration_ms,
                        'budget_ms': 20
                    }
                )
            
            return ValidationResult(
                incident_id=incident.incident_id or effective_incident_id,
                valid=not aggregator.has_errors(),
                errors=aggregator.get_errors(),
                duplicate_incident_ids=duplicate_ids,
                validation_duration_ms=validation_duration_ms
            )
        
        except Exception as e:
            logger.error(
                "Unexpected error during validation",
                exc_info=True,
                extra={
                    'incident_id': incident_id or 'unknown',
                    'error': str(e)
                }
            )
            
            validation_duration_ms = (time.time() - start_time) * 1000
            
            aggregator.add_error(
                field='_internal',
                message=f"Unexpected validation error: {str(e)}",
                error_code='internal_error',
                value=None
            )
            
            return ValidationResult(
                incident_id=incident_id or 'unknown',
                valid=False,
                errors=aggregator.get_errors(),
                validation_duration_ms=validation_duration_ms
            )
    
    def validate_batch(self, incidents: List[Dict[str, Any]]) -> List[ValidationResult]:
        """
        Validate a batch of incident records.
        
        Args:
            incidents: List of incident data dictionaries
        
        Returns:
            List of ValidationResult objects
        """
        results = []
        for i, incident in enumerate(incidents):
            incident_id = incident.get('incident_id', f'batch_item_{i}')
            result = self.validate_single(incident, incident_id)
            results.append(result)
        
        return results
    
    def _validate_location_service_area(self, incident: IncidentInput, 
                                       aggregator: ErrorAggregator) -> None:
        """
        Validate that location is within service area.
        
        For now, uses simple Sydney metro bounds.
        Can be extended to use GIS or database lookup.
        
        Args:
            incident: Validated incident data
            aggregator: Error aggregator to collect errors
        """
        # Sydney metro approximate bounds
        SYDNEY_LAT_MIN = -33.95
        SYDNEY_LAT_MAX = -33.75
        SYDNEY_LON_MIN = 150.95
        SYDNEY_LON_MAX = 151.35
        
        lat = incident.location.latitude
        lon = incident.location.longitude
        
        if not (SYDNEY_LAT_MIN <= lat <= SYDNEY_LAT_MAX and 
                SYDNEY_LON_MIN <= lon <= SYDNEY_LON_MAX):
            aggregator.add_error(
                field='location',
                message=f'Location ({lat}, {lon}) is outside service area',
                error_code='out_of_service_area',
                value=f'({lat}, {lon})'
            )
    
    def _validate_description_quality(self, incident: IncidentInput,
                                     aggregator: ErrorAggregator) -> None:
        """
        Validate description quality (not just length).
        
        Args:
            incident: Validated incident data
            aggregator: Error aggregator to collect errors
        """
        description = incident.description.strip()
        
        # Check for mostly repeated characters (noise)
        if len(description) > 0:
            char_counts = {}
            for char in description.lower():
                if char != ' ':
                    char_counts[char] = char_counts.get(char, 0) + 1
            
            if char_counts:
                max_count = max(char_counts.values())
                repetition_rate = max_count / len(description)
                
                if repetition_rate > 0.7:  # 70% same character
                    aggregator.add_error(
                        field='description',
                        message='Description appears to be mostly repeated characters',
                        error_code='description_quality_low',
                        value=description[:30]
                    )
        
        # Check for minimum word count (at least 2 words)
        word_count = len(description.split())
        if word_count < 2:
            aggregator.add_error(
                field='description',
                message='Description must contain at least 2 words',
                error_code='description_too_short_words',
                value=description[:30]
            )
    
    def validate_with_context(self, incident_data: Dict[str, Any],
                             context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """
        Validate incident with additional context.
        
        Args:
            incident_data: Raw incident data
            context: Optional context dict with keys like 'db_connection', 'cache', etc.
        
        Returns:
            ValidationResult with comprehensive validation
        """
        result = self.validate_single(incident_data)
        
        # Could add context-specific validations here
        # For example: check against database for business rules
        
        return result
    
    def get_duplicate_detector_stats(self) -> Dict[str, int]:
        """
        Get statistics about duplicate detector.
        
        Returns:
            Dictionary with detector statistics
        """
        return {
            'total_incidents_tracked': len(self.duplicate_detector.incident_history),
            'time_window_minutes': self.duplicate_detector.time_window_minutes,
            'distance_threshold_km': self.duplicate_detector.distance_threshold_km
        }
    
    def clear_duplicate_history(self) -> None:
        """Clear duplicate detection history."""
        self.duplicate_detector.clear()
