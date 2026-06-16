"""
Audit logging for data validation operations.

Records all validation operations with detailed metrics including:
- Operation type and status
- Record counts and pass/fail statistics
- Processing timestamps
- Error tracking
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
from collections import defaultdict
import json

from src.data_pipeline.models import AuditLogEntry, ValidationStats
from src.utils.logging_config import get_logger


logger = get_logger(__name__)


class AuditLogger:
    """Tracks and logs validation operations."""
    
    def __init__(self):
        """Initialize audit logger."""
        self.entries: List[AuditLogEntry] = []
        self.operation_stats: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {'success': 0, 'failure': 0, 'partial': 0}
        )
        self.validation_stats = ValidationStats()
    
    def log_operation(
        self,
        operation: str,
        status: str,
        incident_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None
    ) -> AuditLogEntry:
        """
        Log a validation operation.
        
        Args:
            operation: Type of operation
            status: Operation status (success, failure, partial)
            incident_id: Associated incident ID if applicable
            details: Additional operation details
            error_message: Error message if operation failed
        
        Returns:
            Created AuditLogEntry
        """
        entry = AuditLogEntry(
            timestamp=datetime.utcnow(),
            operation=operation,
            incident_id=incident_id,
            status=status,
            details=details or {},
            error_message=error_message
        )
        
        self.entries.append(entry)
        
        # Update operation stats
        self.operation_stats[operation][status] += 1
        
        # Log to system logger
        logger_context = {
            'operation': operation,
            'status': status,
            'incident_id': incident_id,
            'details': details
        }
        
        if status == 'success':
            logger.info(f"Operation completed: {operation}", extra=logger_context)
        elif status == 'partial':
            logger.warning(f"Operation partially completed: {operation}", extra=logger_context)
        else:
            logger_context['error'] = error_message
            logger.error(f"Operation failed: {operation}", extra=logger_context)
        
        return entry
    
    def log_validation_batch(
        self,
        batch_size: int,
        records_valid: int,
        records_invalid: int,
        avg_validation_time_ms: float,
        duplicates_detected: int = 0,
        batch_id: Optional[str] = None
    ) -> AuditLogEntry:
        """
        Log a batch validation operation.
        
        Args:
            batch_size: Total records in batch
            records_valid: Records that passed validation
            records_invalid: Records that failed validation
            avg_validation_time_ms: Average validation time per record
            duplicates_detected: Number of duplicates found
            batch_id: Optional batch identifier
        
        Returns:
            Created AuditLogEntry
        """
        pass_rate = records_valid / batch_size if batch_size > 0 else 0
        
        details = {
            'batch_size': batch_size,
            'records_valid': records_valid,
            'records_invalid': records_invalid,
            'pass_rate': pass_rate,
            'avg_validation_time_ms': avg_validation_time_ms,
            'duplicates_detected': duplicates_detected,
            'batch_id': batch_id
        }
        
        # Update validation stats
        self.validation_stats.total_records += batch_size
        self.validation_stats.valid_records += records_valid
        self.validation_stats.invalid_records += records_invalid
        self.validation_stats.duplicates_detected += duplicates_detected
        self.validation_stats.timestamp = datetime.utcnow()
        
        if self.validation_stats.total_records > 0:
            self.validation_stats.pass_rate = (
                self.validation_stats.valid_records / self.validation_stats.total_records
            )
        
        status = 'partial' if records_invalid > 0 else 'success'
        
        return self.log_operation(
            operation='validation_batch',
            status=status,
            details=details
        )
    
    def log_embedding_operation(
        self,
        description_id: str,
        cached: bool,
        latency_ms: float,
        embedding_size: int = 768,
        error: Optional[str] = None
    ) -> AuditLogEntry:
        """
        Log an embedding operation.
        
        Args:
            description_id: Hash of description
            cached: Whether embedding was from cache
            latency_ms: Time taken for operation
            embedding_size: Size of generated embedding
            error: Error message if operation failed
        
        Returns:
            Created AuditLogEntry
        """
        details = {
            'description_id': description_id,
            'cached': cached,
            'latency_ms': latency_ms,
            'embedding_size': embedding_size
        }
        
        status = 'success' if error is None else 'failure'
        
        return self.log_operation(
            operation='embedding_generation',
            status=status,
            details=details,
            error_message=error
        )
    
    def log_imputation_operation(
        self,
        incident_id: str,
        fields_imputed: List[str],
        imputation_method: str,
        success: bool,
        error: Optional[str] = None
    ) -> AuditLogEntry:
        """
        Log a missing value imputation operation.
        
        Args:
            incident_id: Incident being processed
            fields_imputed: Fields that were imputed
            imputation_method: Method used (survival_analysis, mean, mode, etc.)
            success: Whether imputation succeeded
            error: Error message if operation failed
        
        Returns:
            Created AuditLogEntry
        """
        details = {
            'fields_imputed': fields_imputed,
            'imputation_method': imputation_method,
            'field_count': len(fields_imputed)
        }
        
        status = 'success' if success else 'failure'
        
        return self.log_operation(
            operation='missing_value_imputation',
            status=status,
            incident_id=incident_id,
            details=details,
            error_message=error
        )
    
    def log_duplicate_detection(
        self,
        incident_id: str,
        duplicates_found: List[str],
        detection_method: str = 'spatial_temporal'
    ) -> AuditLogEntry:
        """
        Log duplicate incident detection.
        
        Args:
            incident_id: Incident being checked
            duplicates_found: List of duplicate incident IDs
            detection_method: Method used for detection
        
        Returns:
            Created AuditLogEntry
        """
        details = {
            'duplicates_found': duplicates_found,
            'duplicate_count': len(duplicates_found),
            'detection_method': detection_method
        }
        
        status = 'success'
        
        return self.log_operation(
            operation='duplicate_detection',
            status=status,
            incident_id=incident_id,
            details=details
        )
    
    def get_recent_entries(self, count: int = 100) -> List[AuditLogEntry]:
        """
        Get recent audit log entries.
        
        Args:
            count: Number of entries to return
        
        Returns:
            List of recent AuditLogEntry objects
        """
        return self.entries[-count:]
    
    def get_entries_by_operation(self, operation: str) -> List[AuditLogEntry]:
        """
        Get audit log entries for a specific operation.
        
        Args:
            operation: Operation type to filter by
        
        Returns:
            List of matching entries
        """
        return [e for e in self.entries if e.operation == operation]
    
    def get_entries_by_incident(self, incident_id: str) -> List[AuditLogEntry]:
        """
        Get audit log entries for a specific incident.
        
        Args:
            incident_id: Incident ID to filter by
        
        Returns:
            List of entries related to this incident
        """
        return [e for e in self.entries if e.incident_id == incident_id]
    
    def get_entries_by_status(self, status: str) -> List[AuditLogEntry]:
        """
        Get audit log entries by status.
        
        Args:
            status: Status to filter by (success, failure, partial)
        
        Returns:
            List of entries with this status
        """
        return [e for e in self.entries if e.status == status]
    
    def get_operation_stats(self) -> Dict[str, Dict[str, int]]:
        """
        Get summary statistics for each operation type.
        
        Returns:
            Dictionary with counts by operation and status
        """
        return dict(self.operation_stats)
    
    def get_validation_stats(self) -> ValidationStats:
        """
        Get overall validation statistics.
        
        Returns:
            ValidationStats object
        """
        return self.validation_stats
    
    def get_summary_report(self, operation: Optional[str] = None) -> Dict[str, Any]:
        """
        Get a summary report of audit operations.
        
        Args:
            operation: Optional operation to filter by
        
        Returns:
            Dictionary with summary statistics
        """
        if operation:
            entries = self.get_entries_by_operation(operation)
        else:
            entries = self.entries
        
        total_operations = len(entries)
        successful = sum(1 for e in entries if e.status == 'success')
        failed = sum(1 for e in entries if e.status == 'failure')
        partial = sum(1 for e in entries if e.status == 'partial')
        
        success_rate = successful / total_operations if total_operations > 0 else 0
        
        return {
            'total_operations': total_operations,
            'successful': successful,
            'failed': failed,
            'partial': partial,
            'success_rate': success_rate,
            'validation_stats': self.validation_stats.model_dump(),
            'operation_stats': self.get_operation_stats()
        }
    
    def export_entries(self, format: str = 'json') -> str:
        """
        Export audit log entries in specified format.
        
        Args:
            format: Export format (json, csv)
        
        Returns:
            Formatted string of audit entries
        """
        if format == 'json':
            return json.dumps(
                [e.model_dump(mode='json') for e in self.entries],
                indent=2,
                default=str
            )
        elif format == 'csv':
            # Simple CSV format
            lines = ['timestamp,operation,incident_id,status,details,error_message']
            for entry in self.entries:
                details_str = json.dumps(entry.details).replace(',', ';')
                error_str = entry.error_message or ''
                lines.append(
                    f'{entry.timestamp},{entry.operation},{entry.incident_id or ""},'
                    f'{entry.status},"{details_str}","{error_str}"'
                )
            return '\n'.join(lines)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def clear(self) -> None:
        """Clear all audit log entries."""
        self.entries.clear()
        self.operation_stats.clear()
        self.validation_stats = ValidationStats()


class AuditContext:
    """Context manager for audit logging within a scope."""
    
    def __init__(self, audit_logger: AuditLogger, operation: str, 
                 incident_id: Optional[str] = None):
        """
        Initialize audit context.
        
        Args:
            audit_logger: AuditLogger instance
            operation: Operation type
            incident_id: Optional incident ID
        """
        self.audit_logger = audit_logger
        self.operation = operation
        self.incident_id = incident_id
        self.details: Dict[str, Any] = {}
    
    def __enter__(self):
        """Enter context."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context and log operation."""
        if exc_type is not None:
            status = 'failure'
            error_message = str(exc_val)
        else:
            status = 'success'
            error_message = None
        
        self.audit_logger.log_operation(
            operation=self.operation,
            status=status,
            incident_id=self.incident_id,
            details=self.details,
            error_message=error_message
        )
    
    def add_detail(self, key: str, value: Any) -> None:
        """Add a detail to the operation."""
        self.details[key] = value
    
    def set_details(self, details: Dict[str, Any]) -> None:
        """Set all details for the operation."""
        self.details = details


# Global audit logger instance
_audit_logger_instance: Optional[AuditLogger] = None


def get_audit_logger() -> AuditLogger:
    """
    Get the global audit logger instance.
    
    Returns:
        AuditLogger instance
    """
    global _audit_logger_instance
    if _audit_logger_instance is None:
        _audit_logger_instance = AuditLogger()
    return _audit_logger_instance
