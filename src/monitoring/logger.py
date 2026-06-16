import json
import logging
import os
from datetime import datetime
import uuid

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_obj = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "request_id"):
            log_obj["request_id"] = record.request_id
        if hasattr(record, "incident_id"):
            log_obj["incident_id"] = record.incident_id
        if record.exc_info:
            log_obj["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(log_obj)

def setup_structured_logging():
    handler = logging.StreamHandler()
    handler.setFormatter(JSONFormatter())
    
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers = [] # clear existing
    root_logger.addHandler(handler)

class AuditLogger:
    def __init__(self, log_path=".gridlock/audit.jsonl"):
        self.log_path = log_path
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

    def log_operation(self, user: str, operation_type: str, details: dict):
        """
        Logs immutable audit records.
        """
        # Redaction logic
        safe_details = details.copy()
        if "location" in safe_details:
            # Mask precise coordinates
            safe_details["location"] = "REDACTED_GRID_CELL"
        if "user_ip" in safe_details:
            safe_details["user_ip"] = "REDACTED_IP"
            
        record = {
            "audit_id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "user": user,
            "operation": operation_type,
            "details": safe_details
        }
        
        with open(self.log_path, "a") as f:
            f.write(json.dumps(record) + "\n")
            
audit_logger = AuditLogger()
