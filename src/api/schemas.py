from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime


class LocationInput(BaseModel):
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)


class FeedbackInput(BaseModel):
    approval_status: str
    finalized_manpower: str
    finalized_barricading: str


class IncidentInput(BaseModel):
    incident_id: Optional[str] = None
    location: LocationInput
    timestamp: datetime
    description: str = Field(..., min_length=10)
    incident_type: str
    metadata: Optional[Dict[str, Any]] = None


class IncidentResponse(BaseModel):
    incident_id: str
    status: str
    estimated_completion_ms: int
    request_id: str


class ComponentLatencies(BaseModel):
    data_pipeline: float = 0.0
    module_a: float = 0.0
    module_b: float = 0.0
    playbook: float = 0.0
    shap: float = 0.0
    aggregation: float = 0.0


class PredictionResponse(BaseModel):
    incident_id: str
    submission_time: datetime
    prediction_time: datetime
    predictions: Dict[str, Any]
    playbook: Optional[List[Dict[str, Any]]] = None
    explanations: Optional[Dict[str, Any]] = None
    latency_ms: float
    component_latencies_ms: ComponentLatencies


class HealthResponse(BaseModel):
    status: str
    components: Dict[str, str]
    metrics: Dict[str, float]
