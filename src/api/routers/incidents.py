from fastapi import APIRouter, HTTPException, Request, status, Depends
from ..schemas import IncidentInput, IncidentResponse, FeedbackInput
from ..middleware import verify_api_key
import uuid
import logging

logger = logging.getLogger("api")
router = APIRouter(prefix="/incidents", tags=["incidents"])

# Mock Redis queue
_mock_queue = []

# Cache of submitted incidents for lookup
submitted_incidents = {}


def get_api_key(request: Request) -> str:
    """Extract and validate API key from request."""
    auth_header = request.headers.get("Authorization")

    if not auth_header:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not auth_header.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Authorization header format. Use: Bearer <api_key>",
            headers={"WWW-Authenticate": "Bearer"},
        )

    api_key = auth_header[7:]  # Remove "Bearer " prefix

    # Verify API key
    if not verify_api_key(api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return api_key


@router.post("/", response_model=IncidentResponse, status_code=202)
async def submit_incident(
    incident: IncidentInput, request: Request, api_key: str = Depends(get_api_key)
):
    """
    Submit a new incident for processing and prediction.

    Requires authentication with valid API key.
    Returns 202 Accepted with incident ID and estimated processing time.
    """
    if not incident.incident_id:
        incident.incident_id = str(uuid.uuid4())

    request_id = getattr(request.state, "request_id", "unknown")

    # Validation checks
    if len(incident.description) < 10:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Description must be at least 10 characters long",
        )

    # Validate location bounds
    if not (-90 <= incident.location.latitude <= 90):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Latitude must be between -90 and 90"
        )

    if not (-180 <= incident.location.longitude <= 180):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Longitude must be between -180 and 180"
        )

    # Enqueue to mock Redis and cache it
    incident_dict = incident.model_dump()
    _mock_queue.append({**incident_dict, "api_key": api_key[:8] + "..."})
    submitted_incidents[incident.incident_id] = incident_dict
    logger.info(f"Enqueued incident {incident.incident_id} from API key {api_key[:8]}...")

    return IncidentResponse(
        incident_id=incident.incident_id,
        status="processing",
        estimated_completion_ms=250,
        request_id=request_id,
    )


@router.post("/{incident_id}/feedback", status_code=200)
async def submit_feedback(
    incident_id: str, feedback: FeedbackInput, request: Request, api_key: str = Depends(get_api_key)
):
    """
    Submit HITL operator feedback for model retraining.
    """
    logger.info(f"Feedback received for incident {incident_id}: {feedback.model_dump()}")
    # Here we would normally log this to a DB for model retraining
    return {"status": "success", "message": "Feedback logged successfully"}
