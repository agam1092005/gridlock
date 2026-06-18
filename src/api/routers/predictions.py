from fastapi import APIRouter, HTTPException
from ..schemas import PredictionResponse
from ...orchestration.orchestrator import PredictionOrchestrator
from .incidents import submitted_incidents
import logging

logger = logging.getLogger("api")
router = APIRouter(prefix="/predictions", tags=["predictions"])

orchestrator = PredictionOrchestrator()


@router.get("/{incident_id}", response_model=PredictionResponse)
async def get_prediction(incident_id: str, explainability: bool = False):
    """
    Retrieve predictions for a specific incident.
    """
    try:
        incident = submitted_incidents.get(incident_id)
        if not incident:
            # Fallback to realistic mock context if not found
            incident = {
                "incident_id": incident_id,
                "description": "Heavy multi-vehicle crash blocking lanes on major ORR corridor.",
                "location": {"latitude": 12.9716, "longitude": 77.5946},
                "incident_type": "accident",
                "metadata": {
                    "latitude": 12.9716,
                    "longitude": 77.5946,
                    "priority": "High",
                    "event_cause": "accident",
                    "veh_type": "heavy_vehicle",
                    "corridor": "orr",
                },
            }

        context = {**incident, "explainability": explainability}
        response = await orchestrator.run_pipeline(context)

        # If explainability was not requested, strip it from response
        if not explainability:
            response.explanations = {"enabled": False}

        return response
    except Exception as e:
        logger.error(f"Error fetching prediction: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
