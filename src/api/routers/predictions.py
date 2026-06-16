from fastapi import APIRouter, HTTPException
from ..schemas import PredictionResponse
from ...orchestration.orchestrator import PredictionOrchestrator
import logging

logger = logging.getLogger("api")
router = APIRouter(prefix="/predictions", tags=["predictions"])

orchestrator = PredictionOrchestrator()

@router.get("/{incident_id}", response_model=PredictionResponse)
async def get_prediction(incident_id: str, explainability: bool = False):
    """
    Retrieve predictions for a specific incident.
    """
    # For now, we synchronously call orchestrator for demonstration
    # In production, this would query a DB populated by background workers
    try:
        mock_context = {
            "incident_id": incident_id,
            "severity_score": 75,
            "duration_estimate": 45,
            "incident_type": "accident",
            "explainability": explainability
        }
        response = await orchestrator.run_pipeline(mock_context)
        
        # If explainability was not requested, strip it from response
        if not explainability:
            response.explanations = {"enabled": False}
            
        return response
    except Exception as e:
        logger.error(f"Error fetching prediction: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
