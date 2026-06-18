import pytest
import asyncio
from src.orchestration.orchestrator import PredictionOrchestrator
from src.api.schemas import PredictionResponse


@pytest.mark.asyncio
async def test_prediction_orchestrator_pipeline_success(mocker):
    """Test that the prediction orchestrator successfully executes predictions and returns a schema-conformant response."""
    orchestrator = PredictionOrchestrator()

    # Mock context mimicking an incident submitted to the API
    mock_context = {
        "incident_id": "test-incident-uuid-1234",
        "description": "Multi-car collision on ORR near Peenya causing severe delays.",
        "location": {"latitude": 13.0400041, "longitude": 77.5180991},
        "incident_type": "accident",
        "metadata": {
            "latitude": 13.0400041,
            "longitude": 77.5180991,
            "priority": "High",
            "event_cause": "accident",
            "veh_type": "heavy_vehicle",
            "corridor": "orr",
        },
        "explainability": True,
    }

    # Run pipeline
    response = await orchestrator.run_pipeline(mock_context)

    # Assertions
    assert isinstance(response, PredictionResponse)
    assert response.incident_id == "test-incident-uuid-1234"
    assert "module_a" in response.predictions
    assert "severity_score" in response.predictions["module_a"]
    assert "duration_estimate" in response.predictions["module_a"]

    assert response.playbook is not None
    assert len(response.playbook) > 0

    # Verify latencies are tracked
    assert response.latency_ms > 0
    assert response.component_latencies_ms.module_a >= 0
    assert response.component_latencies_ms.module_b >= 0
    assert response.component_latencies_ms.playbook >= 0
    assert response.component_latencies_ms.shap >= 0
