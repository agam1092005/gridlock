import pytest
import asyncio
from datetime import datetime, timezone
from src.orchestration.orchestrator import PredictionOrchestrator
from src.orchestration.survival_model import (
    SurvivalModelSingleton,
    INCIDENT_TYPE_MAP,
    PRIORITY_MAP,
    HISTORICAL_MEDIANS,
)
from src.api.schemas import PredictionResponse


import os

@pytest.mark.skipif(
    not os.path.exists("models/artifacts/module_a/v1.0/cox_survival.pkl"),
    reason="Model artifact not found in CI"
)
@pytest.mark.asyncio
async def test_survival_model_loaded_and_used():
    """Verify that the Cox Proportional Hazards model is loaded and used for missing end_datetime."""
    # Ensure model singleton is loaded
    cph = SurvivalModelSingleton.get_model()
    assert cph is not None, "Cox model should be successfully loaded from pickle."

    orchestrator = PredictionOrchestrator()

    # Context missing end_datetime
    context_missing_end = {
        "incident_id": "test-survival-missing-end",
        "description": "Multi-car collision on ORR near Peenya causing severe delays.",
        "location": {"latitude": 13.0400, "longitude": 77.5180},
        "incident_type": "accident",
        "metadata": {"priority": "High", "event_cause": "accident"},
        "timestamp": datetime.now(timezone.utc),
        "explainability": False,
    }

    # Run pipeline
    response_missing = await orchestrator.run_pipeline(context_missing_end)
    assert isinstance(response_missing, PredictionResponse)

    # The duration estimate should be positive and derived from the Cox model
    assert response_missing.predictions["module_a"]["duration_estimate"] > 0
    # Historically, the expected resolution time for a High priority accident at current hour
    # is around 500-600 minutes based on training distribution, but definitely > 0.


@pytest.mark.skipif(
    not os.path.exists("models/artifacts/module_a/v1.0/cox_survival.pkl"),
    reason="Model artifact not found in CI"
)
@pytest.mark.asyncio
async def test_survival_model_fallback_on_invalid_data():
    """Verify graceful degradation fallback to historical median on invalid/edge-case values."""
    orchestrator = PredictionOrchestrator()

    # Context with an invalid incident type that does not exist in mapping
    context_invalid = {
        "incident_id": "test-survival-invalid",
        "description": "Unusual event cause that is completely missing.",
        "location": {"latitude": 13.0400, "longitude": 77.5180},
        "incident_type": "completely_novel_type_xyz",
        "metadata": {"priority": "UnknownPriorityLevel", "event_cause": "novel"},
        "timestamp": "invalid-timestamp-string-causes-exception",
        "explainability": False,
    }

    # Run pipeline - should not crash, but degrade gracefully
    response = await orchestrator.run_pipeline(context_invalid)
    assert isinstance(response, PredictionResponse)

    # Should fallback to historical median for generic fallback since type is invalid
    assert (
        response.predictions["module_a"]["duration_estimate"]
        == HISTORICAL_MEDIANS["generic_fallback"]
    )
