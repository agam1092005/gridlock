import os
import pickle
import logging
from typing import Optional
from lifelines import CoxPHFitter

logger = logging.getLogger("survival_model")

# Mappings for model inputs
INCIDENT_TYPE_MAP = {
    'accident': 0.0,
    'congestion': 1.0,
    'road_closure': 2.0,
    'hazard': 3.0,
    'event': 4.0
}

PRIORITY_MAP = {
    'high': 2.0,
    'medium': 1.0,
    'low': 0.0
}

# Historical median durations (fallback/graceful degradation)
HISTORICAL_MEDIANS = {
    "accident": 45.0,
    "congestion": 70.0,
    "road_closure": 120.0,
    "hazard": 60.0,
    "event": 90.0,
    "generic_fallback": 70.0
}

class SurvivalModelSingleton:
    _model: Optional[CoxPHFitter] = None

    @classmethod
    def load_model(cls, model_path: str = "models/artifacts/module_a/v1.0/cox_survival.pkl") -> Optional[CoxPHFitter]:
        """Load the pre-trained Cox proportional hazards model into memory."""
        if cls._model is None:
            if os.path.exists(model_path):
                try:
                    with open(model_path, 'rb') as f:
                        cls._model = pickle.load(f)
                    logger.info(f"Successfully loaded pre-trained Cox model from {model_path}")
                except Exception as e:
                    logger.error(f"Failed to load Cox model from {model_path}: {e}")
            else:
                logger.warning(f"Pre-trained Cox model not found at {model_path}")
        return cls._model

    @classmethod
    def get_model(cls) -> Optional[CoxPHFitter]:
        """Retrieve the loaded Cox model instance, lazily loading if necessary."""
        if cls._model is None:
            cls.load_model()
        return cls._model
