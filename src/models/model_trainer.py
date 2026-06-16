import os
import json
import logging
from datetime import datetime
from .model_registry import ModelRegistry

logger = logging.getLogger("model_trainer")

class ModelTrainer:
    def __init__(self, registry: ModelRegistry):
        self.registry = registry
        self.artifact_base_dir = ".gridlock/models/artifacts"

    def submit_training_job(self, component: str, hyperparameters: dict, dataset_metadata: dict):
        logger.info(f"Starting training job for {component}...")
        
        # 1. Create artifact directory
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(self.artifact_base_dir, component, timestamp)
        os.makedirs(run_dir, exist_ok=True)
        
        # 2. Log metadata
        config = {
            "hyperparameters": hyperparameters,
            "dataset": dataset_metadata,
            "timestamp": timestamp
        }
        with open(os.path.join(run_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=4)
            
        # 3. Execute mock training
        logger.info(f"Simulating training epochs for {component}...")
        
        # Mock metrics outcome
        mock_metrics = {
            "RMSE": 4.2,
            "MAE": 3.1,
            "R2": 0.88,
            "C-index": 0.85
        }
        
        # 4. Compare against production
        active = self.registry.get_active_model(component)
        promote = False
        
        if not active:
            logger.info("No active model found. Promoting as baseline.")
            promote = True
        else:
            old_r2 = active["metrics"].get("R2", 0)
            if mock_metrics["R2"] > old_r2 * 1.05: # >5% improvement
                logger.info(f"New model improves R2 ({old_r2} -> {mock_metrics['R2']}). Marking ready for evaluation.")
                promote = True
            else:
                logger.info("New model did not yield >5% improvement.")
                
        # 5. Register
        version_id = self.registry.register_model(
            component=component,
            metrics=mock_metrics,
            hyperparameters=hyperparameters,
            training_data_version=dataset_metadata.get("version", "v1")
        )
        
        # Auto-promote for testing if it's better
        if promote:
            # Typically this requires human review, but we auto-set for pipeline demo
            self.registry.set_active_model(component, version_id)
            
        return {
            "version_id": version_id,
            "metrics": mock_metrics,
            "promoted": promote
        }
