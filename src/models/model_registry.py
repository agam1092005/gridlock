import json
import os
import uuid
import logging
from datetime import datetime

logger = logging.getLogger("model_registry")


class ModelRegistry:
    def __init__(self, registry_path=".gridlock/registry.json"):
        self.registry_path = registry_path
        self._ensure_registry_exists()

    def _ensure_registry_exists(self):
        os.makedirs(os.path.dirname(self.registry_path), exist_ok=True)
        if not os.path.exists(self.registry_path):
            with open(self.registry_path, "w") as f:
                json.dump({"models": {}}, f)

    def _load(self):
        with open(self.registry_path, "r") as f:
            return json.load(f)

    def _save(self, data):
        with open(self.registry_path, "w") as f:
            json.dump(data, f, indent=4)

    def register_model(
        self,
        component: str,
        metrics: dict,
        hyperparameters: dict,
        training_data_version: str = "v1",
    ):
        data = self._load()
        version_id = str(uuid.uuid4())

        if component not in data["models"]:
            data["models"][component] = []

        model_entry = {
            "version_id": version_id,
            "creation_timestamp": datetime.utcnow().isoformat(),
            "metrics": metrics,
            "hyperparameters": hyperparameters,
            "training_data_version": training_data_version,
            "status": "ready_for_evaluation",
        }

        data["models"][component].append(model_entry)
        self._save(data)
        logger.info(f"Registered new model for {component} with ID {version_id}")
        return version_id

    def set_active_model(self, component: str, version_id: str):
        data = self._load()
        if component not in data["models"]:
            raise ValueError(f"Component {component} not found.")

        # Demote current active
        for m in data["models"][component]:
            if m.get("status") == "active":
                m["status"] = "archived"

        # Promote new
        found = False
        for m in data["models"][component]:
            if m["version_id"] == version_id:
                m["status"] = "active"
                found = True
                break

        if not found:
            raise ValueError(f"Version {version_id} not found.")

        self._save(data)
        logger.info(f"Set {version_id} as active model for {component}")

    def get_active_model(self, component: str):
        data = self._load()
        for m in data["models"].get(component, []):
            if m.get("status") == "active":
                return m
        return None

    def rollback_model(self, component: str, target_version_id: str):
        """Instantly switch from one version to another."""
        self.set_active_model(component, target_version_id)
        logger.warning(f"Rolled back {component} to {target_version_id}")
