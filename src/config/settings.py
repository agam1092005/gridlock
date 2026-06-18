import os
import yaml
from pydantic_settings import BaseSettings


class APISettings(BaseSettings):
    port: int = 8000
    host: str = "0.0.0.0"
    latency_budget_ms: int = 500


class ModelSettings(BaseSettings):
    graph_spatial_radius_km: float = 5.0
    prediction_horizon_minutes: int = 60


class MonitoringSettings(BaseSettings):
    alert_threshold_latency_ms: int = 400
    alert_threshold_error_rate: float = 0.05


class AppSettings(BaseSettings):
    api: APISettings = APISettings()
    models: ModelSettings = ModelSettings()
    monitoring: MonitoringSettings = MonitoringSettings()
    incident_types: list[str] = ["accident", "congestion", "roadwork"]

    @classmethod
    def load_config(cls, config_path: str = "src/config/config.yaml"):
        # Load yaml defaults
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                data = yaml.safe_load(f)
                return cls(**data)
        return cls()


settings = AppSettings.load_config()
