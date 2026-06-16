"""Configuration Loader for Gridlock 2.0."""

import os
from typing import Any, Dict, Optional

import yaml
from pydantic import field_validator
from pydantic_settings import BaseSettings

from .errors import ConfigurationError


class Settings(BaseSettings):
    """Application configuration with validation."""
    
    # API Settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 4
    
    # Database
    database_url: str = "postgresql://user:password@localhost:5432/gridlock"
    database_pool_size: int = 20
    database_max_overflow: int = 40
    
    # Redis
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None
    
    # ML Models
    model_artifacts_dir: str = "./models/artifacts"
    embedding_cache_ttl_seconds: int = 24 * 3600
    prediction_cache_ttl_seconds: int = 7 * 24 * 3600
    
    # Latency Budgets (milliseconds)
    latency_budget_ms: int = 500
    data_pipeline_budget_ms: int = 40
    module_a_budget_ms: int = 130
    module_b_budget_ms: int = 200
    playbook_budget_ms: int = 20
    shap_budget_ms: int = 80
    
    # Logging
    log_level: str = "INFO"
    log_dir: str = "./logs"
    
    # Monitoring
    enable_metrics: bool = True
    metrics_port: int = 9090
    
    # Environment
    environment: str = "development"
    debug: bool = False
    
    class Config:
        env_file = ".env"
        case_sensitive = False
    
    @field_validator('api_port')
    @classmethod
    def validate_api_port(cls, v: int) -> int:
        """Validate API port is in valid range."""
        if not 1024 <= v <= 65535:
            raise ValueError('API port must be between 1024 and 65535')
        return v
    
    @field_validator('redis_port')
    @classmethod
    def validate_redis_port(cls, v: int) -> int:
        """Validate Redis port is in valid range."""
        if not 1024 <= v <= 65535:
            raise ValueError('Redis port must be between 1024 and 65535')
        return v
    
    @field_validator('latency_budget_ms')
    @classmethod
    def validate_latency_budget(cls, v: int) -> int:
        """Validate latency budget is positive."""
        if v <= 0:
            raise ValueError('Latency budget must be positive')
        return v
    
    @field_validator('log_level')
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level is valid."""
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f'Log level must be one of {valid_levels}')
        return v.upper()
    
    @field_validator('environment')
    @classmethod
    def validate_environment(cls, v: str) -> str:
        """Validate environment is valid."""
        valid_envs = ['development', 'testing', 'production']
        if v not in valid_envs:
            raise ValueError(f'Environment must be one of {valid_envs}')
        return v
    
    def validate_all(self):
        """Validate all configuration parameters."""
        errors = []
        
        # Check database URL format
        if not self.database_url.startswith('postgresql://'):
            errors.append('Database URL must start with postgresql://')
        
        # Check latency budget components sum
        total_latency = (
            self.data_pipeline_budget_ms
            + self.module_a_budget_ms
            + self.module_b_budget_ms
            + self.playbook_budget_ms
            + self.shap_budget_ms
        )
        if total_latency > self.latency_budget_ms:
            errors.append(
                f'Component latency budgets ({total_latency}ms) '
                f'exceed total budget ({self.latency_budget_ms}ms)'
            )
        
        # Check model artifacts directory exists or can be created
        if not os.path.exists(self.model_artifacts_dir):
            try:
                os.makedirs(self.model_artifacts_dir, exist_ok=True)
            except OSError as e:
                errors.append(f'Cannot create model artifacts directory: {e}')
        
        # Check log directory exists or can be created
        if not os.path.exists(self.log_dir):
            try:
                os.makedirs(self.log_dir, exist_ok=True)
            except OSError as e:
                errors.append(f'Cannot create log directory: {e}')
        
        if errors:
            raise ConfigurationError(
                f'Configuration validation failed: {"; ".join(errors)}',
                context={'errors': errors}
            )


# Global configuration instance
_config_instance: Optional[Settings] = None


def load_config(config_file: Optional[str] = None) -> Settings:
    """
    Load configuration from file and environment variables.
    
    Args:
        config_file: Path to YAML configuration file
    
    Returns:
        Validated Settings instance
    
    Raises:
        ConfigurationError: If configuration is invalid
    """
    global _config_instance
    
    if _config_instance is not None:
        return _config_instance
    
    # Load from YAML if provided
    if config_file and os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                yaml_config = yaml.safe_load(f) or {}
                
                # Flatten nested YAML into environment variables
                def flatten_dict(d: Dict[str, Any], parent_key: str = '') -> Dict[str, str]:
                    items = []
                    for k, v in d.items():
                        new_key = f"{parent_key}_{k}".upper() if parent_key else k.upper()
                        if isinstance(v, dict):
                            items.extend(flatten_dict(v, new_key).items())
                        else:
                            items.append((new_key, str(v)))
                    return dict(items)
                
                flat_config = flatten_dict(yaml_config)
                for key, value in flat_config.items():
                    if key not in os.environ:  # Don't override existing env vars
                        os.environ[key] = value
        
        except Exception as e:
            raise ConfigurationError(
                f'Failed to load configuration file {config_file}: {e}',
                context={'config_file': config_file},
                original_exception=e,
            )
    
    # Create settings from environment
    try:
        _config_instance = Settings()
        _config_instance.validate_all()
        return _config_instance
    except Exception as e:
        raise ConfigurationError(
            f'Failed to create configuration: {e}',
            original_exception=e,
        )


def get_config() -> Settings:
    """
    Get current configuration instance.
    
    Returns:
        Loaded Settings instance
    
    Raises:
        ConfigurationError: If configuration loading fails
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = load_config()
    return _config_instance


def reset_config():
    """Reset configuration instance (useful for testing)."""
    global _config_instance
    _config_instance = None
