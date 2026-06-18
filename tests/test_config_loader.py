"""Tests for configuration loader."""

import os
import tempfile

import pytest

from src.utils import ConfigurationError, get_config, load_config, reset_config


class TestConfigLoader:
    """Test configuration loading and validation."""

    def test_load_config_from_env(self, monkeypatch):
        """Test loading configuration from environment variables."""
        reset_config()
        monkeypatch.setenv("API_PORT", "8080")
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")

        config = load_config()

        assert config.api_port == 8080
        assert config.log_level == "DEBUG"

    def test_load_config_from_yaml(self):
        """Test loading configuration from YAML file."""
        reset_config()

        yaml_content = """
api:
  host: 127.0.0.1
  port: 9000

logging:
  level: WARNING
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            temp_file = f.name

        try:
            config = load_config(temp_file)
            assert config.api_port == 9000
            assert config.log_level == "WARNING"
        finally:
            os.unlink(temp_file)

    def test_api_port_validation(self, monkeypatch):
        """Test API port validation."""
        reset_config()

        # Invalid port (too low)
        monkeypatch.setenv("API_PORT", "500")
        with pytest.raises(ConfigurationError):
            load_config()

        # Invalid port (too high)
        reset_config()
        monkeypatch.setenv("API_PORT", "70000")
        with pytest.raises(ConfigurationError):
            load_config()

    def test_log_level_validation(self, monkeypatch):
        """Test log level validation."""
        reset_config()
        monkeypatch.setenv("LOG_LEVEL", "INVALID")

        with pytest.raises(ConfigurationError):
            load_config()

    def test_environment_validation(self, monkeypatch):
        """Test environment validation."""
        reset_config()
        monkeypatch.setenv("ENVIRONMENT", "staging")

        with pytest.raises(ConfigurationError):
            load_config()

    def test_latency_budget_validation(self, monkeypatch):
        """Test that component budgets don't exceed total budget."""
        reset_config()
        monkeypatch.setenv("LATENCY_BUDGET_MS", "200")
        monkeypatch.setenv("DATA_PIPELINE_BUDGET_MS", "50")
        monkeypatch.setenv("MODULE_A_BUDGET_MS", "100")
        monkeypatch.setenv("MODULE_B_BUDGET_MS", "100")

        with pytest.raises(ConfigurationError):
            load_config()

    def test_get_config_singleton(self):
        """Test that get_config returns the same instance."""
        reset_config()

        config1 = get_config()
        config2 = get_config()

        assert config1 is config2

    def test_valid_config_creation(self, monkeypatch):
        """Test creating valid configuration."""
        reset_config()
        monkeypatch.setenv("API_PORT", "8000")
        monkeypatch.setenv("LOG_LEVEL", "INFO")
        monkeypatch.setenv("ENVIRONMENT", "production")

        config = load_config()

        assert config.api_port == 8000
        assert config.log_level == "INFO"
        assert config.environment == "production"

    def test_database_url_validation(self, monkeypatch):
        """Test database URL validation."""
        reset_config()
        monkeypatch.setenv("DATABASE_URL", "mysql://user:password@localhost:3306/db")

        with pytest.raises(ConfigurationError):
            load_config()

    def test_config_defaults(self):
        """Test default configuration values."""
        reset_config()

        config = load_config()

        assert config.api_host == "0.0.0.0"
        assert config.redis_port == 6379
        assert config.log_level == "INFO"
        assert config.environment == "development"
