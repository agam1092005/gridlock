#!/usr/bin/env python3
"""
Gridlock 2.0 Startup Initialization Script
Handles startup tasks: environment validation, dependency checks, and service initialization
"""

import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import asyncpg
import redis

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================


def setup_logging(service_name: str) -> logging.Logger:
    """Configure structured logging for startup."""
    log_dir = Path(os.getenv("LOG_DIR", "./logs"))
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / f"{service_name}_startup.log"

    # Configure logging format
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    logger = logging.getLogger("gridlock.startup")
    logger.setLevel(logging.DEBUG)

    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger


# ============================================================================
# ENVIRONMENT VALIDATION
# ============================================================================


class EnvironmentValidator:
    """Validates environment configuration."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.errors = []
        self.warnings = []

    def validate(self) -> bool:
        """Run all validations."""
        self.logger.info("=== Environment Validation ===")

        required_vars = [
            "DATABASE_URL",
            "REDIS_HOST",
            "REDIS_PORT",
            "API_HOST",
            "API_PORT",
            "LOG_LEVEL",
            "SERVICE_NAME",
        ]

        optional_vars = [
            "API_WORKERS",
            "ENVIRONMENT",
            "DEBUG",
        ]

        # Check required variables
        for var in required_vars:
            if not os.getenv(var):
                self.errors.append(f"Required environment variable '{var}' not set")
            else:
                self.logger.info(f"✓ {var} = {self._mask_sensitive(var, os.getenv(var))}")

        # Warn about optional variables
        for var in optional_vars:
            if not os.getenv(var):
                self.warnings.append(
                    f"Optional environment variable '{var}' not set, using default"
                )
            else:
                self.logger.info(f"✓ {var} = {os.getenv(var)}")

        # Validate specific formats
        self._validate_database_url()
        self._validate_ports()

        # Report results
        if self.errors:
            self.logger.error(f"✗ Validation failed with {len(self.errors)} error(s):")
            for error in self.errors:
                self.logger.error(f"  - {error}")
            return False

        if self.warnings:
            for warning in self.warnings:
                self.logger.warning(f"  - {warning}")

        self.logger.info("✓ Environment validation passed")
        return True

    def _validate_database_url(self):
        """Validate DATABASE_URL format."""
        db_url = os.getenv("DATABASE_URL", "")
        if db_url:
            if not db_url.startswith("postgresql://"):
                self.errors.append("DATABASE_URL must start with 'postgresql://'")
            if "@" not in db_url:
                self.errors.append(
                    "DATABASE_URL must contain credentials: postgresql://user:pass@host:port/db"
                )

    def _validate_ports(self):
        """Validate port numbers."""
        ports = {
            "API_PORT": 8000,
            "REDIS_PORT": 6379,
        }

        for var, default in ports.items():
            port_str = os.getenv(var, str(default))
            try:
                port = int(port_str)
                if not (1 <= port <= 65535):
                    self.errors.append(f"{var} must be between 1 and 65535, got {port}")
            except ValueError:
                self.errors.append(f"{var} must be a valid integer, got '{port_str}'")

    @staticmethod
    def _mask_sensitive(var_name: str, value: str) -> str:
        """Mask sensitive values in logs."""
        sensitive_keywords = ["password", "secret", "token", "key", "api_key"]
        if any(keyword in var_name.lower() for keyword in sensitive_keywords):
            return f"{value[:3]}{'*' * (len(value) - 6)}{value[-3:]}" if len(value) > 6 else "***"
        return value


# ============================================================================
# SERVICE HEALTH CHECKS
# ============================================================================


class HealthChecker:
    """Checks health of dependent services."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.max_retries = 30
        self.retry_delay = 2

    async def check_database(self) -> bool:
        """Check PostgreSQL database connectivity."""
        self.logger.info("Checking PostgreSQL database...")

        db_url = os.getenv("DATABASE_URL")
        if not db_url:
            self.logger.error("DATABASE_URL not set")
            return False

        # Parse connection string
        # Format: postgresql://user:password@host:port/database
        try:
            import psycopg2
            from psycopg2 import sql

            # Parse URL manually (asyncpg URL format)
            db_url_parts = db_url.replace("postgresql://", "")
            creds, location = db_url_parts.split("@")
            user, password = creds.split(":")
            host, rest = location.split(":")
            port, database = rest.split("/")
            port = int(port)

            for attempt in range(self.max_retries):
                try:
                    # Try async connection with asyncpg
                    conn = await asyncpg.connect(
                        host=host,
                        port=port,
                        user=user,
                        password=password,
                        database=database,
                        timeout=5,
                    )
                    await conn.close()
                    self.logger.info(f"✓ PostgreSQL is accessible at {host}:{port}/{database}")
                    return True
                except Exception as e:
                    if attempt < self.max_retries - 1:
                        self.logger.debug(f"Attempt {attempt + 1}/{self.max_retries} failed: {e}")
                        await asyncio.sleep(self.retry_delay)
                    else:
                        self.logger.error(f"✗ Failed to connect to PostgreSQL: {e}")
                        return False

        except Exception as e:
            self.logger.error(f"✗ Database URL parsing error: {e}")
            return False

    def check_redis(self) -> bool:
        """Check Redis connectivity."""
        self.logger.info("Checking Redis cache...")

        redis_host = os.getenv("REDIS_HOST", "localhost")
        redis_port = int(os.getenv("REDIS_PORT", "6379"))
        redis_password = os.getenv("REDIS_PASSWORD")

        for attempt in range(self.max_retries):
            try:
                r = redis.Redis(
                    host=redis_host,
                    port=redis_port,
                    password=redis_password,
                    decode_responses=True,
                    socket_connect_timeout=5,
                )
                r.ping()
                self.logger.info(f"✓ Redis is accessible at {redis_host}:{redis_port}")
                return True
            except Exception as e:
                if attempt < self.max_retries - 1:
                    self.logger.debug(f"Attempt {attempt + 1}/{self.max_retries} failed: {e}")
                    time.sleep(self.retry_delay)
                else:
                    self.logger.error(f"✗ Failed to connect to Redis: {e}")
                    return False

    def check_filesystem(self) -> bool:
        """Check filesystem requirements."""
        self.logger.info("Checking filesystem...")

        required_dirs = [
            "logs",
            "models/artifacts",
            "config",
        ]

        for dir_path in required_dirs:
            full_path = Path(dir_path)
            full_path.mkdir(parents=True, exist_ok=True)
            if full_path.exists():
                self.logger.info(f"✓ Directory '{dir_path}' is ready")
            else:
                self.logger.error(f"✗ Failed to create directory '{dir_path}'")
                return False

        return True


# ============================================================================
# INITIALIZATION TASKS
# ============================================================================


class InitializationTasks:
    """Performs initialization tasks for each service."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    async def init_database_migrations(self) -> bool:
        """Run any pending database migrations (placeholder for Alembic)."""
        self.logger.info("Running database migrations...")

        # Note: In production, this would use Alembic
        # For now, schema is loaded via schema.sql in docker-entrypoint
        try:
            db_url = os.getenv("DATABASE_URL")
            import psycopg2

            # Parse connection string
            db_url_parts = db_url.replace("postgresql://", "")
            creds, location = db_url_parts.split("@")
            user, password = creds.split(":")
            host, rest = location.split(":")
            port, database = rest.split("/")

            # Check if migrations have been applied
            conn = await asyncpg.connect(
                host=host,
                port=int(port),
                user=user,
                password=password,
                database=database,
            )

            # Simple check: verify key tables exist
            tables_exist = await conn.fetch(
                """
                SELECT COUNT(*) as table_count
                FROM information_schema.tables
                WHERE table_schema = 'public'
                """
            )

            table_count = tables_exist[0]["table_count"] if tables_exist else 0

            if table_count > 0:
                self.logger.info(f"✓ Found {table_count} tables in database")
            else:
                self.logger.warning("No tables found - schema may not be initialized")

            await conn.close()
            return True

        except Exception as e:
            self.logger.error(f"✗ Migration check failed: {e}")
            return False

    async def init_cache_setup(self) -> bool:
        """Initialize cache with configuration."""
        self.logger.info("Setting up cache...")

        try:
            redis_host = os.getenv("REDIS_HOST", "localhost")
            redis_port = int(os.getenv("REDIS_PORT", "6379"))
            redis_password = os.getenv("REDIS_PASSWORD")

            r = redis.Redis(
                host=redis_host,
                port=redis_port,
                password=redis_password,
                decode_responses=True,
            )

            # Check if cache is already initialized
            if r.get("cache:initialized"):
                self.logger.info("✓ Cache already initialized")
                return True

            # Initialize cache metadata
            cache_metadata = {
                "initialized_at": datetime.utcnow().isoformat(),
                "version": "1.0",
                "service": os.getenv("SERVICE_NAME", "unknown"),
            }

            r.set("cache:metadata", json.dumps(cache_metadata), ex=86400)
            r.set("cache:initialized", "1", ex=86400)

            self.logger.info("✓ Cache initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"✗ Cache initialization failed: {e}")
            return False

    async def init_models_directory(self) -> bool:
        """Initialize models directory structure."""
        self.logger.info("Setting up models directory...")

        try:
            models_dir = Path(os.getenv("MODEL_ARTIFACTS_DIR", "./models/artifacts"))
            models_dir.mkdir(parents=True, exist_ok=True)

            # Create subdirectories for each model
            for model_dir in ["module_a", "module_b"]:
                (models_dir / model_dir).mkdir(exist_ok=True)

            # Create metadata file
            metadata = {
                "initialized_at": datetime.utcnow().isoformat(),
                "models": {
                    "module_a": {"versions": []},
                    "module_b": {"versions": []},
                },
            }

            metadata_file = models_dir / "metadata.json"
            if not metadata_file.exists():
                with open(metadata_file, "w") as f:
                    json.dump(metadata, f, indent=2)

            self.logger.info(f"✓ Models directory ready at {models_dir}")
            return True

        except Exception as e:
            self.logger.error(f"✗ Models directory setup failed: {e}")
            return False


# ============================================================================
# MAIN STARTUP ORCHESTRATION
# ============================================================================


async def main():
    """Main startup orchestration."""
    service_name = os.getenv("SERVICE_NAME", "unknown")
    logger = setup_logging(service_name)

    logger.info(f"Starting Gridlock 2.0 Initialization - Service: {service_name}")
    logger.info(f"Environment: {os.getenv('ENVIRONMENT', 'development')}")
    logger.info(f"Timestamp: {datetime.utcnow().isoformat()}Z")

    # Step 1: Validate environment
    validator = EnvironmentValidator(logger)
    if not validator.validate():
        logger.error("Environment validation failed")
        return 1

    # Step 2: Check filesystem
    checker = HealthChecker(logger)
    if not checker.check_filesystem():
        logger.error("Filesystem check failed")
        return 1

    # Step 3: Check external dependencies
    logger.info("=== Checking Dependencies ===")

    if not checker.check_redis():
        logger.warning("Redis check failed, but continuing...")

    if not await checker.check_database():
        logger.warning("Database check failed, but continuing...")

    # Step 4: Run initialization tasks
    logger.info("=== Running Initialization Tasks ===")
    tasks = InitializationTasks(logger)

    init_results = {
        "database_migrations": await tasks.init_database_migrations(),
        "cache_setup": await tasks.init_cache_setup(),
        "models_directory": await tasks.init_models_directory(),
    }

    # Step 5: Summary
    logger.info("=== Initialization Summary ===")
    for task_name, result in init_results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{status} - {task_name}")

    if all(init_results.values()):
        logger.info("✓ All initialization tasks completed successfully")
        return 0
    else:
        logger.warning("Some initialization tasks failed, but continuing...")
        return 0  # Don't fail, allow service to attempt startup


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
