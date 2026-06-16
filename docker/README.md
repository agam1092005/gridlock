# Docker Directory - Gridlock 2.0 Containerization

This directory contains all Docker-related files for containerizing and deploying Gridlock 2.0.

## Files Overview

### Dockerfiles (2 files)

#### `Dockerfile.api`
- **Base Image:** python:3.11-slim
- **Purpose:** FastAPI backend service
- **Size:** ~1.5GB
- **Ports:** 8000 (API), 9090 (Prometheus metrics)
- **Startup:** Uses `/app/docker/docker-entrypoint.sh` with SERVICE_NAME=api
- **Includes:** PostgreSQL client, Redis CLI, netcat utilities

#### `Dockerfile.dashboard`
- **Base Image:** python:3.11-slim
- **Purpose:** Streamlit real-time dashboard
- **Size:** ~1.2GB
- **Port:** 8501
- **Startup:** Uses `/app/docker/docker-entrypoint.sh` with SERVICE_NAME=dashboard
- **Includes:** Network utilities, Streamlit framework

### Startup Scripts (3 files)

#### `docker-entrypoint.sh`
- **Type:** Bash shell script
- **Purpose:** Main entry point for all containers
- **Size:** ~600 lines
- **Responsibilities:**
  - Route to correct service startup (api, dashboard, migration)
  - Initialize PostgreSQL connection and apply schema
  - Initialize Redis cache
  - Manage service dependencies
  - Handle graceful shutdown (SIGTERM/SIGINT)
  - Provide comprehensive logging
- **Usage:** `ENTRYPOINT ["/app/docker/docker-entrypoint.sh"]`

#### `startup.py`
- **Type:** Python asyncio script
- **Purpose:** Async initialization and validation
- **Size:** ~500 lines
- **Responsibilities:**
  - Validate environment variables
  - Check service dependencies asynchronously
  - Verify database connectivity
  - Initialize cache structures
  - Create models directory
  - Log all operations with structured format
- **Called by:** `docker-entrypoint.sh`

#### `init-db.sh`
- **Type:** Bash shell script
- **Purpose:** PostgreSQL post-schema initialization
- **Called by:** PostgreSQL Docker entrypoint (Phase 2)
- **Responsibilities:**
  - Create PostgreSQL extensions (PostGIS, pgvector, pg_trgm, uuid-ossp, etc.)
  - Create performance indexes
  - Verify schema installation
  - Grant database permissions

### Configuration Files (2 files)

#### `prometheus.yml`
- **Purpose:** Prometheus metrics scraping configuration
- **Size:** ~40 lines
- **Scrapers:**
  - Prometheus self-monitoring
  - Gridlock API metrics (/metrics endpoint)
  - PostgreSQL metrics (if exporter deployed)
  - Redis metrics (if exporter deployed)
- **Intervals:** API: 5s, Others: 15s
- **Retention:** 30 days
- **Mount in docker-compose:** `/etc/prometheus/prometheus.yml`

#### `grafana-datasources.yml`
- **Purpose:** Grafana auto-provisioned datasources
- **Size:** ~30 lines
- **Primary Datasource:** Prometheus at http://prometheus:9090
- **Optional:** Loki, Elasticsearch
- **Mount in docker-compose:** `/etc/grafana/provisioning/datasources/datasources.yml`

## Build Instructions

### Build Individual Images

```bash
# Build API image
docker build -f docker/Dockerfile.api -t gridlock:api:latest .

# Build Dashboard image
docker build -f docker/Dockerfile.dashboard -t gridlock:dashboard:latest .
```

### Build with Docker Compose

```bash
# Build all images
docker-compose build

# Build specific service
docker-compose build api
docker-compose build dashboard

# Force rebuild (no cache)
docker-compose build --no-cache
```

## Directory Structure

```
docker/
├── Dockerfile.api                 # FastAPI backend container
├── Dockerfile.dashboard           # Streamlit dashboard container
├── docker-entrypoint.sh          # Main entry point script
├── startup.py                     # Python initialization
├── init-db.sh                     # PostgreSQL setup
├── prometheus.yml                 # Prometheus configuration
├── grafana-datasources.yml        # Grafana datasources
└── README.md                      # This file
```

## Service Startup Flow

```
docker-compose up -d
    ↓
PostgreSQL starts (healthcheck: pg_isready)
    ├─ Schema applied (schema.sql)
    └─ Extensions created (init-db.sh)
    ↓
Redis starts (healthcheck: PING)
    ├─ Persistence enabled
    └─ Memory limits set
    ↓
db-migrate runs (one-time service)
    ├─ Validates dependencies
    ├─ Applies migrations
    └─ Exits on completion
    ↓
API service starts (SERVICE_NAME=api)
    ├─ Validates environment
    ├─ Connects to PostgreSQL
    ├─ Initializes Redis
    ├─ Loads ML models
    ├─ Starts uvicorn (4 workers)
    └─ Healthcheck: curl /health
    ↓
Dashboard service starts (SERVICE_NAME=dashboard)
    ├─ Waits for API healthy
    ├─ Connects to databases
    ├─ Starts Streamlit server
    └─ Healthcheck: curl /_stcore/health
```

## Environment Variables

### Startup Configuration

```bash
SERVICE_NAME=api|dashboard|migration  # Determines startup behavior
LOG_LEVEL=debug|info|warning|error    # Logging verbosity
ENVIRONMENT=development|production     # Environment mode
DEBUG=true|false                       # Debug flag
```

### Database Configuration

```bash
DATABASE_URL=postgresql://gridlock_user:gridlock_password@postgres:5432/gridlock
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=40
```

### Cache Configuration

```bash
REDIS_HOST=redis
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=
```

### API Configuration

```bash
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4
```

### ML Models Configuration

```bash
MODEL_ARTIFACTS_DIR=./models/artifacts
EMBEDDING_CACHE_TTL_SECONDS=86400
PREDICTION_CACHE_TTL_SECONDS=604800
```

## Health Checks

All containers have health checks configured:

```yaml
healthcheck:
  test: [curl, -f, http://localhost:8000/health]  # or service-specific endpoint
  interval: 30s      # Check every 30 seconds
  timeout: 10s       # Timeout after 10 seconds
  retries: 3         # Mark unhealthy after 3 failures
  start_period: 60s  # Grace period before first check
```

## Logging

All services log to:
- **Console:** Direct STDOUT (colored, human-readable)
- **File:** `./logs/{service}_startup.log` (JSON structured)

Logging Configuration:
```yaml
logging:
  driver: json-file
  options:
    max-size: 10m
    max-file: 3
```

## Docker Compose Integration

Services are defined in `../docker-compose.yml`:

```yaml
services:
  postgres:       # PostgreSQL 15 Alpine
  redis:          # Redis 7 Alpine
  db-migrate:     # One-time migration
  api:            # FastAPI backend
  dashboard:      # Streamlit frontend
  prometheus:     # Metrics (optional, profile: monitoring)
  grafana:        # Visualization (optional, profile: monitoring)
```

## Volume Mounts

### Data Persistence

```yaml
volumes:
  postgres_data: data/postgres/        # Database files
  redis_data: data/redis/              # Cache snapshot
  prometheus_data: data/prometheus/    # Metrics
  grafana_data: data/grafana/          # Dashboards
```

### Source Code (Development)

```yaml
volumes:
  - ./src:/app/src                     # Hot reload
  - ./config:/app/config
  - ./models:/app/models
  - ./logs:/app/logs
```

## Troubleshooting

### Container won't start

```bash
# View detailed logs
docker-compose logs {service_name}

# Check environment variables
docker-compose config | grep -A20 "{service_name}:"

# Rebuild image
docker-compose build --no-cache {service_name}
```

### Health check failing

```bash
# Check service health directly
curl http://localhost:{PORT}/health

# Inspect container
docker inspect gridlock_{service_name}

# Run healthcheck manually
docker-compose exec {service_name} {healthcheck_command}
```

### Network issues

```bash
# Check network connectivity
docker network ls | grep gridlock
docker network inspect gridlock_gridlock_network

# Test DNS resolution
docker-compose exec {service_name} nslookup postgres
```

## Production Deployment

### Pre-deployment Checklist

- [ ] Update .env with production values
- [ ] Change PostgreSQL password
- [ ] Enable Redis authentication
- [ ] Configure TLS/SSL
- [ ] Set resource limits
- [ ] Enable monitoring (--profile monitoring)
- [ ] Configure log rotation
- [ ] Test backup/restore procedure

### Resource Requirements

| Service | CPU | Memory | Storage |
|---------|-----|--------|---------|
| API | 2 cores | 2GB | 1GB |
| Dashboard | 1 core | 1GB | 500MB |
| PostgreSQL | 4 cores | 4GB | 50GB |
| Redis | 2 cores | 2GB | 10GB |
| Prometheus | 1 core | 1GB | 20GB |
| **Total** | **10 cores** | **10GB** | **81.5GB** |

## Support & Documentation

- Full Docker guide: See `../DOCKER.md`
- Implementation summary: See `../DOCKER_IMPLEMENTATION_SUMMARY.md`
- System architecture: See `../design.md`
- Requirements: See `../requirements.md`

## License & Attribution

Gridlock 2.0 - Real-time Incident Severity and Congestion Prediction System
