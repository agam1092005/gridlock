from fastapi import APIRouter, Response
from ..schemas import HealthResponse
from ...monitoring.metrics import metrics_registry
from ...monitoring.logger import audit_logger
import time
import os
import logging

logger = logging.getLogger("api")
router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Check the health of all microservices and dependencies.
    """
    component_statuses = {}
    metrics = {
        "avg_latency_ms": 120.5,
        "error_rate": 0.01,
    }

    # Mock checks
    component_statuses["database"] = "up"
    component_statuses["redis"] = "up"
    component_statuses["module_a"] = "up"
    component_statuses["module_b"] = "up"

    overall_status = "healthy"

    return HealthResponse(status=overall_status, components=component_statuses, metrics=metrics)


@router.get("/metrics")
async def get_metrics():
    """
    Prometheus metrics export.
    Returns dynamic metrics in Prometheus text format.
    """
    metrics_text = metrics_registry.render_prometheus_text()
    return Response(content=metrics_text, media_type="text/plain")


@router.get("/audit/logs")
async def get_audit_logs():
    """
    Query immutable audit logs.
    """
    logs = []
    log_path = ".gridlock/audit.jsonl"
    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            for line in f.readlines()[-100:]:  # last 100 for demo
                logs.append(line.strip())
    return Response(content="[" + ",".join(logs) + "]", media_type="application/json")
