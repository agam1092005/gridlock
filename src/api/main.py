from fastapi import FastAPI, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from contextlib import asynccontextmanager
import logging

from .middleware import TimingAndLoggingMiddleware
from .routers import incidents, predictions, system, ws
from ..utils.errors import GridlockException, ValidationError
from ..monitoring.logger import setup_structured_logging
import asyncio
from ..config.settings import settings
from .routers.ws import ws_manager

setup_structured_logging()
logger = logging.getLogger("api")

from .queue_worker import BackgroundWorker


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup actions
    logger.info(f"Starting Gridlock 2.0 API on port {settings.api.port}...")
    from ..orchestration.survival_model import SurvivalModelSingleton

    SurvivalModelSingleton.load_model()
    worker = BackgroundWorker(incidents._mock_queue)
    worker_task = asyncio.create_task(worker.run())

    yield
    # Shutdown actions
    logger.info("Graceful shutdown initiated. Waiting for in-flight requests (30s timeout)...")
    worker.stop()
    try:
        await asyncio.wait_for(worker_task, timeout=5.0)
    except asyncio.TimeoutError:
        logger.warning("Timeout waiting for background worker to stop.")
    logger.info("Graceful shutdown initiated. Waiting for in-flight requests (30s timeout)...")
    try:
        await asyncio.wait_for(ws_manager.disconnect_all(), timeout=30.0)
    except asyncio.TimeoutError:
        logger.warning("Timeout waiting for websockets to close.")
    logger.info("Flushing cache and finalizing shutdown...")


app = FastAPI(
    title="Gridlock 2.0 API",
    description="Intelligent Traffic Incident Management System",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins
    if hasattr(settings, "cors_origins")
    else ["http://localhost:3000", "http://localhost:8501"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Custom Middleware
app.add_middleware(TimingAndLoggingMiddleware)


# Exception Handlers
@app.exception_handler(GridlockException)
async def gridlock_exception_handler(request, exc: GridlockException):
    """Handle Gridlock-specific exceptions."""
    logger.error(f"Gridlock exception: {exc.error_type.value} - {exc.message}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": exc.message,
            "error_type": exc.error_type.value,
            "context": exc.context,
            "request_id": getattr(request.state, "request_id", "unknown"),
        },
    )


@app.exception_handler(ValidationError)
async def validation_error_handler(request, exc: ValidationError):
    """Handle validation errors."""
    logger.error(f"Validation error: {exc.message}")
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "detail": exc.message,
            "error_type": exc.error_type.value,
            "context": exc.context,
            "request_id": getattr(request.state, "request_id", "unknown"),
        },
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc: RequestValidationError):
    """Handle Pydantic validation errors."""
    logger.error(f"Request validation error: {exc.errors()}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "detail": "Request validation failed",
            "errors": exc.errors(),
            "request_id": getattr(request.state, "request_id", "unknown"),
        },
    )


# Routers
app.include_router(incidents.router, prefix="/api")
app.include_router(predictions.router, prefix="/api")
app.include_router(system.router, prefix="/api")
app.include_router(ws.router)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
