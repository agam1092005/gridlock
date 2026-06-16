# Gridlock 2.0 🚦

An intelligent traffic incident management system capable of processing real-time telemetry to predict incident severities, durations, and secondary congestion ripples using a high-performance ensemble of LightGBM, BiGRU, and Spatial-Temporal Graph Convolutional Networks (STGCN).

## Getting Started

1. **Install Dependencies:**
   Ensure you have Python 3.10+ installed.
   ```bash
   poetry install
   ```
2. **Start the API:**
   ```bash
   poetry run uvicorn src.api.main:app --reload
   ```
3. **Start the Dashboard:**
   In a separate terminal:
   ```bash
   poetry run streamlit run src/dashboard/app.py
   ```

## Key Capabilities
- **Sub-150ms Predictions:** Achieved via an asynchronous `PredictionOrchestrator` that intelligently routes to the cache or inference engines.
- **WebSocket Streaming:** A native React/Streamlit dashboard that automatically receives sub-50ms push updates for active incidents.
- **MLOps Integrated:** Ships with built-in versioning (`ModelRegistry`), training orchestration, and a `ProductionModelMonitor` to track data drift.
- **Production Telemetry:** Emits structured JSON logs and dynamic `/metrics` for Prometheus scraping.
