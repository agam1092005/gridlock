# API Documentation

The Gridlock 2.0 Backend exposes both REST and WebSocket endpoints.

## REST Endpoints

### 1. Submit Incident
**`POST /api/incidents`**
Accepts a raw traffic incident description and processes the entire prediction pipeline asynchronously.
- **Payload:** `{"incident_id": "str", "timestamp": "ISO", "description": "str", "location": {"lat": float, "lng": float}, "incident_type": "str"}`
- **Response:** `202 Accepted` with a tracking ID.

### 2. Get Prediction
**`GET /api/predictions/{incident_id}`**
Retrieves the finalized prediction payload, including severity score, duration estimate, congestion heatmap, and the tailored playbook.

### 3. System Metrics
**`GET /api/metrics`**
Returns live system telemetry (latencies, counts) in Prometheus exposition format.

### 4. Audit Logs
**`GET /api/audit/logs`**
Returns the redacted transaction logs for compliance checks.

## WebSocket Endpoints

### Live Dashboard Feed
**`WS /api/ws/live`**
A low-latency channel that broadcasts finalized `PredictionResponse` JSON objects the moment the orchestrator finishes processing them. Connect via standard `websockets` library.
