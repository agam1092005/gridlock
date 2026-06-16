# System Architecture

Gridlock 2.0 adopts a decoupled, event-driven microservices architecture optimized for low-latency incident response.

## Core Modules

### 1. Data Pipeline
Normalizes raw textual descriptions, parses spatial data, and standardizes numerical fields.

### 2. Module A: Severity & Duration Prediction (LightGBM + BiGRU)
We use a sequential ensemble:
- **LightGBM** rapidly calculates the initial baseline duration and severity score from categorical/tabular features.
- **BiGRU** handles sequential textual descriptions (NLP-encoded sequences) for deeper contextual adjustment.

### 3. Module B: Congestion Prediction (STGCN)
- Spatial-Temporal Graph Convolutional Networks predict ripple effects across the spatial graph radius (`GRAPH_SPATIAL_RADIUS_KM` from config). 

### 4. Orchestrator
The `PredictionOrchestrator` uses Python's `asyncio.gather` to run Modules A and B concurrently, enforcing a strict `<500ms` SLA via the `LatencyMonitor`. If inference breaches the threshold, it degrades gracefully to returning cached outputs.

### 5. Config & Playbooks
Extensible YAML-based configuration enables operators to hot-swap incident rules (`playbooks.yaml`) and tweak system thresholds without touching the Python code.
