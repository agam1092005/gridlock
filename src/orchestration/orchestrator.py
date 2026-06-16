import asyncio
import time
from datetime import datetime
import logging
from ..api.schemas import PredictionResponse, ComponentLatencies

from .playbook import PlaybookEngine
from .shap_explainer import SHAPExplainer
from .latency_monitor import LatencyMonitor
from ..api.websocket import ws_manager
from ..monitoring.metrics import metrics_registry
from ..monitoring.logger import audit_logger

logger = logging.getLogger("orchestrator")

# Global monitor to track cross-request latency degradation
global_latency_monitor = LatencyMonitor()

class PredictionOrchestrator:
    def __init__(self):
        # In a real setup, instantiate the predictor classes
        # self.mod_a = ModuleAPredictor(...)
        # self.mod_b = ModuleBPredictor(...)
        self.playbook_engine = PlaybookEngine()
        self.shap_explainer = SHAPExplainer()

    async def _call_module_a(self, context):
        await asyncio.sleep(0.05) # mock 50ms latency
        return {"severity_score": 75, "duration_estimate": 45}

    async def _call_module_b(self, context):
        await asyncio.sleep(0.1) # mock 100ms latency
        return {"geojson": {"type": "FeatureCollection", "features": []}}

    async def _call_playbook(self, context):
        return self.playbook_engine.generate_playbook(context)

    async def _call_shap(self, context):
        return self.shap_explainer.compute_explanations(context)

    async def run_pipeline(self, context) -> PredictionResponse:
        latencies = ComponentLatencies()
        start_time = time.time()
        
        # 1. Module A (Sequential)
        t0 = time.time()
        res_a = await asyncio.wait_for(self._call_module_a(context), timeout=0.15)
        latencies.module_a = (time.time() - t0) * 1000

        # 2. Module B & Playbook (Concurrent)
        t0 = time.time()
        res_b, res_play = await asyncio.gather(
            self._call_module_b(context),
            self._call_playbook(context),
            return_exceptions=True
        )
        
        # Handle graceful degradation
        if isinstance(res_b, Exception):
            logger.error(f"Module B failed: {res_b}")
            res_b = {"geojson": None}
            
        if isinstance(res_play, Exception):
            logger.error(f"Playbook failed: {res_play}")
            res_play = []
            
        latencies.module_b = (time.time() - t0) * 1000 # Parallel max time
        latencies.playbook = latencies.module_b

        # 3. SHAP Explanations
        t0 = time.time()
        try:
            res_shap = await asyncio.wait_for(self._call_shap(context), timeout=0.1)
        except Exception as e:
            logger.error(f"SHAP failed: {e}")
            res_shap = None
        latencies.shap = (time.time() - t0) * 1000

        total_latency = (time.time() - start_time) * 1000
        global_latency_monitor.record(total_latency)
        
        predictions_merged = {
            "module_a": res_a,
            "module_b": res_b
        }

        response = PredictionResponse(
            incident_id=context.get("incident_id", "unknown"),
            submission_time=datetime.utcnow(),
            prediction_time=datetime.utcnow(),
            predictions=predictions_merged,
            playbook=res_play,
            explanations=res_shap,
            latency_ms=total_latency,
            component_latencies_ms=latencies
        )
        
        # Asynchronously broadcast to any connected websocket clients
        # In a production environment, this might be sent via Redis pub/sub if running multiple Uvicorn workers
        asyncio.create_task(
            ws_manager.broadcast({
                "type": "incident_update",
                "incident_id": response.incident_id,
                "severity_score": res_a.get("severity_score", 0),
                "duration_estimate": res_a.get("duration_estimate", 0),
                "incident_type": context.get("incident_type", "unknown")
            })
        )
        
        # Telemetry updates
        metrics_registry.inc_counter("predictions_total", f"type=\"{context.get('incident_type', 'unknown')}\"")
        metrics_registry.observe_histogram("prediction_latency_ms", total_latency)
        
        audit_logger.log_operation(
            user="system",
            operation_type="prediction_generated",
            details={
                "incident_id": response.incident_id,
                "latency_ms": total_latency,
                "location": context.get("location", {})
            }
        )
        
        return response
