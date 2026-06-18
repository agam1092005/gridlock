import asyncio
import time
from datetime import datetime, timezone
import logging
import numpy as np
from typing import Optional

from ..api.schemas import PredictionResponse, ComponentLatencies
from .playbook import PlaybookEngine
from .shap_explainer import SHAPExplainer
from .latency_monitor import LatencyMonitor
from ..api.websocket import ws_manager
from ..monitoring.metrics import metrics_registry
from ..monitoring.logger import audit_logger

# Import the actual ML models and pipelines
from ..models.module_a.lightgbm_models import LightGBMSeverityModel, LightGBMDurationModel
from ..models.module_b.inference import ModuleBPredictor
from ..data_pipeline.embedding_engine import EmbeddingEngine
from ..data_pipeline.news_fetcher import NewsFetcher

logger = logging.getLogger("orchestrator")

# Global monitor to track cross-request latency degradation
global_latency_monitor = LatencyMonitor()


class PredictionOrchestrator:
    def __init__(self):
        self.model_dir = "models/artifacts/module_a/v1.0"

        # Initialize LightGBM models
        self.lgb_sev = LightGBMSeverityModel()
        self.lgb_dur = LightGBMDurationModel()
        try:
            self.lgb_sev.load(self.model_dir)
            self.lgb_dur.load(self.model_dir)
            logger.info("Successfully loaded Module A ML models in PredictionOrchestrator.")
        except Exception as e:
            logger.error(f"Failed to load Module A ML models: {e}")

        # Initialize Embedding Engine (IndicBERT)
        self.embedder: Optional[EmbeddingEngine] = None
        try:
            self.embedder = EmbeddingEngine()
            logger.info("Successfully loaded EmbeddingEngine in PredictionOrchestrator.")
        except Exception as e:
            logger.error(f"Failed to load EmbeddingEngine in PredictionOrchestrator: {e}")

        # Initialize Module B Predictor (STGCN)
        self.module_b: Optional[ModuleBPredictor] = None
        try:
            self.module_b = ModuleBPredictor(model_path="models/artifacts/module_b/stgcn_model.pth")
            logger.info("Successfully loaded Module B STGCN model in PredictionOrchestrator.")
        except Exception as e:
            logger.error(f"Failed to load Module B STGCN model in PredictionOrchestrator: {e}")
            self.module_b = None

        self.playbook_engine = PlaybookEngine()
        self.shap_explainer = SHAPExplainer(mod_a_sev=self.lgb_sev, mod_a_dur=self.lgb_dur)
        self.news_fetcher = NewsFetcher()

    def _extract_features(self, context):
        """Extract features and generate NLP embeddings from incident context."""
        metadata = context.get("metadata") or {}
        location = context.get("location") or {}

        if isinstance(location, dict):
            lat = metadata.get("latitude") or location.get("latitude", 0.0)
            lon = metadata.get("longitude") or location.get("longitude", 0.0)
        else:
            lat = metadata.get("latitude") or getattr(location, "latitude", 0.0)
            lon = metadata.get("longitude") or getattr(location, "longitude", 0.0)

        priority_str = str(metadata.get("priority", "medium")).lower()
        priority_numeric = 2 if priority_str == "high" else 1 if priority_str == "medium" else 0

        cause = str(metadata.get("event_cause", context.get("incident_type", ""))).lower()
        is_construction = 1.0 if cause == "construction" else 0.0
        is_event = (
            1.0
            if cause in ["public_event", "procession", "protest", "vip_movement", "planned"]
            else 0.0
        )

        veh_type = str(metadata.get("veh_type", context.get("veh_type", ""))).lower()
        is_heavy_vehicle = (
            1.0 if veh_type in ["heavy_vehicle", "truck", "bmtc_bus", "ksrtc_bus"] else 0.0
        )
        is_lcv = 1.0 if veh_type in ["lcv", "private_bus"] else 0.0

        corridor = str(metadata.get("corridor", context.get("corridor", ""))).lower()
        is_major_corridor = 1.0 if any(c in corridor for c in ["orr", "cbd", "tumkur"]) else 0.0

        # 8 numeric features
        # fmt: off
        numeric_feats = np.array([
            float(lat), float(lon), priority_numeric, is_construction,
            is_event, is_heavy_vehicle, is_lcv, is_major_corridor
        ])
        # fmt: on

        # Real NLP description embeddings
        text_desc = context.get("description", context.get("incident_type", ""))
        if self.embedder:
            try:
                real_embed = self.embedder.embed([text_desc])[0]
            except Exception as e:
                logger.error(f"Embedding generation failed: {e}")
                real_embed = np.zeros(768)
        else:
            real_embed = np.zeros(768)

        return np.concatenate([numeric_feats, real_embed]).reshape(1, -1)

    async def _call_module_a(self, context):
        """Inference of Severity and Duration models run in a threadpool."""

        def run_inference():
            try:
                matched_news = self.news_fetcher.check_for_active_keywords(
                    incident_data=context,
                    keywords=["rain", "protest"]
                )
                active_weather_alert = len(matched_news) > 0

                X_input = self._extract_features(context)
                sev_pred = self.lgb_sev.predict(X_input)
                dur_pred = self.lgb_dur.predict(X_input, active_weather_alert=active_weather_alert)

                raw_est = dur_pred.get("estimate", [np.log1p(30)])[0]

                severity_score = float(np.clip(sev_pred.get("score", [50])[0], 0, 100))

                # Check if end_datetime is missing
                end_dt = context.get("end_datetime")
                is_missing_end = end_dt is None or str(end_dt).lower() in ["none", "", "nat"]

                if is_missing_end:
                    from .survival_model import (
                        SurvivalModelSingleton,
                        INCIDENT_TYPE_MAP,
                        PRIORITY_MAP,
                        HISTORICAL_MEDIANS,
                    )

                    inc_type_str = str(context.get("incident_type") or "event").lower()
                    priority_str = str(context.get("metadata", {}).get("priority") or "low").lower()

                    if inc_type_str not in INCIDENT_TYPE_MAP or priority_str not in PRIORITY_MAP:
                        duration_estimate = HISTORICAL_MEDIANS.get(
                            inc_type_str, HISTORICAL_MEDIANS["generic_fallback"]
                        )
                        logger.warning(
                            f"Novel feature encountered (type: {inc_type_str}, priority: {priority_str}). Falling back to historical median: {duration_estimate}"
                        )
                    else:
                        incident_type_val = INCIDENT_TYPE_MAP[inc_type_str]
                        priority_val = PRIORITY_MAP[priority_str]

                        ts = context.get("timestamp")
                        if isinstance(ts, str):
                            try:
                                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                            except Exception:
                                dt = datetime.now(timezone.utc)
                        elif isinstance(ts, datetime):
                            dt = ts
                        else:
                            dt = datetime.now(timezone.utc)
                        hour_of_day_val = float(dt.hour)

                        try:
                            cph = SurvivalModelSingleton.get_model()
                            if cph is not None:
                                import pandas as pd

                                covariates_df = pd.DataFrame(
                                    [
                                        {
                                            "incident_type": incident_type_val,
                                            "priority": priority_val,
                                            "hour_of_day": hour_of_day_val,
                                        }
                                    ]
                                )
                                duration_estimate = float(
                                    cph.predict_expectation(covariates_df).values[0]
                                )
                                logger.info(
                                    f"Calculated survival duration expectation: {duration_estimate:.2f} mins"
                                )
                            else:
                                duration_estimate = HISTORICAL_MEDIANS.get(
                                    inc_type_str, HISTORICAL_MEDIANS["generic_fallback"]
                                )
                                logger.warning(
                                    f"Cox model not loaded, using fallback median for {inc_type_str}: {duration_estimate}"
                                )
                        except Exception as e:
                            duration_estimate = HISTORICAL_MEDIANS.get(
                                inc_type_str, HISTORICAL_MEDIANS["generic_fallback"]
                            )
                            logger.error(
                                f"Survival duration expectation failed, falling back to median: {e}"
                            )
                else:
                    duration_estimate = float(max(0, np.expm1(raw_est)))

                return {
                    "severity_score": severity_score,
                    "duration_estimate": duration_estimate,
                    "model_inputs": X_input,
                }
            except Exception as e:
                logger.error(f"Module A inference failed: {e}")
                return {"severity_score": 50.0, "duration_estimate": 30.0, "model_inputs": None}

        res = await asyncio.to_thread(run_inference)

        # Inject results back to context so subsequent stages can read them
        context["severity_score"] = res["severity_score"]
        context["duration_estimate"] = res["duration_estimate"]
        context["_model_inputs"] = res["model_inputs"]

        return {
            "severity_score": res["severity_score"],
            "duration_estimate": res["duration_estimate"],
        }

    async def _call_module_b(self, context):
        """Inference of STGCN model for localized congestion ripple effect."""
        if not self.module_b:
            return {"geojson": {"type": "FeatureCollection", "features": []}}

        def run_module_b():
            try:
                metadata = context.get("metadata") or {}
                location = context.get("location") or {}

                if isinstance(location, dict):
                    lat = metadata.get("latitude") or location.get("latitude", 0.0)
                    lon = metadata.get("longitude") or location.get("longitude", 0.0)
                else:
                    lat = metadata.get("latitude") or getattr(location, "latitude", 0.0)
                    lon = metadata.get("longitude") or getattr(location, "longitude", 0.0)

                mod_b_context = {
                    "latitude": lat,
                    "longitude": lon,
                    "severity_score": context.get("severity_score", 50.0),
                }
                res = self.module_b.predict(mod_b_context)
                return res.get("geojson")
            except Exception as e:
                logger.error(f"Module B prediction failed: {e}")
                return {"type": "FeatureCollection", "features": []}

        geojson = await asyncio.to_thread(run_module_b)
        return {"geojson": geojson}

    async def _call_playbook(self, context):
        """Lookup playbook recommendations based on incident type and severity."""
        return self.playbook_engine.generate_playbook(context)

    async def _call_shap(self, context):
        """Compute SHAP feature attributions in a threadpool."""
        model_inputs = context.get("_model_inputs")
        if model_inputs is None:
            return {"enabled": False}

        def run_shap():
            try:
                return self.shap_explainer.compute_explanations(context, model_inputs)
            except Exception as e:
                logger.error(f"SHAP explainer run failed: {e}")
                return {"enabled": False}

        return await asyncio.to_thread(run_shap)

    async def run_pipeline(self, context) -> PredictionResponse:
        latencies = ComponentLatencies()
        start_time = time.time()

        # 1. Module A (Sequential)
        t0 = time.time()
        try:
            res_a = await asyncio.wait_for(
                self._call_module_a(context), timeout=1.0
            )  # slightly relaxed timeout for embedding engine lookup
        except asyncio.TimeoutError:
            logger.warning("Module A execution timed out. Falling back to default baseline values.")
            res_a = {"severity_score": 50.0, "duration_estimate": 30.0}
            context["severity_score"] = 50.0
            context["duration_estimate"] = 30.0
            context["_model_inputs"] = None
        latencies.module_a = (time.time() - t0) * 1000

        # 2. Module B & Playbook (Concurrent)
        t0 = time.time()
        res_b, res_play = await asyncio.gather(
            self._call_module_b(context), self._call_playbook(context), return_exceptions=True
        )

        # Handle graceful degradation
        if isinstance(res_b, BaseException):
            logger.error(f"Module B failed: {res_b}")
            res_b = {"geojson": {"type": "FeatureCollection", "features": []}}

        if isinstance(res_play, BaseException):
            logger.error(f"Playbook failed: {res_play}")
            res_play = []

        latencies.module_b = (time.time() - t0) * 1000  # Parallel max time
        latencies.playbook = latencies.module_b

        # 3. SHAP Explanations
        t0 = time.time()
        try:
            res_shap = await asyncio.wait_for(self._call_shap(context), timeout=0.5)
        except Exception as e:
            logger.error(f"SHAP failed: {e}")
            res_shap = {"enabled": False}
        latencies.shap = (time.time() - t0) * 1000

        total_latency = (time.time() - start_time) * 1000
        global_latency_monitor.record(total_latency)

        predictions_merged = {
            "module_a": res_a,
            "module_b": res_b
            if isinstance(res_b, dict)
            else {"geojson": {"type": "FeatureCollection", "features": []}},
        }

        # Extract type-safe playbook actions list
        playbook_actions = []
        if isinstance(res_play, dict):
            playbook_actions = res_play.get("actions", [])
        elif isinstance(res_play, list):
            playbook_actions = res_play

        response = PredictionResponse(
            incident_id=context.get("incident_id", "unknown"),
            submission_time=datetime.now(timezone.utc),
            prediction_time=datetime.now(timezone.utc),
            predictions=predictions_merged,
            playbook=playbook_actions,
            explanations=res_shap,
            latency_ms=total_latency,
            component_latencies_ms=latencies,
        )

        # Asynchronously broadcast to any connected websocket clients
        asyncio.create_task(
            ws_manager.broadcast(
                {
                    "type": "incident_update",
                    "incident_id": response.incident_id,
                    "location": context.get("location"),
                    "description": context.get("description"),
                    "metadata": context.get("metadata"),
                    "severity_score": res_a.get("severity_score", 50.0),
                    "duration_estimate": res_a.get("duration_estimate", 30.0),
                    "incident_type": context.get("incident_type", "unknown"),
                    "module_b_geojson": res_b.get("geojson") if isinstance(res_b, dict) else {},
                    "explanations": res_shap,
                    "playbook": playbook_actions,
                    "api_process_time_ms": total_latency,
                }
            )
        )

        # Telemetry updates
        metrics_registry.inc_counter(
            "predictions_total", f"type=\"{context.get('incident_type', 'unknown')}\""
        )
        metrics_registry.observe_histogram("prediction_latency_ms", total_latency)

        audit_logger.log_operation(
            user="system",
            operation_type="prediction_generated",
            details={
                "incident_id": response.incident_id,
                "latency_ms": total_latency,
                "location": context.get("location", {}),
            },
        )

        return response
