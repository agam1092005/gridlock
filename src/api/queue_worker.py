import asyncio
import logging
from ..orchestration.resilience import retry_with_backoff

logger = logging.getLogger("worker")

from ..models.module_a.lightgbm_models import LightGBMSeverityModel, LightGBMDurationModel
from ..models.module_b.inference import ModuleBPredictor
from ..data_pipeline.news_fetcher import NewsFetcher
from ..data_pipeline.embedding_engine import EmbeddingEngine
import numpy as np
import time
from typing import Optional

class BackgroundWorker:
    def __init__(self, queue):
        self.queue = queue
        self.is_running = False
        self.model_dir = "models/artifacts/module_a/v1.0"
        self.news_fetcher = NewsFetcher()
        
        self.lgb_sev = LightGBMSeverityModel()
        self.lgb_dur = LightGBMDurationModel()
        
        # Load Module B
        self.module_b = ModuleBPredictor(model_path="models/artifacts/module_b/stgcn_model.pth")
        
        # Load Embedding Engine
        self.embedder: Optional[EmbeddingEngine] = None
        try:
            self.embedder = EmbeddingEngine()
            logger.info("Successfully loaded IndicBERT EmbeddingEngine.")
        except Exception as e:
            logger.error(f"Failed to load EmbeddingEngine: {e}")
        
        try:
            self.lgb_sev.load(self.model_dir)
            self.lgb_dur.load(self.model_dir)
            logger.info("Successfully loaded ML models in background worker.")
        except Exception as e:
            logger.error(f"Failed to load ML models: {e}")
            
        from ..orchestration.shap_explainer import SHAPExplainer
        self.shap_explainer = SHAPExplainer(mod_a_sev=self.lgb_sev, mod_a_dur=self.lgb_dur)

    def process_incident(self, incident_data):
        logger.info(f"Processing incident: {incident_data.get('incident_id')}")
        
        # Real ML prediction flow
        start_time = time.time()
        
        # 1. Feature extraction
        metadata = incident_data.get("metadata", {})
        lat = metadata.get("latitude") or incident_data.get("location", {}).get("latitude", 0.0)
        lon = metadata.get("longitude") or incident_data.get("location", {}).get("longitude", 0.0)
        
        priority_str = str(metadata.get("priority", "medium")).lower()
        priority_numeric = 2 if priority_str == 'high' else 1 if priority_str == 'medium' else 0
        
        cause = str(metadata.get("event_cause", incident_data.get("incident_type", ""))).lower()
        is_construction = 1.0 if cause == 'construction' else 0.0
        is_event = 1.0 if cause in ['public_event', 'procession', 'protest', 'vip_movement', 'planned'] else 0.0
        
        veh_type = str(metadata.get("veh_type", incident_data.get("veh_type", ""))).lower()
        is_heavy_vehicle = 1.0 if veh_type in ['heavy_vehicle', 'truck', 'bmtc_bus', 'ksrtc_bus'] else 0.0
        is_lcv = 1.0 if veh_type in ['lcv', 'private_bus'] else 0.0
        
        corridor = str(metadata.get("corridor", incident_data.get("corridor", ""))).lower()
        is_major_corridor = 1.0 if any(c in corridor for c in ['orr', 'cbd', 'tumkur']) else 0.0
        
        # Numeric feats (lat, lon, priority, is_construction, is_event, is_heavy_vehicle, is_lcv, is_major_corridor)
        numeric_feats = np.array([float(lat), float(lon), priority_numeric, is_construction, is_event, is_heavy_vehicle, is_lcv, is_major_corridor])
        
        # Real NLP Embedding
        text_desc = incident_data.get("description", incident_data.get("incident_type", ""))
        if self.embedder:
            try:
                real_embed = self.embedder.embed([text_desc])[0]
            except Exception as e:
                logger.error(f"Embedding failed, falling back to zeros: {e}")
                real_embed = np.zeros(384)
        else:
            real_embed = np.zeros(384)
        
        X_input = np.concatenate([numeric_feats, real_embed]).reshape(1, -1)
        
        # 2. Predict
        try:
            sev_pred = self.lgb_sev.predict(X_input)
            dur_pred = self.lgb_dur.predict(X_input)
            
            # Apply np.expm1 because model predicts log1p(duration)
            raw_est = dur_pred.get("estimate", [np.log1p(30)])[0]
            
            incident_data["severity_score"] = float(np.clip(sev_pred.get("score", [50])[0], 0, 100))
            incident_data["severity_ci"] = [float(np.clip(sev_pred.get("ci_lower", [0])[0], 0, 100)), float(np.clip(sev_pred.get("ci_upper", [100])[0], 0, 100))]
            incident_data["duration_estimate"] = float(max(0, np.expm1(raw_est)))
            incident_data["duration_ci"] = [float(max(0, np.expm1(dur_pred.get("ci_lower", [0])[0]))), float(max(0, np.expm1(dur_pred.get("ci_upper", [100])[0])))]
            incident_data["predicted_by_ai"] = True
            
            # Compute real SHAP explanations
            incident_data["explanations"] = self.shap_explainer.compute_explanations(incident_data, X_input)
            
        except Exception as e:
            logger.error(f"Prediction failed, falling back: {e}")
            incident_data["severity_score"] = 50.0
            incident_data["duration_estimate"] = 30.0
            incident_data["predicted_by_ai"] = False

        # --- LIVE NEWS INTEGRATION (Run regardless of prediction success) ---
        try:
            active_news = self.news_fetcher.check_for_active_keywords()
            incident_data["latest_news"] = self.news_fetcher.get_latest_news(limit=5)
            incident_data["active_news_alerts"] = active_news
            
            if active_news:
                # Dynamically boost AI severity and duration based on real-world news!
                incident_data["severity_score"] = float(np.clip(incident_data.get("severity_score", 50.0) + 15.0, 0, 100))
                incident_data["duration_estimate"] = incident_data.get("duration_estimate", 30.0) * 1.5
        except Exception as e:
            logger.error(f"Live news integration failed: {e}")
            incident_data["latest_news"] = []
            incident_data["active_news_alerts"] = []
        # -----------------------------
            
        # 3. Call Module B with updated severity and location
        try:
            mod_b_context = {
                "latitude": lat,
                "longitude": lon,
                "severity_score": incident_data.get("severity_score", 50.0)
            }
            mod_b_res = self.module_b.predict(mod_b_context)
            incident_data["module_b_geojson"] = mod_b_res["geojson"]
        except Exception as e:
            logger.error(f"Module B prediction failed: {e}")
            
        process_time_ms = (time.time() - start_time) * 1000
        incident_data["api_process_time_ms"] = process_time_ms
        
        # Broadcast via WebSocket
        message = {
            "type": "incident_update",
            **incident_data
        }
        return message

    async def process_and_broadcast(self, incident):
        # Run blocking ML inference and network requests in a threadpool
        import asyncio
        message = await asyncio.to_thread(self.process_incident, incident)
        
        from .websocket import ws_manager
        await ws_manager.broadcast(message)
        logger.info(f"Incident {message.get('incident_id')} processed and broadcast successfully.")

    async def run(self):
        self.is_running = True
        logger.info("Background worker started polling...")
        while self.is_running:
            if self.queue:
                incident = self.queue.pop(0)
                try:
                    await self.process_and_broadcast(incident)
                except Exception as e:
                    logger.error(f"Failed to process incident: {e}")
            else:
                import asyncio
                await asyncio.sleep(1.0) # Poll interval

    def stop(self):
        self.is_running = False
        logger.info("Background worker stopped.")
