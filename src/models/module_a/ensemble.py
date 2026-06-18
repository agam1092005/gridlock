import numpy as np
import os
import json
import logging
from .lightgbm_models import LightGBMSeverityModel, LightGBMDurationModel
from .bigru_model import BiGRUModel
from sklearn.isotonic import IsotonicRegression

logger = logging.getLogger(__name__)


class ModuleAEnsemble:
    def __init__(self, version="1.0"):
        self.version = version
        self.lgb_sev = LightGBMSeverityModel()
        self.lgb_dur = LightGBMDurationModel()
        self.bigru = BiGRUModel()

        # Fusion weights
        self.w_sev_lgb = 0.7
        self.w_sev_bigru = 0.3
        self.w_dur_lgb = 0.6
        self.w_dur_bigru = 0.4

        # Isotonic regression calibration models
        # These learn the mapping: raw_prediction -> calibrated_confidence
        self.severity_calibrator = IsotonicRegression(out_of_bounds="clip")
        self.duration_calibrator = IsotonicRegression(out_of_bounds="clip")

        self.is_calibrated = False

    def calibrate_on_validation(self, val_predictions, val_ground_truth):
        """
        Fit isotonic regression calibration models on validation set.

        Args:
            val_predictions: Array of predicted confidences [0, 1]
            val_ground_truth: Array of actual values (0-100 for severity, minutes for duration)
        """
        try:
            # For severity calibration
            severity_errors = np.abs(val_predictions[:, 0] - val_ground_truth[:, 0])
            # Map to 0-1 range where 0=perfect, 1=max error
            severity_normalized_errors = np.clip(severity_errors / 100.0, 0, 1)

            self.severity_calibrator.fit(
                val_predictions[:, 1],  # Raw confidence from model
                1 - severity_normalized_errors,  # Calibrated confidence (higher = more accurate)
            )

            # For duration calibration
            duration_errors = np.abs(val_predictions[:, 2] - val_ground_truth[:, 1])
            # Normalize duration errors (assuming max duration ~300 minutes)
            duration_normalized_errors = np.clip(duration_errors / 300.0, 0, 1)

            self.duration_calibrator.fit(
                val_predictions[:, 3], 1 - duration_normalized_errors  # Raw confidence from model
            )

            self.is_calibrated = True
            logger.info("Isotonic calibration fitted successfully")
        except Exception as e:
            logger.error(f"Calibration fitting failed: {e}")
            self.is_calibrated = False

    def load_models(self, directory):
        self.lgb_sev.load(directory)
        self.lgb_dur.load(directory)

        import torch

        bigru_path = os.path.join(directory, "bigru_model.pth")
        if os.path.exists(bigru_path):
            self.bigru.load_state_dict(torch.load(bigru_path))
            self.bigru.eval()

    def predict(self, structured_features, embedding, historical_seq=None, incident_type="unknown"):
        # Combine features for LGB
        import torch

        lgb_input = np.concatenate([embedding, structured_features])

        # Fallback logic for unseen incident type
        if incident_type == "unknown":
            logger.warning("Unseen incident type. Fallback model could be used here.")

        # 1. LGB Predictions
        lgb_sev_pred = self.lgb_sev.predict([lgb_input])
        lgb_dur_pred = self.lgb_dur.predict([lgb_input])

        # 2. BiGRU Predictions
        # Convert to tensor
        if historical_seq is None:
            # Mock historical sequence if none provided
            # (seq_len, feature_dim)
            historical_seq = torch.zeros((1, 5, len(lgb_input)))

        with torch.no_grad():
            bigru_sev_pred, bigru_dur_pred, attn = self.bigru(historical_seq)

            bigru_sev_val = bigru_sev_pred.squeeze().item()
            bigru_dur_val = bigru_dur_pred.squeeze().item()

        # 3. Fusion
        fused_sev = self.w_sev_lgb * lgb_sev_pred["score"][0] + self.w_sev_bigru * bigru_sev_val
        fused_dur = self.w_dur_lgb * lgb_dur_pred["estimate"][0] + self.w_dur_bigru * bigru_dur_val

        # 4. Calibration
        if self.is_calibrated:
            try:
                calib_sev = float(self.severity_calibrator.predict([fused_sev])[0])
                calib_dur = float(self.duration_calibrator.predict([fused_dur])[0])
            except Exception as e:
                logger.error(f"Calibration prediction failed: {e}")
                calib_sev = fused_sev
                calib_dur = fused_dur
        else:
            calib_sev = fused_sev
            calib_dur = fused_dur

        # Combine CIs from LGB (simplified)
        return {
            "severity_score": float(np.clip(calib_sev, 0, 100)),
            "severity_ci": [
                float(np.clip(lgb_sev_pred["ci_lower"][0], 0, 100)),
                float(np.clip(lgb_sev_pred["ci_upper"][0], 0, 100)),
            ],
            "duration_estimate": float(max(0, calib_dur)),
            "duration_ci": [
                float(max(0, lgb_dur_pred["ci_lower"][0])),
                float(max(0, lgb_dur_pred["ci_upper"][0])),
            ],
        }
