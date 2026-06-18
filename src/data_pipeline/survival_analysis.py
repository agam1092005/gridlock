"""
Survival Analysis Module for Missing end_datetime Imputation

This module implements Kaplan-Meier and Cox proportional hazards models
to estimate incident duration for missing end_datetime values (94% of incidents).

Key Features:
- Kaplan-Meier curve fitting stratified by incident_type
- Cox proportional hazards model for conditional survival probabilities
- Probabilistic duration estimation with 95% confidence intervals
- Fallback to population-level KM when incident type has <50 samples
- Redis-backed model caching with configurable TTL and refresh
"""

import json
import logging
import pickle
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional, Tuple, List, Any

import numpy as np
import pandas as pd
import redis
from lifelines import KaplanMeierFitter, CoxPHFitter
from scipy import stats

logger = logging.getLogger(__name__)


class SurvivalAnalyzer:
    """
    Analyzes incident survival data to estimate missing end_datetime values.

    Uses Kaplan-Meier curves stratified by incident_type and Cox proportional
    hazards models for conditional survival probability estimation.
    """

    # Minimum number of incidents required to fit type-specific model
    MIN_SAMPLES_FOR_STRATIFIED_KM = 50

    # Feature names for Cox model
    COX_FEATURES = [
        "location_grid_x",
        "location_grid_y",
        "weather_temp",
        "hour_of_day",
        "is_rush_hour",
    ]

    # Cache configuration
    DEFAULT_CACHE_TTL_DAYS = 7
    CACHE_KEY_PREFIX = "survival_model"

    def __init__(
        self,
        redis_client: Optional[redis.Redis] = None,
        cache_ttl_days: int = DEFAULT_CACHE_TTL_DAYS,
    ):
        """
        Initialize SurvivalAnalyzer.

        Args:
            redis_client: Redis client for model caching. If None, caching is disabled.
            cache_ttl_days: Cache TTL in days for fitted models.
        """
        self.redis_client = redis_client
        self.cache_ttl_seconds = cache_ttl_days * 24 * 3600

        # Fitted models - dictionary keyed by incident_type
        self.km_curves: Dict[str, KaplanMeierFitter] = {}

        # Population-level KM (fallback)
        self.population_km: Optional[KaplanMeierFitter] = None

        # Cox proportional hazards model
        self.cox_model: Optional[CoxPHFitter] = None

        # Track model fitting metadata
        self.model_metadata: Dict[str, Any] = {
            "fitted_at": None,
            "km_incident_types": [],
            "cox_features_used": self.COX_FEATURES,
            "num_training_samples": 0,
            "num_samples_per_type": {},
        }

        self._models_fitted = False

    def fit_models(self, historical_incidents_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Fit Kaplan-Meier and Cox models on historical incident data.

        Args:
            historical_incidents_df: DataFrame with columns:
                - start_datetime: incident start time
                - end_datetime: incident end time (may be null for censored observations)
                - incident_type: type of incident (accident, congestion, etc.)
                - location_grid_x: x coordinate of incident
                - location_grid_y: y coordinate of incident
                - weather_temp: temperature at incident
                - hour_of_day: hour of day (0-23)
                - is_rush_hour: boolean flag

        Returns:
            Dictionary with fitting results including:
                - success: bool indicating if fitting succeeded
                - num_samples: total samples used
                - km_incident_types: list of incident types with stratified KM curves
                - cox_features_available: whether Cox model could be fitted
                - error: error message if fitting failed
        """
        logger.info(f"Starting survival model fitting with {len(historical_incidents_df)} samples")

        try:
            # Prepare data
            df = historical_incidents_df.copy()

            # Calculate duration in minutes
            df["duration_min"] = (df["end_datetime"] - df["start_datetime"]).dt.total_seconds() / 60  # type: ignore

            # Mark censored observations (missing end_datetime)
            df["event_observed"] = ~df["end_datetime"].isna()

            # Remove rows with invalid duration or NaT in start_datetime
            df = df[df["duration_min"] > 0]
            df = df[df["start_datetime"].notna()]

            logger.info(
                f"After filtering: {len(df)} valid samples, {df['event_observed'].sum()} events observed, "
                f"{(~df['event_observed']).sum()} censored"
            )

            # Fit population-level KM (baseline, used for fallback)
            self.population_km = KaplanMeierFitter()
            self.population_km.fit(
                durations=df["duration_min"],
                event_observed=df["event_observed"],
                label="population_overall",
            )
            logger.info("Fitted population-level Kaplan-Meier curve")

            # Fit stratified KM curves by incident_type
            self.km_curves = {}
            km_incident_types = []

            for incident_type in df["incident_type"].unique():
                subset = df[df["incident_type"] == incident_type]
                num_samples = len(subset)

                if num_samples >= self.MIN_SAMPLES_FOR_STRATIFIED_KM:
                    kmf = KaplanMeierFitter()
                    kmf.fit(
                        durations=subset["duration_min"],
                        event_observed=subset["event_observed"],
                        label=incident_type,
                    )
                    self.km_curves[incident_type] = kmf
                    km_incident_types.append(incident_type)
                    logger.info(f"Fitted KM curve for {incident_type} ({num_samples} samples)")
                    self.model_metadata["num_samples_per_type"][incident_type] = num_samples
                else:
                    logger.info(
                        f"Skipping KM for {incident_type}: only {num_samples} samples (threshold: {self.MIN_SAMPLES_FOR_STRATIFIED_KM})"
                    )

            # Fit Cox proportional hazards model if sufficient features available
            cox_features_available = all(feat in df.columns for feat in self.COX_FEATURES)

            if cox_features_available:
                try:
                    # Prepare data for Cox model
                    cox_data = df[self.COX_FEATURES + ["duration_min", "event_observed"]].copy()

                    # Drop rows with missing values in Cox features
                    cox_data_clean = cox_data.dropna()

                    if len(cox_data_clean) > 0:
                        # Rename columns for lifelines (expects 'T' for duration, 'E' for event)
                        cox_data_clean = cox_data_clean.rename(
                            columns={"duration_min": "T", "event_observed": "E"}
                        )

                        self.cox_model = CoxPHFitter()
                        self.cox_model.fit(cox_data_clean, duration_col="T", event_col="E")
                        logger.info(f"Fitted Cox PH model with {len(cox_data_clean)} samples")
                    else:
                        logger.warning("No clean data available for Cox model fitting")
                        self.cox_model = None

                except Exception as e:
                    logger.warning(f"Cox model fitting failed: {e}. Continuing without Cox model.")
                    self.cox_model = None
            else:
                logger.warning(
                    f"Missing Cox features: {[f for f in self.COX_FEATURES if f not in df.columns]}"
                )
                self.cox_model = None

            # Update metadata
            self.model_metadata["fitted_at"] = datetime.now(timezone.utc).isoformat()
            self.model_metadata["km_incident_types"] = km_incident_types
            self.model_metadata["num_training_samples"] = len(df)
            self._models_fitted = True

            # Cache models if Redis available
            if self.redis_client:
                self._cache_models()

            result = {
                "success": True,
                "num_samples": len(df),
                "num_samples_per_type": self.model_metadata["num_samples_per_type"],
                "km_incident_types": km_incident_types,
                "cox_features_available": self.cox_model is not None,
                "population_km_fitted": self.population_km is not None,
            }

            logger.info(f"Survival model fitting complete: {result}")
            return result

        except Exception as e:
            logger.error(f"Error fitting survival models: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    def impute_end_datetime(
        self, incident: Dict[str, Any], confidence_level: float = 0.95
    ) -> Dict[str, Any]:
        """
        Generate probabilistic end_datetime estimate for incident.

        Args:
            incident: Dictionary containing incident data:
                - start_datetime: incident start time
                - end_datetime: incident end time (null if missing/ongoing)
                - incident_type: type of incident
                - location_grid_x, location_grid_y, weather_temp, hour_of_day, is_rush_hour (for Cox model)

            confidence_level: Confidence level for intervals (default 0.95 = 95%)

        Returns:
            Dictionary with imputation results:
                - end_datetime: imputed end time
                - duration_estimate_min: median duration estimate
                - duration_ci: (lower, upper) confidence interval bounds
                - confidence_level: confidence level used
                - imputation_method: method used (KM, Cox, or population)
                - percentiles: dict with p25, p50, p75 percentile estimates
                - success: whether imputation succeeded
        """

        # If end_datetime is already known, return it
        if pd.notna(incident.get("end_datetime")):
            return {
                "end_datetime": incident["end_datetime"],
                "duration_estimate_min": None,
                "duration_ci": None,
                "confidence_level": confidence_level,
                "imputation_method": "known_value",
                "percentiles": None,
                "success": True,
            }

        if not self._models_fitted or self.population_km is None:
            logger.warning("Models not fitted. Cannot impute end_datetime.")
            return {
                "end_datetime": None,
                "duration_estimate_min": None,
                "duration_ci": None,
                "confidence_level": confidence_level,
                "imputation_method": None,
                "percentiles": None,
                "success": False,
                "error": "Models not fitted",
            }

        try:
            start_dt_raw = incident.get("start_datetime")
            if not start_dt_raw:
                raise ValueError("start_datetime is missing from incident")

            start_dt = pd.to_datetime(str(start_dt_raw))
            incident_type = incident.get("incident_type", "unknown")

            # Determine which KM curve to use
            if incident_type in self.km_curves and len(self.km_curves[incident_type].durations) > 0:
                km_curve = self.km_curves[incident_type]
                imputation_method = f"KM_{incident_type}"
            else:
                km_curve = self.population_km
                imputation_method = "KM_population"

            # Calculate percentile estimates (10, 25, 50, 75, 90 minutes)
            percentile_durations = [10, 25, 50, 75, 90]
            percentile_estimates: Dict[str, Optional[Dict[str, Any]]] = {}

            for duration in percentile_durations:
                try:
                    survival_prob = km_curve.survival_function_at_times(duration).values[0]
                    percentile_estimates[f"p{duration}"] = {
                        "duration_min": duration,
                        "survival_probability": survival_prob,
                    }
                except Exception:
                    percentile_estimates[f"p{duration}"] = None

            # Get median duration (50th percentile survival = 0.5)
            try:
                # Find the duration at which survival probability is approximately 0.5
                median_duration = km_curve.median_survival_time_
            except Exception:
                # Fallback: calculate from survival function
                median_duration = None

            if median_duration is None or np.isnan(median_duration):
                # Use 50th percentile of observed durations as fallback
                median_duration = km_curve.durations.median() if len(km_curve.durations) > 0 else 30

            # Calculate confidence interval bounds
            # Use a simple percentile-based approach: assume the observed durations
            # follow an approximately normal distribution in log space (common for survival data)
            try:
                # Use quantile-based CI from the KM curve's estimated distribution
                alpha = 1 - confidence_level  # e.g., 0.05 for 95% CI

                # Bootstrap approach: use percentiles of observed event times as CI bounds
                observed_durations = km_curve.durations[km_curve.event_observed]
                if len(observed_durations) > 10:
                    ci_lower = float(np.percentile(observed_durations, alpha / 2 * 100))
                    ci_upper = float(np.percentile(observed_durations, (1 - alpha / 2) * 100))
                    # Ensure bounds are sensible
                    ci_lower = max(1, ci_lower)
                    ci_upper = max(median_duration, ci_upper)
                else:
                    # Fallback for small datasets
                    ci_lower = max(1, median_duration * 0.6)
                    ci_upper = median_duration * 1.8

            except Exception:
                # Fallback to simple scaling approach
                ci_lower = max(1, median_duration * 0.6)
                ci_upper = median_duration * 2.0

            # Apply Cox model adjustment if available and features present
            if self.cox_model is not None and self._has_cox_features(incident):
                cox_adjustment = self._get_cox_adjustment(incident)
                median_duration = median_duration * cox_adjustment
                ci_lower = ci_lower * cox_adjustment
                ci_upper = ci_upper * cox_adjustment
                imputation_method += "+Cox"

            # Generate imputed end_datetime
            imputed_end_dt = start_dt + pd.Timedelta(minutes=median_duration)

            return {
                "end_datetime": imputed_end_dt,
                "duration_estimate_min": median_duration,
                "duration_ci": (ci_lower, ci_upper),
                "confidence_level": confidence_level,
                "imputation_method": imputation_method,
                "percentiles": percentile_estimates,
                "success": True,
            }

        except Exception as e:
            logger.error(f"Error imputing end_datetime: {e}", exc_info=True)
            return {
                "end_datetime": None,
                "duration_estimate_min": None,
                "duration_ci": None,
                "confidence_level": confidence_level,
                "imputation_method": None,
                "percentiles": None,
                "success": False,
                "error": str(e),
            }

    def _has_cox_features(self, incident: Dict[str, Any]) -> bool:
        """Check if incident has all required Cox model features."""
        return all(incident.get(feat) is not None for feat in self.COX_FEATURES)

    def _get_cox_adjustment(self, incident: Dict[str, Any]) -> float:
        """
        Calculate Cox model adjustment factor for incident.

        Returns a multiplier to apply to baseline duration estimate.
        """
        try:
            if self.cox_model is None:
                return 1.0

            # Prepare feature vector for Cox model
            cox_features_dict = {feat: incident.get(feat) for feat in self.COX_FEATURES}

            # Calculate partial hazard (relative risk)
            # Higher hazard = faster resolution = shorter duration = multiply by <1
            # Lower hazard = slower resolution = longer duration = multiply by >1
            partial_hazard = self.cox_model.predict_partial_hazard(
                pd.DataFrame([cox_features_dict])
            ).values[0]

            # Inverse relationship: partial hazard vs duration
            # Normalize to mean of 1.0
            adjustment = 1.0 / max(partial_hazard, 0.1)  # Avoid division by zero

            return adjustment

        except Exception as e:
            logger.debug(f"Cox adjustment calculation failed: {e}")
            return 1.0

    def _cache_models(self) -> bool:
        """
        Cache fitted models to Redis.

        Returns:
            True if caching succeeded, False otherwise
        """
        if self.redis_client is None:
            return False

        try:
            # Serialize KM curves
            km_data = {}
            for incident_type, km in self.km_curves.items():
                try:
                    km_data[incident_type] = pickle.dumps(km)
                except Exception as e:
                    logger.warning(f"Failed to serialize KM curve for {incident_type}: {e}")

            # Serialize Cox model
            cox_data = None
            if self.cox_model is not None:
                try:
                    cox_data = pickle.dumps(self.cox_model)
                except Exception as e:
                    logger.warning(f"Failed to serialize Cox model: {e}")

            # Serialize population KM
            population_km_data = None
            if self.population_km is not None:
                try:
                    population_km_data = pickle.dumps(self.population_km)
                except Exception as e:
                    logger.warning(f"Failed to serialize population KM: {e}")

            # Create cache payload
            cache_payload = {
                "km_curves": km_data,
                "cox_model": cox_data,
                "population_km": population_km_data,
                "metadata": self.model_metadata,
                "cached_at": datetime.now(timezone.utc).isoformat(),
            }

            # Store in Redis
            cache_key = f"{self.CACHE_KEY_PREFIX}:models"
            self.redis_client.setex(
                cache_key,
                self.cache_ttl_seconds,
                json.dumps(
                    {"metadata": self.model_metadata, "cached_at": cache_payload["cached_at"]}
                ),
            )

            # Store serialized models separately
            self.redis_client.setex(
                f"{cache_key}:km_curves", self.cache_ttl_seconds, pickle.dumps(km_data)
            )
            self.redis_client.setex(
                f"{cache_key}:cox_model", self.cache_ttl_seconds, cox_data or b""
            )
            self.redis_client.setex(
                f"{cache_key}:population_km", self.cache_ttl_seconds, population_km_data or b""
            )

            logger.info(f"Cached survival models to Redis with TTL {self.cache_ttl_seconds}s")
            return True

        except Exception as e:
            logger.error(f"Error caching models to Redis: {e}")
            return False

    def load_models_from_cache(self) -> bool:
        """
        Load fitted models from Redis cache.

        Returns:
            True if models were loaded from cache, False otherwise
        """
        if self.redis_client is None:
            return False

        try:
            cache_key = f"{self.CACHE_KEY_PREFIX}:models"

            # Check if cache exists
            if not self.redis_client.exists(cache_key):
                logger.debug("No cached models found in Redis")
                return False

            # Load metadata
            metadata_json = self.redis_client.get(cache_key)
            if metadata_json:
                cache_data = json.loads(metadata_json)
                self.model_metadata = cache_data.get("metadata", {})

            # Load KM curves
            km_data = self.redis_client.get(f"{cache_key}:km_curves")
            if km_data:
                if isinstance(km_data, str):
                    km_data = km_data.encode("latin-1")
                km_dict = pickle.loads(km_data)
                self.km_curves = km_dict

            # Load Cox model
            cox_data = self.redis_client.get(f"{cache_key}:cox_model")
            if cox_data and len(cox_data) > 0:
                if isinstance(cox_data, str):
                    cox_data = cox_data.encode("latin-1")
                self.cox_model = pickle.loads(cox_data)

            # Load population KM
            population_km_data = self.redis_client.get(f"{cache_key}:population_km")
            if population_km_data and len(population_km_data) > 0:
                if isinstance(population_km_data, str):
                    population_km_data = population_km_data.encode("latin-1")
                self.population_km = pickle.loads(population_km_data)

            self._models_fitted = True
            logger.info("Loaded survival models from Redis cache")
            return True

        except Exception as e:
            logger.warning(f"Error loading models from cache: {e}")
            return False

    def clear_cache(self) -> bool:
        """Clear cached models from Redis."""
        if self.redis_client is None:
            return False

        try:
            cache_key_prefix = f"{self.CACHE_KEY_PREFIX}:models"
            keys_to_delete = [
                cache_key_prefix,
                f"{cache_key_prefix}:km_curves",
                f"{cache_key_prefix}:cox_model",
                f"{cache_key_prefix}:population_km",
            ]

            for key in keys_to_delete:
                self.redis_client.delete(key)

            logger.info("Cleared survival models cache")
            return True

        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return False

    def get_model_status(self) -> Dict[str, Any]:
        """Get status information about fitted models."""
        return {
            "models_fitted": self._models_fitted,
            "population_km_fitted": self.population_km is not None,
            "num_km_curves": len(self.km_curves),
            "km_incident_types": list(self.km_curves.keys()),
            "cox_model_fitted": self.cox_model is not None,
            "metadata": self.model_metadata,
        }
