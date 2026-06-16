"""
Feature encoding and standardization for Gridlock 2.0 data pipeline.

Provides one-hot encoding, z-score normalization, derived feature generation,
and missing value imputation with data quality monitoring.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)


class FeatureEncoder:
    """
    Encodes and standardizes incident features for ML model input.
    
    Handles:
    - One-hot encoding for categorical variables (incident_type, day_of_week, weather_category)
    - Z-score normalization for numerical features
    - Derived feature generation (distance_to_highway, location_grid_cell, time_of_day_bins)
    - Missing value imputation (mean for numerical, mode for categorical)
    - Data quality validation (>40% missing detection)
    """
    
    def __init__(self, scaler: Optional[StandardScaler] = None, categorical_encoder: Optional[OneHotEncoder] = None):
        """
        Initialize feature encoder.
        
        Args:
            scaler: Pre-fitted StandardScaler for inference mode. If None, creates new scaler.
            categorical_encoder: Pre-fitted OneHotEncoder for inference mode. If None, creates new encoder.
        """
        self.scaler = scaler or StandardScaler()
        self.categorical_encoder = categorical_encoder or OneHotEncoder(
            sparse_output=False,
            handle_unknown='ignore'
        )
        
        # Numerical features to standardize
        self.numerical_features = [
            'temperature',
            'wind_speed',
            'precipitation',
            'humidity',
            'visibility',
            'severity_initial',
            'num_lanes_blocked',
            'num_vehicles',
        ]
        
        # Categorical features to one-hot encode
        self.categorical_features = [
            'incident_type',
            'day_of_week',
            'weather_category',
        ]
        
        # Track feature means for imputation
        self.feature_means: Dict[str, float] = {}
        self.feature_modes: Dict[str, str] = {}
        self.missing_percentages: Dict[str, float] = {}
        self.is_fitted = False
    
    def fit(self, incidents_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Fit encoder on training data.
        
        Args:
            incidents_df: DataFrame with incident features
            
        Returns:
            Dictionary with fit statistics including:
            - num_samples: Number of training samples
            - feature_means: Mean values for numerical features
            - feature_modes: Mode values for categorical features
            - missing_percentages: % missing for each feature
            - warnings: List of data quality warnings
        """
        df = incidents_df.copy()
        warnings = []
        
        # Compute statistics for imputation
        for feature in self.numerical_features:
            if feature in df.columns:
                self.feature_means[feature] = df[feature].mean()
                missing_pct = (df[feature].isna().sum() / len(df)) * 100
                self.missing_percentages[feature] = missing_pct
                
                if missing_pct > 40:
                    warnings.append(
                        f"Feature '{feature}' has {missing_pct:.1f}% missing values (>40% threshold)"
                    )
        
        for feature in self.categorical_features:
            if feature in df.columns:
                self.feature_modes[feature] = df[feature].mode()[0] if not df[feature].mode().empty else 'unknown'
                missing_pct = (df[feature].isna().sum() / len(df)) * 100
                self.missing_percentages[feature] = missing_pct
                
                if missing_pct > 40:
                    warnings.append(
                        f"Feature '{feature}' has {missing_pct:.1f}% missing values (>40% threshold)"
                    )
        
        # Fit StandardScaler on numerical features
        numerical_cols = [f for f in self.numerical_features if f in df.columns]
        if numerical_cols:
            df_numerical = df[numerical_cols].fillna(0)
            self.scaler.fit(df_numerical)
        
        # Fit OneHotEncoder on categorical features
        categorical_cols = [f for f in self.categorical_features if f in df.columns]
        if categorical_cols:
            df_categorical = df[categorical_cols].fillna('unknown')
            self.categorical_encoder.fit(df_categorical)
        
        if warnings:
            for warning in warnings:
                logger.warning(f"Data quality warning during encoder fit: {warning}")
        
        self.is_fitted = True
        
        return {
            'num_samples': len(df),
            'feature_means': self.feature_means,
            'feature_modes': self.feature_modes,
            'missing_percentages': self.missing_percentages,
            'warnings': warnings,
        }
    
    def _impute_missing_values(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Impute missing values in features.
        
        Uses mean imputation for numerical features and mode imputation for categorical features.
        
        Args:
            df: DataFrame to impute
            
        Returns:
            Tuple of (imputed DataFrame, list of imputed features)
        """
        df_imputed = df.copy()
        imputed_features = []
        
        # Mean imputation for numerical features
        for feature in self.numerical_features:
            if feature in df_imputed.columns and df_imputed[feature].isna().any():
                mean_value = self.feature_means.get(feature, df_imputed[feature].mean())
                df_imputed[feature].fillna(mean_value, inplace=True)
                imputed_features.append(feature)
        
        # Mode imputation for categorical features
        for feature in self.categorical_features:
            if feature in df_imputed.columns and df_imputed[feature].isna().any():
                mode_value = self.feature_modes.get(feature, 'unknown')
                df_imputed[feature].fillna(mode_value, inplace=True)
                imputed_features.append(feature)
        
        return df_imputed, imputed_features
    
    def _generate_derived_features(self, incident: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate derived features from raw incident data.
        
        Generates:
        - distance_to_highway: Estimated distance to nearest highway in km
        - location_grid_cell: Grid cell ID based on lat/lon
        - is_rush_hour: Boolean flag for rush hours (7-9 AM or 5-7 PM)
        - time_of_day_bins: Categorical time period (early_morning, morning, afternoon, evening, night)
        
        Args:
            incident: Dictionary with incident data
            
        Returns:
            Dictionary with added derived features
        """
        derived = incident.copy()
        
        # distance_to_highway: Simplified estimation based on location
        location = incident.get('location', {})
        lat = location.get('latitude', -33.8688)
        lon = location.get('longitude', 151.2093)
        
        # Simplified heuristic: compute distance proxy (in real system, query GIS database)
        # Using a simple grid-based approximation
        grid_x = int((lon + 180) / 0.01)
        grid_y = int((lat + 90) / 0.01)
        distance_to_highway_km = abs(grid_x % 10 - 5) * 0.5 + abs(grid_y % 10 - 5) * 0.5
        derived['distance_to_highway'] = min(distance_to_highway_km, 50.0)  # Cap at 50km
        
        # location_grid_cell: Map lat/lon to grid cell
        grid_cell_id = f"cell_{grid_x // 10}_{grid_y // 10}"
        derived['location_grid_cell'] = grid_cell_id
        
        # is_rush_hour: Check if incident is during typical rush hours
        timestamp = incident.get('timestamp')
        if isinstance(timestamp, str):
            try:
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            except (ValueError, AttributeError):
                dt = datetime.now()
        elif isinstance(timestamp, datetime):
            dt = timestamp
        else:
            dt = datetime.now()
        
        hour = dt.hour
        is_rush_hour = (7 <= hour < 9) or (17 <= hour < 19)
        derived['is_rush_hour'] = is_rush_hour
        
        # time_of_day_bins: Categorize time period
        if 5 <= hour < 9:
            time_bin = 'morning'
        elif 9 <= hour < 12:
            time_bin = 'mid_morning'
        elif 12 <= hour < 17:
            time_bin = 'afternoon'
        elif 17 <= hour < 21:
            time_bin = 'evening'
        else:
            time_bin = 'night'
        
        derived['time_of_day_bins'] = time_bin
        
        return derived
    
    def _extract_day_of_week(self, timestamp: Any) -> str:
        """Extract day of week from timestamp."""
        if isinstance(timestamp, str):
            try:
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            except (ValueError, AttributeError):
                dt = datetime.now()
        elif isinstance(timestamp, datetime):
            dt = timestamp
        else:
            dt = datetime.now()
        
        days = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
        return days[dt.weekday()]
    
    def encode(self, incident: Dict[str, Any]) -> Dict[str, Any]:
        """
        Encode a single incident for ML model input.
        
        Args:
            incident: Dictionary with incident data
            
        Returns:
            Dictionary with encoded features, standardized numerical features, and metadata
        """
        if not self.is_fitted:
            logger.warning("Encoder not fitted. Results may be inaccurate.")
        
        # Generate derived features
        incident_with_derived = self._generate_derived_features(incident)
        
        # Add day of week
        timestamp = incident.get('timestamp')
        incident_with_derived['day_of_week'] = self._extract_day_of_week(timestamp)
        
        # Extract weather category and flatten weather fields
        weather = incident.get('weather', {})
        temp = weather.get('temperature')
        if temp is None:
            weather_category = 'unknown'
        elif temp < 0:
            weather_category = 'cold'
        elif temp < 15:
            weather_category = 'cool'
        elif temp < 25:
            weather_category = 'mild'
        else:
            weather_category = 'warm'
        incident_with_derived['weather_category'] = weather_category
        
        # Flatten weather fields into incident data
        if weather:
            for weather_key, weather_value in weather.items():
                if weather_key not in incident_with_derived:
                    incident_with_derived[weather_key] = weather_value
        
        # Convert to DataFrame for easier processing
        df = pd.DataFrame([incident_with_derived])
        
        # Impute missing values
        df_imputed, imputed_features = self._impute_missing_values(df)
        
        # Standardize numerical features
        numerical_cols = [f for f in self.numerical_features if f in df_imputed.columns]
        encoded_data = incident_with_derived.copy()
        
        if numerical_cols:
            df_numerical = df_imputed[numerical_cols]
            df_standardized = pd.DataFrame(
                self.scaler.transform(df_numerical),
                columns=numerical_cols
            )
            
            for col in numerical_cols:
                encoded_data[f'{col}_standardized'] = df_standardized[col].values[0]
        
        # One-hot encode categorical features
        categorical_cols = [f for f in self.categorical_features if f in df_imputed.columns]
        if categorical_cols:
            df_categorical = df_imputed[categorical_cols]
            try:
                encoded_categorical = self.categorical_encoder.transform(df_categorical)
                feature_names = self.categorical_encoder.get_feature_names_out(categorical_cols)
                
                for i, feature_name in enumerate(feature_names):
                    encoded_data[feature_name] = encoded_categorical[0, i]
            except Exception as e:
                logger.debug(f"Categorical encoding partial (may be expected on unfitted encoder): {e}")
        
        # Add metadata
        encoded_data['_encoding_metadata'] = {
            'imputed_features': imputed_features,
            'encoded_datetime': datetime.now().isoformat(),
        }
        
        return encoded_data
    
    def encode_batch(self, incidents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Encode a batch of incidents.
        
        Args:
            incidents: List of incident dictionaries
            
        Returns:
            List of encoded incidents
        """
        return [self.encode(incident) for incident in incidents]
    
    def get_feature_statistics(self) -> Dict[str, Any]:
        """
        Get statistics from encoder fit.
        
        Returns:
            Dictionary with:
            - numerical_features: List of numerical features
            - categorical_features: List of categorical features
            - feature_means: Mean values (for inference)
            - feature_modes: Mode values (for inference)
            - missing_percentages: % missing for each feature
            - is_fitted: Whether encoder has been fitted
        """
        return {
            'numerical_features': self.numerical_features,
            'categorical_features': self.categorical_features,
            'feature_means': self.feature_means,
            'feature_modes': self.feature_modes,
            'missing_percentages': self.missing_percentages,
            'is_fitted': self.is_fitted,
        }


class FeatureStore:
    """
    Maintains and manages feature vectors with versioning.
    
    Stores:
    - Encoded feature vectors linked to incident IDs
    - Feature vector version (linked to dataset version)
    - Metadata about features and encoding parameters
    - Retrieval and comparison capabilities
    """
    
    def __init__(self, dataset_version: str = "v1.0"):
        """
        Initialize feature store.
        
        Args:
            dataset_version: Version identifier for the dataset/features
        """
        self.dataset_version = dataset_version
        self.store: Dict[str, Dict[str, Any]] = {}
        self.version_history: List[str] = []
        self.created_at = datetime.now()
    
    def add_feature_vector(self, incident_id: str, features: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Add an encoded feature vector to the store.
        
        Args:
            incident_id: Unique identifier for the incident
            features: Dictionary of encoded features
            metadata: Optional metadata about the feature vector
        """
        self.store[incident_id] = {
            'features': features,
            'metadata': metadata or {},
            'dataset_version': self.dataset_version,
            'stored_at': datetime.now().isoformat(),
        }
    
    def get_feature_vector(self, incident_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a feature vector by incident ID.
        
        Args:
            incident_id: Incident identifier
            
        Returns:
            Dictionary with features and metadata, or None if not found
        """
        return self.store.get(incident_id)
    
    def has_feature_vector(self, incident_id: str) -> bool:
        """
        Check if feature vector exists for incident.
        
        Args:
            incident_id: Incident identifier
            
        Returns:
            True if vector exists, False otherwise
        """
        return incident_id in self.store
    
    def get_all_incident_ids(self) -> List[str]:
        """
        Get list of all incident IDs in store.
        
        Returns:
            List of incident IDs
        """
        return list(self.store.keys())
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the feature store.
        
        Returns:
            Dictionary with:
            - total_vectors: Total number of stored vectors
            - dataset_version: Current dataset version
            - created_at: When store was created
            - feature_count: Number of features per vector
            - storage_size_mb: Approximate storage size in MB
        """
        total_vectors = len(self.store)
        
        if total_vectors == 0:
            feature_count = 0
            storage_size_mb = 0
        else:
            first_vector = next(iter(self.store.values()))
            feature_count = len(first_vector.get('features', {}))
            
            # Rough estimate: each feature ~8 bytes (float64)
            storage_size_mb = (total_vectors * feature_count * 8) / (1024 * 1024)
        
        return {
            'total_vectors': total_vectors,
            'dataset_version': self.dataset_version,
            'created_at': self.created_at.isoformat(),
            'feature_count': feature_count,
            'storage_size_mb': storage_size_mb,
        }
    
    def clear(self) -> None:
        """Clear all feature vectors from store."""
        self.store.clear()
    
    def delete_feature_vector(self, incident_id: str) -> bool:
        """
        Delete a feature vector by incident ID.
        
        Args:
            incident_id: Incident identifier
            
        Returns:
            True if deleted, False if not found
        """
        if incident_id in self.store:
            del self.store[incident_id]
            return True
        return False
