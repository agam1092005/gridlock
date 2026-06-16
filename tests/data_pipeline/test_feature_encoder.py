"""
Unit tests for feature encoding and feature store.

Tests FeatureEncoder for one-hot encoding, standardization, derived feature generation,
and imputation. Tests FeatureStore for vector storage and versioning.
"""

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.data_pipeline.feature_encoder import FeatureEncoder, FeatureStore


# ============================================================================
# FeatureEncoder Tests
# ============================================================================


class TestFeatureEncoderFit:
    """Tests for FeatureEncoder.fit method."""
    
    def test_fit_basic(self):
        """Test basic encoder fitting on sample data."""
        encoder = FeatureEncoder()
        
        # Create sample training data
        data = pd.DataFrame({
            'incident_type': ['accident', 'congestion', 'accident'],
            'day_of_week': ['monday', 'tuesday', 'wednesday'],
            'weather_category': ['cold', 'mild', 'warm'],
            'temperature': [5.0, 15.0, 25.0],
            'wind_speed': [10.0, 5.0, 3.0],
            'precipitation': [0.0, 2.0, 0.0],
            'humidity': [60.0, 70.0, 50.0],
            'visibility': [5000.0, 8000.0, 10000.0],
        })
        
        stats = encoder.fit(data)
        
        assert encoder.is_fitted
        assert stats['num_samples'] == 3
        assert 'temperature' in stats['feature_means']
        assert stats['feature_means']['temperature'] == pytest.approx(15.0)
        assert 'incident_type' in stats['feature_modes']
        assert stats['feature_modes']['incident_type'] == 'accident'
        assert len(stats['warnings']) == 0
    
    def test_fit_with_missing_values(self):
        """Test fitting with missing values in features."""
        encoder = FeatureEncoder()
        
        data = pd.DataFrame({
            'incident_type': ['accident', None, 'accident'],
            'temperature': [5.0, None, 25.0],
            'wind_speed': [10.0, 5.0, None],
        })
        
        stats = encoder.fit(data)
        
        assert encoder.is_fitted
        assert 'temperature' in stats['missing_percentages']
        assert stats['missing_percentages']['temperature'] == pytest.approx(33.33, abs=0.1)
    
    def test_fit_high_missing_data_warning(self):
        """Test warning when >40% of feature is missing."""
        encoder = FeatureEncoder()
        
        data = pd.DataFrame({
            'temperature': [5.0, None, None, None, 25.0],
            'wind_speed': [10.0, 5.0, 3.0, 2.0, 1.0],
        })
        
        stats = encoder.fit(data)
        
        assert len(stats['warnings']) > 0
        assert 'temperature' in stats['warnings'][0]
        assert '>40%' in stats['warnings'][0]
    
    def test_fit_computes_means_and_modes(self):
        """Test that fit computes correct means and modes."""
        encoder = FeatureEncoder()
        
        data = pd.DataFrame({
            'incident_type': ['accident', 'accident', 'congestion'],
            'temperature': [10.0, 20.0, 30.0],
            'wind_speed': [5.0, 10.0, 15.0],
        })
        
        encoder.fit(data)
        
        assert encoder.feature_means['temperature'] == pytest.approx(20.0)
        assert encoder.feature_means['wind_speed'] == pytest.approx(10.0)
        assert encoder.feature_modes['incident_type'] == 'accident'


class TestFeatureEncoderImputeMissingValues:
    """Tests for imputation of missing values."""
    
    def test_impute_numerical_features_mean(self):
        """Test mean imputation for numerical features."""
        encoder = FeatureEncoder()
        
        # Fit encoder
        train_data = pd.DataFrame({
            'temperature': [10.0, 20.0, 30.0],
            'wind_speed': [5.0, 10.0, 15.0],
        })
        encoder.fit(train_data)
        
        # Test data with missing values
        test_data = pd.DataFrame({
            'temperature': [10.0, None, 30.0],
            'wind_speed': [5.0, 10.0, None],
        })
        
        imputed, imputed_features = encoder._impute_missing_values(test_data)
        
        assert not imputed['temperature'].isna().any()
        assert not imputed['wind_speed'].isna().any()
        assert 'temperature' in imputed_features
        assert 'wind_speed' in imputed_features
        assert imputed.iloc[1]['temperature'] == pytest.approx(20.0)  # Mean of [10, 20, 30]
    
    def test_impute_categorical_features_mode(self):
        """Test mode imputation for categorical features."""
        encoder = FeatureEncoder()
        
        # Fit encoder
        train_data = pd.DataFrame({
            'incident_type': ['accident', 'accident', 'congestion'],
            'day_of_week': ['monday', 'tuesday', 'wednesday'],
        })
        encoder.fit(train_data)
        
        # Test data with missing values
        test_data = pd.DataFrame({
            'incident_type': ['accident', None, 'congestion'],
            'day_of_week': ['monday', 'tuesday', None],
        })
        
        imputed, imputed_features = encoder._impute_missing_values(test_data)
        
        assert not imputed['incident_type'].isna().any()
        assert not imputed['day_of_week'].isna().any()
        assert imputed.iloc[1]['incident_type'] == 'accident'  # Mode
        assert 'incident_type' in imputed_features
        assert 'day_of_week' in imputed_features


class TestFeatureEncoderDerivedFeatures:
    """Tests for derived feature generation."""
    
    def test_generate_distance_to_highway(self):
        """Test distance_to_highway derived feature."""
        encoder = FeatureEncoder()
        encoder.fit(pd.DataFrame({'temperature': [20.0]}))
        
        incident = {
            'location': {'latitude': -33.8688, 'longitude': 151.2093},
            'timestamp': '2024-01-15T14:30:00Z',
        }
        
        result = encoder._generate_derived_features(incident)
        
        assert 'distance_to_highway' in result
        assert isinstance(result['distance_to_highway'], (int, float))
        assert 0 <= result['distance_to_highway'] <= 50
    
    def test_generate_location_grid_cell(self):
        """Test location_grid_cell derived feature."""
        encoder = FeatureEncoder()
        encoder.fit(pd.DataFrame({'temperature': [20.0]}))
        
        incident = {
            'location': {'latitude': -33.8688, 'longitude': 151.2093},
            'timestamp': '2024-01-15T14:30:00Z',
        }
        
        result = encoder._generate_derived_features(incident)
        
        assert 'location_grid_cell' in result
        assert isinstance(result['location_grid_cell'], str)
        assert result['location_grid_cell'].startswith('cell_')
    
    def test_generate_is_rush_hour_true(self):
        """Test is_rush_hour during rush hours (7-9 AM)."""
        encoder = FeatureEncoder()
        encoder.fit(pd.DataFrame({'temperature': [20.0]}))
        
        incident = {
            'location': {'latitude': -33.8688, 'longitude': 151.2093},
            'timestamp': '2024-01-15T08:30:00Z',  # 8:30 AM
        }
        
        result = encoder._generate_derived_features(incident)
        
        assert result['is_rush_hour'] is True
    
    def test_generate_is_rush_hour_false(self):
        """Test is_rush_hour outside rush hours."""
        encoder = FeatureEncoder()
        encoder.fit(pd.DataFrame({'temperature': [20.0]}))
        
        incident = {
            'location': {'latitude': -33.8688, 'longitude': 151.2093},
            'timestamp': '2024-01-15T12:30:00Z',  # 12:30 PM
        }
        
        result = encoder._generate_derived_features(incident)
        
        assert result['is_rush_hour'] is False
    
    def test_generate_is_rush_hour_evening_peak(self):
        """Test is_rush_hour during evening peak (5-7 PM)."""
        encoder = FeatureEncoder()
        encoder.fit(pd.DataFrame({'temperature': [20.0]}))
        
        incident = {
            'location': {'latitude': -33.8688, 'longitude': 151.2093},
            'timestamp': '2024-01-15T18:30:00Z',  # 6:30 PM
        }
        
        result = encoder._generate_derived_features(incident)
        
        assert result['is_rush_hour'] is True
    
    def test_generate_time_of_day_bins(self):
        """Test time_of_day_bins derived feature."""
        encoder = FeatureEncoder()
        encoder.fit(pd.DataFrame({'temperature': [20.0]}))
        
        test_cases = [
            ('2024-01-15T06:00:00Z', 'morning'),
            ('2024-01-15T12:00:00Z', 'afternoon'),
            ('2024-01-15T18:00:00Z', 'evening'),
            ('2024-01-15T02:00:00Z', 'night'),
        ]
        
        for timestamp, expected_bin in test_cases:
            incident = {
                'location': {'latitude': -33.8688, 'longitude': 151.2093},
                'timestamp': timestamp,
            }
            result = encoder._generate_derived_features(incident)
            assert result['time_of_day_bins'] == expected_bin


class TestFeatureEncoderExtractDayOfWeek:
    """Tests for day-of-week extraction."""
    
    def test_extract_day_of_week_from_string(self):
        """Test extracting day of week from ISO timestamp string."""
        encoder = FeatureEncoder()
        encoder.fit(pd.DataFrame({'temperature': [20.0]}))
        
        # 2024-01-15 was a Monday
        day = encoder._extract_day_of_week('2024-01-15T14:30:00Z')
        assert day == 'monday'
    
    def test_extract_day_of_week_from_datetime(self):
        """Test extracting day of week from datetime object."""
        encoder = FeatureEncoder()
        encoder.fit(pd.DataFrame({'temperature': [20.0]}))
        
        dt = datetime(2024, 1, 15)  # Monday
        day = encoder._extract_day_of_week(dt)
        assert day == 'monday'
    
    def test_extract_day_of_week_all_days(self):
        """Test all days of the week."""
        encoder = FeatureEncoder()
        encoder.fit(pd.DataFrame({'temperature': [20.0]}))
        
        # 2024-01-15 (Monday) to 2024-01-21 (Sunday)
        expected_days = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
        
        for i, expected_day in enumerate(expected_days):
            date = datetime(2024, 1, 15 + i)
            day = encoder._extract_day_of_week(date)
            assert day == expected_day


class TestFeatureEncoderEncode:
    """Tests for encoding single incidents."""
    
    def test_encode_single_incident(self):
        """Test encoding a single incident."""
        encoder = FeatureEncoder()
        
        # Fit encoder
        train_data = pd.DataFrame({
            'incident_type': ['accident', 'congestion'],
            'day_of_week': ['monday', 'tuesday'],
            'weather_category': ['cold', 'mild'],
            'temperature': [10.0, 20.0],
            'wind_speed': [5.0, 10.0],
        })
        encoder.fit(train_data)
        
        # Encode incident
        incident = {
            'incident_id': 'test-123',
            'location': {'latitude': -33.8688, 'longitude': 151.2093},
            'timestamp': '2024-01-15T14:30:00Z',
            'description': 'Test incident',
            'incident_type': 'accident',
            'weather': {'temperature': 15.0, 'wind_speed': 7.0},
        }
        
        encoded = encoder.encode(incident)
        
        assert 'temperature_standardized' in encoded
        assert 'wind_speed_standardized' in encoded
        assert 'distance_to_highway' in encoded
        assert 'location_grid_cell' in encoded
        assert 'is_rush_hour' in encoded
        assert 'time_of_day_bins' in encoded
        assert '_encoding_metadata' in encoded
    
    def test_encode_handles_missing_values(self):
        """Test encoding with missing values."""
        encoder = FeatureEncoder()
        
        train_data = pd.DataFrame({
            'temperature': [10.0, 20.0, 30.0],
            'wind_speed': [5.0, 10.0, 15.0],
        })
        encoder.fit(train_data)
        
        incident = {
            'location': {'latitude': -33.8688, 'longitude': 151.2093},
            'timestamp': '2024-01-15T14:30:00Z',
            'weather': {'temperature': None, 'wind_speed': None},
        }
        
        encoded = encoder.encode(incident)
        
        # Should not raise an error and should have imputed values
        assert 'temperature_standardized' in encoded or 'temperature' in encoded
        assert '_encoding_metadata' in encoded
        imputed_features = encoded['_encoding_metadata']['imputed_features']
        assert len(imputed_features) > 0
    
    def test_encode_preserves_original_data(self):
        """Test that encoding preserves original incident data."""
        encoder = FeatureEncoder()
        encoder.fit(pd.DataFrame({'temperature': [20.0]}))
        
        incident = {
            'incident_id': 'test-123',
            'location': {'latitude': -33.8688, 'longitude': 151.2093},
            'timestamp': '2024-01-15T14:30:00Z',
            'description': 'Test incident',
            'incident_type': 'accident',
        }
        
        encoded = encoder.encode(incident)
        
        assert encoded['incident_id'] == 'test-123'
        assert encoded['description'] == 'Test incident'
        assert encoded['incident_type'] == 'accident'
    
    def test_encode_includes_metadata(self):
        """Test that encoded output includes metadata."""
        encoder = FeatureEncoder()
        encoder.fit(pd.DataFrame({'temperature': [20.0]}))
        
        incident = {
            'location': {'latitude': -33.8688, 'longitude': 151.2093},
            'timestamp': '2024-01-15T14:30:00Z',
        }
        
        encoded = encoder.encode(incident)
        
        assert '_encoding_metadata' in encoded
        assert 'imputed_features' in encoded['_encoding_metadata']
        assert 'encoded_datetime' in encoded['_encoding_metadata']


class TestFeatureEncoderEncodeBatch:
    """Tests for batch encoding."""
    
    def test_encode_batch(self):
        """Test batch encoding multiple incidents."""
        encoder = FeatureEncoder()
        encoder.fit(pd.DataFrame({'temperature': [20.0]}))
        
        incidents = [
            {
                'incident_id': f'test-{i}',
                'location': {'latitude': -33.8688, 'longitude': 151.2093},
                'timestamp': '2024-01-15T14:30:00Z',
            }
            for i in range(5)
        ]
        
        encoded_batch = encoder.encode_batch(incidents)
        
        assert len(encoded_batch) == 5
        for encoded in encoded_batch:
            assert '_encoding_metadata' in encoded
    
    def test_encode_batch_preserves_order(self):
        """Test that batch encoding preserves incident order."""
        encoder = FeatureEncoder()
        encoder.fit(pd.DataFrame({'temperature': [20.0]}))
        
        incidents = [
            {'incident_id': f'test-{i}', 'location': {'latitude': -33.8688, 'longitude': 151.2093},
             'timestamp': '2024-01-15T14:30:00Z'}
            for i in range(10)
        ]
        
        encoded_batch = encoder.encode_batch(incidents)
        
        for i, encoded in enumerate(encoded_batch):
            assert encoded['incident_id'] == f'test-{i}'


class TestFeatureEncoderStatistics:
    """Tests for getting encoder statistics."""
    
    def test_get_feature_statistics(self):
        """Test retrieving feature statistics."""
        encoder = FeatureEncoder()
        
        train_data = pd.DataFrame({
            'incident_type': ['accident', 'congestion'],
            'temperature': [10.0, 20.0],
        })
        encoder.fit(train_data)
        
        stats = encoder.get_feature_statistics()
        
        assert stats['is_fitted'] is True
        assert 'numerical_features' in stats
        assert 'categorical_features' in stats
        assert 'feature_means' in stats
        assert 'feature_modes' in stats
        assert 'missing_percentages' in stats


# ============================================================================
# FeatureStore Tests
# ============================================================================


class TestFeatureStoreBasicOperations:
    """Tests for basic FeatureStore operations."""
    
    def test_create_feature_store(self):
        """Test creating a new feature store."""
        store = FeatureStore(dataset_version="v1.0")
        
        assert store.dataset_version == "v1.0"
        assert len(store.get_all_incident_ids()) == 0
    
    def test_add_feature_vector(self):
        """Test adding a feature vector to the store."""
        store = FeatureStore()
        
        features = {'temperature': 20.0, 'wind_speed': 5.0}
        store.add_feature_vector('incident-1', features)
        
        assert store.has_feature_vector('incident-1')
    
    def test_get_feature_vector(self):
        """Test retrieving a feature vector."""
        store = FeatureStore()
        
        features = {'temperature': 20.0, 'wind_speed': 5.0}
        metadata = {'source': 'test'}
        store.add_feature_vector('incident-1', features, metadata)
        
        retrieved = store.get_feature_vector('incident-1')
        
        assert retrieved is not None
        assert retrieved['features'] == features
        assert retrieved['metadata'] == metadata
        assert retrieved['dataset_version'] == store.dataset_version
    
    def test_get_nonexistent_feature_vector(self):
        """Test retrieving a nonexistent feature vector."""
        store = FeatureStore()
        
        retrieved = store.get_feature_vector('nonexistent')
        
        assert retrieved is None
    
    def test_has_feature_vector_true(self):
        """Test has_feature_vector when vector exists."""
        store = FeatureStore()
        
        store.add_feature_vector('incident-1', {'temp': 20.0})
        
        assert store.has_feature_vector('incident-1') is True
    
    def test_has_feature_vector_false(self):
        """Test has_feature_vector when vector doesn't exist."""
        store = FeatureStore()
        
        assert store.has_feature_vector('nonexistent') is False


class TestFeatureStoreRetrievalOperations:
    """Tests for feature vector retrieval."""
    
    def test_get_all_incident_ids(self):
        """Test getting all incident IDs from store."""
        store = FeatureStore()
        
        for i in range(5):
            store.add_feature_vector(f'incident-{i}', {'temp': 20.0})
        
        ids = store.get_all_incident_ids()
        
        assert len(ids) == 5
        assert all(f'incident-{i}' in ids for i in range(5))
    
    def test_get_all_incident_ids_empty_store(self):
        """Test getting incident IDs from empty store."""
        store = FeatureStore()
        
        ids = store.get_all_incident_ids()
        
        assert len(ids) == 0


class TestFeatureStoreDeletionOperations:
    """Tests for feature vector deletion."""
    
    def test_delete_feature_vector_existing(self):
        """Test deleting an existing feature vector."""
        store = FeatureStore()
        store.add_feature_vector('incident-1', {'temp': 20.0})
        
        deleted = store.delete_feature_vector('incident-1')
        
        assert deleted is True
        assert not store.has_feature_vector('incident-1')
    
    def test_delete_feature_vector_nonexistent(self):
        """Test deleting a nonexistent feature vector."""
        store = FeatureStore()
        
        deleted = store.delete_feature_vector('nonexistent')
        
        assert deleted is False
    
    def test_clear_feature_store(self):
        """Test clearing all feature vectors."""
        store = FeatureStore()
        
        for i in range(5):
            store.add_feature_vector(f'incident-{i}', {'temp': 20.0})
        
        store.clear()
        
        assert len(store.get_all_incident_ids()) == 0


class TestFeatureStoreStatistics:
    """Tests for feature store statistics."""
    
    def test_get_statistics_empty_store(self):
        """Test getting statistics from empty store."""
        store = FeatureStore(dataset_version="v1.0")
        
        stats = store.get_statistics()
        
        assert stats['total_vectors'] == 0
        assert stats['dataset_version'] == "v1.0"
        assert stats['feature_count'] == 0
        assert stats['storage_size_mb'] == 0
    
    def test_get_statistics_with_vectors(self):
        """Test getting statistics with stored vectors."""
        store = FeatureStore(dataset_version="v2.0")
        
        features = {'temp': 20.0, 'wind': 5.0, 'humidity': 60.0}
        for i in range(10):
            store.add_feature_vector(f'incident-{i}', features)
        
        stats = store.get_statistics()
        
        assert stats['total_vectors'] == 10
        assert stats['dataset_version'] == "v2.0"
        assert stats['feature_count'] == 3
        assert stats['storage_size_mb'] > 0
    
    def test_get_statistics_includes_timestamps(self):
        """Test that statistics include created_at timestamp."""
        store = FeatureStore()
        
        stats = store.get_statistics()
        
        assert 'created_at' in stats
        assert isinstance(stats['created_at'], str)


class TestFeatureStoreVersioning:
    """Tests for feature store versioning."""
    
    def test_feature_store_preserves_dataset_version(self):
        """Test that added vectors preserve dataset version."""
        store = FeatureStore(dataset_version="v3.0")
        
        store.add_feature_vector('incident-1', {'temp': 20.0})
        
        retrieved = store.get_feature_vector('incident-1')
        
        assert retrieved['dataset_version'] == "v3.0"
    
    def test_multiple_stores_different_versions(self):
        """Test multiple stores with different versions."""
        store_v1 = FeatureStore(dataset_version="v1.0")
        store_v2 = FeatureStore(dataset_version="v2.0")
        
        store_v1.add_feature_vector('incident-1', {'temp': 20.0})
        store_v2.add_feature_vector('incident-1', {'temp': 20.0})
        
        v1_retrieved = store_v1.get_feature_vector('incident-1')
        v2_retrieved = store_v2.get_feature_vector('incident-1')
        
        assert v1_retrieved['dataset_version'] == "v1.0"
        assert v2_retrieved['dataset_version'] == "v2.0"


class TestFeatureStoreMetadata:
    """Tests for feature vector metadata handling."""
    
    def test_add_vector_with_metadata(self):
        """Test adding vector with metadata."""
        store = FeatureStore()
        
        features = {'temp': 20.0}
        metadata = {'source': 'api', 'quality_score': 0.95}
        
        store.add_feature_vector('incident-1', features, metadata)
        retrieved = store.get_feature_vector('incident-1')
        
        assert retrieved['metadata'] == metadata
    
    def test_add_vector_without_metadata(self):
        """Test adding vector without metadata uses default."""
        store = FeatureStore()
        
        features = {'temp': 20.0}
        store.add_feature_vector('incident-1', features)
        
        retrieved = store.get_feature_vector('incident-1')
        
        assert retrieved['metadata'] == {}
    
    def test_vector_includes_stored_timestamp(self):
        """Test that stored vectors include timestamp."""
        store = FeatureStore()
        
        store.add_feature_vector('incident-1', {'temp': 20.0})
        retrieved = store.get_feature_vector('incident-1')
        
        assert 'stored_at' in retrieved
        assert isinstance(retrieved['stored_at'], str)


# ============================================================================
# Integration Tests
# ============================================================================


class TestFeatureEncoderStoreIntegration:
    """Integration tests for FeatureEncoder and FeatureStore."""
    
    def test_encode_and_store_workflow(self):
        """Test complete workflow of encoding and storing features."""
        # Train encoder
        encoder = FeatureEncoder()
        train_data = pd.DataFrame({
            'incident_type': ['accident', 'congestion'],
            'temperature': [10.0, 20.0],
            'wind_speed': [5.0, 10.0],
        })
        encoder.fit(train_data)
        
        # Create store
        store = FeatureStore(dataset_version="v1.0")
        
        # Encode and store incidents
        incidents = [
            {
                'incident_id': f'incident-{i}',
                'location': {'latitude': -33.8688, 'longitude': 151.2093},
                'timestamp': '2024-01-15T14:30:00Z',
                'incident_type': 'accident',
                'weather': {'temperature': 15.0, 'wind_speed': 7.0},
            }
            for i in range(5)
        ]
        
        for incident in incidents:
            encoded = encoder.encode(incident)
            store.add_feature_vector(incident['incident_id'], encoded, {'source': 'test'})
        
        # Verify storage
        assert store.get_statistics()['total_vectors'] == 5
        
        for i in range(5):
            vector = store.get_feature_vector(f'incident-{i}')
            assert vector is not None
            assert vector['dataset_version'] == "v1.0"
    
    def test_large_batch_encoding_and_storage(self):
        """Test encoding and storing a large batch of incidents."""
        encoder = FeatureEncoder()
        train_data = pd.DataFrame({
            'temperature': np.linspace(0, 40, 100),
            'wind_speed': np.linspace(0, 20, 100),
        })
        encoder.fit(train_data)
        
        store = FeatureStore(dataset_version="v1.0")
        
        # Create 100 incidents
        incidents = [
            {
                'incident_id': f'incident-{i:04d}',
                'location': {'latitude': -33.8688, 'longitude': 151.2093},
                'timestamp': '2024-01-15T14:30:00Z',
                'weather': {'temperature': float(i), 'wind_speed': float(i % 20)},
            }
            for i in range(100)
        ]
        
        encoded_batch = encoder.encode_batch(incidents)
        
        for encoded in encoded_batch:
            store.add_feature_vector(encoded['incident_id'], encoded)
        
        assert store.get_statistics()['total_vectors'] == 100
