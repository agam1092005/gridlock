-- Gridlock 2.0 PostgreSQL Schema
-- Initialize core tables for incidents, predictions, models, features, and audit logs

-- Enable extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS vector;

-- Incident type enumeration
CREATE TYPE incident_type_enum AS ENUM ('accident', 'congestion', 'roadwork', 'weather', 'unknown');

-- Incident severity level enumeration
CREATE TYPE severity_level_enum AS ENUM ('low', 'medium', 'high', 'critical');

-- Prediction status enumeration
CREATE TYPE prediction_status_enum AS ENUM ('pending', 'processing', 'completed', 'failed');

-- Model status enumeration
CREATE TYPE model_status_enum AS ENUM ('active', 'archived', 'testing', 'deprecated');

-- ============================================================================
-- INCIDENTS TABLE
-- ============================================================================
CREATE TABLE incidents (
    incident_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    location_latitude DECIMAL(10, 8) NOT NULL,
    location_longitude DECIMAL(11, 8) NOT NULL,
    location_geom GEOMETRY(POINT, 4326) NOT NULL,  -- PostGIS geometry for spatial queries
    timestamp TIMESTAMPTZ NOT NULL,
    start_datetime TIMESTAMPTZ NOT NULL,
    end_datetime TIMESTAMPTZ,
    description TEXT NOT NULL,
    incident_type incident_type_enum NOT NULL DEFAULT 'unknown',
    severity_initial DECIMAL(5, 2),  -- 0-100, optional initial severity from reporter
    status prediction_status_enum DEFAULT 'pending',
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    
    -- Metadata
    is_ongoing BOOLEAN DEFAULT FALSE,  -- True if end_datetime is missing and incident is still active
    data_source VARCHAR(100),  -- Source of incident report (API, sensor, manual, etc.)
    reporter_id VARCHAR(100),  -- Optional reporter identifier
    
    CONSTRAINT valid_location CHECK (
        location_latitude >= -90 AND location_latitude <= 90 AND
        location_longitude >= -180 AND location_longitude <= 180
    )
);

CREATE INDEX idx_incidents_timestamp ON incidents(timestamp DESC);
CREATE INDEX idx_incidents_location ON incidents USING GIST(location_geom);
CREATE INDEX idx_incidents_type ON incidents(incident_type);
CREATE INDEX idx_incidents_status ON incidents(status);
CREATE INDEX idx_incidents_created_at ON incidents(created_at DESC);

-- ============================================================================
-- PREDICTIONS TABLE
-- ============================================================================
CREATE TABLE predictions (
    prediction_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    incident_id UUID NOT NULL REFERENCES incidents(incident_id) ON DELETE CASCADE,
    
    -- Module A: Severity & Duration Predictions
    severity_score DECIMAL(5, 2),  -- 0-100
    severity_confidence_lower DECIMAL(5, 2),  -- 95% CI lower bound
    severity_confidence_upper DECIMAL(5, 2),  -- 95% CI upper bound
    severity_confidence_level DECIMAL(3, 2),  -- Default 0.95
    
    duration_estimate_minutes DECIMAL(8, 2),  -- Predicted duration in minutes
    duration_confidence_lower DECIMAL(8, 2),  -- 95% CI lower bound
    duration_confidence_upper DECIMAL(8, 2),  -- 95% CI upper bound
    duration_confidence_level DECIMAL(3, 2),  -- Default 0.95
    
    -- Module B: Spatial-Temporal Congestion Predictions
    congestion_forecast_horizon_minutes INT,  -- Prediction horizon (typically 30)
    congestion_predictions JSONB,  -- Array of {timestamp, occupancy_percent, affected_nodes}
    congestion_heatmap_geojson JSONB,  -- GeoJSON feature collection for visualization
    
    -- Component latencies (milliseconds)
    latency_data_pipeline_ms DECIMAL(8, 2),
    latency_module_a_ms DECIMAL(8, 2),
    latency_module_b_ms DECIMAL(8, 2),
    latency_playbook_ms DECIMAL(8, 2),
    latency_shap_ms DECIMAL(8, 2),
    latency_aggregation_ms DECIMAL(8, 2),
    latency_total_ms DECIMAL(8, 2),  -- End-to-end latency
    
    -- Status and timestamps
    status prediction_status_enum DEFAULT 'pending',
    error_message TEXT,  -- If status is 'failed'
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    
    -- Model versions used
    module_a_model_version VARCHAR(100),
    module_b_model_version VARCHAR(100),
    
    -- Processing metadata
    request_id VARCHAR(100),  -- Correlate with API logs
    is_fallback BOOLEAN DEFAULT FALSE  -- True if any component used fallback
);

CREATE INDEX idx_predictions_incident_id ON predictions(incident_id);
CREATE INDEX idx_predictions_status ON predictions(status);
CREATE INDEX idx_predictions_created_at ON predictions(created_at DESC);
CREATE INDEX idx_predictions_latency_total ON predictions(latency_total_ms);

-- ============================================================================
-- PLAYBOOK_RECOMMENDATIONS TABLE
-- ============================================================================
CREATE TABLE playbook_recommendations (
    playbook_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    prediction_id UUID NOT NULL REFERENCES predictions(prediction_id) ON DELETE CASCADE,
    incident_id UUID NOT NULL REFERENCES incidents(incident_id) ON DELETE CASCADE,
    
    -- Playbook content
    incident_summary TEXT,  -- Natural language summary
    recommended_actions JSONB,  -- Array of {priority, action, timeline}
    driver_alert TEXT,  -- Public-facing driver communication
    emergency_services_notification TEXT,  -- Internal alert to emergency services
    internal_operations_alert TEXT,  -- Internal traffic operations alert
    
    -- Monitoring recommendations
    monitoring_plan JSONB,  -- {check_points, escalation_triggers}
    
    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    
    -- Feedback
    operator_feedback TEXT,
    feedback_timestamp TIMESTAMPTZ
);

CREATE INDEX idx_playbook_prediction_id ON playbook_recommendations(prediction_id);
CREATE INDEX idx_playbook_incident_id ON playbook_recommendations(incident_id);

-- ============================================================================
-- SHAP_EXPLANATIONS TABLE
-- ============================================================================
CREATE TABLE shap_explanations (
    explanation_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    prediction_id UUID NOT NULL REFERENCES predictions(prediction_id) ON DELETE CASCADE,
    
    -- Severity explanations
    severity_base_value DECIMAL(5, 2),
    severity_predicted_value DECIMAL(5, 2),
    severity_feature_contributions JSONB,  -- Array of {feature_name, shap_value, contribution_direction}
    
    -- Duration explanations
    duration_base_value DECIMAL(8, 2),
    duration_predicted_value DECIMAL(8, 2),
    duration_feature_contributions JSONB,
    
    -- Module B explanations
    congestion_influential_nodes JSONB,  -- Array of {node_id, influence_score}
    congestion_historical_patterns JSONB,  -- Historical incidents that influenced prediction
    
    -- Metadata
    computation_time_ms DECIMAL(8, 2),
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_shap_explanation_prediction_id ON shap_explanations(prediction_id);

-- ============================================================================
-- MODELS TABLE (Model Registry & Versioning)
-- ============================================================================
CREATE TABLE models (
    model_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_name VARCHAR(255) NOT NULL,  -- 'module_a_severity', 'module_a_duration', 'module_b_congestion', etc.
    version VARCHAR(100) NOT NULL,  -- e.g., 'v1', '20240115_001', semantic versioning
    status model_status_enum DEFAULT 'testing',
    
    -- Model metadata
    model_type VARCHAR(100),  -- 'lightgbm', 'bigru', 'stgcn', 'cox', etc.
    framework VARCHAR(100),  -- 'lightgbm', 'pytorch', 'lifelines', etc.
    artifact_path VARCHAR(500),  -- Path to serialized model (relative to models/artifacts/)
    
    -- Training information
    training_dataset_version VARCHAR(100),  -- Link to dataset version
    training_date TIMESTAMPTZ,
    training_duration_seconds INT,
    training_samples INT,
    
    -- Metrics
    validation_rmse DECIMAL(10, 4),
    validation_mae DECIMAL(10, 4),
    validation_r2_score DECIMAL(5, 4),
    validation_c_index DECIMAL(5, 4),  -- For survival models
    validation_f1_score DECIMAL(5, 4),
    test_rmse DECIMAL(10, 4),
    test_mae DECIMAL(10, 4),
    test_r2_score DECIMAL(5, 4),
    
    -- Hyperparameters
    hyperparameters JSONB,  -- All training hyperparameters
    
    -- Deployment
    deployment_date TIMESTAMPTZ,
    deployment_notes TEXT,
    previous_model_id UUID REFERENCES models(model_id),  -- Link to prior version
    
    -- Monitoring
    prediction_count INT DEFAULT 0,
    average_prediction_latency_ms DECIMAL(8, 2),
    error_rate DECIMAL(5, 4),  -- Errors / total predictions
    accuracy_degradation_percent DECIMAL(5, 2),  -- Degradation vs baseline
    
    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT unique_model_version UNIQUE (model_name, version)
);

CREATE INDEX idx_models_name_status ON models(model_name, status);
CREATE INDEX idx_models_deployment_date ON models(deployment_date DESC);
CREATE INDEX idx_models_model_type ON models(model_type);

-- ============================================================================
-- FEATURES TABLE (Feature Store & Version Control)
-- ============================================================================
CREATE TABLE features (
    feature_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    incident_id UUID NOT NULL REFERENCES incidents(incident_id) ON DELETE CASCADE,
    
    -- Text embedding (IndicBERT)
    description_embedding VECTOR(768),  -- 768-dimensional embedding vector
    embedding_hash VARCHAR(64),  -- SHA256 hash of description for cache lookup
    
    -- Structured features
    structured_features JSONB,  -- {incident_type, hour_of_day, day_of_week, is_rush_hour, weather_temp, ...}
    
    -- Historical context
    historical_context JSONB,  -- {past_incidents, historical_congestion_pattern, ...}
    
    -- Derived features
    distance_to_highway_km DECIMAL(8, 2),
    location_grid_cell_x INT,
    location_grid_cell_y INT,
    proximity_to_intersections INT,
    
    -- Feature versions
    feature_schema_version VARCHAR(100),  -- Version of feature definition
    dataset_version VARCHAR(100),  -- Version of dataset these features came from
    
    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    extracted_at TIMESTAMPTZ  -- When features were extracted from incident
);

CREATE INDEX idx_features_incident_id ON features(incident_id);
CREATE INDEX idx_features_embedding_hash ON features(embedding_hash);
CREATE INDEX idx_features_dataset_version ON features(dataset_version);

-- ============================================================================
-- AUDIT_LOGS TABLE (Compliance & Debugging)
-- ============================================================================
CREATE TABLE audit_logs (
    log_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    component VARCHAR(100),  -- 'data_pipeline', 'module_a', 'module_b', 'api', etc.
    action VARCHAR(255),  -- 'incident_submitted', 'validation_failed', 'prediction_generated', etc.
    severity VARCHAR(20),  -- 'debug', 'info', 'warning', 'error', 'critical'
    message TEXT,
    
    -- Context
    incident_id UUID REFERENCES incidents(incident_id) ON DELETE SET NULL,
    prediction_id UUID REFERENCES predictions(prediction_id) ON DELETE SET NULL,
    request_id VARCHAR(100),  -- Correlate with API request logs
    user_id VARCHAR(100),  -- If applicable
    api_key_id VARCHAR(100),  -- If API-based access
    
    -- Structured data
    metadata JSONB,  -- Additional context (latencies, counts, error details, etc.)
    
    -- Performance
    duration_ms DECIMAL(10, 2),  -- Operation duration
    
    -- Compliance
    data_processed_records INT,  -- Number of records processed in this operation
    data_validation_failed_count INT,
    data_validation_passed_count INT
);

CREATE INDEX idx_audit_logs_timestamp ON audit_logs(timestamp DESC);
CREATE INDEX idx_audit_logs_component ON audit_logs(component);
CREATE INDEX idx_audit_logs_action ON audit_logs(action);
CREATE INDEX idx_audit_logs_incident_id ON audit_logs(incident_id);
CREATE INDEX idx_audit_logs_request_id ON audit_logs(request_id);

-- ============================================================================
-- LOCATION_GRID_CELLS TABLE (Spatial Reference for Congestion Mapping)
-- ============================================================================
CREATE TABLE location_grid_cells (
    cell_id VARCHAR(50) PRIMARY KEY,  -- e.g., 'grid_001_002'
    grid_x INT NOT NULL,
    grid_y INT NOT NULL,
    geometry GEOMETRY(POLYGON, 4326) NOT NULL,
    center_lat DECIMAL(10, 8),
    center_lon DECIMAL(11, 8),
    
    -- Cell properties
    road_class VARCHAR(50),  -- 'highway', 'arterial', 'local'
    typical_capacity INT,  -- Vehicles per hour
    historical_congestion_baseline DECIMAL(5, 2),  -- 0-100 occupancy baseline
    
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_grid_cells_geometry ON location_grid_cells USING GIST(geometry);
CREATE INDEX idx_grid_cells_coordinates ON location_grid_cells(grid_x, grid_y);

-- ============================================================================
-- ROAD_NETWORK_GRAPH TABLE (Spatial-Temporal Graph Nodes)
-- ============================================================================
CREATE TABLE road_network_graph (
    node_id VARCHAR(100) PRIMARY KEY,  -- e.g., 'seg_001', 'node_highway_i95_mile_10'
    node_type VARCHAR(50),  -- 'road_segment', 'grid_cell', 'intersection'
    geometry GEOMETRY(LINESTRING, 4326),  -- For road segments
    center_geom GEOMETRY(POINT, 4326),  -- Center point for spatial lookup
    
    -- Node properties
    road_class VARCHAR(50),  -- 'highway', 'arterial', 'local'
    speed_limit_kmh INT,
    capacity_vehicles_per_hour INT,
    length_meters INT,
    
    -- Connectivity
    adjacent_node_ids TEXT[],  -- Array of neighboring node IDs
    
    -- Historical baseline
    historical_occupancy_baseline DECIMAL(5, 2),  -- Average occupancy 0-100%
    
    -- Graph metadata
    last_updated TIMESTAMPTZ,
    graph_version VARCHAR(100)
);

CREATE INDEX idx_road_network_geom ON road_network_graph USING GIST(center_geom);
CREATE INDEX idx_road_network_road_class ON road_network_graph(road_class);

-- ============================================================================
-- API_KEYS TABLE (Authentication)
-- ============================================================================
CREATE TABLE api_keys (
    api_key_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    api_key_hash VARCHAR(64) NOT NULL UNIQUE,  -- SHA256 hash of actual key
    description VARCHAR(255),
    is_active BOOLEAN DEFAULT TRUE,
    rate_limit_requests_per_second INT DEFAULT 10,
    
    -- Permissions
    can_submit_incidents BOOLEAN DEFAULT TRUE,
    can_query_predictions BOOLEAN DEFAULT TRUE,
    can_access_health BOOLEAN DEFAULT TRUE,
    
    -- Tracking
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    last_used_at TIMESTAMPTZ,
    created_by VARCHAR(100),
    
    -- Audit
    disabled_at TIMESTAMPTZ,
    disabled_reason VARCHAR(255)
);

CREATE INDEX idx_api_keys_hash ON api_keys(api_key_hash);
CREATE INDEX idx_api_keys_active ON api_keys(is_active);

-- ============================================================================
-- DATA_QUALITY_METRICS TABLE (Monitoring)
-- ============================================================================
CREATE TABLE data_quality_metrics (
    metric_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    
    -- Counters
    total_records_received INT,
    validation_passed_count INT,
    validation_failed_count INT,
    records_with_missing_end_datetime INT,
    
    -- Quality percentages
    overall_pass_rate DECIMAL(5, 4),  -- 0.0 to 1.0
    completeness_percent DECIMAL(5, 2),
    
    -- Aggregation window
    window_start_time TIMESTAMPTZ,
    window_end_time TIMESTAMPTZ,
    window_duration_minutes INT
);

CREATE INDEX idx_data_quality_metrics_timestamp ON data_quality_metrics(timestamp DESC);

-- ============================================================================
-- SYSTEM HEALTH TABLE (Monitoring)
-- ============================================================================
CREATE TABLE system_health_snapshots (
    health_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    
    -- Service status
    data_pipeline_status VARCHAR(20),  -- 'up', 'degraded', 'down'
    module_a_status VARCHAR(20),
    module_b_status VARCHAR(20),
    database_status VARCHAR(20),
    redis_status VARCHAR(20),
    
    -- Performance
    data_pipeline_latency_ms DECIMAL(8, 2),
    module_a_latency_ms DECIMAL(8, 2),
    module_b_latency_ms DECIMAL(8, 2),
    api_latency_p95_ms DECIMAL(8, 2),
    api_latency_p99_ms DECIMAL(8, 2),
    
    -- Throughput
    predictions_per_minute INT,
    active_requests INT,
    
    -- Resource usage
    database_pool_size INT,
    redis_memory_bytes BIGINT,
    
    -- Overall
    overall_status VARCHAR(20)  -- 'healthy', 'degraded', 'unhealthy'
);

CREATE INDEX idx_health_snapshots_timestamp ON system_health_snapshots(timestamp DESC);

-- ============================================================================
-- MATERIALIZED VIEW: Recent Predictions Summary
-- ============================================================================
CREATE MATERIALIZED VIEW recent_predictions_summary AS
SELECT
    i.incident_id,
    i.timestamp,
    i.location_latitude,
    i.location_longitude,
    i.incident_type,
    i.description,
    p.severity_score,
    p.duration_estimate_minutes,
    p.status,
    p.latency_total_ms,
    p.created_at AS prediction_created_at
FROM incidents i
LEFT JOIN predictions p ON i.incident_id = p.incident_id
WHERE i.created_at > CURRENT_TIMESTAMP - INTERVAL '7 days'
ORDER BY i.timestamp DESC;

CREATE INDEX idx_recent_predictions_incident_id ON recent_predictions_summary(incident_id);

-- ============================================================================
-- FUNCTION: Update updated_at timestamp automatically
-- ============================================================================
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply trigger to incidents table
CREATE TRIGGER trigger_incidents_updated_at
BEFORE UPDATE ON incidents
FOR EACH ROW
EXECUTE FUNCTION update_updated_at_column();

-- Apply trigger to predictions table
CREATE TRIGGER trigger_predictions_updated_at
BEFORE UPDATE ON predictions
FOR EACH ROW
EXECUTE FUNCTION update_updated_at_column();

-- Apply trigger to playbook_recommendations table
CREATE TRIGGER trigger_playbook_updated_at
BEFORE UPDATE ON playbook_recommendations
FOR EACH ROW
EXECUTE FUNCTION update_updated_at_column();

-- Apply trigger to models table
CREATE TRIGGER trigger_models_updated_at
BEFORE UPDATE ON models
FOR EACH ROW
EXECUTE FUNCTION update_updated_at_column();

-- ============================================================================
-- COMMENTS FOR DOCUMENTATION
-- ============================================================================

COMMENT ON TABLE incidents IS 'Core incident data: location, time, description, type. Base entity for all predictions and monitoring.';
COMMENT ON TABLE predictions IS 'ML model predictions: severity scores, duration estimates, congestion forecasts, component latencies.';
COMMENT ON TABLE playbook_recommendations IS 'Generated operator actions and recommendations for each incident prediction.';
COMMENT ON TABLE shap_explanations IS 'Feature importance scores and prediction explanations for model transparency.';
COMMENT ON TABLE models IS 'Model registry and versioning: all trained models with metrics, deployments, performance monitoring.';
COMMENT ON TABLE features IS 'Feature store: extracted features for each incident including embeddings and structured features.';
COMMENT ON TABLE audit_logs IS 'Audit trail for compliance, debugging, and performance analysis.';

-- ============================================================================
-- SEED DATA (Optional: Initial Configuration)
-- ============================================================================

-- Insert initial model registry entries (will be updated during training)
INSERT INTO models (model_name, version, status, model_type, framework, artifact_path, created_at)
VALUES
    ('module_a_severity', 'v0', 'archived', 'lightgbm', 'lightgbm', 'module_a/v0/lightgbm_severity.pkl', CURRENT_TIMESTAMP),
    ('module_a_duration', 'v0', 'archived', 'lightgbm', 'lightgbm', 'module_a/v0/lightgbm_duration.pkl', CURRENT_TIMESTAMP),
    ('module_b_congestion', 'v0', 'archived', 'stgcn', 'pytorch', 'module_b/v0/stgcn_model.pth', CURRENT_TIMESTAMP)
ON CONFLICT DO NOTHING;

-- ============================================================================
-- COMMENTS AND NOTES
-- ============================================================================
-- 
-- PostGIS Note:
-- - location_geom column requires: CREATE EXTENSION postgis;
-- - Spatial queries: SELECT * FROM incidents WHERE ST_DWithin(location_geom, ST_GeomFromText('POINT(...)', 4326), 2000);
--
-- pgvector Extension (Optional):
-- - CREATE EXTENSION vector;
-- - For vector similarity search on embeddings: CREATE INDEX ON features USING ivfflat (description_embedding vector_cosine_ops);
--
-- Maintenance:
-- - Run VACUUM ANALYZE periodically to maintain index performance
-- - Archive old audit logs and predictions (>90 days) to archive tables
-- - Refresh materialized views: REFRESH MATERIALIZED VIEW recent_predictions_summary;
--
