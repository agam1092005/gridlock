#!/bin/bash

# PostgreSQL Initialization Script for Gridlock 2.0
# This script runs after schema.sql to perform additional setup tasks
# It's called automatically by the Docker entrypoint

set -e

echo "=========================================="
echo "PostgreSQL Initialization"
echo "=========================================="

# Create extensions if not already present
echo "Creating PostgreSQL extensions..."

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    -- UUID generation
    CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
    
    -- PostGIS for geospatial queries
    CREATE EXTENSION IF NOT EXISTS postgis;
    
    -- Text search
    CREATE EXTENSION IF NOT EXISTS pg_trgm;
    
    -- Vector similarity search (for embeddings)
    CREATE EXTENSION IF NOT EXISTS vector;
    
    -- Statistics functions
    CREATE EXTENSION IF NOT EXISTS tablefunc;

    GRANT ALL PRIVILEGES ON DATABASE "$POSTGRES_DB" TO "$POSTGRES_USER";
    
    SELECT version();
EOSQL

echo "✓ PostgreSQL extensions created successfully"

# Create indexes for performance
echo "Creating indexes for common queries..."

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    -- Index for vector similarity search on embeddings (if pgvector is available)
    -- Uncomment this after pgvector extension is installed
    -- CREATE INDEX IF NOT EXISTS idx_features_embedding ON features 
    --   USING ivfflat (description_embedding vector_cosine_ops) 
    --   WITH (lists = 100);

    -- Additional indexes for common queries
    CREATE INDEX IF NOT EXISTS idx_incidents_timestamp_desc ON incidents(timestamp DESC);
    CREATE INDEX IF NOT EXISTS idx_predictions_created_at_desc ON predictions(created_at DESC);
    
    ANALYZE;

    SELECT 'Index creation completed';
EOSQL

echo "✓ Indexes created successfully"

# Verify schema
echo "Verifying schema installation..."

TABLE_COUNT=$(psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" -tc "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public';")
TABLE_COUNT=$(echo "$TABLE_COUNT" | xargs)

if [ "$TABLE_COUNT" -gt 0 ]; then
    echo "✓ Schema verification passed - found $TABLE_COUNT tables"
else
    echo "⚠ Warning: No tables found in schema"
fi

echo "=========================================="
echo "PostgreSQL initialization complete"
echo "=========================================="
