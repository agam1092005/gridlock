#!/bin/bash

# Docker Entrypoint Script for Gridlock 2.0
# Handles startup initialization for all services
# Usage: Called by docker-compose during container startup

set -e  # Exit on any error
set -u  # Exit on undefined variable

# ============================================================================
# CONFIGURATION
# ============================================================================

SERVICE_NAME="${SERVICE_NAME:-unknown}"
LOG_FILE="${LOG_DIR:-./logs}/startup.log"
MAX_STARTUP_RETRIES=30
RETRY_INTERVAL=2

# Colors for output (only if stdout is a terminal)
if [ -t 1 ]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    NC='\033[0m'  # No Color
else
    RED=''
    GREEN=''
    YELLOW=''
    NC=''
fi

# ============================================================================
# LOGGING
# ============================================================================

mkdir -p "$(dirname "$LOG_FILE")"

log_info() {
    local msg="$1"
    echo -e "${GREEN}[INFO]${NC} [$(date '+%Y-%m-%d %H:%M:%S')] $msg" | tee -a "$LOG_FILE"
}

log_warn() {
    local msg="$1"
    echo -e "${YELLOW}[WARN]${NC} [$(date '+%Y-%m-%d %H:%M:%S')] $msg" | tee -a "$LOG_FILE"
}

log_error() {
    local msg="$1"
    echo -e "${RED}[ERROR]${NC} [$(date '+%Y-%m-%d %H:%M:%S')] $msg" | tee -a "$LOG_FILE"
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

wait_for_service() {
    local host="$1"
    local port="$2"
    local service_name="$3"
    local max_attempts="${4:-30}"
    
    log_info "Waiting for $service_name at $host:$port..."
    
    local attempt=1
    while [ $attempt -le $max_attempts ]; do
        if nc -z "$host" "$port" 2>/dev/null; then
            log_info "✓ $service_name is ready"
            return 0
        fi
        
        if [ $attempt -eq 1 ]; then
            log_warn "$service_name not ready yet, retrying (attempt $attempt/$max_attempts)..."
        elif [ $((attempt % 5)) -eq 0 ]; then
            log_warn "Still waiting for $service_name (attempt $attempt/$max_attempts)..."
        fi
        
        sleep "$RETRY_INTERVAL"
        ((attempt++))
    done
    
    log_error "Failed to connect to $service_name after $max_attempts attempts"
    return 1
}

check_command() {
    local cmd="$1"
    if ! command -v "$cmd" &> /dev/null; then
        log_error "Required command '$cmd' not found in PATH"
        return 1
    fi
    return 0
}

# ============================================================================
# DATABASE INITIALIZATION
# ============================================================================

init_database() {
    log_info "=== Database Initialization ===" 
    
    # Parse DATABASE_URL
    # Format: postgresql://user:password@host:port/dbname
    if [ -z "${DATABASE_URL:-}" ]; then
        log_error "DATABASE_URL not set"
        return 1
    fi
    
    # Extract connection components
    PROTO_REMOVED="${DATABASE_URL#*://}"
    DB_USER="${PROTO_REMOVED%%:*}"
    
    AUTH="${PROTO_REMOVED%%@*}"
    DB_PASSWORD="${AUTH#*:}"
    
    HOST_PORT_DB="${PROTO_REMOVED#*@}"
    DB_HOST="${HOST_PORT_DB%%:*}"
    
    PORT_DB="${HOST_PORT_DB#*:}"
    DB_PORT="${PORT_DB%%/*}"
    
    DB_NAME="${PORT_DB#*/}"
    
    log_info "Connecting to PostgreSQL: $DB_HOST:$DB_PORT as $DB_USER"
    
    # Wait for PostgreSQL
    wait_for_service "$DB_HOST" "$DB_PORT" "PostgreSQL" || return 1
    
    # Set up PGPASSWORD for passwordless auth
    export PGPASSWORD="$DB_PASSWORD"
    
    # Check if database exists
    log_info "Checking if database '$DB_NAME' exists..."
    if psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -lqt | cut -d'|' -f1 | grep -qw "$DB_NAME"; then
        log_info "✓ Database '$DB_NAME' already exists"
        
        # Verify schema exists
        log_info "Verifying schema..."
        if psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "SELECT 1 FROM information_schema.tables LIMIT 1;" 2>/dev/null | grep -q 1; then
            log_info "✓ Schema already initialized"
            return 0
        fi
    fi
    
    # Create database if it doesn't exist
    log_info "Creating database '$DB_NAME'..."
    createdb -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" "$DB_NAME" 2>/dev/null || log_warn "Database already exists or creation skipped"
    
    # Apply schema
    log_info "Applying database schema..."
    if [ -f "/app/config/schema.sql" ]; then
        psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" < /app/config/schema.sql
        if [ $? -eq 0 ]; then
            log_info "✓ Schema applied successfully"
        else
            log_error "Failed to apply schema"
            return 1
        fi
    else
        log_warn "Schema file not found at /app/config/schema.sql"
    fi
    
    # Unset password variable
    unset PGPASSWORD
    
    log_info "✓ Database initialization complete"
    return 0
}

# ============================================================================
# CACHE (REDIS) INITIALIZATION
# ============================================================================

init_cache() {
    log_info "=== Cache (Redis) Initialization ===" 
    
    if [ -z "${REDIS_HOST:-}" ] || [ -z "${REDIS_PORT:-}" ]; then
        log_error "REDIS_HOST or REDIS_PORT not set"
        return 1
    fi
    
    log_info "Connecting to Redis: $REDIS_HOST:$REDIS_PORT"
    
    # Wait for Redis
    wait_for_service "$REDIS_HOST" "$REDIS_PORT" "Redis" || return 1
    
    # Run Python redis initialization script if it exists
    if [ -f "/app/config/redis_init.py" ]; then
        log_info "Running Redis initialization script..."
        
        cd /app
        python -u /app/config/redis_init.py
        if [ $? -eq 0 ]; then
            log_info "✓ Redis initialization complete"
            return 0
        else
            log_error "Redis initialization script failed"
            return 1
        fi
    else
        log_warn "Redis initialization script not found"
        return 0
    fi
}

# ============================================================================
# SERVICE-SPECIFIC STARTUP
# ============================================================================

start_api_service() {
    log_info "=== Starting API Service ===" 
    
    # Initialize dependencies
    init_database || return 1
    init_cache || return 1
    
    log_info "Starting FastAPI server..."
    
    # Run the API server
    exec uvicorn \
        "src.api.main:app" \
        --host "${API_HOST:-0.0.0.0}" \
        --port "${API_PORT:-8000}" \
        --workers "${API_WORKERS:-4}" \
        --log-level "${LOG_LEVEL:-info}" \
        --access-log
}

start_dashboard_service() {
    log_info "=== Starting Dashboard Service ===" 
    
    # Wait for dependencies
    log_info "Waiting for API service..."
    API_HOST="${API_HOST:-api}"
    API_PORT="${API_PORT:-8000}"
    wait_for_service "$API_HOST" "$API_PORT" "API Service" || return 1
    
    log_info "Starting Streamlit dashboard..."
    
    # Run the Streamlit dashboard
    exec streamlit run \
        "src/dashboard/app.py" \
        --server.port "${STREAMLIT_PORT:-8501}" \
        --server.address "0.0.0.0" \
        --logger.level="${LOG_LEVEL:-info}" \
        --client.showErrorDetails=true
}

start_migration_service() {
    log_info "=== Database Migration Service ===" 
    
    # Initialize database only
    init_database || return 1
    
    log_info "✓ Migration service completed successfully"
    exit 0
}

# ============================================================================
# MAIN STARTUP LOGIC
# ============================================================================

main() {
    log_info "Starting Gridlock 2.0 - Service: $SERVICE_NAME"
    log_info "Environment: ${ENVIRONMENT:-development}"
    
    # Validate required tools
    check_command "nc" || log_warn "nc (netcat) not found, health checks may fail"
    
    # Ensure logs directory exists
    mkdir -p "${LOG_DIR:-./logs}"
    
    # Route based on service type
    case "${SERVICE_NAME}" in
        api)
            start_api_service
            ;;
        dashboard)
            start_dashboard_service
            ;;
        migration)
            start_migration_service
            ;;
        *)
            log_error "Unknown service: $SERVICE_NAME"
            log_info "Valid services: api, dashboard, migration"
            exit 1
            ;;
    esac
}

# ============================================================================
# SIGNAL HANDLING FOR GRACEFUL SHUTDOWN
# ============================================================================

trap 'log_info "Received SIGTERM, shutting down gracefully..."; exit 0' SIGTERM
trap 'log_info "Received SIGINT, shutting down gracefully..."; exit 0' SIGINT

# Run main function
main "$@"
