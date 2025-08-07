#!/bin/bash
# Persistent Cognitive Multi-Agent System Startup Script
# Generated during deployment

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${SCRIPT_DIR}/config/production/production_config.yaml"

echo "🚀 Starting Persistent Cognitive Multi-Agent System..."
echo "Configuration: $CONFIG_FILE"
echo "Timestamp: $(date)"

# Set Python path
export PYTHONPATH="${SCRIPT_DIR}/src:${PYTHONPATH:-}"

# Start the system
python3 "${SCRIPT_DIR}/src/startup/persistent_cognitive_startup.py" \
    --config "$CONFIG_FILE" \
    --environment production \
    --host 0.0.0.0 \
    --port 8080

