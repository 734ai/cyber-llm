#!/bin/bash
# Simple system startup script

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}/src:${PYTHONPATH:-}"

echo "🚀 Starting Persistent Cognitive Multi-Agent System..."
echo "Project Root: $SCRIPT_DIR"
echo "Python Path: $PYTHONPATH"
echo ""

# Check if databases exist
if [ -f "/home/o1/Desktop/data/cognitive/cognitive_system.db" ]; then
    echo "✅ Cognitive database found"
else
    echo "❌ Cognitive database not found - run ./setup_databases.sh first"
    exit 1
fi

if [ -f "/home/o1/Desktop/data/server/server_system.db" ]; then
    echo "✅ Server database found"
else
    echo "❌ Server database not found - run ./setup_databases.sh first"
    exit 1
fi

echo ""
echo "Starting server on http://localhost:8080"
echo "Press Ctrl+C to stop"
echo ""

# Start the cognitive system
python3 "${SCRIPT_DIR}/src/startup/persistent_cognitive_startup.py" \
    --host 0.0.0.0 \
    --port 8080 \
    --cognitive-db "/home/o1/Desktop/data/cognitive/cognitive_system.db" \
    --server-db "/home/o1/Desktop/data/server/server_system.db" \
    --environment production
