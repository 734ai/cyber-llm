#!/bin/bash
# System monitoring script for Persistent Cognitive Multi-Agent System

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
API_URL="http://localhost:8080"

echo "🔍 Cognitive System Health Check - $(date)"
echo "=================================================================="

# Check if system is running
if curl -s "${API_URL}/health" > /dev/null 2>&1; then
    echo "✅ System is responding"
    
    # Get health status
    HEALTH_RESPONSE=$(curl -s "${API_URL}/health")
    echo "📊 Health Status:"
    echo "$HEALTH_RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$HEALTH_RESPONSE"
    
    # Get system stats
    if curl -s "${API_URL}/admin/stats" > /dev/null 2>&1; then
        echo -e "\n📈 System Statistics:"
        STATS_RESPONSE=$(curl -s "${API_URL}/admin/stats")
        echo "$STATS_RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$STATS_RESPONSE"
    fi
    
else
    echo "❌ System is not responding"
    echo "🔧 Check if the service is running: systemctl status cognitive-system"
fi

echo "=================================================================="
