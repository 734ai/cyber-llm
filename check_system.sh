#!/bin/bash
# Simple system health check

echo "🔍 System Health Check - $(date)"
echo "=================================="

# Check if system is responding
if command -v curl &> /dev/null; then
    if curl -s http://localhost:8080/health > /dev/null 2>&1; then
        echo "✅ System is responding"
        echo ""
        echo "📊 Health Status:"
        curl -s http://localhost:8080/health | python3 -m json.tool 2>/dev/null || echo "API response received"
    else
        echo "❌ System is not responding on http://localhost:8080"
        echo "💡 Try starting with: ./start_system.sh"
    fi
else
    echo "ℹ️ curl not available - manual check: http://localhost:8080/health"
fi

echo ""
echo "📁 Database Status:"
if [ -f "/home/o1/Desktop/data/cognitive/cognitive_system.db" ]; then
    DB_SIZE=$(du -h /home/o1/Desktop/data/cognitive/cognitive_system.db | cut -f1)
    echo "   Cognitive DB: $DB_SIZE"
else
    echo "   ❌ Cognitive DB not found"
fi

if [ -f "/home/o1/Desktop/data/server/server_system.db" ]; then
    DB_SIZE=$(du -h /home/o1/Desktop/data/server/server_system.db | cut -f1)
    echo "   Server DB: $DB_SIZE"
else
    echo "   ❌ Server DB not found"
fi

echo "=================================="
