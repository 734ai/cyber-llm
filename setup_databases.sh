#!/bin/bash
#
# Database Setup Script for Persistent Cognitive Multi-Agent System
# Initializes SQLite databases with proper schemas and configurations
#

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
COGNITIVE_DB="${PROJECT_ROOT}/data/cognitive/cognitive_system.db"
SERVER_DB="${PROJECT_ROOT}/data/server/server_system.db"

# Color codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}🗄️ Initializing Database Systems${NC}"
echo "=================================================="

# Create database directories
mkdir -p "${PROJECT_ROOT}/data/cognitive"
mkdir -p "${PROJECT_ROOT}/data/server"
mkdir -p "${PROJECT_ROOT}/data/backups"

# Initialize Cognitive Database
echo -e "${GREEN}✅ Creating Cognitive Database Schema${NC}"
sqlite3 "$COGNITIVE_DB" << 'EOF'
-- Cognitive System Database Schema
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA cache_size=10000;
PRAGMA temp_store=memory;

-- Episodic Memory Table
CREATE TABLE IF NOT EXISTS episodic_memory (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    agent_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    event_data TEXT NOT NULL,
    context TEXT,
    importance_score REAL DEFAULT 0.5,
    emotional_valence REAL DEFAULT 0.0,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Semantic Memory Table  
CREATE TABLE IF NOT EXISTS semantic_memory (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id TEXT NOT NULL,
    concept TEXT NOT NULL,
    knowledge_data TEXT NOT NULL,
    confidence_score REAL DEFAULT 0.5,
    source TEXT,
    last_accessed DATETIME DEFAULT CURRENT_TIMESTAMP,
    access_count INTEGER DEFAULT 0,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Working Memory Table
CREATE TABLE IF NOT EXISTS working_memory (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    item_type TEXT NOT NULL,
    content TEXT NOT NULL,
    priority REAL DEFAULT 0.5,
    expires_at DATETIME,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Procedural Memory Table
CREATE TABLE IF NOT EXISTS procedural_memory (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id TEXT NOT NULL,
    skill_name TEXT NOT NULL,
    procedure_data TEXT NOT NULL,
    success_rate REAL DEFAULT 0.0,
    usage_count INTEGER DEFAULT 0,
    last_used DATETIME,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Strategic Memory Table
CREATE TABLE IF NOT EXISTS strategic_memory (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id TEXT NOT NULL,
    goal_id TEXT NOT NULL,
    goal_description TEXT NOT NULL,
    strategy_data TEXT NOT NULL,
    priority REAL DEFAULT 0.5,
    status TEXT DEFAULT 'active',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Reasoning Chains Table
CREATE TABLE IF NOT EXISTS reasoning_chains (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id TEXT NOT NULL,
    chain_id TEXT UNIQUE NOT NULL,
    reasoning_type TEXT NOT NULL,
    input_data TEXT NOT NULL,
    reasoning_steps TEXT NOT NULL,
    conclusion TEXT,
    confidence REAL DEFAULT 0.5,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Memory Relationships Table
CREATE TABLE IF NOT EXISTS memory_relationships (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_type TEXT NOT NULL,
    source_id INTEGER NOT NULL,
    target_type TEXT NOT NULL,
    target_id INTEGER NOT NULL,
    relationship_type TEXT NOT NULL,
    strength REAL DEFAULT 0.5,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Create Indices for Performance
CREATE INDEX IF NOT EXISTS idx_episodic_agent_time ON episodic_memory(agent_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_semantic_agent_concept ON semantic_memory(agent_id, concept);
CREATE INDEX IF NOT EXISTS idx_working_agent_session ON working_memory(agent_id, session_id);
CREATE INDEX IF NOT EXISTS idx_procedural_agent_skill ON procedural_memory(agent_id, skill_name);
CREATE INDEX IF NOT EXISTS idx_strategic_agent_goal ON strategic_memory(agent_id, goal_id);
CREATE INDEX IF NOT EXISTS idx_reasoning_agent_type ON reasoning_chains(agent_id, reasoning_type);
CREATE INDEX IF NOT EXISTS idx_relationships_source ON memory_relationships(source_type, source_id);

-- Initial System Metadata
INSERT OR IGNORE INTO semantic_memory (agent_id, concept, knowledge_data, source) VALUES
('system', 'database_version', '1.0.0', 'initialization'),
('system', 'schema_created', datetime('now'), 'initialization'),
('system', 'cognitive_capabilities', '["persistent_memory", "advanced_reasoning", "strategic_planning"]', 'initialization');
EOF

# Initialize Server Database
echo -e "${GREEN}✅ Creating Server Database Schema${NC}"
sqlite3 "$SERVER_DB" << 'EOF'
-- Server System Database Schema
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;  
PRAGMA cache_size=10000;
PRAGMA temp_store=memory;

-- Agent Sessions Table
CREATE TABLE IF NOT EXISTS agent_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT UNIQUE NOT NULL,
    agent_id TEXT NOT NULL,
    agent_type TEXT NOT NULL,
    status TEXT DEFAULT 'active',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    last_activity DATETIME DEFAULT CURRENT_TIMESTAMP,
    expires_at DATETIME,
    session_data TEXT
);

-- Task Queue Table
CREATE TABLE IF NOT EXISTS task_queue (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id TEXT UNIQUE NOT NULL,
    agent_id TEXT NOT NULL,
    task_type TEXT NOT NULL,
    task_data TEXT NOT NULL,
    priority INTEGER DEFAULT 0,
    status TEXT DEFAULT 'pending',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    started_at DATETIME,
    completed_at DATETIME,
    result TEXT
);

-- System Metrics Table
CREATE TABLE IF NOT EXISTS system_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    metric_name TEXT NOT NULL,
    metric_value REAL NOT NULL,
    metric_unit TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    tags TEXT
);

-- Agent State Table
CREATE TABLE IF NOT EXISTS agent_state (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id TEXT NOT NULL,
    state_key TEXT NOT NULL,
    state_value TEXT NOT NULL,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(agent_id, state_key)
);

-- System Configuration Table
CREATE TABLE IF NOT EXISTS system_config (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    config_key TEXT UNIQUE NOT NULL,
    config_value TEXT NOT NULL,
    config_type TEXT DEFAULT 'string',
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Background Jobs Table
CREATE TABLE IF NOT EXISTS background_jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT UNIQUE NOT NULL,
    job_type TEXT NOT NULL,
    job_data TEXT NOT NULL,
    schedule_expr TEXT,
    last_run DATETIME,
    next_run DATETIME,
    status TEXT DEFAULT 'active',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Create Indices for Performance
CREATE INDEX IF NOT EXISTS idx_sessions_agent ON agent_sessions(agent_id, status);
CREATE INDEX IF NOT EXISTS idx_tasks_agent_status ON task_queue(agent_id, status);
CREATE INDEX IF NOT EXISTS idx_metrics_name_time ON system_metrics(metric_name, timestamp);
CREATE INDEX IF NOT EXISTS idx_agent_state_agent ON agent_state(agent_id);
CREATE INDEX IF NOT EXISTS idx_jobs_type_status ON background_jobs(job_type, status);

-- Initial System Configuration
INSERT OR IGNORE INTO system_config (config_key, config_value, config_type) VALUES
('system_version', '2.0.0', 'string'),
('database_initialized', datetime('now'), 'datetime'),
('cognitive_mode', 'persistent', 'string'),
('server_mode', 'continuous', 'string'),
('memory_consolidation_interval', '21600', 'integer'),
('strategic_planning_interval', '7200', 'integer');

-- Initial Background Jobs
INSERT OR IGNORE INTO background_jobs (job_id, job_type, job_data, schedule_expr) VALUES
('memory_consolidation', 'consolidate_memory', '{"type": "episodic", "threshold": 0.3}', '0 */6 * * *'),
('strategic_coordination', 'agent_coordination', '{"type": "strategic_planning"}', '0 */2 * * *'),
('system_cleanup', 'cleanup', '{"type": "expired_sessions"}', '0 0 * * *');
EOF

# Set proper permissions
chmod 600 "$COGNITIVE_DB"
chmod 600 "$SERVER_DB"

# Verify database creation
echo -e "${GREEN}✅ Verifying Database Integrity${NC}"

# Check Cognitive DB
if sqlite3 "$COGNITIVE_DB" "SELECT COUNT(*) FROM sqlite_master WHERE type='table';" > /dev/null 2>&1; then
    TABLE_COUNT=$(sqlite3 "$COGNITIVE_DB" "SELECT COUNT(*) FROM sqlite_master WHERE type='table';")
    echo "   Cognitive Database: $TABLE_COUNT tables created"
else
    echo "   ERROR: Cognitive Database verification failed"
    exit 1
fi

# Check Server DB
if sqlite3 "$SERVER_DB" "SELECT COUNT(*) FROM sqlite_master WHERE type='table';" > /dev/null 2>&1; then
    TABLE_COUNT=$(sqlite3 "$SERVER_DB" "SELECT COUNT(*) FROM sqlite_master WHERE type='table';")
    echo "   Server Database: $TABLE_COUNT tables created"
else
    echo "   ERROR: Server Database verification failed"
    exit 1
fi

echo -e "${GREEN}✅ Database initialization completed successfully${NC}"
echo "   Cognitive DB: $COGNITIVE_DB"
echo "   Server DB: $SERVER_DB"
echo "   Database mode: WAL (Write-Ahead Logging)"
echo "   Backup location: ${PROJECT_ROOT}/data/backups"

# Create initial backup
echo -e "${GREEN}✅ Creating initial database backup${NC}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
sqlite3 "$COGNITIVE_DB" ".backup ${PROJECT_ROOT}/data/backups/cognitive_${TIMESTAMP}.db"
sqlite3 "$SERVER_DB" ".backup ${PROJECT_ROOT}/data/backups/server_${TIMESTAMP}.db"

echo "=================================================="
echo -e "${BLUE}🗄️ Database Systems Ready for Production${NC}"
