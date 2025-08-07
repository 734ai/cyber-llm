"""
Advanced Long-term Memory Architecture for Persistent Agent Memory
Implements cross-session memory persistence with intelligent retrieval
"""
import sqlite3
import json
import hashlib
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class MemoryRecord:
    """Individual memory record with metadata"""
    id: str
    content: str
    memory_type: str  # episodic, semantic, procedural, strategic
    timestamp: datetime
    importance: float
    access_count: int
    last_accessed: datetime
    embedding: Optional[List[float]] = None
    tags: List[str] = None
    agent_id: str = ""
    session_id: str = ""
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []

class LongTermMemoryManager:
    """Advanced persistent memory system with cross-session capabilities"""
    
    def __init__(self, db_path: str = "data/cognitive/long_term_memory.db"):
        """Initialize long-term memory system"""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
        self._memory_cache = {}
        self._embeddings_model = None
        
    def _init_database(self):
        """Initialize database schemas"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS long_term_memory (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    importance REAL NOT NULL,
                    access_count INTEGER DEFAULT 0,
                    last_accessed TEXT NOT NULL,
                    embedding TEXT,
                    tags TEXT,
                    agent_id TEXT,
                    session_id TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memory_associations (
                    id TEXT PRIMARY KEY,
                    memory_id_1 TEXT,
                    memory_id_2 TEXT,
                    association_type TEXT,
                    strength REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (memory_id_1) REFERENCES long_term_memory(id),
                    FOREIGN KEY (memory_id_2) REFERENCES long_term_memory(id)
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memory_consolidation_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    consolidation_type TEXT,
                    memories_processed INTEGER,
                    patterns_discovered INTEGER,
                    timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                    details TEXT
                )
            """)
            
            # Create indices for performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_type ON long_term_memory(memory_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_agent_id ON long_term_memory(agent_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_importance ON long_term_memory(importance)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON long_term_memory(timestamp)")
            
    def store_memory(self, content: str, memory_type: str, 
                    importance: float = 0.5, agent_id: str = "",
                    session_id: str = "", tags: List[str] = None) -> str:
        """Store a new memory with intelligent categorization"""
        try:
            memory_id = hashlib.sha256(f"{content}{memory_type}{datetime.now().isoformat()}".encode()).hexdigest()
            
            record = MemoryRecord(
                id=memory_id,
                content=content,
                memory_type=memory_type,
                timestamp=datetime.now(),
                importance=importance,
                access_count=0,
                last_accessed=datetime.now(),
                tags=tags or [],
                agent_id=agent_id,
                session_id=session_id
            )
            
            # Generate embedding for semantic search
            if self._embeddings_model:
                record.embedding = self._generate_embedding(content)
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO long_term_memory (
                        id, content, memory_type, timestamp, importance,
                        access_count, last_accessed, embedding, tags, agent_id, session_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    record.id, record.content, record.memory_type,
                    record.timestamp.isoformat(), record.importance,
                    record.access_count, record.last_accessed.isoformat(),
                    json.dumps(record.embedding) if record.embedding else None,
                    json.dumps(record.tags), record.agent_id, record.session_id
                ))
            
            logger.info(f"Stored long-term memory: {memory_id[:8]}... ({memory_type})")
            return memory_id
            
        except Exception as e:
            logger.error(f"Error storing memory: {e}")
            return ""
    
    def retrieve_memories(self, query: str = "", memory_type: str = "",
                         agent_id: str = "", limit: int = 10,
                         importance_threshold: float = 0.0) -> List[MemoryRecord]:
        """Retrieve memories with intelligent filtering and ranking"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conditions = []
                params = []
                
                if query:
                    conditions.append("content LIKE ?")
                    params.append(f"%{query}%")
                
                if memory_type:
                    conditions.append("memory_type = ?")
                    params.append(memory_type)
                
                if agent_id:
                    conditions.append("agent_id = ?")
                    params.append(agent_id)
                
                if importance_threshold > 0:
                    conditions.append("importance >= ?")
                    params.append(importance_threshold)
                
                where_clause = " AND ".join(conditions) if conditions else "1=1"
                
                cursor = conn.execute(f"""
                    SELECT * FROM long_term_memory 
                    WHERE {where_clause}
                    ORDER BY importance DESC, access_count DESC, timestamp DESC
                    LIMIT ?
                """, params + [limit])
                
                memories = []
                for row in cursor.fetchall():
                    memory = MemoryRecord(
                        id=row[0],
                        content=row[1],
                        memory_type=row[2],
                        timestamp=datetime.fromisoformat(row[3]),
                        importance=row[4],
                        access_count=row[5],
                        last_accessed=datetime.fromisoformat(row[6]),
                        embedding=json.loads(row[7]) if row[7] else None,
                        tags=json.loads(row[8]) if row[8] else [],
                        agent_id=row[9] or "",
                        session_id=row[10] or ""
                    )
                    memories.append(memory)
                    
                    # Update access statistics
                    self._update_access_stats(memory.id)
                
                logger.info(f"Retrieved {len(memories)} memories for query: {query[:50]}...")
                return memories
                
        except Exception as e:
            logger.error(f"Error retrieving memories: {e}")
            return []
    
    def consolidate_memories(self) -> Dict[str, int]:
        """Advanced memory consolidation with pattern discovery"""
        try:
            stats = {
                'memories_processed': 0,
                'patterns_discovered': 0,
                'associations_created': 0,
                'memories_merged': 0
            }
            
            with sqlite3.connect(self.db_path) as conn:
                # Get all memories for consolidation
                cursor = conn.execute("""
                    SELECT * FROM long_term_memory 
                    ORDER BY timestamp DESC
                """)
                
                memories = cursor.fetchall()
                stats['memories_processed'] = len(memories)
                
                # Pattern discovery through content similarity
                for i, memory1 in enumerate(memories):
                    for j, memory2 in enumerate(memories[i+1:], i+1):
                        similarity = self._calculate_semantic_similarity(
                            memory1[1], memory2[1]
                        )
                        
                        if similarity > 0.8:  # High similarity threshold
                            self._create_memory_association(
                                memory1[0], memory2[0], "semantic_similarity", similarity
                            )
                            stats['associations_created'] += 1
                            stats['patterns_discovered'] += 1
                
                # Temporal pattern detection
                self._detect_temporal_patterns(memories)
                
                # Log consolidation results
                conn.execute("""
                    INSERT INTO memory_consolidation_log (
                        consolidation_type, memories_processed, 
                        patterns_discovered, details
                    ) VALUES (?, ?, ?, ?)
                """, (
                    "full_consolidation", stats['memories_processed'],
                    stats['patterns_discovered'], json.dumps(stats)
                ))
            
            logger.info(f"Memory consolidation complete: {stats}")
            return stats
            
        except Exception as e:
            logger.error(f"Error during memory consolidation: {e}")
            return {'error': str(e)}
    
    def get_cross_session_context(self, agent_id: str, limit: int = 20) -> List[MemoryRecord]:
        """Retrieve cross-session context for agent continuity"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT * FROM long_term_memory 
                    WHERE agent_id = ? 
                    ORDER BY importance DESC, last_accessed DESC, timestamp DESC
                    LIMIT ?
                """, (agent_id, limit))
                
                memories = []
                for row in cursor.fetchall():
                    memory = MemoryRecord(
                        id=row[0],
                        content=row[1],
                        memory_type=row[2],
                        timestamp=datetime.fromisoformat(row[3]),
                        importance=row[4],
                        access_count=row[5],
                        last_accessed=datetime.fromisoformat(row[6]),
                        embedding=json.loads(row[7]) if row[7] else None,
                        tags=json.loads(row[8]) if row[8] else [],
                        agent_id=row[9] or "",
                        session_id=row[10] or ""
                    )
                    memories.append(memory)
                
                logger.info(f"Retrieved {len(memories)} cross-session memories for agent {agent_id}")
                return memories
                
        except Exception as e:
            logger.error(f"Error retrieving cross-session context: {e}")
            return []
    
    def _update_access_stats(self, memory_id: str):
        """Update memory access statistics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    UPDATE long_term_memory 
                    SET access_count = access_count + 1,
                        last_accessed = ?,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                """, (datetime.now().isoformat(), memory_id))
                
        except Exception as e:
            logger.error(f"Error updating access stats: {e}")
    
    def _generate_embedding(self, content: str) -> List[float]:
        """Generate embeddings for semantic search (placeholder)"""
        # In production, use a proper embedding model
        # For now, return a simple hash-based vector
        hash_val = hash(content)
        return [float((hash_val >> i) & 1) for i in range(128)]
    
    def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between texts"""
        # Simple word overlap similarity (replace with proper embeddings)
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def _create_memory_association(self, memory_id_1: str, memory_id_2: str,
                                  association_type: str, strength: float):
        """Create association between memories"""
        try:
            association_id = hashlib.sha256(
                f"{memory_id_1}{memory_id_2}{association_type}".encode()
            ).hexdigest()
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO memory_associations (
                        id, memory_id_1, memory_id_2, association_type, strength
                    ) VALUES (?, ?, ?, ?, ?)
                """, (association_id, memory_id_1, memory_id_2, association_type, strength))
                
        except Exception as e:
            logger.error(f"Error creating memory association: {e}")
    
    def _detect_temporal_patterns(self, memories: List[Tuple]):
        """Detect temporal patterns in memory sequences"""
        # Group memories by agent and detect sequences
        agent_memories = {}
        for memory in memories:
            agent_id = memory[9] or "unknown"
            if agent_id not in agent_memories:
                agent_memories[agent_id] = []
            agent_memories[agent_id].append(memory)
        
        # Analyze patterns within each agent's memory timeline
        for agent_id, agent_mem_list in agent_memories.items():
            # Sort by timestamp
            agent_mem_list.sort(key=lambda x: x[3])  # timestamp is at index 3
            
            # Detect recurring patterns or sequences
            # This is a simplified pattern detection
            for i in range(len(agent_mem_list) - 2):
                # Look for sequences of similar operations
                mem1, mem2, mem3 = agent_mem_list[i:i+3]
                
                # Check for similar memory types in sequence
                if mem1[2] == mem2[2] == mem3[2]:  # same memory_type
                    self._create_memory_association(
                        mem1[0], mem3[0], "temporal_sequence", 0.7
                    )
    
    def get_memory_statistics(self) -> Dict[str, Any]:
        """Get comprehensive memory system statistics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                stats = {}
                
                # Basic counts
                cursor = conn.execute("SELECT COUNT(*) FROM long_term_memory")
                stats['total_memories'] = cursor.fetchone()[0]
                
                # Memory type distribution
                cursor = conn.execute("""
                    SELECT memory_type, COUNT(*) 
                    FROM long_term_memory 
                    GROUP BY memory_type
                """)
                stats['memory_types'] = dict(cursor.fetchall())
                
                # Agent distribution
                cursor = conn.execute("""
                    SELECT agent_id, COUNT(*) 
                    FROM long_term_memory 
                    WHERE agent_id != ''
                    GROUP BY agent_id
                """)
                stats['agent_distribution'] = dict(cursor.fetchall())
                
                # Importance distribution
                cursor = conn.execute("""
                    SELECT 
                        CASE 
                            WHEN importance >= 0.8 THEN 'high'
                            WHEN importance >= 0.5 THEN 'medium'
                            ELSE 'low'
                        END as importance_level,
                        COUNT(*)
                    FROM long_term_memory
                    GROUP BY importance_level
                """)
                stats['importance_distribution'] = dict(cursor.fetchall())
                
                # Association statistics
                cursor = conn.execute("SELECT COUNT(*) FROM memory_associations")
                stats['total_associations'] = cursor.fetchone()[0]
                
                return stats
                
        except Exception as e:
            logger.error(f"Error getting memory statistics: {e}")
            return {'error': str(e)}

# Export the main class
__all__ = ['LongTermMemoryManager', 'MemoryRecord']
