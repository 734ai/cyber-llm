"""
Persistent Memory Architecture for Advanced Cognitive Agents
Long-term memory systems with cross-session persistence and strategic thinking
"""

import sqlite3
import json
import pickle
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union, Set
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import logging
from abc import ABC, abstractmethod
from collections import defaultdict, deque
import asyncio
import threading
import time
from enum import Enum
import hashlib
import uuid
from pathlib import Path

class MemoryType(Enum):
    EPISODIC = "episodic"           # Events and experiences
    SEMANTIC = "semantic"           # Facts and knowledge
    PROCEDURAL = "procedural"       # Skills and procedures
    WORKING = "working"             # Temporary active memory
    STRATEGIC = "strategic"         # Long-term goals and plans

class ReasoningType(Enum):
    DEDUCTIVE = "deductive"         # General to specific
    INDUCTIVE = "inductive"         # Specific to general
    ABDUCTIVE = "abductive"         # Best explanation
    ANALOGICAL = "analogical"       # Pattern matching
    CAUSAL = "causal"              # Cause and effect
    STRATEGIC = "strategic"         # Goal-oriented
    COUNTERFACTUAL = "counterfactual"  # What-if scenarios
    METACOGNITIVE = "metacognitive"    # Thinking about thinking

@dataclass
class MemoryItem:
    """Base class for memory items"""
    memory_id: str
    memory_type: MemoryType
    content: Dict[str, Any]
    timestamp: str
    importance: float  # 0.0 to 1.0
    access_count: int
    last_accessed: str
    tags: List[str]
    metadata: Dict[str, Any]
    expires_at: Optional[str] = None

@dataclass
class EpisodicMemory(MemoryItem):
    """Specific events and experiences"""
    event_type: str
    context: Dict[str, Any]
    outcome: Dict[str, Any]
    learned_patterns: List[str]
    emotional_valence: float  # -1.0 (negative) to 1.0 (positive)
    
    def __post_init__(self):
        self.memory_type = MemoryType.EPISODIC

@dataclass
class SemanticMemory(MemoryItem):
    """Facts and general knowledge"""
    concept: str
    properties: Dict[str, Any]
    relationships: List[Dict[str, Any]]
    confidence: float
    evidence: List[str]
    
    def __post_init__(self):
        self.memory_type = MemoryType.SEMANTIC

@dataclass
class ProceduralMemory(MemoryItem):
    """Skills and procedures"""
    skill_name: str
    steps: List[Dict[str, Any]]
    conditions: Dict[str, Any]
    success_rate: float
    optimization_history: List[Dict[str, Any]]
    
    def __post_init__(self):
        self.memory_type = MemoryType.PROCEDURAL

@dataclass
class WorkingMemory(MemoryItem):
    """Temporary active memory"""
    current_goal: str
    active_context: Dict[str, Any]
    attention_focus: List[str]
    processing_state: Dict[str, Any]
    
    def __post_init__(self):
        self.memory_type = MemoryType.WORKING

@dataclass
class StrategicMemory(MemoryItem):
    """Long-term goals and strategic plans"""
    goal: str
    plan_steps: List[Dict[str, Any]]
    progress: float
    deadline: Optional[str]
    priority: int
    dependencies: List[str]
    success_criteria: Dict[str, Any]
    
    def __post_init__(self):
        self.memory_type = MemoryType.STRATEGIC

@dataclass
class ReasoningChain:
    """Represents a chain of reasoning"""
    chain_id: str
    reasoning_type: ReasoningType
    premise: Dict[str, Any]
    steps: List[Dict[str, Any]]
    conclusion: Dict[str, Any]
    confidence: float
    evidence: List[str]
    timestamp: str
    agent_id: str
    context: Dict[str, Any]

class MemoryConsolidator:
    """Consolidates and optimizes memory over time"""
    
    def __init__(self, database_path: str):
        self.database_path = database_path
        self.logger = logging.getLogger(__name__)
        self.consolidation_rules = self._init_consolidation_rules()
    
    def _init_consolidation_rules(self) -> Dict[str, Any]:
        """Initialize memory consolidation rules"""
        return {
            'episodic_to_semantic': {
                'min_occurrences': 3,
                'similarity_threshold': 0.8,
                'time_window_days': 30
            },
            'importance_decay': {
                'decay_rate': 0.95,
                'min_importance': 0.1,
                'access_boost': 1.1
            },
            'working_memory_cleanup': {
                'max_age_hours': 24,
                'max_items': 100,
                'importance_threshold': 0.3
            },
            'strategic_plan_updates': {
                'progress_review_days': 7,
                'priority_adjustment': True,
                'dependency_check': True
            }
        }
    
    async def consolidate_memories(self, agent_id: str) -> Dict[str, Any]:
        """Perform memory consolidation for an agent"""
        consolidation_results = {
            'episodic_consolidation': 0,
            'semantic_updates': 0,
            'procedural_optimizations': 0,
            'working_memory_cleanup': 0,
            'strategic_updates': 0,
            'total_processing_time': 0
        }
        
        start_time = time.time()
        
        try:
            # Episodic to semantic consolidation
            consolidation_results['episodic_consolidation'] = await self._consolidate_episodic_to_semantic(agent_id)
            
            # Update semantic relationships
            consolidation_results['semantic_updates'] = await self._update_semantic_relationships(agent_id)
            
            # Optimize procedural memories
            consolidation_results['procedural_optimizations'] = await self._optimize_procedural_memories(agent_id)
            
            # Clean working memory
            consolidation_results['working_memory_cleanup'] = await self._cleanup_working_memory(agent_id)
            
            # Update strategic plans
            consolidation_results['strategic_updates'] = await self._update_strategic_plans(agent_id)
            
            consolidation_results['total_processing_time'] = time.time() - start_time
            
            self.logger.info(f"Memory consolidation completed for agent {agent_id}: {consolidation_results}")
            
        except Exception as e:
            self.logger.error(f"Error during memory consolidation for agent {agent_id}: {e}")
        
        return consolidation_results
    
    async def _consolidate_episodic_to_semantic(self, agent_id: str) -> int:
        """Convert repeated episodic memories to semantic knowledge"""
        consolidated_count = 0
        
        with sqlite3.connect(self.database_path) as conn:
            # Find similar episodic memories
            cursor = conn.execute("""
                SELECT memory_id, content, timestamp, importance, access_count
                FROM memory_items 
                WHERE agent_id = ? AND memory_type = 'episodic' 
                ORDER BY timestamp DESC LIMIT 1000
            """, (agent_id,))
            
            episodic_memories = cursor.fetchall()
            
            # Group similar memories
            memory_groups = self._group_similar_memories(episodic_memories)
            
            for group in memory_groups:
                if len(group) >= self.consolidation_rules['episodic_to_semantic']['min_occurrences']:
                    # Create semantic memory from pattern
                    semantic_memory = self._create_semantic_from_episodic_group(group, agent_id)
                    
                    if semantic_memory:
                        # Insert semantic memory
                        conn.execute("""
                            INSERT INTO memory_items 
                            (memory_id, agent_id, memory_type, content, timestamp, importance, 
                             access_count, last_accessed, tags, metadata)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            semantic_memory.memory_id,
                            agent_id,
                            semantic_memory.memory_type.value,
                            json.dumps(asdict(semantic_memory)),
                            semantic_memory.timestamp,
                            semantic_memory.importance,
                            semantic_memory.access_count,
                            semantic_memory.last_accessed,
                            json.dumps(semantic_memory.tags),
                            json.dumps(semantic_memory.metadata)
                        ))
                        
                        consolidated_count += 1
        
        return consolidated_count
    
    def _group_similar_memories(self, memories: List[Tuple]) -> List[List[Dict]]:
        """Group similar episodic memories together"""
        memory_groups = []
        processed_memories = set()
        
        for i, memory in enumerate(memories):
            if i in processed_memories:
                continue
                
            current_group = [memory]
            memory_content = json.loads(memory[1])
            
            for j, other_memory in enumerate(memories[i+1:], i+1):
                if j in processed_memories:
                    continue
                
                other_content = json.loads(other_memory[1])
                similarity = self._calculate_memory_similarity(memory_content, other_content)
                
                if similarity >= self.consolidation_rules['episodic_to_semantic']['similarity_threshold']:
                    current_group.append(other_memory)
                    processed_memories.add(j)
            
            if len(current_group) > 1:
                memory_groups.append(current_group)
            
            processed_memories.add(i)
        
        return memory_groups
    
    def _calculate_memory_similarity(self, content1: Dict, content2: Dict) -> float:
        """Calculate similarity between two memory contents"""
        # Simple similarity based on common keys and values
        common_keys = set(content1.keys()) & set(content2.keys())
        
        if not common_keys:
            return 0.0
        
        similarity_scores = []
        
        for key in common_keys:
            val1, val2 = content1[key], content2[key]
            
            if isinstance(val1, str) and isinstance(val2, str):
                # String similarity (simplified)
                similarity_scores.append(1.0 if val1 == val2 else 0.5 if val1.lower() in val2.lower() else 0.0)
            elif isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                # Numeric similarity
                max_val = max(abs(val1), abs(val2))
                if max_val > 0:
                    similarity_scores.append(1.0 - abs(val1 - val2) / max_val)
                else:
                    similarity_scores.append(1.0)
            else:
                # Default similarity
                similarity_scores.append(1.0 if val1 == val2 else 0.0)
        
        return sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0.0
    
    def _create_semantic_from_episodic_group(self, memory_group: List[Tuple], agent_id: str) -> Optional[SemanticMemory]:
        """Create semantic memory from a group of similar episodic memories"""
        try:
            # Extract common patterns and concepts
            all_contents = [json.loads(memory[1]) for memory in memory_group]
            
            # Find common concept
            common_elements = set(all_contents[0].keys())
            for content in all_contents[1:]:
                common_elements &= set(content.keys())
            
            if not common_elements:
                return None
            
            # Create semantic concept
            concept_name = f"pattern_{len(memory_group)}_occurrences_{int(time.time())}"
            
            properties = {}
            for key in common_elements:
                values = [content[key] for content in all_contents]
                if len(set(map(str, values))) == 1:
                    properties[key] = values[0]  # Consistent value
                else:
                    properties[f"{key}_variations"] = list(set(map(str, values)))
            
            # Calculate confidence based on consistency and frequency
            confidence = min(1.0, len(memory_group) / 10.0)
            
            semantic_memory = SemanticMemory(
                memory_id=f"semantic_{uuid.uuid4().hex[:8]}",
                memory_type=MemoryType.SEMANTIC,
                content={},
                timestamp=datetime.now().isoformat(),
                importance=sum(memory[3] for memory in memory_group) / len(memory_group),
                access_count=0,
                last_accessed=datetime.now().isoformat(),
                tags=["consolidated", "pattern"],
                metadata={"source_episodic_count": len(memory_group)},
                concept=concept_name,
                properties=properties,
                relationships=[],
                confidence=confidence,
                evidence=[memory[0] for memory in memory_group]
            )
            
            return semantic_memory
            
        except Exception as e:
            self.logger.error(f"Error creating semantic memory from episodic group: {e}")
            return None
    
    async def _update_semantic_relationships(self, agent_id: str) -> int:
        """Update relationships between semantic memories"""
        updates_count = 0
        
        with sqlite3.connect(self.database_path) as conn:
            # Get all semantic memories
            cursor = conn.execute("""
                SELECT memory_id, content FROM memory_items 
                WHERE agent_id = ? AND memory_type = 'semantic'
            """, (agent_id,))
            
            semantic_memories = cursor.fetchall()
            
            # Find and update relationships
            for i, memory1 in enumerate(semantic_memories):
                memory1_content = json.loads(memory1[1])
                
                for memory2 in semantic_memories[i+1:]:
                    memory2_content = json.loads(memory2[1])
                    
                    # Check for potential relationships
                    relationship = self._identify_semantic_relationship(memory1_content, memory2_content)
                    
                    if relationship:
                        # Update both memories with the relationship
                        self._update_memory_relationships(conn, memory1[0], relationship)
                        self._update_memory_relationships(conn, memory2[0], relationship)
                        updates_count += 1
        
        return updates_count
    
    def _identify_semantic_relationship(self, content1: Dict, content2: Dict) -> Optional[Dict[str, Any]]:
        """Identify relationships between semantic memories"""
        # Simple relationship detection based on content overlap
        common_properties = set()
        
        if 'properties' in content1 and 'properties' in content2:
            props1 = content1['properties']
            props2 = content2['properties']
            
            for key in props1:
                if key in props2 and props1[key] == props2[key]:
                    common_properties.add(key)
        
        if len(common_properties) >= 2:
            return {
                'type': 'similarity',
                'strength': len(common_properties) / max(len(content1.get('properties', {})), len(content2.get('properties', {})), 1),
                'common_properties': list(common_properties)
            }
        
        return None
    
    def _update_memory_relationships(self, conn: sqlite3.Connection, memory_id: str, relationship: Dict[str, Any]):
        """Update memory with new relationship"""
        cursor = conn.execute("SELECT content FROM memory_items WHERE memory_id = ?", (memory_id,))
        result = cursor.fetchone()
        
        if result:
            content = json.loads(result[0])
            if 'relationships' not in content:
                content['relationships'] = []
            
            content['relationships'].append(relationship)
            
            conn.execute(
                "UPDATE memory_items SET content = ?, last_accessed = ? WHERE memory_id = ?",
                (json.dumps(content), datetime.now().isoformat(), memory_id)
            )
    
    async def _optimize_procedural_memories(self, agent_id: str) -> int:
        """Optimize procedural memories based on success rates"""
        optimizations = 0
        
        with sqlite3.connect(self.database_path) as conn:
            cursor = conn.execute("""
                SELECT memory_id, content FROM memory_items 
                WHERE agent_id = ? AND memory_type = 'procedural'
            """, (agent_id,))
            
            procedural_memories = cursor.fetchall()
            
            for memory_id, content_json in procedural_memories:
                content = json.loads(content_json)
                
                if 'success_rate' in content and content['success_rate'] < 0.7:
                    # Optimize low-performing procedures
                    optimized_steps = self._optimize_procedure_steps(content.get('steps', []))
                    
                    if optimized_steps != content.get('steps', []):
                        content['steps'] = optimized_steps
                        content['optimization_history'] = content.get('optimization_history', [])
                        content['optimization_history'].append({
                            'timestamp': datetime.now().isoformat(),
                            'type': 'step_optimization',
                            'previous_success_rate': content.get('success_rate', 0.0)
                        })
                        
                        conn.execute(
                            "UPDATE memory_items SET content = ?, last_accessed = ? WHERE memory_id = ?",
                            (json.dumps(content), datetime.now().isoformat(), memory_id)
                        )
                        
                        optimizations += 1
        
        return optimizations
    
    def _optimize_procedure_steps(self, steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Optimize procedure steps for better success rate"""
        # Simple optimization: reorder steps by success probability
        optimized_steps = sorted(steps, key=lambda x: x.get('success_probability', 0.5), reverse=True)
        
        # Add validation steps
        for step in optimized_steps:
            if 'validation' not in step:
                step['validation'] = {
                    'check_conditions': True,
                    'verify_outcome': True,
                    'rollback_on_failure': True
                }
        
        return optimized_steps
    
    async def _cleanup_working_memory(self, agent_id: str) -> int:
        """Clean up old and low-importance working memory items"""
        cleanup_count = 0
        
        with sqlite3.connect(self.database_path) as conn:
            # Remove old working memory items
            cutoff_time = (datetime.now() - timedelta(
                hours=self.consolidation_rules['working_memory_cleanup']['max_age_hours']
            )).isoformat()
            
            cursor = conn.execute("""
                DELETE FROM memory_items 
                WHERE agent_id = ? AND memory_type = 'working' 
                AND (timestamp < ? OR importance < ?)
            """, (agent_id, cutoff_time, self.consolidation_rules['working_memory_cleanup']['importance_threshold']))
            
            cleanup_count = cursor.rowcount
            
            # Limit working memory to max items
            cursor = conn.execute("""
                SELECT memory_id FROM memory_items 
                WHERE agent_id = ? AND memory_type = 'working' 
                ORDER BY importance DESC, last_accessed DESC
            """, (agent_id,))
            
            working_memories = cursor.fetchall()
            max_items = self.consolidation_rules['working_memory_cleanup']['max_items']
            
            if len(working_memories) > max_items:
                memories_to_delete = working_memories[max_items:]
                for memory_id_tuple in memories_to_delete:
                    conn.execute("DELETE FROM memory_items WHERE memory_id = ?", memory_id_tuple)
                    cleanup_count += 1
        
        return cleanup_count
    
    async def _update_strategic_plans(self, agent_id: str) -> int:
        """Update strategic plans based on progress and dependencies"""
        updates = 0
        
        with sqlite3.connect(self.database_path) as conn:
            cursor = conn.execute("""
                SELECT memory_id, content FROM memory_items 
                WHERE agent_id = ? AND memory_type = 'strategic'
            """, (agent_id,))
            
            strategic_memories = cursor.fetchall()
            
            for memory_id, content_json in strategic_memories:
                content = json.loads(content_json)
                updated = False
                
                # Update progress based on completed steps
                if 'plan_steps' in content:
                    completed_steps = sum(1 for step in content['plan_steps'] if step.get('completed', False))
                    total_steps = len(content['plan_steps'])
                    
                    if total_steps > 0:
                        new_progress = completed_steps / total_steps
                        if new_progress != content.get('progress', 0.0):
                            content['progress'] = new_progress
                            updated = True
                
                # Check deadlines and adjust priorities
                if 'deadline' in content and content['deadline']:
                    deadline = datetime.fromisoformat(content['deadline'])
                    days_until_deadline = (deadline - datetime.now()).days
                    
                    if days_until_deadline <= 7 and content.get('priority', 0) < 8:
                        content['priority'] = min(10, content.get('priority', 0) + 2)
                        updated = True
                
                # Check dependencies
                if 'dependencies' in content:
                    resolved_dependencies = []
                    for dep in content['dependencies']:
                        if self._is_dependency_resolved(conn, agent_id, dep):
                            resolved_dependencies.append(dep)
                    
                    if resolved_dependencies:
                        content['dependencies'] = [dep for dep in content['dependencies'] 
                                                 if dep not in resolved_dependencies]
                        updated = True
                
                if updated:
                    conn.execute(
                        "UPDATE memory_items SET content = ?, last_accessed = ? WHERE memory_id = ?",
                        (json.dumps(content), datetime.now().isoformat(), memory_id)
                    )
                    updates += 1
        
        return updates
    
    def _is_dependency_resolved(self, conn: sqlite3.Connection, agent_id: str, dependency: str) -> bool:
        """Check if a strategic dependency has been resolved"""
        cursor = conn.execute("""
            SELECT COUNT(*) FROM memory_items 
            WHERE agent_id = ? AND memory_type = 'strategic' 
            AND content LIKE ? AND content LIKE '%"progress": 1.0%'
        """, (agent_id, f'%{dependency}%'))
        
        return cursor.fetchone()[0] > 0

class PersistentMemorySystem:
    """Main persistent memory system for cognitive agents"""
    
    def __init__(self, database_path: str = "data/cognitive/persistent_memory.db"):
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        self.consolidator = MemoryConsolidator(str(self.database_path))
        
        # Initialize database
        self._init_database()
        
        # Background consolidation
        self.consolidation_running = False
        self.consolidation_interval = 6 * 60 * 60  # 6 hours
    
    def _init_database(self):
        """Initialize SQLite database for persistent memory"""
        with sqlite3.connect(self.database_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA cache_size=10000")
            conn.execute("PRAGMA temp_store=memory")
            
            # Memory items table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memory_items (
                    memory_id TEXT PRIMARY KEY,
                    agent_id TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    importance REAL NOT NULL,
                    access_count INTEGER DEFAULT 0,
                    last_accessed TEXT NOT NULL,
                    tags TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    expires_at TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Reasoning chains table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS reasoning_chains (
                    chain_id TEXT PRIMARY KEY,
                    agent_id TEXT NOT NULL,
                    reasoning_type TEXT NOT NULL,
                    premise TEXT NOT NULL,
                    steps TEXT NOT NULL,
                    conclusion TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    evidence TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    context TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Memory associations table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memory_associations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    memory_id_1 TEXT NOT NULL,
                    memory_id_2 TEXT NOT NULL,
                    association_type TEXT NOT NULL,
                    strength REAL NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (memory_id_1) REFERENCES memory_items (memory_id),
                    FOREIGN KEY (memory_id_2) REFERENCES memory_items (memory_id)
                )
            """)
            
            # Create indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_agent_type ON memory_items (agent_id, memory_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_timestamp ON memory_items (timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_importance ON memory_items (importance)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_reasoning_agent ON reasoning_chains (agent_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_reasoning_type ON reasoning_chains (reasoning_type)")
    
    async def store_memory(self, agent_id: str, memory: MemoryItem) -> bool:
        """Store a memory item"""
        try:
            with sqlite3.connect(self.database_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO memory_items 
                    (memory_id, agent_id, memory_type, content, timestamp, importance, 
                     access_count, last_accessed, tags, metadata, expires_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    memory.memory_id,
                    agent_id,
                    memory.memory_type.value,
                    json.dumps(asdict(memory)),
                    memory.timestamp,
                    memory.importance,
                    memory.access_count,
                    memory.last_accessed,
                    json.dumps(memory.tags),
                    json.dumps(memory.metadata),
                    memory.expires_at
                ))
            
            self.logger.debug(f"Stored memory {memory.memory_id} for agent {agent_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error storing memory {memory.memory_id} for agent {agent_id}: {e}")
            return False
    
    async def retrieve_memories(self, agent_id: str, memory_type: Optional[MemoryType] = None,
                              tags: Optional[List[str]] = None, limit: int = 100) -> List[MemoryItem]:
        """Retrieve memories for an agent"""
        memories = []
        
        try:
            with sqlite3.connect(self.database_path) as conn:
                query = "SELECT * FROM memory_items WHERE agent_id = ?"
                params = [agent_id]
                
                if memory_type:
                    query += " AND memory_type = ?"
                    params.append(memory_type.value)
                
                if tags:
                    tag_conditions = " AND (" + " OR ".join(["tags LIKE ?" for _ in tags]) + ")"
                    query += tag_conditions
                    params.extend([f"%{tag}%" for tag in tags])
                
                query += " ORDER BY importance DESC, last_accessed DESC LIMIT ?"
                params.append(limit)
                
                cursor = conn.execute(query, params)
                rows = cursor.fetchall()
                
                for row in rows:
                    # Update access count
                    conn.execute(
                        "UPDATE memory_items SET access_count = access_count + 1, last_accessed = ? WHERE memory_id = ?",
                        (datetime.now().isoformat(), row[0])
                    )
                    
                    # Reconstruct memory object
                    memory_data = json.loads(row[3])
                    memory_type_enum = MemoryType(row[2])
                    
                    if memory_type_enum == MemoryType.EPISODIC:
                        memory = EpisodicMemory(**memory_data)
                    elif memory_type_enum == MemoryType.SEMANTIC:
                        memory = SemanticMemory(**memory_data)
                    elif memory_type_enum == MemoryType.PROCEDURAL:
                        memory = ProceduralMemory(**memory_data)
                    elif memory_type_enum == MemoryType.WORKING:
                        memory = WorkingMemory(**memory_data)
                    elif memory_type_enum == MemoryType.STRATEGIC:
                        memory = StrategicMemory(**memory_data)
                    else:
                        memory = MemoryItem(**memory_data)
                    
                    memories.append(memory)
            
            self.logger.debug(f"Retrieved {len(memories)} memories for agent {agent_id}")
            
        except Exception as e:
            self.logger.error(f"Error retrieving memories for agent {agent_id}: {e}")
        
        return memories
    
    async def store_reasoning_chain(self, reasoning_chain: ReasoningChain) -> bool:
        """Store a reasoning chain"""
        try:
            with sqlite3.connect(self.database_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO reasoning_chains 
                    (chain_id, agent_id, reasoning_type, premise, steps, conclusion, 
                     confidence, evidence, timestamp, context)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    reasoning_chain.chain_id,
                    reasoning_chain.agent_id,
                    reasoning_chain.reasoning_type.value,
                    json.dumps(reasoning_chain.premise),
                    json.dumps(reasoning_chain.steps),
                    json.dumps(reasoning_chain.conclusion),
                    reasoning_chain.confidence,
                    json.dumps(reasoning_chain.evidence),
                    reasoning_chain.timestamp,
                    json.dumps(reasoning_chain.context)
                ))
            
            self.logger.debug(f"Stored reasoning chain {reasoning_chain.chain_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error storing reasoning chain {reasoning_chain.chain_id}: {e}")
            return False
    
    async def retrieve_reasoning_chains(self, agent_id: str, reasoning_type: Optional[ReasoningType] = None,
                                      limit: int = 50) -> List[ReasoningChain]:
        """Retrieve reasoning chains for an agent"""
        chains = []
        
        try:
            with sqlite3.connect(self.database_path) as conn:
                query = "SELECT * FROM reasoning_chains WHERE agent_id = ?"
                params = [agent_id]
                
                if reasoning_type:
                    query += " AND reasoning_type = ?"
                    params.append(reasoning_type.value)
                
                query += " ORDER BY confidence DESC, timestamp DESC LIMIT ?"
                params.append(limit)
                
                cursor = conn.execute(query, params)
                rows = cursor.fetchall()
                
                for row in rows:
                    chain = ReasoningChain(
                        chain_id=row[0],
                        agent_id=row[1],
                        reasoning_type=ReasoningType(row[2]),
                        premise=json.loads(row[3]),
                        steps=json.loads(row[4]),
                        conclusion=json.loads(row[5]),
                        confidence=row[6],
                        evidence=json.loads(row[7]),
                        timestamp=row[8],
                        context=json.loads(row[9])
                    )
                    chains.append(chain)
            
            self.logger.debug(f"Retrieved {len(chains)} reasoning chains for agent {agent_id}")
            
        except Exception as e:
            self.logger.error(f"Error retrieving reasoning chains for agent {agent_id}: {e}")
        
        return chains
    
    async def create_memory_association(self, memory_id_1: str, memory_id_2: str, 
                                      association_type: str, strength: float) -> bool:
        """Create an association between two memories"""
        try:
            with sqlite3.connect(self.database_path) as conn:
                conn.execute("""
                    INSERT INTO memory_associations (memory_id_1, memory_id_2, association_type, strength)
                    VALUES (?, ?, ?, ?)
                """, (memory_id_1, memory_id_2, association_type, strength))
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error creating memory association: {e}")
            return False
    
    async def find_associated_memories(self, memory_id: str, min_strength: float = 0.5) -> List[Tuple[str, str, float]]:
        """Find memories associated with a given memory"""
        associations = []
        
        try:
            with sqlite3.connect(self.database_path) as conn:
                cursor = conn.execute("""
                    SELECT memory_id_2, association_type, strength 
                    FROM memory_associations 
                    WHERE memory_id_1 = ? AND strength >= ?
                    UNION
                    SELECT memory_id_1, association_type, strength 
                    FROM memory_associations 
                    WHERE memory_id_2 = ? AND strength >= ?
                    ORDER BY strength DESC
                """, (memory_id, min_strength, memory_id, min_strength))
                
                associations = cursor.fetchall()
            
        except Exception as e:
            self.logger.error(f"Error finding associated memories for {memory_id}: {e}")
        
        return associations
    
    def start_background_consolidation(self):
        """Start background memory consolidation process"""
        if self.consolidation_running:
            return
        
        self.consolidation_running = True
        
        def consolidation_loop():
            while self.consolidation_running:
                try:
                    # Get all agents with memories
                    with sqlite3.connect(self.database_path) as conn:
                        cursor = conn.execute("SELECT DISTINCT agent_id FROM memory_items")
                        agent_ids = [row[0] for row in cursor.fetchall()]
                    
                    # Consolidate memories for each agent
                    for agent_id in agent_ids:
                        asyncio.run(self.consolidator.consolidate_memories(agent_id))
                    
                    # Sleep until next consolidation cycle
                    time.sleep(self.consolidation_interval)
                    
                except Exception as e:
                    self.logger.error(f"Error in background consolidation: {e}")
                    time.sleep(300)  # Wait 5 minutes before retrying
        
        consolidation_thread = threading.Thread(target=consolidation_loop, daemon=True)
        consolidation_thread.start()
        
        self.logger.info("Started background memory consolidation")
    
    def stop_background_consolidation(self):
        """Stop background memory consolidation process"""
        self.consolidation_running = False
        self.logger.info("Stopped background memory consolidation")
    
    def get_memory_statistics(self, agent_id: str) -> Dict[str, Any]:
        """Get memory statistics for an agent"""
        stats = {}
        
        try:
            with sqlite3.connect(self.database_path) as conn:
                # Total memory counts by type
                cursor = conn.execute("""
                    SELECT memory_type, COUNT(*) FROM memory_items 
                    WHERE agent_id = ? GROUP BY memory_type
                """, (agent_id,))
                
                memory_counts = dict(cursor.fetchall())
                stats['memory_counts'] = memory_counts
                
                # Total memories
                stats['total_memories'] = sum(memory_counts.values())
                
                # Memory importance distribution
                cursor = conn.execute("""
                    SELECT AVG(importance), MIN(importance), MAX(importance) 
                    FROM memory_items WHERE agent_id = ?
                """, (agent_id,))
                
                importance_stats = cursor.fetchone()
                stats['importance_stats'] = {
                    'average': importance_stats[0] or 0.0,
                    'minimum': importance_stats[1] or 0.0,
                    'maximum': importance_stats[2] or 0.0
                }
                
                # Recent activity
                cursor = conn.execute("""
                    SELECT COUNT(*) FROM memory_items 
                    WHERE agent_id = ? AND last_accessed >= ?
                """, (agent_id, (datetime.now() - timedelta(days=1)).isoformat()))
                
                stats['recent_access_count'] = cursor.fetchone()[0]
                
                # Reasoning chain stats
                cursor = conn.execute("""
                    SELECT reasoning_type, COUNT(*) FROM reasoning_chains 
                    WHERE agent_id = ? GROUP BY reasoning_type
                """, (agent_id,))
                
                reasoning_counts = dict(cursor.fetchall())
                stats['reasoning_counts'] = reasoning_counts
                stats['total_reasoning_chains'] = sum(reasoning_counts.values())
                
                # Association stats
                cursor = conn.execute("""
                    SELECT COUNT(*) FROM memory_associations ma
                    JOIN memory_items mi1 ON ma.memory_id_1 = mi1.memory_id
                    JOIN memory_items mi2 ON ma.memory_id_2 = mi2.memory_id
                    WHERE mi1.agent_id = ? OR mi2.agent_id = ?
                """, (agent_id, agent_id))
                
                stats['association_count'] = cursor.fetchone()[0]
            
        except Exception as e:
            self.logger.error(f"Error getting memory statistics for agent {agent_id}: {e}")
            stats = {'error': str(e)}
        
        return stats

# Example usage and testing
if __name__ == "__main__":
    print("🧠 Persistent Memory Architecture Testing:")
    print("=" * 50)
    
    # Initialize persistent memory system
    memory_system = PersistentMemorySystem()
    
    # Start background consolidation
    memory_system.start_background_consolidation()
    
    async def test_memory_operations():
        agent_id = "test_agent_001"
        
        # Test episodic memory storage
        print("\n📚 Testing episodic memory storage...")
        episodic_memory = EpisodicMemory(
            memory_id="episode_001",
            memory_type=MemoryType.EPISODIC,
            content={},
            timestamp=datetime.now().isoformat(),
            importance=0.8,
            access_count=0,
            last_accessed=datetime.now().isoformat(),
            tags=["security_incident", "network_scan"],
            metadata={"source": "ids_alert"},
            event_type="network_scan_detected",
            context={"source_ip": "192.168.1.100", "target_ports": [22, 80, 443]},
            outcome={"blocked": True, "alert_generated": True},
            learned_patterns=["port_scan_pattern"],
            emotional_valence=0.2
        )
        
        success = await memory_system.store_memory(agent_id, episodic_memory)
        print(f"  Stored episodic memory: {success}")
        
        # Test semantic memory storage
        print("\n🧠 Testing semantic memory storage...")
        semantic_memory = SemanticMemory(
            memory_id="semantic_001",
            memory_type=MemoryType.SEMANTIC,
            content={},
            timestamp=datetime.now().isoformat(),
            importance=0.9,
            access_count=0,
            last_accessed=datetime.now().isoformat(),
            tags=["cybersecurity_knowledge", "network_security"],
            metadata={"domain": "network_security"},
            concept="port_scanning",
            properties={
                "definition": "Systematic probing of network ports to identify services",
                "indicators": ["sequential_port_access", "connection_attempts", "timeout_patterns"],
                "countermeasures": ["port_blocking", "rate_limiting", "intrusion_detection"]
            },
            relationships=[],
            confidence=0.95,
            evidence=["rfc_standards", "security_literature"]
        )
        
        success = await memory_system.store_memory(agent_id, semantic_memory)
        print(f"  Stored semantic memory: {success}")
        
        # Test procedural memory storage
        print("\n⚙️ Testing procedural memory storage...")
        procedural_memory = ProceduralMemory(
            memory_id="procedure_001",
            memory_type=MemoryType.PROCEDURAL,
            content={},
            timestamp=datetime.now().isoformat(),
            importance=0.7,
            access_count=0,
            last_accessed=datetime.now().isoformat(),
            tags=["incident_response", "network_security"],
            metadata={"category": "defensive_procedures"},
            skill_name="network_scan_response",
            steps=[
                {"step": 1, "action": "identify_source", "success_probability": 0.9},
                {"step": 2, "action": "block_source_ip", "success_probability": 0.95},
                {"step": 3, "action": "generate_alert", "success_probability": 1.0},
                {"step": 4, "action": "investigate_context", "success_probability": 0.8}
            ],
            conditions={"trigger": "port_scan_detected", "confidence": ">0.8"},
            success_rate=0.85,
            optimization_history=[]
        )
        
        success = await memory_system.store_memory(agent_id, procedural_memory)
        print(f"  Stored procedural memory: {success}")
        
        # Test strategic memory storage
        print("\n🎯 Testing strategic memory storage...")
        strategic_memory = StrategicMemory(
            memory_id="strategic_001",
            memory_type=MemoryType.STRATEGIC,
            content={},
            timestamp=datetime.now().isoformat(),
            importance=1.0,
            access_count=0,
            last_accessed=datetime.now().isoformat(),
            tags=["long_term_goal", "security_posture"],
            metadata={"category": "defensive_strategy"},
            goal="improve_network_security_posture",
            plan_steps=[
                {"step": 1, "description": "Deploy additional IDS sensors", "completed": False, "target_date": "2025-08-15"},
                {"step": 2, "description": "Implement rate limiting", "completed": False, "target_date": "2025-08-20"},
                {"step": 3, "description": "Update response procedures", "completed": False, "target_date": "2025-08-25"}
            ],
            progress=0.0,
            deadline=(datetime.now() + timedelta(days=30)).isoformat(),
            priority=8,
            dependencies=["budget_approval", "technical_resources"],
            success_criteria={"scan_detection_rate": ">95%", "response_time": "<60s"}
        )
        
        success = await memory_system.store_memory(agent_id, strategic_memory)
        print(f"  Stored strategic memory: {success}")
        
        # Test reasoning chain storage
        print("\n🔗 Testing reasoning chain storage...")
        reasoning_chain = ReasoningChain(
            chain_id="reasoning_001",
            reasoning_type=ReasoningType.DEDUCTIVE,
            premise={
                "observation": "Multiple connection attempts to various ports from single IP",
                "pattern": "Sequential port access with short intervals"
            },
            steps=[
                {"step": 1, "reasoning": "Sequential port access indicates systematic scanning"},
                {"step": 2, "reasoning": "Single source IP suggests coordinated effort"},
                {"step": 3, "reasoning": "Pattern matches known port scanning signatures"}
            ],
            conclusion={
                "assessment": "Network port scan detected",
                "confidence_level": "high",
                "recommended_action": "block_and_investigate"
            },
            confidence=0.92,
            evidence=["network_logs", "ids_patterns", "historical_data"],
            timestamp=datetime.now().isoformat(),
            agent_id=agent_id,
            context={"alert_id": "alert_12345", "network_segment": "dmz"}
        )
        
        success = await memory_system.store_reasoning_chain(reasoning_chain)
        print(f"  Stored reasoning chain: {success}")
        
        # Test memory retrieval
        print("\n🔍 Testing memory retrieval...")
        
        # Retrieve all memories
        all_memories = await memory_system.retrieve_memories(agent_id, limit=10)
        print(f"  Retrieved {len(all_memories)} total memories")
        
        # Retrieve specific memory types
        episodic_memories = await memory_system.retrieve_memories(agent_id, MemoryType.EPISODIC)
        print(f"  Retrieved {len(episodic_memories)} episodic memories")
        
        semantic_memories = await memory_system.retrieve_memories(agent_id, MemoryType.SEMANTIC)
        print(f"  Retrieved {len(semantic_memories)} semantic memories")
        
        # Retrieve by tags
        security_memories = await memory_system.retrieve_memories(agent_id, tags=["security_incident"])
        print(f"  Retrieved {len(security_memories)} security-related memories")
        
        # Test reasoning chain retrieval
        reasoning_chains = await memory_system.retrieve_reasoning_chains(agent_id)
        print(f"  Retrieved {len(reasoning_chains)} reasoning chains")
        
        # Test memory associations
        print("\n🔗 Testing memory associations...")
        success = await memory_system.create_memory_association(
            "episode_001", "semantic_001", "relates_to", 0.8
        )
        print(f"  Created memory association: {success}")
        
        associations = await memory_system.find_associated_memories("episode_001")
        print(f"  Found {len(associations)} associations")
        
        # Test memory statistics
        print("\n📊 Testing memory statistics...")
        stats = memory_system.get_memory_statistics(agent_id)
        print(f"  Memory statistics: {stats}")
        
        # Test memory consolidation
        print("\n🔄 Testing memory consolidation...")
        consolidation_results = await memory_system.consolidator.consolidate_memories(agent_id)
        print(f"  Consolidation results: {consolidation_results}")
        
        return True
    
    # Run async tests
    import asyncio
    asyncio.run(test_memory_operations())
    
    # Stop background consolidation for testing
    memory_system.stop_background_consolidation()
    
    print("\n✅ Persistent Memory Architecture implemented and tested")
    print(f"  Database: {memory_system.database_path}")
    print(f"  Features: Episodic, Semantic, Procedural, Working, Strategic Memory")
    print(f"  Capabilities: Cross-session persistence, automated consolidation, reasoning chains")
