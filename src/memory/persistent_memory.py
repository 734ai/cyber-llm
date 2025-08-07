"""
Advanced Persistent Memory System for Cyber-LLM
Long-term memory, reasoning chains, and strategic planning capabilities

Author: Muzan Sano <sanosensei36@gmail.com>
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import sqlite3
import pickle
import hashlib
from pathlib import Path
import numpy as np
from collections import defaultdict, deque
import threading
import time

from ..utils.logging_system import CyberLLMLogger, CyberLLMError, ErrorCategory
from ..utils.secrets_manager import get_secrets_manager

class MemoryType(Enum):
    """Types of memory in the system"""
    EPISODIC = "episodic"          # Specific events and experiences
    SEMANTIC = "semantic"          # General knowledge and facts
    PROCEDURAL = "procedural"      # Skills and procedures
    WORKING = "working"            # Temporary active information
    STRATEGIC = "strategic"        # Long-term goals and plans

class ReasoningType(Enum):
    """Types of reasoning chains"""
    DEDUCTIVE = "deductive"        # From general to specific
    INDUCTIVE = "inductive"        # From specific to general
    ABDUCTIVE = "abductive"        # Best explanation inference
    CAUSAL = "causal"             # Cause-effect relationships
    STRATEGIC = "strategic"        # Goal-oriented planning
    COUNTERFACTUAL = "counterfactual"  # What-if scenarios

@dataclass
class MemoryItem:
    """Individual memory item"""
    memory_id: str
    memory_type: MemoryType
    content: Dict[str, Any]
    
    # Temporal information
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    
    # Memory strength and importance
    importance_score: float = 0.5  # 0-1 scale
    confidence: float = 1.0
    decay_rate: float = 0.1
    
    # Associations and context
    associated_memories: List[str] = field(default_factory=list)
    context_tags: List[str] = field(default_factory=list)
    agent_id: Optional[str] = None
    
    # Metadata
    source: str = "unknown"
    validated: bool = False
    compressed: bool = False

@dataclass
class ReasoningChain:
    """Multi-step reasoning chain"""
    chain_id: str
    reasoning_type: ReasoningType
    goal: str
    
    # Reasoning steps
    steps: List[Dict[str, Any]] = field(default_factory=list)
    premises: List[str] = field(default_factory=list)
    conclusions: List[str] = field(default_factory=list)
    
    # Chain metadata
    created_at: datetime = field(default_factory=datetime.now)
    completed: bool = False
    confidence: float = 0.0
    agent_id: Optional[str] = None
    
    # Execution tracking
    current_step: int = 0
    execution_time: float = 0.0
    memory_references: List[str] = field(default_factory=list)

@dataclass
class StrategicPlan:
    """Long-term strategic plan"""
    plan_id: str
    objective: str
    timeline: timedelta
    
    # Plan structure
    phases: List[Dict[str, Any]] = field(default_factory=list)
    milestones: List[Dict[str, Any]] = field(default_factory=list)
    dependencies: Dict[str, List[str]] = field(default_factory=dict)
    
    # Execution tracking
    created_at: datetime = field(default_factory=datetime.now)
    status: str = "planning"  # planning, executing, completed, failed
    progress: float = 0.0
    
    # Adaptation and learning
    adaptations: List[Dict[str, Any]] = field(default_factory=list)
    lessons_learned: List[str] = field(default_factory=list)

class PersistentMemoryManager:
    """Advanced persistent memory system with reasoning capabilities"""
    
    def __init__(self, 
                 memory_db_path: str = "data/persistent_memory.db",
                 max_memory_items: int = 100000,
                 memory_consolidation_interval: int = 3600,
                 logger: Optional[CyberLLMLogger] = None):
        
        self.logger = logger or CyberLLMLogger(name="persistent_memory")
        self.memory_db_path = Path(memory_db_path)
        self.max_memory_items = max_memory_items
        self.consolidation_interval = memory_consolidation_interval
        
        # Memory stores
        self.episodic_memory = {}  # Recent experiences
        self.semantic_memory = {}  # General knowledge
        self.working_memory = deque(maxlen=50)  # Active information
        self.strategic_plans = {}  # Long-term plans
        self.reasoning_chains = {}  # Active reasoning
        
        # Memory indexing and retrieval
        self.memory_index = defaultdict(set)  # Tag-based indexing
        self.association_graph = defaultdict(set)  # Memory associations
        
        # Background processes
        self.consolidation_running = False
        self.consolidation_thread = None
        
        # Initialize memory system
        asyncio.create_task(self._initialize_memory_system())
        
        self.logger.info("Persistent Memory Manager initialized")
    
    async def _initialize_memory_system(self):
        """Initialize the persistent memory system"""
        
        try:
            # Create database structure
            self.memory_db_path.parent.mkdir(parents=True, exist_ok=True)
            
            conn = sqlite3.connect(self.memory_db_path)
            cursor = conn.cursor()
            
            # Memory items table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS memory_items (
                    memory_id TEXT PRIMARY KEY,
                    memory_type TEXT NOT NULL,
                    content BLOB,
                    created_at TIMESTAMP,
                    last_accessed TIMESTAMP,
                    access_count INTEGER DEFAULT 0,
                    importance_score REAL DEFAULT 0.5,
                    confidence REAL DEFAULT 1.0,
                    decay_rate REAL DEFAULT 0.1,
                    associated_memories TEXT,  -- JSON
                    context_tags TEXT,  -- JSON
                    agent_id TEXT,
                    source TEXT,
                    validated BOOLEAN DEFAULT FALSE,
                    compressed BOOLEAN DEFAULT FALSE
                )
            """)
            
            # Reasoning chains table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS reasoning_chains (
                    chain_id TEXT PRIMARY KEY,
                    reasoning_type TEXT NOT NULL,
                    goal TEXT,
                    steps TEXT,  -- JSON
                    premises TEXT,  -- JSON
                    conclusions TEXT,  -- JSON
                    created_at TIMESTAMP,
                    completed BOOLEAN DEFAULT FALSE,
                    confidence REAL DEFAULT 0.0,
                    agent_id TEXT,
                    current_step INTEGER DEFAULT 0,
                    execution_time REAL DEFAULT 0.0,
                    memory_references TEXT  -- JSON
                )
            """)
            
            # Strategic plans table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS strategic_plans (
                    plan_id TEXT PRIMARY KEY,
                    objective TEXT NOT NULL,
                    timeline INTEGER,  -- Seconds
                    phases TEXT,  -- JSON
                    milestones TEXT,  -- JSON
                    dependencies TEXT,  -- JSON
                    created_at TIMESTAMP,
                    status TEXT DEFAULT 'planning',
                    progress REAL DEFAULT 0.0,
                    adaptations TEXT,  -- JSON
                    lessons_learned TEXT  -- JSON
                )
            """)
            
            # Memory associations table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS memory_associations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    memory_id_1 TEXT,
                    memory_id_2 TEXT,
                    association_strength REAL DEFAULT 0.5,
                    association_type TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (memory_id_1) REFERENCES memory_items(memory_id),
                    FOREIGN KEY (memory_id_2) REFERENCES memory_items(memory_id)
                )
            """)
            
            conn.commit()
            conn.close()
            
            # Load existing memories
            await self._load_persistent_memories()
            
            # Start background consolidation process
            self._start_memory_consolidation()
            
            self.logger.info("Memory system database initialized and loaded")
            
        except Exception as e:
            self.logger.error("Failed to initialize memory system", error=str(e))
            raise CyberLLMError("Memory system initialization failed", ErrorCategory.SYSTEM)
    
    async def store_memory(self, 
                          memory_type: MemoryType,
                          content: Dict[str, Any],
                          importance: float = 0.5,
                          context_tags: List[str] = None,
                          agent_id: str = None) -> str:
        """Store a new memory item"""
        
        memory_id = f"{memory_type.value}_{hashlib.md5(str(content).encode()).hexdigest()[:8]}"
        
        memory_item = MemoryItem(
            memory_id=memory_id,
            memory_type=memory_type,
            content=content,
            created_at=datetime.now(),
            last_accessed=datetime.now(),
            importance_score=importance,
            context_tags=context_tags or [],
            agent_id=agent_id,
            source="direct_storage"
        )
        
        # Store in appropriate memory system
        if memory_type == MemoryType.EPISODIC:
            self.episodic_memory[memory_id] = memory_item
        elif memory_type == MemoryType.SEMANTIC:
            self.semantic_memory[memory_id] = memory_item
        elif memory_type == MemoryType.WORKING:
            self.working_memory.append(memory_item)
        
        # Update indexes
        for tag in context_tags or []:
            self.memory_index[tag].add(memory_id)
        
        # Persist to database
        await self._persist_memory_item(memory_item)
        
        self.logger.debug(f"Stored memory: {memory_id}", memory_type=memory_type.value)
        return memory_id
    
    async def retrieve_memories(self, 
                              query: str,
                              memory_types: List[MemoryType] = None,
                              limit: int = 10,
                              min_relevance: float = 0.3) -> List[MemoryItem]:
        """Retrieve memories based on query"""
        
        if not memory_types:
            memory_types = [MemoryType.EPISODIC, MemoryType.SEMANTIC]
        
        relevant_memories = []
        
        # Search through different memory types
        for memory_type in memory_types:
            if memory_type == MemoryType.EPISODIC:
                memories = self.episodic_memory.values()
            elif memory_type == MemoryType.SEMANTIC:
                memories = self.semantic_memory.values()
            elif memory_type == MemoryType.WORKING:
                memories = list(self.working_memory)
            else:
                continue
            
            for memory in memories:
                relevance = await self._calculate_relevance(query, memory)
                if relevance >= min_relevance:
                    relevant_memories.append((memory, relevance))
                    # Update access information
                    memory.last_accessed = datetime.now()
                    memory.access_count += 1
        
        # Sort by relevance and return top results
        relevant_memories.sort(key=lambda x: x[1], reverse=True)
        return [memory for memory, _ in relevant_memories[:limit]]
    
    async def create_reasoning_chain(self, 
                                   reasoning_type: ReasoningType,
                                   goal: str,
                                   premises: List[str],
                                   agent_id: str = None) -> str:
        """Create a new reasoning chain"""
        
        chain_id = f"reasoning_{reasoning_type.value}_{int(time.time())}"
        
        reasoning_chain = ReasoningChain(
            chain_id=chain_id,
            reasoning_type=reasoning_type,
            goal=goal,
            premises=premises,
            agent_id=agent_id
        )
        
        self.reasoning_chains[chain_id] = reasoning_chain
        
        # Persist to database
        await self._persist_reasoning_chain(reasoning_chain)
        
        self.logger.info(f"Created reasoning chain: {chain_id}", 
                        reasoning_type=reasoning_type.value,
                        goal=goal)
        
        return chain_id
    
    async def execute_reasoning_step(self, 
                                   chain_id: str,
                                   step_content: Dict[str, Any]) -> bool:
        """Execute a single reasoning step"""
        
        if chain_id not in self.reasoning_chains:
            raise CyberLLMError(f"Reasoning chain not found: {chain_id}", ErrorCategory.VALIDATION)
        
        chain = self.reasoning_chains[chain_id]
        
        try:
            start_time = time.time()
            
            # Add step to chain
            step = {
                "step_number": len(chain.steps) + 1,
                "content": step_content,
                "timestamp": datetime.now().isoformat(),
                "execution_time": 0.0
            }
            
            # Execute reasoning based on type
            if chain.reasoning_type == ReasoningType.DEDUCTIVE:
                result = await self._execute_deductive_step(chain, step_content)
            elif chain.reasoning_type == ReasoningType.INDUCTIVE:
                result = await self._execute_inductive_step(chain, step_content)
            elif chain.reasoning_type == ReasoningType.CAUSAL:
                result = await self._execute_causal_step(chain, step_content)
            elif chain.reasoning_type == ReasoningType.STRATEGIC:
                result = await self._execute_strategic_step(chain, step_content)
            else:
                result = await self._execute_generic_step(chain, step_content)
            
            # Update step with result
            step["result"] = result
            step["execution_time"] = time.time() - start_time
            
            chain.steps.append(step)
            chain.current_step += 1
            chain.execution_time += step["execution_time"]
            
            # Update confidence based on step success
            if result.get("success", False):
                chain.confidence = min(1.0, chain.confidence + 0.1)
            else:
                chain.confidence = max(0.0, chain.confidence - 0.1)
            
            # Update persistent storage
            await self._persist_reasoning_chain(chain)
            
            return result.get("success", False)
            
        except Exception as e:
            self.logger.error(f"Failed to execute reasoning step: {chain_id}", error=str(e))
            return False
    
    async def create_strategic_plan(self, 
                                  objective: str,
                                  timeline: timedelta,
                                  initial_phases: List[Dict[str, Any]] = None) -> str:
        """Create a new strategic plan"""
        
        plan_id = f"strategic_{hashlib.md5(objective.encode()).hexdigest()[:8]}"
        
        strategic_plan = StrategicPlan(
            plan_id=plan_id,
            objective=objective,
            timeline=timeline,
            phases=initial_phases or []
        )
        
        self.strategic_plans[plan_id] = strategic_plan
        
        # Persist to database
        await self._persist_strategic_plan(strategic_plan)
        
        self.logger.info(f"Created strategic plan: {plan_id}", objective=objective)
        return plan_id
    
    async def update_strategic_plan(self, 
                                  plan_id: str,
                                  updates: Dict[str, Any]) -> bool:
        """Update an existing strategic plan"""
        
        if plan_id not in self.strategic_plans:
            return False
        
        plan = self.strategic_plans[plan_id]
        
        # Apply updates
        for key, value in updates.items():
            if hasattr(plan, key):
                setattr(plan, key, value)
        
        # Track adaptation
        adaptation = {
            "timestamp": datetime.now().isoformat(),
            "changes": updates,
            "reason": updates.get("adaptation_reason", "Unknown")
        }
        plan.adaptations.append(adaptation)
        
        # Update persistent storage
        await self._persist_strategic_plan(plan)
        
        return True
    
    async def consolidate_memories(self):
        """Perform memory consolidation and cleanup"""
        
        try:
            # Decay unused memories
            current_time = datetime.now()
            
            for memory_store in [self.episodic_memory, self.semantic_memory]:
                to_remove = []
                
                for memory_id, memory in memory_store.items():
                    # Calculate memory decay
                    time_since_access = (current_time - memory.last_accessed).total_seconds()
                    decay_factor = memory.decay_rate * (time_since_access / 3600)  # Per hour
                    
                    memory.importance_score *= (1 - decay_factor)
                    
                    # Remove very low importance memories
                    if memory.importance_score < 0.1 and memory.access_count < 2:
                        to_remove.append(memory_id)
                
                # Remove decayed memories
                for memory_id in to_remove:
                    del memory_store[memory_id]
                    await self._remove_memory_from_db(memory_id)
            
            # Strengthen associated memories
            await self._strengthen_memory_associations()
            
            # Compress old memories
            await self._compress_old_memories()
            
            self.logger.info("Memory consolidation completed")
            
        except Exception as e:
            self.logger.error("Memory consolidation failed", error=str(e))
    
    def _start_memory_consolidation(self):
        """Start background memory consolidation process"""
        
        def consolidation_worker():
            while self.consolidation_running:
                try:
                    asyncio.run(self.consolidate_memories())
                    time.sleep(self.consolidation_interval)
                except Exception as e:
                    self.logger.error("Consolidation worker error", error=str(e))
                    time.sleep(60)  # Wait before retrying
        
        self.consolidation_running = True
        self.consolidation_thread = threading.Thread(target=consolidation_worker, daemon=True)
        self.consolidation_thread.start()
    
    async def _load_persistent_memories(self):
        """Load memories from persistent storage"""
        
        try:
            conn = sqlite3.connect(self.memory_db_path)
            cursor = conn.cursor()
            
            # Load memory items
            cursor.execute("SELECT * FROM memory_items ORDER BY last_accessed DESC LIMIT ?", 
                         (self.max_memory_items,))
            
            rows = cursor.fetchall()
            
            for row in rows:
                memory_item = MemoryItem(
                    memory_id=row[0],
                    memory_type=MemoryType(row[1]),
                    content=pickle.loads(row[2]),
                    created_at=datetime.fromisoformat(row[3]),
                    last_accessed=datetime.fromisoformat(row[4]),
                    access_count=row[5],
                    importance_score=row[6],
                    confidence=row[7],
                    decay_rate=row[8],
                    associated_memories=json.loads(row[9]) if row[9] else [],
                    context_tags=json.loads(row[10]) if row[10] else [],
                    agent_id=row[11],
                    source=row[12],
                    validated=bool(row[13]),
                    compressed=bool(row[14])
                )
                
                # Store in appropriate memory system
                if memory_item.memory_type == MemoryType.EPISODIC:
                    self.episodic_memory[memory_item.memory_id] = memory_item
                elif memory_item.memory_type == MemoryType.SEMANTIC:
                    self.semantic_memory[memory_item.memory_id] = memory_item
            
            # Load reasoning chains
            cursor.execute("SELECT * FROM reasoning_chains WHERE completed = FALSE")
            
            for row in cursor.fetchall():
                reasoning_chain = ReasoningChain(
                    chain_id=row[0],
                    reasoning_type=ReasoningType(row[1]),
                    goal=row[2],
                    steps=json.loads(row[3]) if row[3] else [],
                    premises=json.loads(row[4]) if row[4] else [],
                    conclusions=json.loads(row[5]) if row[5] else [],
                    created_at=datetime.fromisoformat(row[6]),
                    completed=bool(row[7]),
                    confidence=row[8],
                    agent_id=row[9],
                    current_step=row[10],
                    execution_time=row[11],
                    memory_references=json.loads(row[12]) if row[12] else []
                )
                
                self.reasoning_chains[reasoning_chain.chain_id] = reasoning_chain
            
            # Load strategic plans
            cursor.execute("SELECT * FROM strategic_plans WHERE status != 'completed'")
            
            for row in cursor.fetchall():
                strategic_plan = StrategicPlan(
                    plan_id=row[0],
                    objective=row[1],
                    timeline=timedelta(seconds=row[2]),
                    phases=json.loads(row[3]) if row[3] else [],
                    milestones=json.loads(row[4]) if row[4] else [],
                    dependencies=json.loads(row[5]) if row[5] else {},
                    created_at=datetime.fromisoformat(row[6]),
                    status=row[7],
                    progress=row[8],
                    adaptations=json.loads(row[9]) if row[9] else [],
                    lessons_learned=json.loads(row[10]) if row[10] else []
                )
                
                self.strategic_plans[strategic_plan.plan_id] = strategic_plan
            
            conn.close()
            
            self.logger.info(f"Loaded persistent memories: {len(self.episodic_memory + self.semantic_memory)} items")
            
        except Exception as e:
            self.logger.error("Failed to load persistent memories", error=str(e))
    
    async def _calculate_relevance(self, query: str, memory: MemoryItem) -> float:
        """Calculate relevance score between query and memory"""
        
        # Simple relevance calculation (would use embeddings in production)
        query_words = set(query.lower().split())
        memory_text = str(memory.content).lower()
        memory_words = set(memory_text.split())
        
        # Jaccard similarity
        intersection = len(query_words.intersection(memory_words))
        union = len(query_words.union(memory_words))
        
        if union == 0:
            return 0.0
        
        base_similarity = intersection / union
        
        # Boost based on importance and recency
        importance_boost = memory.importance_score * 0.3
        recency_boost = min(0.2, 1.0 / ((datetime.now() - memory.last_accessed).days + 1))
        
        return min(1.0, base_similarity + importance_boost + recency_boost)
    
    async def _execute_deductive_step(self, chain: ReasoningChain, step_content: Dict[str, Any]) -> Dict[str, Any]:
        """Execute deductive reasoning step"""
        
        # Deductive reasoning: apply general rules to specific cases
        rule = step_content.get("rule")
        case = step_content.get("case")
        
        if not rule or not case:
            return {"success": False, "error": "Missing rule or case for deductive reasoning"}
        
        # Simple rule application (would be more sophisticated in production)
        conclusion = f"If {rule} and {case}, then conclusion follows"
        
        return {
            "success": True,
            "conclusion": conclusion,
            "reasoning": f"Applied rule '{rule}' to case '{case}'"
        }
    
    async def _execute_inductive_step(self, chain: ReasoningChain, step_content: Dict[str, Any]) -> Dict[str, Any]:
        """Execute inductive reasoning step"""
        
        # Inductive reasoning: generalize from specific examples
        examples = step_content.get("examples", [])
        
        if len(examples) < 2:
            return {"success": False, "error": "Need at least 2 examples for inductive reasoning"}
        
        # Simple pattern detection
        pattern = f"Pattern derived from {len(examples)} examples"
        
        return {
            "success": True,
            "pattern": pattern,
            "reasoning": f"Generalized from {len(examples)} specific examples"
        }
    
    async def _execute_causal_step(self, chain: ReasoningChain, step_content: Dict[str, Any]) -> Dict[str, Any]:
        """Execute causal reasoning step"""
        
        # Causal reasoning: identify cause-effect relationships
        cause = step_content.get("cause")
        effect = step_content.get("effect")
        
        if not cause or not effect:
            return {"success": False, "error": "Missing cause or effect for causal reasoning"}
        
        # Simple causal analysis
        causal_link = f"'{cause}' causes '{effect}'"
        
        return {
            "success": True,
            "causal_link": causal_link,
            "reasoning": f"Established causal relationship between cause and effect"
        }
    
    async def _execute_strategic_step(self, chain: ReasoningChain, step_content: Dict[str, Any]) -> Dict[str, Any]:
        """Execute strategic reasoning step"""
        
        # Strategic reasoning: goal decomposition and planning
        goal = step_content.get("goal")
        constraints = step_content.get("constraints", [])
        resources = step_content.get("resources", [])
        
        if not goal:
            return {"success": False, "error": "Missing goal for strategic reasoning"}
        
        # Simple strategic analysis
        strategy = f"Strategy for achieving '{goal}' given constraints and resources"
        
        return {
            "success": True,
            "strategy": strategy,
            "reasoning": f"Developed strategy considering {len(constraints)} constraints and {len(resources)} resources"
        }
    
    async def _execute_generic_step(self, chain: ReasoningChain, step_content: Dict[str, Any]) -> Dict[str, Any]:
        """Execute generic reasoning step"""
        
        return {
            "success": True,
            "result": "Generic reasoning step completed",
            "reasoning": "Applied general reasoning principles"
        }
    
    async def _persist_memory_item(self, memory_item: MemoryItem):
        """Persist memory item to database"""
        
        try:
            conn = sqlite3.connect(self.memory_db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO memory_items
                (memory_id, memory_type, content, created_at, last_accessed, access_count,
                 importance_score, confidence, decay_rate, associated_memories, context_tags,
                 agent_id, source, validated, compressed)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                memory_item.memory_id,
                memory_item.memory_type.value,
                pickle.dumps(memory_item.content),
                memory_item.created_at.isoformat(),
                memory_item.last_accessed.isoformat(),
                memory_item.access_count,
                memory_item.importance_score,
                memory_item.confidence,
                memory_item.decay_rate,
                json.dumps(memory_item.associated_memories),
                json.dumps(memory_item.context_tags),
                memory_item.agent_id,
                memory_item.source,
                memory_item.validated,
                memory_item.compressed
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Failed to persist memory item: {memory_item.memory_id}", error=str(e))
    
    async def _persist_reasoning_chain(self, chain: ReasoningChain):
        """Persist reasoning chain to database"""
        
        try:
            conn = sqlite3.connect(self.memory_db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO reasoning_chains
                (chain_id, reasoning_type, goal, steps, premises, conclusions,
                 created_at, completed, confidence, agent_id, current_step,
                 execution_time, memory_references)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                chain.chain_id,
                chain.reasoning_type.value,
                chain.goal,
                json.dumps(chain.steps),
                json.dumps(chain.premises),
                json.dumps(chain.conclusions),
                chain.created_at.isoformat(),
                chain.completed,
                chain.confidence,
                chain.agent_id,
                chain.current_step,
                chain.execution_time,
                json.dumps(chain.memory_references)
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Failed to persist reasoning chain: {chain.chain_id}", error=str(e))
    
    async def _persist_strategic_plan(self, plan: StrategicPlan):
        """Persist strategic plan to database"""
        
        try:
            conn = sqlite3.connect(self.memory_db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO strategic_plans
                (plan_id, objective, timeline, phases, milestones, dependencies,
                 created_at, status, progress, adaptations, lessons_learned)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                plan.plan_id,
                plan.objective,
                int(plan.timeline.total_seconds()),
                json.dumps(plan.phases),
                json.dumps(plan.milestones),
                json.dumps(plan.dependencies),
                plan.created_at.isoformat(),
                plan.status,
                plan.progress,
                json.dumps(plan.adaptations),
                json.dumps(plan.lessons_learned)
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Failed to persist strategic plan: {plan.plan_id}", error=str(e))
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory system statistics"""
        
        return {
            "episodic_memories": len(self.episodic_memory),
            "semantic_memories": len(self.semantic_memory),
            "working_memory_items": len(self.working_memory),
            "active_reasoning_chains": len([c for c in self.reasoning_chains.values() if not c.completed]),
            "strategic_plans": len(self.strategic_plans),
            "memory_associations": len(self.association_graph),
            "consolidation_running": self.consolidation_running
        }

# Factory function
def create_persistent_memory_manager(**kwargs) -> PersistentMemoryManager:
    """Create persistent memory manager with configuration"""
    return PersistentMemoryManager(**kwargs)
