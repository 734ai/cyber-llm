"""
Advanced Reasoning Engine with Persistent Memory
Implements long-term thinking, strategic planning, and persistent memory systems

Author: Cyber-LLM Development Team
Date: August 6, 2025
Version: 2.0.0
"""

import asyncio
import json
import logging
import sqlite3
import pickle
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, field
from enum import Enum
import threading
import time
from collections import defaultdict, deque
from pathlib import Path
import numpy as np

# Advanced reasoning imports
from abc import ABC, abstractmethod
import uuid
import networkx as nx
import yaml

class ReasoningType(Enum):
    """Types of reasoning supported by the system"""
    DEDUCTIVE = "deductive"  # General to specific
    INDUCTIVE = "inductive"  # Specific to general  
    ABDUCTIVE = "abductive"  # Best explanation
    ANALOGICAL = "analogical"  # Similarity-based
    CAUSAL = "causal"  # Cause-effect relationships
    STRATEGIC = "strategic"  # Long-term planning
    COUNTERFACTUAL = "counterfactual"  # What-if scenarios
    META_COGNITIVE = "meta_cognitive"  # Reasoning about reasoning

class MemoryType(Enum):
    """Types of memory in the system"""
    WORKING = "working"  # Short-term active memory
    EPISODIC = "episodic"  # Specific experiences
    SEMANTIC = "semantic"  # General knowledge
    PROCEDURAL = "procedural"  # Skills and procedures
    STRATEGIC = "strategic"  # Long-term plans and goals

@dataclass
class ReasoningStep:
    """Individual step in a reasoning chain"""
    step_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    reasoning_type: ReasoningType = ReasoningType.DEDUCTIVE
    premise: str = ""
    inference_rule: str = ""
    conclusion: str = ""
    confidence: float = 0.0
    evidence: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    dependencies: List[str] = field(default_factory=list)
    
@dataclass 
class ReasoningChain:
    """Complete reasoning chain with multiple steps"""
    chain_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    topic: str = ""
    goal: str = ""
    steps: List[ReasoningStep] = field(default_factory=list)
    conclusion: str = ""
    confidence: float = 0.0
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    success: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MemoryEntry:
    """Entry in the persistent memory system"""
    memory_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    memory_type: MemoryType = MemoryType.EPISODIC
    content: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    importance: float = 0.0
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.now)
    decay_rate: float = 0.1
    tags: Set[str] = field(default_factory=set)

@dataclass
class StrategicPlan:
    """Long-term strategic plan with goals and milestones"""
    plan_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    description: str = ""
    primary_goal: str = ""
    sub_goals: List[str] = field(default_factory=list)
    timeline: Dict[str, datetime] = field(default_factory=dict)
    milestones: List[Dict[str, Any]] = field(default_factory=list)
    success_criteria: List[str] = field(default_factory=list)
    risk_factors: List[str] = field(default_factory=list)
    resources_required: List[str] = field(default_factory=list)
    current_status: str = "planning"
    progress_percentage: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

class PersistentMemoryManager:
    """Advanced persistent memory system for agents"""
    
    def __init__(self, db_path: str = "data/agent_memory.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger("persistent_memory")
        
        # Memory organization
        self.working_memory = deque(maxlen=100)  # Active memories
        self.memory_graph = nx.DiGraph()  # Semantic relationships
        self.memory_cache = {}  # LRU cache for fast access
        
        # Initialize database
        self._init_database()
        
        # Background processes
        self.consolidation_thread = None
        self.decay_thread = None
        self._start_background_processes()
    
    def _init_database(self):
        """Initialize the SQLite database for persistent storage"""
        
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.execute("PRAGMA foreign_keys = ON")
        
        # Memory entries table
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS memory_entries (
            memory_id TEXT PRIMARY KEY,
            memory_type TEXT NOT NULL,
            content BLOB NOT NULL,
            timestamp REAL NOT NULL,
            importance REAL NOT NULL,
            access_count INTEGER DEFAULT 0,
            last_accessed REAL NOT NULL,
            decay_rate REAL NOT NULL,
            tags TEXT DEFAULT ''
        )
        """)
        
        # Reasoning chains table
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS reasoning_chains (
            chain_id TEXT PRIMARY KEY,
            topic TEXT NOT NULL,
            goal TEXT NOT NULL,
            steps BLOB NOT NULL,
            conclusion TEXT NOT NULL,
            confidence REAL NOT NULL,
            start_time REAL NOT NULL,
            end_time REAL,
            success BOOLEAN NOT NULL,
            metadata BLOB DEFAULT ''
        )
        """)
        
        # Strategic plans table
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS strategic_plans (
            plan_id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            description TEXT NOT NULL,
            primary_goal TEXT NOT NULL,
            sub_goals BLOB NOT NULL,
            timeline BLOB NOT NULL,
            milestones BLOB NOT NULL,
            success_criteria BLOB NOT NULL,
            risk_factors BLOB NOT NULL,
            resources_required BLOB NOT NULL,
            current_status TEXT NOT NULL,
            progress_percentage REAL NOT NULL,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL
        )
        """)
        
        # Memory relationships table
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS memory_relationships (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_memory_id TEXT NOT NULL,
            target_memory_id TEXT NOT NULL,
            relationship_type TEXT NOT NULL,
            strength REAL NOT NULL,
            created_at REAL NOT NULL,
            FOREIGN KEY (source_memory_id) REFERENCES memory_entries (memory_id),
            FOREIGN KEY (target_memory_id) REFERENCES memory_entries (memory_id)
        )
        """)
        
        self.conn.commit()
        self.logger.info("Persistent memory database initialized")
    
    async def store_memory(self, memory_entry: MemoryEntry) -> str:
        """Store a memory entry in persistent storage"""
        
        try:
            # Store in database
            self.conn.execute("""
            INSERT OR REPLACE INTO memory_entries 
            (memory_id, memory_type, content, timestamp, importance, 
             access_count, last_accessed, decay_rate, tags)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                memory_entry.memory_id,
                memory_entry.memory_type.value,
                pickle.dumps(memory_entry.content),
                memory_entry.timestamp.timestamp(),
                memory_entry.importance,
                memory_entry.access_count,
                memory_entry.last_accessed.timestamp(),
                memory_entry.decay_rate,
                json.dumps(list(memory_entry.tags))
            ))
            
            self.conn.commit()
            
            # Add to working memory if important
            if memory_entry.importance > 0.5:
                self.working_memory.append(memory_entry)
            
            # Update cache
            self.memory_cache[memory_entry.memory_id] = memory_entry
            
            self.logger.debug(f"Stored memory: {memory_entry.memory_id}")
            return memory_entry.memory_id
            
        except Exception as e:
            self.logger.error(f"Error storing memory: {e}")
            return None
    
    async def retrieve_memory(self, memory_id: str) -> Optional[MemoryEntry]:
        """Retrieve a specific memory by ID"""
        
        # Check cache first
        if memory_id in self.memory_cache:
            memory = self.memory_cache[memory_id]
            memory.access_count += 1
            memory.last_accessed = datetime.now()
            return memory
        
        try:
            cursor = self.conn.execute("""
            SELECT * FROM memory_entries WHERE memory_id = ?
            """, (memory_id,))
            
            row = cursor.fetchone()
            if row:
                memory = MemoryEntry(
                    memory_id=row[0],
                    memory_type=MemoryType(row[1]),
                    content=pickle.loads(row[2]),
                    timestamp=datetime.fromtimestamp(row[3]),
                    importance=row[4],
                    access_count=row[5] + 1,
                    last_accessed=datetime.now(),
                    decay_rate=row[7],
                    tags=set(json.loads(row[8]))
                )
                
                # Update access count
                self.conn.execute("""
                UPDATE memory_entries 
                SET access_count = ?, last_accessed = ?
                WHERE memory_id = ?
                """, (memory.access_count, memory.last_accessed.timestamp(), memory_id))
                self.conn.commit()
                
                # Cache the memory
                self.memory_cache[memory_id] = memory
                
                return memory
                
        except Exception as e:
            self.logger.error(f"Error retrieving memory {memory_id}: {e}")
        
        return None
    
    async def search_memories(self, query: str, memory_types: List[MemoryType] = None, 
                            limit: int = 50) -> List[MemoryEntry]:
        """Search memories based on content and type"""
        
        memories = []
        
        try:
            # Build query conditions
            conditions = []
            params = []
            
            if memory_types:
                type_conditions = " OR ".join(["memory_type = ?"] * len(memory_types))
                conditions.append(f"({type_conditions})")
                params.extend([mt.value for mt in memory_types])
            
            where_clause = " AND ".join(conditions) if conditions else "1=1"
            
            cursor = self.conn.execute(f"""
            SELECT * FROM memory_entries 
            WHERE {where_clause}
            ORDER BY importance DESC, last_accessed DESC
            LIMIT ?
            """, params + [limit])
            
            for row in cursor.fetchall():
                memory = MemoryEntry(
                    memory_id=row[0],
                    memory_type=MemoryType(row[1]),
                    content=pickle.loads(row[2]),
                    timestamp=datetime.fromtimestamp(row[3]),
                    importance=row[4],
                    access_count=row[5],
                    last_accessed=datetime.fromtimestamp(row[6]),
                    decay_rate=row[7],
                    tags=set(json.loads(row[8]))
                )
                
                # Simple text matching (can be enhanced with vector similarity)
                if self._matches_query(memory, query):
                    memories.append(memory)
            
        except Exception as e:
            self.logger.error(f"Error searching memories: {e}")
        
        return sorted(memories, key=lambda m: m.importance, reverse=True)
    
    def _matches_query(self, memory: MemoryEntry, query: str) -> bool:
        """Simple text matching for memory search"""
        query_lower = query.lower()
        
        # Search in content
        content_str = json.dumps(memory.content).lower()
        if query_lower in content_str:
            return True
        
        # Search in tags
        for tag in memory.tags:
            if query_lower in tag.lower():
                return True
        
        return False
    
    async def consolidate_memories(self):
        """Consolidate and organize memories"""
        
        try:
            # Get all working memories
            working_memories = list(self.working_memory)
            
            # Group related memories
            memory_groups = self._group_related_memories(working_memories)
            
            # Create consolidated memories
            for group in memory_groups:
                if len(group) > 1:
                    consolidated = await self._create_consolidated_memory(group)
                    await self.store_memory(consolidated)
            
            self.logger.info(f"Consolidated {len(memory_groups)} memory groups")
            
        except Exception as e:
            self.logger.error(f"Error consolidating memories: {e}")
    
    def _group_related_memories(self, memories: List[MemoryEntry]) -> List[List[MemoryEntry]]:
        """Group related memories together"""
        groups = []
        processed = set()
        
        for memory in memories:
            if memory.memory_id in processed:
                continue
            
            # Find related memories
            related = [memory]
            for other_memory in memories:
                if (other_memory.memory_id != memory.memory_id and
                    other_memory.memory_id not in processed):
                    
                    # Simple relatedness check (can be enhanced)
                    if self._are_memories_related(memory, other_memory):
                        related.append(other_memory)
                        processed.add(other_memory.memory_id)
            
            if related:
                groups.append(related)
                for mem in related:
                    processed.add(mem.memory_id)
        
        return groups
    
    def _are_memories_related(self, mem1: MemoryEntry, mem2: MemoryEntry) -> bool:
        """Check if two memories are related"""
        
        # Check temporal proximity
        time_diff = abs((mem1.timestamp - mem2.timestamp).total_seconds())
        if time_diff < 3600:  # Within 1 hour
            return True
        
        # Check tag overlap
        tag_overlap = len(mem1.tags.intersection(mem2.tags))
        if tag_overlap > 0:
            return True
        
        # Check content similarity (simple approach)
        content1 = json.dumps(mem1.content).lower()
        content2 = json.dumps(mem2.content).lower()
        
        # Simple word overlap
        words1 = set(content1.split())
        words2 = set(content2.split())
        overlap_ratio = len(words1.intersection(words2)) / max(len(words1), len(words2))
        
        return overlap_ratio > 0.3
    
    async def _create_consolidated_memory(self, memories: List[MemoryEntry]) -> MemoryEntry:
        """Create a consolidated memory from related memories"""
        
        # Combine content
        consolidated_content = {
            "type": "consolidated",
            "source_memories": [mem.memory_id for mem in memories],
            "combined_content": [mem.content for mem in memories],
            "themes": self._extract_themes(memories)
        }
        
        # Calculate importance
        importance = max(mem.importance for mem in memories)
        
        # Combine tags
        all_tags = set()
        for mem in memories:
            all_tags.update(mem.tags)
        all_tags.add("consolidated")
        
        return MemoryEntry(
            memory_type=MemoryType.SEMANTIC,
            content=consolidated_content,
            importance=importance,
            tags=all_tags
        )
    
    def _extract_themes(self, memories: List[MemoryEntry]) -> List[str]:
        """Extract common themes from memories"""
        
        # Simple theme extraction (can be enhanced with NLP)
        all_text = " ".join([
            json.dumps(mem.content) for mem in memories
        ]).lower()
        
        # Common cybersecurity themes
        themes = []
        security_themes = [
            "vulnerability", "threat", "attack", "exploit", "malware",
            "phishing", "social engineering", "network security", "encryption",
            "authentication", "authorization", "firewall", "intrusion"
        ]
        
        for theme in security_themes:
            if theme in all_text:
                themes.append(theme)
        
        return themes
    
    def _start_background_processes(self):
        """Start background memory management processes"""
        
        def consolidation_worker():
            while True:
                try:
                    time.sleep(300)  # Every 5 minutes
                    asyncio.run(self.consolidate_memories())
                except Exception as e:
                    self.logger.error(f"Consolidation error: {e}")
        
        def decay_worker():
            while True:
                try:
                    time.sleep(600)  # Every 10 minutes
                    self._apply_memory_decay()
                except Exception as e:
                    self.logger.error(f"Decay error: {e}")
        
        # Start background threads
        self.consolidation_thread = threading.Thread(target=consolidation_worker, daemon=True)
        self.decay_thread = threading.Thread(target=decay_worker, daemon=True)
        
        self.consolidation_thread.start()
        self.decay_thread.start()
        
        self.logger.info("Background memory processes started")
    
    def _apply_memory_decay(self):
        """Apply decay to memories over time"""
        
        try:
            cursor = self.conn.execute("""
            SELECT memory_id, importance, last_accessed, decay_rate 
            FROM memory_entries
            """)
            
            updates = []
            current_time = datetime.now().timestamp()
            
            for row in cursor.fetchall():
                memory_id, importance, last_accessed, decay_rate = row
                
                # Calculate time since last access
                time_since_access = current_time - last_accessed
                
                # Apply decay (exponential decay)
                decay_factor = np.exp(-decay_rate * time_since_access / 86400)  # Days
                new_importance = importance * decay_factor
                
                # Minimum importance threshold
                if new_importance < 0.01:
                    new_importance = 0.01
                
                updates.append((new_importance, memory_id))
            
            # Batch update
            self.conn.executemany("""
            UPDATE memory_entries SET importance = ? WHERE memory_id = ?
            """, updates)
            
            self.conn.commit()
            self.logger.debug(f"Applied decay to {len(updates)} memories")
            
        except Exception as e:
            self.logger.error(f"Error applying memory decay: {e}")

class AdvancedReasoningEngine:
    """Advanced reasoning engine with multiple reasoning types"""
    
    def __init__(self, memory_manager: PersistentMemoryManager):
        self.memory_manager = memory_manager
        self.logger = logging.getLogger("reasoning_engine")
        
        # Reasoning components
        self.inference_rules = self._load_inference_rules()
        self.reasoning_strategies = {
            ReasoningType.DEDUCTIVE: self._deductive_reasoning,
            ReasoningType.INDUCTIVE: self._inductive_reasoning,
            ReasoningType.ABDUCTIVE: self._abductive_reasoning,
            ReasoningType.ANALOGICAL: self._analogical_reasoning,
            ReasoningType.CAUSAL: self._causal_reasoning,
            ReasoningType.STRATEGIC: self._strategic_reasoning,
            ReasoningType.COUNTERFACTUAL: self._counterfactual_reasoning,
            ReasoningType.META_COGNITIVE: self._meta_cognitive_reasoning
        }
        
        # Active reasoning chains
        self.active_chains = {}
        
    def _load_inference_rules(self) -> Dict[str, Dict[str, Any]]:
        """Load inference rules for different reasoning types"""
        
        return {
            "modus_ponens": {
                "pattern": "If P then Q, P is true",
                "conclusion": "Q is true",
                "confidence_base": 0.9
            },
            "modus_tollens": {
                "pattern": "If P then Q, Q is false",
                "conclusion": "P is false", 
                "confidence_base": 0.85
            },
            "hypothetical_syllogism": {
                "pattern": "If P then Q, If Q then R",
                "conclusion": "If P then R",
                "confidence_base": 0.8
            },
            "disjunctive_syllogism": {
                "pattern": "P or Q, not P",
                "conclusion": "Q",
                "confidence_base": 0.8
            },
            "causal_inference": {
                "pattern": "Event A precedes Event B, correlation observed",
                "conclusion": "A may cause B",
                "confidence_base": 0.6
            }
        }
    
    async def start_reasoning_chain(self, topic: str, goal: str, 
                                  reasoning_type: ReasoningType = ReasoningType.DEDUCTIVE) -> str:
        """Start a new reasoning chain"""
        
        chain = ReasoningChain(
            topic=topic,
            goal=goal,
            metadata={"reasoning_type": reasoning_type.value}
        )
        
        self.active_chains[chain.chain_id] = chain
        
        # Store in memory
        memory_entry = MemoryEntry(
            memory_type=MemoryType.PROCEDURAL,
            content={
                "type": "reasoning_chain_start",
                "chain_id": chain.chain_id,
                "topic": topic,
                "goal": goal,
                "reasoning_type": reasoning_type.value
            },
            importance=0.7,
            tags={"reasoning", "chain_start", reasoning_type.value}
        )
        
        await self.memory_manager.store_memory(memory_entry)
        
        self.logger.info(f"Started reasoning chain: {chain.chain_id}")
        return chain.chain_id
    
    async def add_reasoning_step(self, chain_id: str, premise: str, 
                               inference_rule: str = "", evidence: List[str] = None) -> str:
        """Add a step to an active reasoning chain"""
        
        if chain_id not in self.active_chains:
            self.logger.error(f"Reasoning chain {chain_id} not found")
            return None
        
        chain = self.active_chains[chain_id]
        evidence = evidence or []
        
        # Determine reasoning type from chain metadata
        reasoning_type = ReasoningType(chain.metadata.get("reasoning_type", "deductive"))
        
        # Apply reasoning strategy
        reasoning_func = self.reasoning_strategies.get(reasoning_type, self._deductive_reasoning)
        conclusion, confidence = await reasoning_func(premise, inference_rule, evidence, chain)
        
        # Create reasoning step
        step = ReasoningStep(
            reasoning_type=reasoning_type,
            premise=premise,
            inference_rule=inference_rule,
            conclusion=conclusion,
            confidence=confidence,
            evidence=evidence,
            dependencies=[s.step_id for s in chain.steps[-3:]]  # Last 3 steps
        )
        
        chain.steps.append(step)
        
        # Store step in memory
        memory_entry = MemoryEntry(
            memory_type=MemoryType.PROCEDURAL,
            content={
                "type": "reasoning_step",
                "chain_id": chain_id,
                "step_id": step.step_id,
                "premise": premise,
                "conclusion": conclusion,
                "confidence": confidence,
                "inference_rule": inference_rule
            },
            importance=confidence,
            tags={"reasoning", "step", reasoning_type.value}
        )
        
        await self.memory_manager.store_memory(memory_entry)
        
        self.logger.debug(f"Added reasoning step to chain {chain_id}")
        return step.step_id
    
    async def _deductive_reasoning(self, premise: str, inference_rule: str, 
                                 evidence: List[str], chain: ReasoningChain) -> Tuple[str, float]:
        """Apply deductive reasoning"""
        
        # Look up inference rule
        if inference_rule in self.inference_rules:
            rule = self.inference_rules[inference_rule]
            base_confidence = rule["confidence_base"]
            
            # Apply rule logic (simplified)
            if "modus_ponens" in inference_rule.lower():
                conclusion = f"Therefore, the consequent follows from the premise: {premise}"
                confidence = base_confidence
            else:
                conclusion = f"Following {inference_rule}: {premise}"
                confidence = base_confidence * 0.8
        else:
            # Default deductive reasoning
            conclusion = f"Based on logical deduction from: {premise}"
            confidence = 0.7
        
        # Adjust confidence based on evidence
        if evidence:
            confidence = min(confidence + len(evidence) * 0.05, 0.95)
        
        return conclusion, confidence
    
    async def _inductive_reasoning(self, premise: str, inference_rule: str,
                                 evidence: List[str], chain: ReasoningChain) -> Tuple[str, float]:
        """Apply inductive reasoning"""
        
        # Inductive reasoning builds general conclusions from specific observations
        pattern_strength = len(evidence) / max(len(chain.steps) + 1, 1)
        
        conclusion = f"Based on observed pattern in {len(evidence)} cases: {premise}"
        confidence = min(0.3 + pattern_strength * 0.4, 0.8)  # Inductive reasoning is less certain
        
        return conclusion, confidence
    
    async def _abductive_reasoning(self, premise: str, inference_rule: str,
                                 evidence: List[str], chain: ReasoningChain) -> Tuple[str, float]:
        """Apply abductive reasoning (inference to best explanation)"""
        
        # Abductive reasoning finds the best explanation for observations
        explanation_quality = len(evidence) * 0.1
        
        conclusion = f"Best explanation for '{premise}' given available evidence"
        confidence = min(0.5 + explanation_quality, 0.75)  # Moderate confidence
        
        return conclusion, confidence
    
    async def _analogical_reasoning(self, premise: str, inference_rule: str,
                                  evidence: List[str], chain: ReasoningChain) -> Tuple[str, float]:
        """Apply analogical reasoning"""
        
        # Search for similar past experiences in memory
        similar_memories = await self.memory_manager.search_memories(
            premise, [MemoryType.EPISODIC], limit=5
        )
        
        if similar_memories:
            analogy_strength = len(similar_memories) * 0.15
            conclusion = f"By analogy to {len(similar_memories)} similar cases: {premise}"
            confidence = min(0.4 + analogy_strength, 0.7)
        else:
            conclusion = f"No strong analogies found for: {premise}"
            confidence = 0.3
        
        return conclusion, confidence
    
    async def _causal_reasoning(self, premise: str, inference_rule: str,
                              evidence: List[str], chain: ReasoningChain) -> Tuple[str, float]:
        """Apply causal reasoning"""
        
        # Look for temporal and correlational patterns
        causal_indicators = ["caused by", "resulted in", "led to", "triggered"]
        
        causal_strength = sum(1 for indicator in causal_indicators if indicator in premise.lower())
        temporal_evidence = len([e for e in evidence if "time" in e.lower() or "sequence" in e.lower()])
        
        conclusion = f"Causal relationship identified: {premise}"
        confidence = min(0.4 + (causal_strength * 0.1) + (temporal_evidence * 0.1), 0.8)
        
        return conclusion, confidence
    
    async def _strategic_reasoning(self, premise: str, inference_rule: str,
                                 evidence: List[str], chain: ReasoningChain) -> Tuple[str, float]:
        """Apply strategic reasoning for long-term planning"""
        
        # Strategic reasoning considers multiple steps and long-term goals
        strategic_depth = len(chain.steps)
        goal_alignment = 0.8 if chain.goal.lower() in premise.lower() else 0.5
        
        conclusion = f"Strategic implication: {premise} aligns with long-term objectives"
        confidence = min(goal_alignment + (strategic_depth * 0.05), 0.85)
        
        return conclusion, confidence
    
    async def _counterfactual_reasoning(self, premise: str, inference_rule: str,
                                      evidence: List[str], chain: ReasoningChain) -> Tuple[str, float]:
        """Apply counterfactual reasoning (what-if scenarios)"""
        
        # Counterfactual reasoning explores alternative scenarios
        scenario_plausibility = 0.6  # Default plausibility
        
        if "what if" in premise.lower() or "if not" in premise.lower():
            scenario_plausibility += 0.1
        
        conclusion = f"Counterfactual analysis: {premise} would lead to alternative outcomes"
        confidence = min(scenario_plausibility, 0.7)  # Inherently speculative
        
        return conclusion, confidence
    
    async def _meta_cognitive_reasoning(self, premise: str, inference_rule: str,
                                      evidence: List[str], chain: ReasoningChain) -> Tuple[str, float]:
        """Apply meta-cognitive reasoning (reasoning about reasoning)"""
        
        # Meta-cognitive reasoning evaluates the reasoning process itself
        reasoning_quality = sum(step.confidence for step in chain.steps) / max(len(chain.steps), 1)
        
        conclusion = f"Meta-analysis of reasoning quality: {reasoning_quality:.2f} average confidence"
        confidence = reasoning_quality
        
        return conclusion, confidence
    
    async def complete_reasoning_chain(self, chain_id: str) -> Optional[ReasoningChain]:
        """Complete a reasoning chain and store results"""
        
        if chain_id not in self.active_chains:
            self.logger.error(f"Reasoning chain {chain_id} not found")
            return None
        
        chain = self.active_chains[chain_id]
        chain.end_time = datetime.now()
        
        # Generate final conclusion
        if chain.steps:
            # Combine conclusions from all steps
            step_conclusions = [step.conclusion for step in chain.steps]
            chain.conclusion = f"Final reasoning conclusion: {' → '.join(step_conclusions[-3:])}"
            
            # Calculate overall confidence
            confidences = [step.confidence for step in chain.steps]
            chain.confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            chain.success = chain.confidence > 0.5
        else:
            chain.conclusion = "No reasoning steps completed"
            chain.success = False
        
        # Store completed chain in database
        try:
            self.memory_manager.conn.execute("""
            INSERT OR REPLACE INTO reasoning_chains
            (chain_id, topic, goal, steps, conclusion, confidence, 
             start_time, end_time, success, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                chain.chain_id,
                chain.topic,
                chain.goal,
                pickle.dumps(chain.steps),
                chain.conclusion,
                chain.confidence,
                chain.start_time.timestamp(),
                chain.end_time.timestamp(),
                chain.success,
                pickle.dumps(chain.metadata)
            ))
            
            self.memory_manager.conn.commit()
            
            # Store in episodic memory
            memory_entry = MemoryEntry(
                memory_type=MemoryType.EPISODIC,
                content={
                    "type": "completed_reasoning_chain",
                    "chain_id": chain.chain_id,
                    "topic": chain.topic,
                    "conclusion": chain.conclusion,
                    "success": chain.success,
                    "duration": (chain.end_time - chain.start_time).total_seconds()
                },
                importance=chain.confidence,
                tags={"reasoning", "completed", chain.metadata.get("reasoning_type", "unknown")}
            )
            
            await self.memory_manager.store_memory(memory_entry)
            
            # Remove from active chains
            del self.active_chains[chain_id]
            
            self.logger.info(f"Completed reasoning chain: {chain_id}")
            return chain
            
        except Exception as e:
            self.logger.error(f"Error completing reasoning chain: {e}")
            return None

class StrategicPlanningEngine:
    """Long-term strategic planning and goal decomposition"""
    
    def __init__(self, memory_manager: PersistentMemoryManager, reasoning_engine: AdvancedReasoningEngine):
        self.memory_manager = memory_manager
        self.reasoning_engine = reasoning_engine
        self.logger = logging.getLogger("strategic_planning")
        
        # Planning templates
        self.planning_templates = self._load_planning_templates()
        
        # Active plans
        self.active_plans = {}
        
    def _load_planning_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load strategic planning templates"""
        
        return {
            "cybersecurity_assessment": {
                "phases": [
                    "reconnaissance",
                    "vulnerability_analysis", 
                    "threat_modeling",
                    "risk_assessment",
                    "mitigation_planning",
                    "implementation",
                    "monitoring"
                ],
                "typical_duration": 30,  # days
                "success_criteria": [
                    "Complete security posture assessment",
                    "Identified all critical vulnerabilities",
                    "Developed mitigation strategies",
                    "Implemented security controls"
                ]
            },
            "penetration_testing": {
                "phases": [
                    "scoping",
                    "information_gathering",
                    "threat_modeling",
                    "vulnerability_assessment",
                    "exploitation",
                    "post_exploitation",
                    "reporting"
                ],
                "typical_duration": 14,  # days
                "success_criteria": [
                    "Identified exploitable vulnerabilities",
                    "Demonstrated business impact",
                    "Provided remediation recommendations"
                ]
            },
            "incident_response": {
                "phases": [
                    "detection",
                    "analysis", 
                    "containment",
                    "eradication",
                    "recovery",
                    "lessons_learned"
                ],
                "typical_duration": 7,  # days
                "success_criteria": [
                    "Contained security incident",
                    "Minimized business impact",
                    "Prevented future incidents"
                ]
            }
        }
    
    async def create_strategic_plan(self, title: str, primary_goal: str, 
                                  template_type: str = "cybersecurity_assessment") -> str:
        """Create a new strategic plan"""
        
        template = self.planning_templates.get(template_type, {})
        
        # Decompose primary goal into sub-goals
        sub_goals = await self._decompose_goal(primary_goal, template)
        
        # Create timeline
        timeline = self._create_timeline(template, sub_goals)
        
        # Generate milestones
        milestones = self._generate_milestones(sub_goals, timeline)
        
        # Assess risks
        risk_factors = await self._assess_risks(primary_goal, sub_goals)
        
        # Determine resources
        resources_required = self._determine_resources(template, sub_goals)
        
        plan = StrategicPlan(
            title=title,
            description=f"Strategic plan for {primary_goal}",
            primary_goal=primary_goal,
            sub_goals=sub_goals,
            timeline=timeline,
            milestones=milestones,
            success_criteria=template.get("success_criteria", []),
            risk_factors=risk_factors,
            resources_required=resources_required,
            current_status="planning"
        )
        
        # Store in database
        try:
            self.memory_manager.conn.execute("""
            INSERT INTO strategic_plans
            (plan_id, title, description, primary_goal, sub_goals, timeline,
             milestones, success_criteria, risk_factors, resources_required,
             current_status, progress_percentage, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                plan.plan_id,
                plan.title,
                plan.description,
                plan.primary_goal,
                pickle.dumps(plan.sub_goals),
                pickle.dumps(plan.timeline),
                pickle.dumps(plan.milestones),
                pickle.dumps(plan.success_criteria),
                pickle.dumps(plan.risk_factors),
                pickle.dumps(plan.resources_required),
                plan.current_status,
                plan.progress_percentage,
                plan.created_at.timestamp(),
                plan.updated_at.timestamp()
            ))
            
            self.memory_manager.conn.commit()
            
            # Add to active plans
            self.active_plans[plan.plan_id] = plan
            
            # Store in episodic memory
            memory_entry = MemoryEntry(
                memory_type=MemoryType.STRATEGIC,
                content={
                    "type": "strategic_plan_created",
                    "plan_id": plan.plan_id,
                    "title": title,
                    "primary_goal": primary_goal,
                    "sub_goals_count": len(sub_goals)
                },
                importance=0.8,
                tags={"strategic_planning", "plan_created", template_type}
            )
            
            await self.memory_manager.store_memory(memory_entry)
            
            self.logger.info(f"Created strategic plan: {plan.plan_id}")
            return plan.plan_id
            
        except Exception as e:
            self.logger.error(f"Error creating strategic plan: {e}")
            return None
    
    async def _decompose_goal(self, primary_goal: str, template: Dict[str, Any]) -> List[str]:
        """Decompose primary goal into actionable sub-goals"""
        
        # Start reasoning chain for goal decomposition
        chain_id = await self.reasoning_engine.start_reasoning_chain(
            topic=f"Goal Decomposition: {primary_goal}",
            goal="Break down primary goal into actionable sub-goals",
            reasoning_type=ReasoningType.STRATEGIC
        )
        
        sub_goals = []
        
        # Use template phases if available
        if "phases" in template:
            for phase in template["phases"]:
                sub_goal = f"Complete {phase} phase for {primary_goal}"
                sub_goals.append(sub_goal)
                
                # Add reasoning step
                await self.reasoning_engine.add_reasoning_step(
                    chain_id,
                    f"Phase {phase} is essential for achieving {primary_goal}",
                    "strategic_decomposition"
                )
        else:
            # Generic decomposition
            generic_phases = [
                "planning and preparation",
                "implementation and execution", 
                "monitoring and evaluation",
                "optimization and improvement"
            ]
            
            for phase in generic_phases:
                sub_goal = f"Complete {phase} for {primary_goal}"
                sub_goals.append(sub_goal)
        
        # Complete reasoning chain
        await self.reasoning_engine.complete_reasoning_chain(chain_id)
        
        return sub_goals
    
    def _create_timeline(self, template: Dict[str, Any], sub_goals: List[str]) -> Dict[str, datetime]:
        """Create timeline for strategic plan"""
        
        timeline = {}
        start_date = datetime.now()
        
        # Total duration from template or estimate
        total_duration = template.get("typical_duration", len(sub_goals) * 3)  # days
        duration_per_goal = total_duration / len(sub_goals) if sub_goals else 1
        
        current_date = start_date
        
        for i, sub_goal in enumerate(sub_goals):
            timeline[f"sub_goal_{i}_start"] = current_date
            timeline[f"sub_goal_{i}_end"] = current_date + timedelta(days=duration_per_goal)
            current_date = timeline[f"sub_goal_{i}_end"]
        
        timeline["plan_start"] = start_date
        timeline["plan_end"] = current_date
        
        return timeline
    
    def _generate_milestones(self, sub_goals: List[str], timeline: Dict[str, datetime]) -> List[Dict[str, Any]]:
        """Generate milestones for strategic plan"""
        
        milestones = []
        
        for i, sub_goal in enumerate(sub_goals):
            milestone = {
                "milestone_id": str(uuid.uuid4()),
                "title": f"Milestone {i+1}: {sub_goal}",
                "description": f"Complete sub-goal: {sub_goal}",
                "target_date": timeline.get(f"sub_goal_{i}_end", datetime.now()),
                "success_criteria": [f"Successfully complete {sub_goal}"],
                "status": "pending",
                "progress_percentage": 0.0
            }
            
            milestones.append(milestone)
        
        return milestones
    
    async def _assess_risks(self, primary_goal: str, sub_goals: List[str]) -> List[str]:
        """Assess potential risks for the strategic plan"""
        
        # Start reasoning chain for risk assessment
        chain_id = await self.reasoning_engine.start_reasoning_chain(
            topic=f"Risk Assessment: {primary_goal}",
            goal="Identify potential risks and mitigation strategies",
            reasoning_type=ReasoningType.STRATEGIC
        )
        
        # Common cybersecurity risks
        common_risks = [
            "Technical complexity may exceed available expertise",
            "Timeline constraints may impact quality",
            "Resource availability may be limited",
            "External dependencies may cause delays",
            "Changing requirements may affect scope",
            "Security vulnerabilities may be discovered during implementation",
            "Stakeholder availability may be limited"
        ]
        
        # Assess relevance of each risk
        relevant_risks = []
        
        for risk in common_risks:
            # Add reasoning step for each risk
            await self.reasoning_engine.add_reasoning_step(
                chain_id,
                f"Risk consideration: {risk}",
                "risk_assessment"
            )
            
            relevant_risks.append(risk)
        
        # Complete reasoning chain
        await self.reasoning_engine.complete_reasoning_chain(chain_id)
        
        return relevant_risks
    
    def _determine_resources(self, template: Dict[str, Any], sub_goals: List[str]) -> List[str]:
        """Determine required resources for strategic plan"""
        
        # Common resources for cybersecurity plans
        base_resources = [
            "Cybersecurity expertise",
            "Technical infrastructure access",
            "Documentation and reporting tools",
            "Communication and collaboration platforms"
        ]
        
        # Template-specific resources
        if "resources" in template:
            base_resources.extend(template["resources"])
        
        # Add resources based on sub-goals
        specialized_resources = []
        
        for sub_goal in sub_goals:
            if "vulnerability" in sub_goal.lower():
                specialized_resources.append("Vulnerability scanning tools")
            elif "penetration" in sub_goal.lower():
                specialized_resources.append("Penetration testing tools")
            elif "monitoring" in sub_goal.lower():
                specialized_resources.append("Security monitoring platforms")
        
        return list(set(base_resources + specialized_resources))
    
    async def update_plan_progress(self, plan_id: str, milestone_id: str = None, 
                                 progress_percentage: float = None, status: str = None) -> bool:
        """Update progress of strategic plan"""
        
        try:
            if plan_id not in self.active_plans:
                # Load from database
                plan = await self._load_plan(plan_id)
                if not plan:
                    self.logger.error(f"Plan {plan_id} not found")
                    return False
                self.active_plans[plan_id] = plan
            
            plan = self.active_plans[plan_id]
            
            # Update milestone if specified
            if milestone_id:
                for milestone in plan.milestones:
                    if milestone["milestone_id"] == milestone_id:
                        if progress_percentage is not None:
                            milestone["progress_percentage"] = progress_percentage
                        if status:
                            milestone["status"] = status
                        break
            
            # Update overall plan progress
            if progress_percentage is not None:
                plan.progress_percentage = progress_percentage
            
            if status:
                plan.current_status = status
            
            plan.updated_at = datetime.now()
            
            # Update database
            self.memory_manager.conn.execute("""
            UPDATE strategic_plans
            SET milestones = ?, progress_percentage = ?, 
                current_status = ?, updated_at = ?
            WHERE plan_id = ?
            """, (
                pickle.dumps(plan.milestones),
                plan.progress_percentage,
                plan.current_status,
                plan.updated_at.timestamp(),
                plan_id
            ))
            
            self.memory_manager.conn.commit()
            
            # Store progress update in memory
            memory_entry = MemoryEntry(
                memory_type=MemoryType.EPISODIC,
                content={
                    "type": "plan_progress_update",
                    "plan_id": plan_id,
                    "milestone_id": milestone_id,
                    "progress_percentage": progress_percentage,
                    "status": status
                },
                importance=0.6,
                tags={"strategic_planning", "progress_update"}
            )
            
            await self.memory_manager.store_memory(memory_entry)
            
            self.logger.info(f"Updated plan progress: {plan_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating plan progress: {e}")
            return False
    
    async def _load_plan(self, plan_id: str) -> Optional[StrategicPlan]:
        """Load strategic plan from database"""
        
        try:
            cursor = self.memory_manager.conn.execute("""
            SELECT * FROM strategic_plans WHERE plan_id = ?
            """, (plan_id,))
            
            row = cursor.fetchone()
            if row:
                return StrategicPlan(
                    plan_id=row[0],
                    title=row[1],
                    description=row[2],
                    primary_goal=row[3],
                    sub_goals=pickle.loads(row[4]),
                    timeline=pickle.loads(row[5]),
                    milestones=pickle.loads(row[6]),
                    success_criteria=pickle.loads(row[7]),
                    risk_factors=pickle.loads(row[8]),
                    resources_required=pickle.loads(row[9]),
                    current_status=row[10],
                    progress_percentage=row[11],
                    created_at=datetime.fromtimestamp(row[12]),
                    updated_at=datetime.fromtimestamp(row[13])
                )
                
        except Exception as e:
            self.logger.error(f"Error loading plan {plan_id}: {e}")
        
        return None

# Integration class that brings everything together
class PersistentCognitiveSystem:
    """Main system that integrates persistent memory, reasoning, and strategic planning"""
    
    def __init__(self, db_path: str = "data/cognitive_system.db"):
        # Initialize components
        self.memory_manager = PersistentMemoryManager(db_path)
        self.reasoning_engine = AdvancedReasoningEngine(self.memory_manager)
        self.strategic_planner = StrategicPlanningEngine(self.memory_manager, self.reasoning_engine)
        
        self.logger = logging.getLogger("persistent_cognitive_system")
        self.logger.info("Persistent cognitive system initialized")
    
    async def process_complex_scenario(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Process a complex cybersecurity scenario using all cognitive capabilities"""
        
        scenario_id = str(uuid.uuid4())
        self.logger.info(f"Processing complex scenario: {scenario_id}")
        
        results = {
            "scenario_id": scenario_id,
            "timestamp": datetime.now().isoformat(),
            "results": {}
        }
        
        try:
            # Step 1: Store scenario in memory
            scenario_memory = MemoryEntry(
                memory_type=MemoryType.EPISODIC,
                content=scenario,
                importance=0.8,
                tags={"scenario", "complex", "cybersecurity"}
            )
            
            memory_id = await self.memory_manager.store_memory(scenario_memory)
            results["results"]["memory_id"] = memory_id
            
            # Step 2: Start strategic planning if it's a long-term objective
            if scenario.get("type") == "strategic" or scenario.get("requires_planning", False):
                plan_id = await self.strategic_planner.create_strategic_plan(
                    title=scenario.get("title", f"Scenario {scenario_id}"),
                    primary_goal=scenario.get("objective", "Complete cybersecurity scenario"),
                    template_type=scenario.get("template", "cybersecurity_assessment")
                )
                
                results["results"]["plan_id"] = plan_id
            
            # Step 3: Apply reasoning to understand the scenario
            reasoning_types = scenario.get("reasoning_types", [ReasoningType.DEDUCTIVE])
            reasoning_results = {}
            
            for reasoning_type in reasoning_types:
                chain_id = await self.reasoning_engine.start_reasoning_chain(
                    topic=f"Scenario Analysis: {scenario.get('title', scenario_id)}",
                    goal="Analyze and understand the cybersecurity scenario",
                    reasoning_type=reasoning_type
                )
                
                # Add reasoning steps based on scenario details
                for detail in scenario.get("details", []):
                    await self.reasoning_engine.add_reasoning_step(
                        chain_id,
                        detail,
                        "scenario_analysis",
                        scenario.get("evidence", [])
                    )
                
                # Complete reasoning
                chain = await self.reasoning_engine.complete_reasoning_chain(chain_id)
                reasoning_results[reasoning_type.value] = {
                    "chain_id": chain_id,
                    "conclusion": chain.conclusion if chain else "Failed to complete",
                    "confidence": chain.confidence if chain else 0.0
                }
            
            results["results"]["reasoning"] = reasoning_results
            
            # Step 4: Generate recommendations
            recommendations = await self._generate_recommendations(scenario, reasoning_results)
            results["results"]["recommendations"] = recommendations
            
            # Step 5: Update long-term memory with insights
            insight_memory = MemoryEntry(
                memory_type=MemoryType.SEMANTIC,
                content={
                    "type": "scenario_insight",
                    "scenario_id": scenario_id,
                    "key_learnings": recommendations,
                    "confidence_scores": {k: v["confidence"] for k, v in reasoning_results.items()}
                },
                importance=0.7,
                tags={"insight", "learning", "cybersecurity"}
            )
            
            await self.memory_manager.store_memory(insight_memory)
            
            results["status"] = "success"
            self.logger.info(f"Successfully processed scenario: {scenario_id}")
            
        except Exception as e:
            results["status"] = "error"
            results["error"] = str(e)
            self.logger.error(f"Error processing scenario {scenario_id}: {e}")
        
        return results
    
    async def _generate_recommendations(self, scenario: Dict[str, Any], 
                                      reasoning_results: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on scenario analysis"""
        
        recommendations = []
        
        # Base recommendations based on scenario type
        scenario_type = scenario.get("type", "general")
        
        if scenario_type == "vulnerability_assessment":
            recommendations.extend([
                "Conduct comprehensive vulnerability scan",
                "Prioritize critical vulnerabilities for immediate remediation",
                "Implement security patches and updates",
                "Establish regular vulnerability monitoring"
            ])
        elif scenario_type == "incident_response":
            recommendations.extend([
                "Immediately contain the security incident",
                "Preserve forensic evidence",
                "Assess scope and impact of the incident",
                "Implement recovery procedures",
                "Conduct post-incident analysis"
            ])
        elif scenario_type == "penetration_testing":
            recommendations.extend([
                "Define clear scope and objectives",
                "Follow structured testing methodology",
                "Document all findings and evidence",
                "Provide actionable remediation guidance"
            ])
        else:
            recommendations.extend([
                "Assess current security posture",
                "Identify key risk areas",
                "Develop mitigation strategies",
                "Implement monitoring and detection"
            ])
        
        # Add reasoning-based recommendations
        for reasoning_type, results in reasoning_results.items():
            if results["confidence"] > 0.7:
                recommendations.append(f"High confidence in {reasoning_type} analysis suggests prioritizing related actions")
        
        # Search for similar past experiences
        similar_memories = await self.memory_manager.search_memories(
            scenario.get("title", ""), [MemoryType.EPISODIC], limit=3
        )
        
        if similar_memories:
            recommendations.append(f"Apply lessons learned from {len(similar_memories)} similar past scenarios")
        
        return recommendations[:10]  # Limit to top 10 recommendations

# Factory function for easy instantiation
def create_persistent_cognitive_system(db_path: str = "data/cognitive_system.db") -> PersistentCognitiveSystem:
    """Create and initialize the persistent cognitive system"""
    return PersistentCognitiveSystem(db_path)

# Main execution for testing
if __name__ == "__main__":
    import asyncio
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    async def test_system():
        """Test the persistent cognitive system"""
        
        # Create system
        system = create_persistent_cognitive_system()
        
        # Test scenario
        test_scenario = {
            "type": "vulnerability_assessment",
            "title": "Web Application Security Assessment",
            "objective": "Assess security posture of critical web application",
            "details": [
                "Web application handles sensitive customer data",
                "Application has not been tested in 12 months",
                "Recent security incidents in similar applications reported"
            ],
            "evidence": [
                "Previous vulnerability scan results",
                "Security incident reports from industry",
                "Application architecture documentation"
            ],
            "reasoning_types": [ReasoningType.DEDUCTIVE, ReasoningType.CAUSAL],
            "requires_planning": True,
            "template": "cybersecurity_assessment"
        }
        
        # Process scenario
        results = await system.process_complex_scenario(test_scenario)
        
        print("=== Persistent Cognitive System Test Results ===")
        print(json.dumps(results, indent=2, default=str))
        
        # Test memory search
        memories = await system.memory_manager.search_memories("vulnerability", limit=5)
        print(f"\n=== Found {len(memories)} memories related to 'vulnerability' ===")
        
        for memory in memories:
            print(f"- {memory.memory_id}: {memory.content.get('type', 'Unknown')} (importance: {memory.importance:.2f})")
    
    # Run test
    asyncio.run(test_system())
