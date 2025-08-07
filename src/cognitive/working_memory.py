"""
Working Memory Management System with Attention-based Focus and Context Switching
Implements dynamic attention mechanisms and context management for cognitive agents
"""
import sqlite3
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import logging
from pathlib import Path
import heapq
import threading
import time

logger = logging.getLogger(__name__)

@dataclass
class WorkingMemoryItem:
    """Individual item in working memory"""
    id: str
    content: str
    item_type: str  # goal, observation, hypothesis, plan, etc.
    priority: float  # 0.0-1.0, higher is more important
    activation_level: float  # 0.0-1.0, current activation
    created_at: datetime
    last_accessed: datetime
    access_count: int
    decay_rate: float  # how quickly activation decays
    context_tags: List[str]
    source_agent: str
    related_items: List[str]  # IDs of related items

@dataclass
class AttentionFocus:
    """Current attention focus with weighted priorities"""
    focus_id: str
    focus_type: str  # task, threat, goal, etc.
    focus_items: List[str]  # Working memory item IDs
    attention_weight: float  # 0.0-1.0
    duration: timedelta
    created_at: datetime
    metadata: Dict[str, Any]

class WorkingMemoryManager:
    """Advanced working memory with attention-based focus management"""
    
    def __init__(self, db_path: str = "data/cognitive/working_memory.db",
                 capacity: int = 50, decay_interval: float = 30.0):
        """Initialize working memory system"""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.capacity = capacity  # Maximum items in working memory
        self.decay_interval = decay_interval  # Seconds between decay updates
        
        self._init_database()
        self._memory_items = {}  # In-memory cache
        self._attention_focus = None
        self._attention_history = []
        
        # Start background decay process
        self._decay_thread = threading.Thread(target=self._decay_loop, daemon=True)
        self._decay_running = True
        self._decay_thread.start()
        
        # Load existing items
        self._load_working_memory()
    
    def _init_database(self):
        """Initialize database schemas"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS working_memory_items (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    item_type TEXT NOT NULL,
                    priority REAL NOT NULL,
                    activation_level REAL NOT NULL,
                    created_at TEXT NOT NULL,
                    last_accessed TEXT NOT NULL,
                    access_count INTEGER DEFAULT 0,
                    decay_rate REAL DEFAULT 0.1,
                    context_tags TEXT,
                    source_agent TEXT,
                    related_items TEXT,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS attention_focus_log (
                    id TEXT PRIMARY KEY,
                    focus_type TEXT NOT NULL,
                    focus_items TEXT,
                    attention_weight REAL NOT NULL,
                    duration_seconds REAL,
                    created_at TEXT NOT NULL,
                    ended_at TEXT,
                    metadata TEXT,
                    agent_id TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS context_switches (
                    id TEXT PRIMARY KEY,
                    from_focus TEXT,
                    to_focus TEXT,
                    switch_reason TEXT,
                    switch_cost REAL,
                    timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                    agent_id TEXT
                )
            """)
            
            # Create indices
            conn.execute("CREATE INDEX IF NOT EXISTS idx_wm_priority ON working_memory_items(priority DESC)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_wm_activation ON working_memory_items(activation_level DESC)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_wm_type ON working_memory_items(item_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_wm_agent ON working_memory_items(source_agent)")
    
    def add_item(self, content: str, item_type: str, priority: float = 0.5,
                source_agent: str = "", context_tags: List[str] = None) -> str:
        """Add item to working memory with attention-based priority"""
        try:
            item_id = str(uuid.uuid4())
            
            item = WorkingMemoryItem(
                id=item_id,
                content=content,
                item_type=item_type,
                priority=priority,
                activation_level=priority,  # Initial activation equals priority
                created_at=datetime.now(),
                last_accessed=datetime.now(),
                access_count=0,
                decay_rate=0.1,  # Default decay rate
                context_tags=context_tags or [],
                source_agent=source_agent,
                related_items=[]
            )
            
            # Check capacity and evict if necessary
            if len(self._memory_items) >= self.capacity:
                self._evict_lowest_activation()
            
            # Store in memory and database
            self._memory_items[item_id] = item
            self._store_item_to_db(item)
            
            # Update attention focus if this is high priority
            if priority > 0.7 and (not self._attention_focus or 
                                  priority > self._attention_focus.attention_weight):
                self._update_attention_focus(item_id, item_type, priority)
            
            logger.info(f"Added working memory item: {item_type} (priority: {priority:.2f})")
            return item_id
            
        except Exception as e:
            logger.error(f"Error adding working memory item: {e}")
            return ""
    
    def get_item(self, item_id: str) -> Optional[WorkingMemoryItem]:
        """Retrieve item from working memory and update activation"""
        try:
            if item_id in self._memory_items:
                item = self._memory_items[item_id]
                
                # Update access statistics
                item.last_accessed = datetime.now()
                item.access_count += 1
                
                # Boost activation on access (but cap at 1.0)
                activation_boost = min(0.2, 1.0 - item.activation_level)
                item.activation_level = min(1.0, item.activation_level + activation_boost)
                
                # Update in database
                self._update_item_in_db(item)
                
                logger.debug(f"Retrieved working memory item: {item_id[:8]}...")
                return item
            
            logger.warning(f"Working memory item not found: {item_id}")
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving working memory item: {e}")
            return None
    
    def get_active_items(self, min_activation: float = 0.3,
                        item_type: str = "", limit: int = 20) -> List[WorkingMemoryItem]:
        """Get currently active items above activation threshold"""
        try:
            active_items = []
            
            for item in self._memory_items.values():
                if (item.activation_level >= min_activation and
                    (not item_type or item.item_type == item_type)):
                    active_items.append(item)
            
            # Sort by activation level (highest first)
            active_items.sort(key=lambda x: x.activation_level, reverse=True)
            
            logger.info(f"Retrieved {len(active_items[:limit])} active items")
            return active_items[:limit]
            
        except Exception as e:
            logger.error(f"Error getting active items: {e}")
            return []
    
    def focus_attention(self, focus_type: str, item_ids: List[str],
                       attention_weight: float = 0.8, agent_id: str = "") -> str:
        """Focus attention on specific items"""
        try:
            focus_id = str(uuid.uuid4())
            
            # End current focus if exists
            if self._attention_focus:
                self._end_attention_focus()
            
            # Create new attention focus
            new_focus = AttentionFocus(
                focus_id=focus_id,
                focus_type=focus_type,
                focus_items=item_ids,
                attention_weight=attention_weight,
                duration=timedelta(0),
                created_at=datetime.now(),
                metadata={'agent_id': agent_id}
            )
            
            self._attention_focus = new_focus
            
            # Boost activation of focused items
            for item_id in item_ids:
                if item_id in self._memory_items:
                    item = self._memory_items[item_id]
                    item.activation_level = min(1.0, item.activation_level + 0.3)
                    self._update_item_in_db(item)
            
            # Store focus in database
            self._store_attention_focus(new_focus)
            
            logger.info(f"Focused attention on {focus_type}: {len(item_ids)} items")
            return focus_id
            
        except Exception as e:
            logger.error(f"Error focusing attention: {e}")
            return ""
    
    def switch_context(self, new_focus_type: str, new_item_ids: List[str],
                      switch_reason: str = "", agent_id: str = "") -> Dict[str, Any]:
        """Switch attention context with cost calculation"""
        try:
            switch_result = {
                'switch_id': str(uuid.uuid4()),
                'from_focus': None,
                'to_focus': new_focus_type,
                'switch_cost': 0.0,
                'success': False
            }
            
            # Calculate switch cost based on current focus
            if self._attention_focus:
                switch_result['from_focus'] = self._attention_focus.focus_type
                
                # Cost factors:
                # 1. How long we've been in current focus
                current_duration = datetime.now() - self._attention_focus.created_at
                duration_cost = min(current_duration.total_seconds() / 300.0, 0.3)  # Max 5min
                
                # 2. Number of active items being abandoned
                abandoned_items = len(self._attention_focus.focus_items)
                abandonment_cost = min(abandoned_items * 0.1, 0.4)
                
                # 3. Similarity between old and new focus
                similarity_discount = self._calculate_focus_similarity(
                    self._attention_focus.focus_items, new_item_ids
                )
                
                total_cost = duration_cost + abandonment_cost - similarity_discount
                switch_result['switch_cost'] = max(0.0, min(total_cost, 1.0))
                
                # Record context switch
                self._record_context_switch(
                    self._attention_focus.focus_type,
                    new_focus_type,
                    switch_reason,
                    switch_result['switch_cost'],
                    agent_id
                )
            
            # Perform the switch
            focus_id = self.focus_attention(new_focus_type, new_item_ids, agent_id=agent_id)
            switch_result['success'] = bool(focus_id)
            
            logger.info(f"Context switch: {switch_result['from_focus']} -> {new_focus_type} (cost: {switch_result['switch_cost']:.3f})")
            return switch_result
            
        except Exception as e:
            logger.error(f"Error switching context: {e}")
            return {'error': str(e)}
    
    def get_current_focus(self) -> Optional[AttentionFocus]:
        """Get current attention focus"""
        return self._attention_focus
    
    def decay_memory(self):
        """Apply decay to all working memory items"""
        try:
            decayed_count = 0
            evicted_items = []
            
            for item_id, item in list(self._memory_items.items()):
                # Apply decay based on time since last access
                time_since_access = datetime.now() - item.last_accessed
                decay_amount = item.decay_rate * (time_since_access.total_seconds() / 60.0)
                
                item.activation_level = max(0.0, item.activation_level - decay_amount)
                decayed_count += 1
                
                # Evict items with very low activation
                if item.activation_level < 0.05:
                    evicted_items.append(item_id)
                else:
                    # Update in database
                    self._update_item_in_db(item)
            
            # Remove evicted items
            for item_id in evicted_items:
                del self._memory_items[item_id]
                self._remove_item_from_db(item_id)
            
            if evicted_items:
                logger.info(f"Memory decay: {decayed_count} items decayed, {len(evicted_items)} evicted")
                
        except Exception as e:
            logger.error(f"Error during memory decay: {e}")
    
    def find_related_items(self, item_id: str, max_items: int = 5) -> List[WorkingMemoryItem]:
        """Find items related to the given item"""
        try:
            if item_id not in self._memory_items:
                return []
            
            source_item = self._memory_items[item_id]
            related_items = []
            
            for other_id, other_item in self._memory_items.items():
                if other_id == item_id:
                    continue
                
                # Calculate relatedness score
                relatedness = 0.0
                
                # Same type bonus
                if source_item.item_type == other_item.item_type:
                    relatedness += 0.3
                
                # Shared context tags
                shared_tags = set(source_item.context_tags) & set(other_item.context_tags)
                if shared_tags:
                    relatedness += len(shared_tags) * 0.2
                
                # Same source agent
                if source_item.source_agent == other_item.source_agent:
                    relatedness += 0.2
                
                # Temporal proximity
                time_diff = abs((source_item.created_at - other_item.created_at).total_seconds())
                if time_diff < 300:  # Within 5 minutes
                    relatedness += 0.3 * (300 - time_diff) / 300
                
                if relatedness > 0.1:  # Minimum relatedness threshold
                    related_items.append((other_item, relatedness))
            
            # Sort by relatedness and return top items
            related_items.sort(key=lambda x: x[1], reverse=True)
            
            return [item for item, score in related_items[:max_items]]
            
        except Exception as e:
            logger.error(f"Error finding related items: {e}")
            return []
    
    def _update_attention_focus(self, item_id: str, item_type: str, priority: float):
        """Update current attention focus"""
        if self._attention_focus:
            self._end_attention_focus()
        
        self.focus_attention(item_type, [item_id], priority)
    
    def _end_attention_focus(self):
        """End current attention focus"""
        if self._attention_focus:
            # Update duration
            self._attention_focus.duration = datetime.now() - self._attention_focus.created_at
            
            # Add to history
            self._attention_history.append(self._attention_focus)
            
            # Update in database
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    UPDATE attention_focus_log SET
                        ended_at = ?,
                        duration_seconds = ?
                    WHERE id = ?
                """, (
                    datetime.now().isoformat(),
                    self._attention_focus.duration.total_seconds(),
                    self._attention_focus.focus_id
                ))
            
            self._attention_focus = None
    
    def _evict_lowest_activation(self):
        """Evict item with lowest activation to make space"""
        if not self._memory_items:
            return
        
        lowest_item_id = min(
            self._memory_items.keys(),
            key=lambda x: self._memory_items[x].activation_level
        )
        
        del self._memory_items[lowest_item_id]
        self._remove_item_from_db(lowest_item_id)
        
        logger.debug(f"Evicted working memory item: {lowest_item_id[:8]}...")
    
    def _calculate_focus_similarity(self, items1: List[str], items2: List[str]) -> float:
        """Calculate similarity between two sets of focus items"""
        if not items1 or not items2:
            return 0.0
        
        set1 = set(items1)
        set2 = set(items2)
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0
    
    def _record_context_switch(self, from_focus: str, to_focus: str,
                              reason: str, cost: float, agent_id: str):
        """Record context switch in database"""
        try:
            switch_id = str(uuid.uuid4())
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO context_switches (
                        id, from_focus, to_focus, switch_reason,
                        switch_cost, agent_id
                    ) VALUES (?, ?, ?, ?, ?, ?)
                """, (switch_id, from_focus, to_focus, reason, cost, agent_id))
                
        except Exception as e:
            logger.error(f"Error recording context switch: {e}")
    
    def _decay_loop(self):
        """Background thread for memory decay"""
        while self._decay_running:
            try:
                time.sleep(self.decay_interval)
                self.decay_memory()
            except Exception as e:
                logger.error(f"Error in decay loop: {e}")
    
    def _load_working_memory(self):
        """Load working memory items from database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT * FROM working_memory_items 
                    ORDER BY activation_level DESC 
                    LIMIT ?
                """, (self.capacity,))
                
                for row in cursor.fetchall():
                    item = WorkingMemoryItem(
                        id=row[0],
                        content=row[1],
                        item_type=row[2],
                        priority=row[3],
                        activation_level=row[4],
                        created_at=datetime.fromisoformat(row[5]),
                        last_accessed=datetime.fromisoformat(row[6]),
                        access_count=row[7],
                        decay_rate=row[8],
                        context_tags=json.loads(row[9]) if row[9] else [],
                        source_agent=row[10] or "",
                        related_items=json.loads(row[11]) if row[11] else []
                    )
                    self._memory_items[item.id] = item
                
                logger.info(f"Loaded {len(self._memory_items)} working memory items")
                
        except Exception as e:
            logger.error(f"Error loading working memory: {e}")
    
    def _store_item_to_db(self, item: WorkingMemoryItem):
        """Store item to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO working_memory_items (
                        id, content, item_type, priority, activation_level,
                        created_at, last_accessed, access_count, decay_rate,
                        context_tags, source_agent, related_items
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    item.id, item.content, item.item_type, item.priority,
                    item.activation_level, item.created_at.isoformat(),
                    item.last_accessed.isoformat(), item.access_count,
                    item.decay_rate, json.dumps(item.context_tags),
                    item.source_agent, json.dumps(item.related_items)
                ))
                
        except Exception as e:
            logger.error(f"Error storing item to database: {e}")
    
    def _update_item_in_db(self, item: WorkingMemoryItem):
        """Update item in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    UPDATE working_memory_items SET
                        activation_level = ?, last_accessed = ?,
                        access_count = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                """, (
                    item.activation_level, item.last_accessed.isoformat(),
                    item.access_count, item.id
                ))
                
        except Exception as e:
            logger.error(f"Error updating item in database: {e}")
    
    def _remove_item_from_db(self, item_id: str):
        """Remove item from database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM working_memory_items WHERE id = ?", (item_id,))
                
        except Exception as e:
            logger.error(f"Error removing item from database: {e}")
    
    def _store_attention_focus(self, focus: AttentionFocus):
        """Store attention focus in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO attention_focus_log (
                        id, focus_type, focus_items, attention_weight,
                        created_at, metadata, agent_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    focus.focus_id, focus.focus_type,
                    json.dumps(focus.focus_items), focus.attention_weight,
                    focus.created_at.isoformat(), json.dumps(focus.metadata),
                    focus.metadata.get('agent_id', '')
                ))
                
        except Exception as e:
            logger.error(f"Error storing attention focus: {e}")
    
    def get_working_memory_statistics(self) -> Dict[str, Any]:
        """Get comprehensive working memory statistics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                stats = {
                    'current_capacity': len(self._memory_items),
                    'max_capacity': self.capacity,
                    'utilization': len(self._memory_items) / self.capacity
                }
                
                # Activation distribution
                if self._memory_items:
                    activations = [item.activation_level for item in self._memory_items.values()]
                    stats['avg_activation'] = sum(activations) / len(activations)
                    stats['max_activation'] = max(activations)
                    stats['min_activation'] = min(activations)
                
                # Item type distribution
                type_counts = {}
                for item in self._memory_items.values():
                    type_counts[item.item_type] = type_counts.get(item.item_type, 0) + 1
                stats['item_types'] = type_counts
                
                # Context switch statistics
                cursor = conn.execute("""
                    SELECT COUNT(*), AVG(switch_cost) 
                    FROM context_switches 
                    WHERE timestamp > datetime('now', '-1 hour')
                """)
                row = cursor.fetchone()
                stats['recent_switches'] = row[0] or 0
                stats['avg_switch_cost'] = row[1] or 0.0
                
                # Current focus
                if self._attention_focus:
                    stats['current_focus'] = {
                        'type': self._attention_focus.focus_type,
                        'items': len(self._attention_focus.focus_items),
                        'weight': self._attention_focus.attention_weight,
                        'duration_seconds': (datetime.now() - self._attention_focus.created_at).total_seconds()
                    }
                else:
                    stats['current_focus'] = None
                
                return stats
                
        except Exception as e:
            logger.error(f"Error getting working memory statistics: {e}")
            return {'error': str(e)}
    
    def cleanup(self):
        """Cleanup resources"""
        self._decay_running = False
        if self._decay_thread.is_alive():
            self._decay_thread.join(timeout=1.0)

# Export the main classes
__all__ = ['WorkingMemoryManager', 'WorkingMemoryItem', 'AttentionFocus']
