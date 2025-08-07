"""
Episodic Memory System for Experience Replay and Learning
Captures temporal sequences of agent experiences for learning
"""
import sqlite3
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import logging
from pathlib import Path
import pickle

logger = logging.getLogger(__name__)

@dataclass
class Episode:
    """Individual episode with temporal sequence"""
    id: str
    agent_id: str
    session_id: str
    start_time: datetime
    end_time: Optional[datetime]
    episode_type: str  # operation, training, evaluation, etc.
    context: Dict[str, Any]
    actions: List[Dict[str, Any]]
    observations: List[Dict[str, Any]]
    rewards: List[float]
    outcome: Optional[str]
    success: bool
    metadata: Dict[str, Any]

@dataclass
class ExperienceReplay:
    """Experience replay record for learning"""
    episode_id: str
    replay_count: int
    last_replayed: datetime
    replay_effectiveness: float
    learning_insights: List[str]

class EpisodicMemorySystem:
    """Advanced episodic memory with experience replay capabilities"""
    
    def __init__(self, db_path: str = "data/cognitive/episodic_memory.db"):
        """Initialize episodic memory system"""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
        self._active_episodes = {}
        
    def _init_database(self):
        """Initialize database schemas"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS episodes (
                    id TEXT PRIMARY KEY,
                    agent_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    start_time TEXT NOT NULL,
                    end_time TEXT,
                    episode_type TEXT NOT NULL,
                    context TEXT,
                    actions TEXT,
                    observations TEXT,
                    rewards TEXT,
                    outcome TEXT,
                    success BOOLEAN,
                    metadata TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS experience_replay (
                    id TEXT PRIMARY KEY,
                    episode_id TEXT,
                    replay_count INTEGER DEFAULT 0,
                    last_replayed TEXT,
                    replay_effectiveness REAL DEFAULT 0.0,
                    learning_insights TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (episode_id) REFERENCES episodes(id)
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS episode_patterns (
                    id TEXT PRIMARY KEY,
                    pattern_type TEXT,
                    pattern_description TEXT,
                    episodes TEXT,
                    frequency INTEGER,
                    success_rate REAL,
                    discovered_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indices
            conn.execute("CREATE INDEX IF NOT EXISTS idx_agent_episodes ON episodes(agent_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_episode_type ON episodes(episode_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_episode_success ON episodes(success)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_episode_start_time ON episodes(start_time)")
    
    def start_episode(self, agent_id: str, session_id: str, 
                     episode_type: str, context: Dict[str, Any] = None) -> str:
        """Start a new episode recording"""
        try:
            episode_id = str(uuid.uuid4())
            
            episode = Episode(
                id=episode_id,
                agent_id=agent_id,
                session_id=session_id,
                start_time=datetime.now(),
                end_time=None,
                episode_type=episode_type,
                context=context or {},
                actions=[],
                observations=[],
                rewards=[],
                outcome=None,
                success=False,
                metadata={}
            )
            
            self._active_episodes[episode_id] = episode
            
            # Store initial episode data
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO episodes (
                        id, agent_id, session_id, start_time, episode_type,
                        context, actions, observations, rewards, success, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    episode.id, episode.agent_id, episode.session_id,
                    episode.start_time.isoformat(), episode.episode_type,
                    json.dumps(episode.context), json.dumps(episode.actions),
                    json.dumps(episode.observations), json.dumps(episode.rewards),
                    episode.success, json.dumps(episode.metadata)
                ))
            
            logger.info(f"Started episode {episode_id} for agent {agent_id}")
            return episode_id
            
        except Exception as e:
            logger.error(f"Error starting episode: {e}")
            return ""
    
    def record_action(self, episode_id: str, action: Dict[str, Any]):
        """Record an action in the current episode"""
        try:
            if episode_id in self._active_episodes:
                episode = self._active_episodes[episode_id]
                action['timestamp'] = datetime.now().isoformat()
                episode.actions.append(action)
                
                logger.debug(f"Recorded action in episode {episode_id}: {action.get('type', 'unknown')}")
            else:
                logger.warning(f"Episode {episode_id} not active")
                
        except Exception as e:
            logger.error(f"Error recording action: {e}")
    
    def record_observation(self, episode_id: str, observation: Dict[str, Any]):
        """Record an observation in the current episode"""
        try:
            if episode_id in self._active_episodes:
                episode = self._active_episodes[episode_id]
                observation['timestamp'] = datetime.now().isoformat()
                episode.observations.append(observation)
                
                logger.debug(f"Recorded observation in episode {episode_id}")
            else:
                logger.warning(f"Episode {episode_id} not active")
                
        except Exception as e:
            logger.error(f"Error recording observation: {e}")
    
    def record_reward(self, episode_id: str, reward: float):
        """Record a reward signal in the current episode"""
        try:
            if episode_id in self._active_episodes:
                episode = self._active_episodes[episode_id]
                episode.rewards.append(reward)
                
                logger.debug(f"Recorded reward in episode {episode_id}: {reward}")
            else:
                logger.warning(f"Episode {episode_id} not active")
                
        except Exception as e:
            logger.error(f"Error recording reward: {e}")
    
    def end_episode(self, episode_id: str, success: bool = False, 
                   outcome: str = "", metadata: Dict[str, Any] = None):
        """End an episode and store final results"""
        try:
            if episode_id in self._active_episodes:
                episode = self._active_episodes[episode_id]
                episode.end_time = datetime.now()
                episode.success = success
                episode.outcome = outcome
                if metadata:
                    episode.metadata.update(metadata)
                
                # Update database with final episode data
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("""
                        UPDATE episodes SET
                            end_time = ?, actions = ?, observations = ?,
                            rewards = ?, outcome = ?, success = ?, metadata = ?
                        WHERE id = ?
                    """, (
                        episode.end_time.isoformat(),
                        json.dumps(episode.actions),
                        json.dumps(episode.observations),
                        json.dumps(episode.rewards),
                        episode.outcome,
                        episode.success,
                        json.dumps(episode.metadata),
                        episode_id
                    ))
                
                # Create experience replay record
                self._create_replay_record(episode_id)
                
                # Remove from active episodes
                del self._active_episodes[episode_id]
                
                logger.info(f"Ended episode {episode_id}: success={success}, outcome={outcome}")
            else:
                logger.warning(f"Episode {episode_id} not active")
                
        except Exception as e:
            logger.error(f"Error ending episode: {e}")
    
    def get_episodes_for_replay(self, agent_id: str = "", episode_type: str = "",
                               success_only: bool = False, limit: int = 10) -> List[Episode]:
        """Get episodes suitable for experience replay"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conditions = []
                params = []
                
                if agent_id:
                    conditions.append("agent_id = ?")
                    params.append(agent_id)
                
                if episode_type:
                    conditions.append("episode_type = ?")
                    params.append(episode_type)
                
                if success_only:
                    conditions.append("success = 1")
                
                where_clause = " AND ".join(conditions) if conditions else "1=1"
                
                cursor = conn.execute(f"""
                    SELECT * FROM episodes 
                    WHERE {where_clause} AND end_time IS NOT NULL
                    ORDER BY start_time DESC
                    LIMIT ?
                """, params + [limit])
                
                episodes = []
                for row in cursor.fetchall():
                    episode = Episode(
                        id=row[0],
                        agent_id=row[1],
                        session_id=row[2],
                        start_time=datetime.fromisoformat(row[3]),
                        end_time=datetime.fromisoformat(row[4]) if row[4] else None,
                        episode_type=row[5],
                        context=json.loads(row[6]) if row[6] else {},
                        actions=json.loads(row[7]) if row[7] else [],
                        observations=json.loads(row[8]) if row[8] else [],
                        rewards=json.loads(row[9]) if row[9] else [],
                        outcome=row[10],
                        success=bool(row[11]),
                        metadata=json.loads(row[12]) if row[12] else {}
                    )
                    episodes.append(episode)
                
                logger.info(f"Retrieved {len(episodes)} episodes for replay")
                return episodes
                
        except Exception as e:
            logger.error(f"Error getting episodes for replay: {e}")
            return []
    
    def replay_experience(self, episode_id: str) -> Dict[str, Any]:
        """Replay an episode and extract learning insights"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT * FROM episodes WHERE id = ?", (episode_id,))
                row = cursor.fetchone()
                
                if not row:
                    return {'error': 'Episode not found'}
                
                episode = Episode(
                    id=row[0],
                    agent_id=row[1],
                    session_id=row[2],
                    start_time=datetime.fromisoformat(row[3]),
                    end_time=datetime.fromisoformat(row[4]) if row[4] else None,
                    episode_type=row[5],
                    context=json.loads(row[6]) if row[6] else {},
                    actions=json.loads(row[7]) if row[7] else [],
                    observations=json.loads(row[8]) if row[8] else [],
                    rewards=json.loads(row[9]) if row[9] else [],
                    outcome=row[10],
                    success=bool(row[11]),
                    metadata=json.loads(row[12]) if row[12] else {}
                )
                
                # Analyze episode for learning insights
                insights = self._analyze_episode_for_insights(episode)
                
                # Update replay statistics
                self._update_replay_stats(episode_id, insights)
                
                logger.info(f"Replayed episode {episode_id} with {len(insights)} insights")
                return {
                    'episode': episode,
                    'insights': insights,
                    'replay_time': datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Error replaying experience: {e}")
            return {'error': str(e)}
    
    def discover_patterns(self) -> Dict[str, Any]:
        """Discover patterns across episodes"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Get all completed episodes
                cursor = conn.execute("""
                    SELECT * FROM episodes 
                    WHERE end_time IS NOT NULL
                    ORDER BY start_time
                """)
                
                episodes = cursor.fetchall()
                patterns = {
                    'action_patterns': self._discover_action_patterns(episodes),
                    'success_patterns': self._discover_success_patterns(episodes),
                    'temporal_patterns': self._discover_temporal_patterns(episodes),
                    'agent_patterns': self._discover_agent_patterns(episodes)
                }
                
                # Store discovered patterns
                for pattern_type, pattern_list in patterns.items():
                    for pattern in pattern_list:
                        self._store_pattern(pattern_type, pattern)
                
                logger.info(f"Discovered patterns: {sum(len(p) for p in patterns.values())} total")
                return patterns
                
        except Exception as e:
            logger.error(f"Error discovering patterns: {e}")
            return {'error': str(e)}
    
    def _create_replay_record(self, episode_id: str):
        """Create experience replay record"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO experience_replay (id, episode_id, last_replayed)
                    VALUES (?, ?, ?)
                """, (str(uuid.uuid4()), episode_id, datetime.now().isoformat()))
                
        except Exception as e:
            logger.error(f"Error creating replay record: {e}")
    
    def _analyze_episode_for_insights(self, episode: Episode) -> List[str]:
        """Analyze episode and extract learning insights"""
        insights = []
        
        try:
            # Action sequence analysis
            if len(episode.actions) > 1:
                action_types = [a.get('type', 'unknown') for a in episode.actions]
                unique_actions = len(set(action_types))
                insights.append(f"Used {unique_actions} different action types in sequence")
            
            # Reward trajectory analysis
            if episode.rewards:
                total_reward = sum(episode.rewards)
                avg_reward = total_reward / len(episode.rewards)
                insights.append(f"Average reward per step: {avg_reward:.3f}")
                
                # Reward trend
                if len(episode.rewards) > 2:
                    if episode.rewards[-1] > episode.rewards[0]:
                        insights.append("Improving performance throughout episode")
                    else:
                        insights.append("Declining performance throughout episode")
            
            # Success factor analysis
            if episode.success:
                insights.append(f"Success achieved with {len(episode.actions)} actions")
                if episode.outcome:
                    insights.append(f"Success outcome: {episode.outcome}")
            else:
                insights.append(f"Failed after {len(episode.actions)} actions")
                if episode.outcome:
                    insights.append(f"Failure reason: {episode.outcome}")
            
            # Context relevance
            if episode.context:
                context_keys = list(episode.context.keys())
                insights.append(f"Context factors: {', '.join(context_keys[:3])}")
            
        except Exception as e:
            logger.error(f"Error analyzing episode insights: {e}")
            insights.append(f"Analysis error: {str(e)}")
        
        return insights
    
    def _update_replay_stats(self, episode_id: str, insights: List[str]):
        """Update replay statistics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Calculate effectiveness based on insights
                effectiveness = min(len(insights) / 10.0, 1.0)  # Scale to 0-1
                
                conn.execute("""
                    UPDATE experience_replay SET
                        replay_count = replay_count + 1,
                        last_replayed = ?,
                        replay_effectiveness = ?,
                        learning_insights = ?
                    WHERE episode_id = ?
                """, (
                    datetime.now().isoformat(),
                    effectiveness,
                    json.dumps(insights),
                    episode_id
                ))
                
        except Exception as e:
            logger.error(f"Error updating replay stats: {e}")
    
    def _discover_action_patterns(self, episodes: List[Tuple]) -> List[Dict[str, Any]]:
        """Discover common action patterns"""
        patterns = []
        action_sequences = {}
        
        for episode in episodes:
            if episode[7]:  # actions column
                actions = json.loads(episode[7])
                action_types = [a.get('type', 'unknown') for a in actions]
                
                # Look for sequences of length 2-4
                for seq_len in range(2, min(5, len(action_types) + 1)):
                    for i in range(len(action_types) - seq_len + 1):
                        sequence = tuple(action_types[i:i + seq_len])
                        if sequence not in action_sequences:
                            action_sequences[sequence] = {'count': 0, 'success_count': 0}
                        action_sequences[sequence]['count'] += 1
                        if episode[11]:  # success column
                            action_sequences[sequence]['success_count'] += 1
        
        # Convert to patterns
        for sequence, stats in action_sequences.items():
            if stats['count'] >= 3:  # Minimum frequency
                success_rate = stats['success_count'] / stats['count']
                patterns.append({
                    'pattern': ' -> '.join(sequence),
                    'frequency': stats['count'],
                    'success_rate': success_rate
                })
        
        return sorted(patterns, key=lambda x: x['frequency'], reverse=True)
    
    def _discover_success_patterns(self, episodes: List[Tuple]) -> List[Dict[str, Any]]:
        """Discover patterns that lead to success"""
        patterns = []
        success_factors = {}
        
        for episode in episodes:
            # Analyze context factors for all episodes
            if episode[6]:  # context column
                context = json.loads(episode[6])
                for key, value in context.items():
                    factor_key = f"{key}={value}"
                    if factor_key not in success_factors:
                        success_factors[factor_key] = {'success': 0, 'total': 0}
                    success_factors[factor_key]['total'] += 1
                    if episode[11]:  # success column
                        success_factors[factor_key]['success'] += 1
        
        # Convert to patterns
        for factor, stats in success_factors.items():
            if stats['total'] >= 3:  # Minimum frequency
                success_rate = stats['success'] / stats['total'] if stats['total'] > 0 else 0
                if success_rate > 0.7:  # High success rate threshold
                    patterns.append({
                        'pattern': f"Context factor: {factor}",
                        'frequency': stats['total'],
                        'success_rate': success_rate
                    })
        
        return sorted(patterns, key=lambda x: x['success_rate'], reverse=True)
    
    def _discover_temporal_patterns(self, episodes: List[Tuple]) -> List[Dict[str, Any]]:
        """Discover temporal patterns in episodes"""
        patterns = []
        
        # Group episodes by hour of day
        hour_stats = {}
        for episode in episodes:
            start_time = datetime.fromisoformat(episode[3])
            hour = start_time.hour
            
            if hour not in hour_stats:
                hour_stats[hour] = {'total': 0, 'success': 0}
            
            hour_stats[hour]['total'] += 1
            if episode[11]:  # success column
                hour_stats[hour]['success'] += 1
        
        # Find patterns
        for hour, stats in hour_stats.items():
            if stats['total'] >= 2:  # Minimum episodes
                success_rate = stats['success'] / stats['total']
                patterns.append({
                    'pattern': f"Episodes at hour {hour}",
                    'frequency': stats['total'],
                    'success_rate': success_rate
                })
        
        return sorted(patterns, key=lambda x: x['frequency'], reverse=True)
    
    def _discover_agent_patterns(self, episodes: List[Tuple]) -> List[Dict[str, Any]]:
        """Discover agent-specific patterns"""
        patterns = []
        agent_stats = {}
        
        for episode in episodes:
            agent_id = episode[1]  # agent_id column
            episode_type = episode[5]  # episode_type column
            
            key = f"{agent_id}:{episode_type}"
            if key not in agent_stats:
                agent_stats[key] = {'total': 0, 'success': 0}
            
            agent_stats[key]['total'] += 1
            if episode[11]:  # success column
                agent_stats[key]['success'] += 1
        
        # Convert to patterns
        for key, stats in agent_stats.items():
            if stats['total'] >= 3:  # Minimum episodes
                success_rate = stats['success'] / stats['total']
                patterns.append({
                    'pattern': f"Agent pattern: {key}",
                    'frequency': stats['total'],
                    'success_rate': success_rate
                })
        
        return sorted(patterns, key=lambda x: x['success_rate'], reverse=True)
    
    def _store_pattern(self, pattern_type: str, pattern: Dict[str, Any]):
        """Store discovered pattern"""
        try:
            pattern_id = str(uuid.uuid4())
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO episode_patterns (
                        id, pattern_type, pattern_description, 
                        frequency, success_rate
                    ) VALUES (?, ?, ?, ?, ?)
                """, (
                    pattern_id, pattern_type, pattern['pattern'],
                    pattern['frequency'], pattern['success_rate']
                ))
                
        except Exception as e:
            logger.error(f"Error storing pattern: {e}")
    
    def get_episodic_statistics(self) -> Dict[str, Any]:
        """Get comprehensive episodic memory statistics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                stats = {}
                
                # Basic episode counts
                cursor = conn.execute("SELECT COUNT(*) FROM episodes")
                stats['total_episodes'] = cursor.fetchone()[0]
                
                cursor = conn.execute("SELECT COUNT(*) FROM episodes WHERE success = 1")
                stats['successful_episodes'] = cursor.fetchone()[0]
                
                # Episode type distribution
                cursor = conn.execute("""
                    SELECT episode_type, COUNT(*), 
                           SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successes
                    FROM episodes 
                    GROUP BY episode_type
                """)
                
                episode_types = {}
                for row in cursor.fetchall():
                    episode_types[row[0]] = {
                        'total': row[1],
                        'successes': row[2],
                        'success_rate': row[2] / row[1] if row[1] > 0 else 0
                    }
                stats['episode_types'] = episode_types
                
                # Agent performance
                cursor = conn.execute("""
                    SELECT agent_id, COUNT(*),
                           SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successes
                    FROM episodes
                    GROUP BY agent_id
                """)
                
                agent_performance = {}
                for row in cursor.fetchall():
                    agent_performance[row[0]] = {
                        'total_episodes': row[1],
                        'successes': row[2],
                        'success_rate': row[2] / row[1] if row[1] > 0 else 0
                    }
                stats['agent_performance'] = agent_performance
                
                # Replay statistics
                cursor = conn.execute("SELECT COUNT(*) FROM experience_replay")
                stats['total_replays'] = cursor.fetchone()[0]
                
                cursor = conn.execute("SELECT AVG(replay_effectiveness) FROM experience_replay")
                avg_effectiveness = cursor.fetchone()[0]
                stats['average_replay_effectiveness'] = avg_effectiveness or 0.0
                
                # Pattern discovery
                cursor = conn.execute("SELECT COUNT(*) FROM episode_patterns")
                stats['discovered_patterns'] = cursor.fetchone()[0]
                
                return stats
                
        except Exception as e:
            logger.error(f"Error getting episodic statistics: {e}")
            return {'error': str(e)}

# Export the main classes
__all__ = ['EpisodicMemorySystem', 'Episode', 'ExperienceReplay']
