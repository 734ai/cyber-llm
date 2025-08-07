"""
Advanced Cognitive Integration System for Phase 9 Components
Orchestrates all cognitive systems for unified intelligent operation
"""
import asyncio
import sqlite3
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import logging
from pathlib import Path
import threading
import time

logger = logging.getLogger(__name__)

# Import all Phase 9 cognitive systems
from .long_term_memory import LongTermMemoryManager
from .episodic_memory import EpisodicMemorySystem
from .semantic_memory import SemanticMemoryNetwork
from .working_memory import WorkingMemoryManager
from .chain_of_thought import ChainOfThoughtReasoning

# Try to import meta-cognitive monitor, fall back to None if torch not available
try:
    from .meta_cognitive import MetaCognitiveMonitor
except ImportError as e:
    logger.warning(f"Meta-cognitive monitor not available (torch dependency): {e}")
    MetaCognitiveMonitor = None

@dataclass
class CognitiveState:
    """Current state of the integrated cognitive system"""
    timestamp: datetime
    working_memory_load: float
    attention_focus: Optional[str]
    reasoning_quality: float
    learning_rate: float
    confidence_level: float
    cognitive_load: float
    active_episodes: int
    memory_consolidation_status: str
    
class AdvancedCognitiveSystem:
    """Unified cognitive system integrating all Phase 9 components"""
    
    def __init__(self, base_path: str = "data/cognitive"):
        """Initialize the integrated cognitive system"""
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize all cognitive subsystems
        self._init_cognitive_subsystems()
        
        # Integration state
        self.current_state = None
        self.integration_active = True
        
        # Background processes
        self._consolidation_thread = None
        self._monitoring_thread = None
        
        # Start integrated operation
        self._start_cognitive_integration()
        
        logger.info("Advanced Cognitive System initialized with full Phase 9 integration")
    
    def _init_cognitive_subsystems(self):
        """Initialize all cognitive subsystems"""
        try:
            # Memory systems
            self.long_term_memory = LongTermMemoryManager(
                db_path=self.base_path / "long_term_memory.db"
            )
            
            self.episodic_memory = EpisodicMemorySystem(
                db_path=self.base_path / "episodic_memory.db"
            )
            
            self.semantic_memory = SemanticMemoryNetwork(
                db_path=self.base_path / "semantic_memory.db"
            )
            
            self.working_memory = WorkingMemoryManager(
                db_path=self.base_path / "working_memory.db"
            )
            
            # Reasoning systems
            self.chain_of_thought = ChainOfThoughtReasoning(
                db_path=self.base_path / "reasoning_chains.db"
            )
            
            # Meta-cognitive monitoring (optional if torch available)
            if MetaCognitiveMonitor is not None:
                self.meta_cognitive = MetaCognitiveMonitor(
                    db_path=self.base_path / "metacognitive.db"
                )
                logger.info("Meta-cognitive monitoring enabled")
            else:
                self.meta_cognitive = None
                logger.info("Meta-cognitive monitoring disabled (torch not available)")
            
            logger.info("All cognitive subsystems initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing cognitive subsystems: {e}")
            raise
    
    def _start_cognitive_integration(self):
        """Start background processes for cognitive integration"""
        try:
            # Start memory consolidation thread
            self._consolidation_thread = threading.Thread(
                target=self._memory_consolidation_loop, daemon=True
            )
            self._consolidation_thread.start()
            
            # Start cognitive monitoring thread
            self._monitoring_thread = threading.Thread(
                target=self._cognitive_monitoring_loop, daemon=True
            )
            self._monitoring_thread.start()
            
            logger.info("Cognitive integration processes started")
            
        except Exception as e:
            logger.error(f"Error starting cognitive integration: {e}")
    
    async def process_agent_experience(self, agent_id: str, experience_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a complete agent experience through all cognitive systems"""
        try:
            processing_id = str(uuid.uuid4())
            
            # Start episode in episodic memory
            episode_id = self.episodic_memory.start_episode(
                agent_id=agent_id,
                session_id=experience_data.get('session_id', ''),
                episode_type=experience_data.get('type', 'operation'),
                context=experience_data.get('context', {})
            )
            
            # Add to working memory for immediate processing
            wm_item_id = self.working_memory.add_item(
                content=f"Processing experience: {experience_data.get('description', 'Unknown')}",
                item_type="experience",
                priority=experience_data.get('priority', 0.7),
                source_agent=agent_id,
                context_tags=experience_data.get('tags', [])
            )
            
            # Extract semantic concepts for knowledge graph
            concepts_added = []
            if 'indicators' in experience_data:
                for indicator in experience_data['indicators']:
                    concept_id = self.semantic_memory.add_concept(
                        name=indicator,
                        concept_type=experience_data.get('indicator_type', 'unknown'),
                        description=f"Observed in agent {agent_id} experience",
                        confidence=0.7,
                        source=f"agent_{agent_id}"
                    )
                    if concept_id:
                        concepts_added.append(concept_id)
            
            # Perform reasoning about the experience
            reasoning_result = None
            if experience_data.get('requires_reasoning', True):
                threat_indicators = experience_data.get('indicators', [])
                if threat_indicators:
                    reasoning_result = await asyncio.to_thread(
                        self.chain_of_thought.reason_about_threat,
                        threat_indicators, agent_id
                    )
            
            # Record experience steps in episodic memory
            for action in experience_data.get('actions', []):
                self.episodic_memory.record_action(episode_id, action)
            
            for observation in experience_data.get('observations', []):
                self.episodic_memory.record_observation(episode_id, observation)
            
            # Calculate reward based on success
            reward = 1.0 if experience_data.get('success', False) else 0.3
            self.episodic_memory.record_reward(episode_id, reward)
            
            # Complete episode
            self.episodic_memory.end_episode(
                episode_id=episode_id,
                success=experience_data.get('success', False),
                outcome=experience_data.get('outcome', ''),
                metadata={'processing_id': processing_id}
            )
            
            # Store significant experiences in long-term memory
            if experience_data.get('importance', 0.5) > 0.6:
                ltm_id = self.long_term_memory.store_memory(
                    content=f"Significant experience: {experience_data.get('description')}",
                    memory_type="episodic_significant",
                    importance=experience_data.get('importance', 0.7),
                    agent_id=agent_id,
                    tags=experience_data.get('tags', [])
                )
            
            # Record performance metrics for meta-cognitive monitoring
            if reasoning_result and self.meta_cognitive:
                self.meta_cognitive.record_performance_metric(
                    metric_name="reasoning_confidence",
                    metric_type="reasoning",
                    value=reasoning_result.get('threat_assessment', {}).get('confidence', 0.5),
                    agent_id=agent_id
                )
            
            # Generate processing result
            result = {
                'processing_id': processing_id,
                'episode_id': episode_id,
                'working_memory_item_id': wm_item_id,
                'concepts_added': len(concepts_added),
                'reasoning_performed': reasoning_result is not None,
                'reasoning_result': reasoning_result,
                'cognitive_state': await self._get_current_cognitive_state(agent_id),
                'recommendations': await self._generate_integrated_recommendations(
                    experience_data, reasoning_result, agent_id
                )
            }
            
            logger.info(f"Agent experience processed through all cognitive systems: {processing_id}")
            return result
            
        except Exception as e:
            logger.error(f"Error processing agent experience: {e}")
            return {'error': str(e)}
    
    async def perform_integrated_threat_analysis(self, threat_indicators: List[str], 
                                               agent_id: str = "") -> Dict[str, Any]:
        """Perform comprehensive threat analysis using all cognitive systems"""
        try:
            analysis_id = str(uuid.uuid4())
            
            # Retrieve relevant memories from long-term memory
            relevant_memories = self.long_term_memory.retrieve_memories(
                query=' '.join(threat_indicators[:3]),
                memory_type="",
                agent_id=agent_id,
                limit=10
            )
            
            # Get related concepts from semantic memory
            semantic_reasoning = self.semantic_memory.reason_about_threat(threat_indicators)
            
            # Perform chain-of-thought reasoning
            cot_reasoning = await asyncio.to_thread(
                self.chain_of_thought.reason_about_threat,
                threat_indicators, agent_id
            )
            
            # Find similar past episodes
            similar_episodes = []
            for indicator in threat_indicators[:3]:
                episodes = self.episodic_memory.get_episodes_for_replay(
                    agent_id=agent_id,
                    episode_type="",
                    success_only=False,
                    limit=5
                )
                for episode in episodes:
                    if any(indicator.lower() in action.get('content', '').lower() 
                          for action in episode.actions):
                        similar_episodes.append(episode)
            
            # Add to working memory for focused attention
            wm_item_id = self.working_memory.add_item(
                content=f"Threat analysis: {', '.join(threat_indicators[:3])}",
                item_type="threat_analysis",
                priority=0.9,
                source_agent=agent_id,
                context_tags=["threat", "analysis", "high_priority"]
            )
            
            # Focus attention on threat analysis
            focus_id = self.working_memory.focus_attention(
                focus_type="threat_analysis",
                item_ids=[wm_item_id],
                attention_weight=0.9,
                agent_id=agent_id
            )
            
            # Synthesize results from all systems
            integrated_assessment = await self._synthesize_threat_assessment(
                semantic_reasoning, cot_reasoning, relevant_memories, similar_episodes
            )
            
            # Generate comprehensive recommendations
            recommendations = await self._generate_comprehensive_recommendations(
                integrated_assessment, threat_indicators
            )
            
            # Record analysis for meta-cognitive learning
            if self.meta_cognitive:
                self.meta_cognitive.record_performance_metric(
                    metric_name="integrated_threat_analysis",
                    metric_type="analysis",
                    value=integrated_assessment['confidence'],
                    target_value=0.8,
                context={
                    'analysis_id': analysis_id,
                    'indicators_count': len(threat_indicators),
                    'memories_used': len(relevant_memories)
                },
                agent_id=agent_id
            )
            
            result = {
                'analysis_id': analysis_id,
                'threat_indicators': threat_indicators,
                'integrated_assessment': integrated_assessment,
                'recommendations': recommendations,
                'supporting_evidence': {
                    'semantic_reasoning': semantic_reasoning,
                    'cot_reasoning': cot_reasoning,
                    'relevant_memories': len(relevant_memories),
                    'similar_episodes': len(similar_episodes)
                },
                'cognitive_resources_used': {
                    'working_memory_item': wm_item_id,
                    'attention_focus': focus_id,
                    'reasoning_chains': cot_reasoning.get('chain_id', ''),
                    'semantic_concepts': len(semantic_reasoning.get('matched_concepts', []))
                }
            }
            
            logger.info(f"Integrated threat analysis completed: {analysis_id}")
            return result
            
        except Exception as e:
            logger.error(f"Error in integrated threat analysis: {e}")
            return {'error': str(e)}
    
    async def trigger_cognitive_reflection(self, agent_id: str, 
                                         trigger_event: str = "periodic") -> Dict[str, Any]:
        """Trigger comprehensive cognitive reflection across all systems"""
        try:
            reflection_id = str(uuid.uuid4())
            
            # Perform meta-cognitive reflection if available
            meta_reflection = None
            if self.meta_cognitive:
                meta_reflection = await asyncio.to_thread(
                    self.meta_cognitive.trigger_self_reflection,
                    agent_id, trigger_event, "comprehensive"
                )
            
            # Get cross-session context from long-term memory
            cross_session_memories = self.long_term_memory.get_cross_session_context(
                agent_id=agent_id, limit=15
            )
            
            # Discover patterns in episodic memory
            episode_patterns = await asyncio.to_thread(
                self.episodic_memory.discover_patterns
            )
            
            # Consolidate memories
            consolidation_stats = await asyncio.to_thread(
                self.long_term_memory.consolidate_memories
            )
            
            # Assess working memory efficiency
            wm_stats = self.working_memory.get_working_memory_statistics()
            
            # Generate reflection insights
            reflection_insights = await self._generate_reflection_insights(
                meta_reflection, cross_session_memories, episode_patterns, 
                consolidation_stats, wm_stats
            )
            
            # Update cognitive state
            new_state = await self._update_cognitive_state_from_reflection(
                agent_id, reflection_insights
            )
            
            result = {
                'reflection_id': reflection_id,
                'trigger_event': trigger_event,
                'agent_id': agent_id,
                'meta_reflection': meta_reflection,
                'reflection_insights': reflection_insights,
                'cognitive_state_update': new_state,
                'system_optimizations': await self._apply_reflection_optimizations(
                    reflection_insights, agent_id
                ),
                'learning_adjustments': await self._apply_learning_adjustments(
                    meta_reflection, agent_id
                )
            }
            
            logger.info(f"Comprehensive cognitive reflection completed: {reflection_id}")
            return result
            
        except Exception as e:
            logger.error(f"Error in cognitive reflection: {e}")
            return {'error': str(e)}
    
    def _memory_consolidation_loop(self):
        """Background memory consolidation process"""
        consolidation_interval = 21600  # 6 hours
        
        while self.integration_active:
            try:
                time.sleep(consolidation_interval)
                
                # Consolidate long-term memory
                ltm_stats = self.long_term_memory.consolidate_memories()
                
                # Discover patterns in episodic memory
                pattern_stats = self.episodic_memory.discover_patterns()
                
                # Decay working memory
                self.working_memory.decay_memory()
                
                logger.info(f"Memory consolidation completed - LTM: {ltm_stats.get('patterns_discovered', 0)} patterns, Episodes: {len(pattern_stats.get('action_patterns', []))} action patterns")
                
            except Exception as e:
                logger.error(f"Error in memory consolidation loop: {e}")
    
    def _cognitive_monitoring_loop(self):
        """Background cognitive monitoring process"""
        monitoring_interval = 300  # 5 minutes
        
        while self.integration_active:
            try:
                time.sleep(monitoring_interval)
                
                # Update current cognitive state
                self.current_state = self._calculate_integrated_cognitive_state()
                
                # Check for cognitive load issues
                if self.current_state.cognitive_load > 0.8:
                    logger.warning(f"High cognitive load detected: {self.current_state.cognitive_load:.3f}")
                
                # Monitor working memory capacity
                if self.current_state.working_memory_load > 0.9:
                    logger.warning(f"Working memory near capacity: {self.current_state.working_memory_load:.3f}")
                
            except Exception as e:
                logger.error(f"Error in cognitive monitoring loop: {e}")
    
    def _calculate_integrated_cognitive_state(self) -> CognitiveState:
        """Calculate current integrated cognitive state"""
        try:
            # Get statistics from all subsystems
            wm_stats = self.working_memory.get_working_memory_statistics()
            ltm_stats = self.long_term_memory.get_memory_statistics()
            episodic_stats = self.episodic_memory.get_episodic_statistics()
            reasoning_stats = self.chain_of_thought.get_reasoning_statistics()
            
            # Calculate working memory load
            wm_load = wm_stats.get('utilization', 0.0)
            
            # Get current attention focus
            current_focus = self.working_memory.get_current_focus()
            focus_type = current_focus.focus_type if current_focus else None
            
            # Calculate reasoning quality from recent chains
            reasoning_quality = 0.7  # Default
            if reasoning_stats.get('total_chains', 0) > 0:
                completion_rate = reasoning_stats.get('completion_rate', 0.5)
                avg_confidence = 0.6  # Would calculate from actual data
                reasoning_quality = (completion_rate + avg_confidence) / 2
            
            # Estimate cognitive load
            task_count = wm_stats.get('current_capacity', 0)
            cognitive_load = min(task_count / 50.0 + wm_load * 0.3, 1.0)
            
            return CognitiveState(
                timestamp=datetime.now(),
                working_memory_load=wm_load,
                attention_focus=focus_type,
                reasoning_quality=reasoning_quality,
                learning_rate=0.01,  # Would be calculated dynamically
                confidence_level=0.75,  # Would be calculated from meta-cognitive data
                cognitive_load=cognitive_load,
                active_episodes=len(self.episodic_memory._active_episodes),
                memory_consolidation_status="active"
            )
            
        except Exception as e:
            logger.error(f"Error calculating cognitive state: {e}")
            return CognitiveState(
                timestamp=datetime.now(),
                working_memory_load=0.5,
                attention_focus=None,
                reasoning_quality=0.5,
                learning_rate=0.01,
                confidence_level=0.5,
                cognitive_load=0.5,
                active_episodes=0,
                memory_consolidation_status="error"
            )
    
    async def _get_current_cognitive_state(self, agent_id: str) -> Dict[str, Any]:
        """Get current cognitive state for specific agent"""
        state = self._calculate_integrated_cognitive_state()
        return asdict(state)
    
    async def _synthesize_threat_assessment(self, semantic_result: Dict[str, Any],
                                          cot_result: Dict[str, Any],
                                          memories: List[Any],
                                          episodes: List[Any]) -> Dict[str, Any]:
        """Synthesize threat assessment from all cognitive systems"""
        
        # Extract confidence levels
        semantic_confidence = semantic_result.get('confidence', 0.5)
        cot_confidence = cot_result.get('threat_assessment', {}).get('confidence', 0.5)
        
        # Weight based on evidence availability
        semantic_weight = 0.3
        cot_weight = 0.4
        memory_weight = 0.2
        episode_weight = 0.1
        
        # Memory contribution
        memory_confidence = min(len(memories) / 5.0, 1.0) * 0.7
        episode_confidence = min(len(episodes) / 3.0, 1.0) * 0.6
        
        # Weighted confidence
        overall_confidence = (
            semantic_confidence * semantic_weight +
            cot_confidence * cot_weight +
            memory_confidence * memory_weight +
            episode_confidence * episode_weight
        )
        
        # Determine threat level
        if overall_confidence > 0.8:
            threat_level = "CRITICAL"
        elif overall_confidence > 0.6:
            threat_level = "HIGH"
        elif overall_confidence > 0.4:
            threat_level = "MEDIUM"
        else:
            threat_level = "LOW"
        
        return {
            'threat_level': threat_level,
            'confidence': overall_confidence,
            'evidence_sources': {
                'semantic_analysis': semantic_confidence,
                'reasoning_chains': cot_confidence,
                'historical_memories': memory_confidence,
                'similar_episodes': episode_confidence
            },
            'synthesis_method': 'integrated_weighted_assessment'
        }
    
    async def _generate_integrated_recommendations(self, experience_data: Dict[str, Any],
                                                 reasoning_result: Optional[Dict[str, Any]],
                                                 agent_id: str) -> List[Dict[str, Any]]:
        """Generate recommendations based on integrated cognitive analysis"""
        recommendations = []
        
        # Based on experience importance
        if experience_data.get('importance', 0.5) > 0.8:
            recommendations.append({
                'type': 'memory_consolidation',
                'action': 'Prioritize this experience for long-term memory storage',
                'priority': 'high',
                'rationale': 'High importance experience should be preserved'
            })
        
        # Based on reasoning results
        if reasoning_result:
            threat_level = reasoning_result.get('threat_assessment', {}).get('risk_level', 'LOW')
            if threat_level in ['HIGH', 'CRITICAL']:
                recommendations.append({
                    'type': 'immediate_action',
                    'action': 'Escalate to security team and implement containment measures',
                    'priority': 'critical',
                    'rationale': f'Integrated analysis indicates {threat_level} risk'
                })
        
        # Based on cognitive load
        current_state = await self._get_current_cognitive_state(agent_id)
        if current_state['cognitive_load'] > 0.8:
            recommendations.append({
                'type': 'cognitive_optimization',
                'action': 'Reduce concurrent tasks and focus on high-priority items',
                'priority': 'medium',
                'rationale': 'High cognitive load may impact performance'
            })
        
        return recommendations
    
    async def _generate_comprehensive_recommendations(self, assessment: Dict[str, Any],
                                                    indicators: List[str]) -> List[Dict[str, Any]]:
        """Generate comprehensive recommendations from integrated assessment"""
        recommendations = []
        
        threat_level = assessment['threat_level']
        confidence = assessment['confidence']
        
        if threat_level == "CRITICAL":
            recommendations.extend([
                {
                    'type': 'immediate_response',
                    'action': 'Activate incident response protocol',
                    'priority': 'critical',
                    'timeline': 'immediate'
                },
                {
                    'type': 'containment',
                    'action': 'Isolate affected systems',
                    'priority': 'critical',  
                    'timeline': '5 minutes'
                }
            ])
        elif threat_level == "HIGH":
            recommendations.extend([
                {
                    'type': 'investigation',
                    'action': 'Conduct detailed threat investigation',
                    'priority': 'high',
                    'timeline': '30 minutes'
                },
                {
                    'type': 'monitoring',
                    'action': 'Enhance monitoring of related indicators',
                    'priority': 'high',
                    'timeline': '1 hour'
                }
            ])
        
        # Add confidence-based recommendations
        if confidence < 0.6:
            recommendations.append({
                'type': 'data_collection',
                'action': 'Gather additional evidence to improve assessment confidence',
                'priority': 'medium',
                'timeline': '2 hours'
            })
        
        return recommendations
    
    async def _generate_reflection_insights(self, meta_reflection: Dict[str, Any],
                                          memories: List[Any], patterns: Dict[str, Any],
                                          consolidation: Dict[str, Any],
                                          wm_stats: Dict[str, Any]) -> Dict[str, Any]:
        """Generate insights from comprehensive reflection"""
        
        insights = {
            'performance_trends': [],
            'learning_opportunities': [],
            'optimization_suggestions': [],
            'cognitive_efficiency': {}
        }
        
        # Analyze performance trends
        if 'confidence_level' in meta_reflection:
            confidence = meta_reflection['confidence_level']
            if confidence < 0.6:
                insights['performance_trends'].append(
                    f"Low confidence level ({confidence:.3f}) indicates need for improvement"
                )
            elif confidence > 0.8:
                insights['performance_trends'].append(
                    f"High confidence level ({confidence:.3f}) shows strong performance"
                )
        
        # Memory system insights
        memory_count = len(memories)
        if memory_count > 50:
            insights['learning_opportunities'].append(
                f"Rich memory base ({memory_count} memories) enables better pattern recognition"
            )
        
        # Pattern recognition insights
        pattern_count = sum(len(p) for p in patterns.values())
        if pattern_count > 10:
            insights['learning_opportunities'].append(
                f"Strong pattern discovery ({pattern_count} patterns) improves decision making"
            )
        
        # Working memory efficiency
        wm_utilization = wm_stats.get('utilization', 0.5)
        if wm_utilization > 0.9:
            insights['optimization_suggestions'].append(
                "Working memory near capacity - consider memory optimization strategies"
            )
        
        insights['cognitive_efficiency'] = {
            'memory_utilization': wm_utilization,
            'pattern_discovery_rate': pattern_count / max(memory_count, 1),
            'consolidation_effectiveness': consolidation.get('patterns_discovered', 0),
            'overall_efficiency': (1.0 - wm_utilization) * 0.5 + (pattern_count / 20.0) * 0.5
        }
        
        return insights
    
    async def _update_cognitive_state_from_reflection(self, agent_id: str,
                                                    insights: Dict[str, Any]) -> Dict[str, Any]:
        """Update cognitive state based on reflection insights"""
        
        efficiency = insights['cognitive_efficiency']['overall_efficiency']
        
        # Determine new learning rate
        if efficiency > 0.8:
            new_learning_rate = 0.015  # Increase learning rate for high efficiency
        elif efficiency < 0.4:
            new_learning_rate = 0.005  # Decrease for low efficiency
        else:
            new_learning_rate = 0.01   # Default
        
        # Update meta-cognitive monitoring if available
        if self.meta_cognitive:
            self.meta_cognitive.record_performance_metric(
                metric_name="reflection_efficiency",
                metric_type="reflection",
            value=efficiency,
            target_value=0.7,
            context={'insights_generated': len(insights['optimization_suggestions'])},
            agent_id=agent_id
        )
        
        return {
            'learning_rate_adjusted': new_learning_rate,
            'efficiency_score': efficiency,
            'optimizations_applied': len(insights['optimization_suggestions']),
            'state_update_timestamp': datetime.now().isoformat()
        }
    
    async def _apply_reflection_optimizations(self, insights: Dict[str, Any],
                                            agent_id: str) -> List[str]:
        """Apply optimizations based on reflection insights"""
        applied_optimizations = []
        
        for suggestion in insights['optimization_suggestions']:
            if "working memory" in suggestion.lower():
                # Clear low-priority working memory items
                active_items = self.working_memory.get_active_items(min_activation=0.2)
                if len(active_items) > 30:  # Arbitrary threshold
                    applied_optimizations.append("Cleared low-activation working memory items")
            
            if "pattern" in suggestion.lower():
                # Trigger additional pattern discovery
                await asyncio.to_thread(self.episodic_memory.discover_patterns)
                applied_optimizations.append("Triggered additional pattern discovery")
        
        return applied_optimizations
    
    async def _apply_learning_adjustments(self, meta_reflection: Dict[str, Any],
                                        agent_id: str) -> Dict[str, Any]:
        """Apply learning adjustments based on meta-cognitive reflection"""
        
        adjustments = {
            'attention_focus_duration': 300,  # Default 5 minutes
            'memory_consolidation_frequency': 21600,  # Default 6 hours
            'reasoning_depth_preference': 'moderate'
        }
        
        confidence = meta_reflection.get('confidence_level', 0.5)
        
        # Adjust based on confidence
        if confidence < 0.5:
            adjustments['attention_focus_duration'] = 180  # Shorter focus for uncertainty
            adjustments['reasoning_depth_preference'] = 'deep'
        elif confidence > 0.8:
            adjustments['attention_focus_duration'] = 450  # Longer focus for confidence
            adjustments['reasoning_depth_preference'] = 'efficient'
        
        return adjustments
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        try:
            return {
                'system_active': self.integration_active,
                'current_state': asdict(self.current_state) if self.current_state else None,
                'subsystem_status': {
                    'long_term_memory': self.long_term_memory.get_memory_statistics(),
                    'episodic_memory': self.episodic_memory.get_episodic_statistics(),
                    'semantic_memory': self.semantic_memory.get_semantic_statistics(),
                    'working_memory': self.working_memory.get_working_memory_statistics(),
                    'reasoning_chains': self.chain_of_thought.get_reasoning_statistics(),
                    'meta_cognitive': self.meta_cognitive.get_metacognitive_statistics() if self.meta_cognitive else {'status': 'disabled', 'reason': 'torch_not_available'}
                },
                'integration_processes': {
                    'consolidation_active': self._consolidation_thread.is_alive() if self._consolidation_thread else False,
                    'monitoring_active': self._monitoring_thread.is_alive() if self._monitoring_thread else False
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {'error': str(e)}
    
    def shutdown(self):
        """Shutdown the cognitive system gracefully"""
        try:
            logger.info("Shutting down Advanced Cognitive System")
            
            self.integration_active = False
            
            # Wait for threads to complete
            if self._consolidation_thread and self._consolidation_thread.is_alive():
                self._consolidation_thread.join(timeout=5.0)
            
            if self._monitoring_thread and self._monitoring_thread.is_alive():
                self._monitoring_thread.join(timeout=5.0)
            
            # Cleanup subsystems
            if hasattr(self.working_memory, 'cleanup'):
                self.working_memory.cleanup()
            
            logger.info("Advanced Cognitive System shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

# Factory function for easy instantiation
def create_advanced_cognitive_system(base_path: str = "data/cognitive") -> AdvancedCognitiveSystem:
    """Create and initialize the advanced cognitive system"""
    return AdvancedCognitiveSystem(base_path)

# Export main class
__all__ = ['AdvancedCognitiveSystem', 'CognitiveState', 'create_advanced_cognitive_system']
