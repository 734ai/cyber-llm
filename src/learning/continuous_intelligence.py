"""
Phase 7 Integration Module for Cyber-LLM
Integrates all continuous intelligence and evolution components into a cohesive system.

Author: Muzan Sano <sanosensei36@gmail.com>
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import yaml
from pathlib import Path

# Import Phase 7 components
from .online_learning import OnlineLearningManager, LearningEvent, LearningEventType
from .federated_learning import FederatedLearningCoordinator, FederatedLearningParticipant
from .meta_learning import MetaLearningManager, MetaLearningStrategy
from .research_collaboration import ResearchCollaborationManager, CollaborationType
from .constitutional_ai import ConstitutionalAIManager

from ..utils.logging_system import CyberLLMLogger

# Configure logging
logger = CyberLLMLogger(__name__).get_logger()

class ContinuousIntelligenceMode(Enum):
    """Modes of continuous intelligence operation"""
    CONSERVATIVE = "conservative"     # Minimal learning, high safety
    BALANCED = "balanced"            # Balanced learning and safety
    AGGRESSIVE = "aggressive"        # Maximum learning, calculated risks
    RESEARCH = "research"            # Research-focused with collaboration
    PRODUCTION = "production"        # Production-optimized stability

@dataclass
class ContinuousIntelligenceConfig:
    """Configuration for continuous intelligence system"""
    
    # General settings
    mode: ContinuousIntelligenceMode = ContinuousIntelligenceMode.BALANCED
    organization_name: str = "CyberLLM-Org"
    enable_online_learning: bool = True
    enable_federated_learning: bool = True
    enable_meta_learning: bool = True
    enable_research_collaboration: bool = True
    enable_constitutional_ai: bool = True
    
    # Learning parameters
    learning_rate_multiplier: float = 1.0
    adaptation_threshold: float = 0.8
    meta_learning_batch_size: int = 4
    collaboration_sensitivity_level: str = "consortium"
    
    # Safety parameters
    constitutional_strict_mode: bool = True
    human_oversight_threshold: float = 0.8
    max_autonomous_adaptations: int = 10
    
    # Performance parameters
    update_frequency_minutes: int = 60
    batch_processing_size: int = 100
    max_memory_usage_gb: float = 8.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

class ContinuousIntelligenceOrchestrator:
    """Main orchestrator for all continuous intelligence components"""
    
    def __init__(self, 
                 config: ContinuousIntelligenceConfig,
                 model,
                 tokenizer):
        
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        
        # Initialize components based on configuration
        self.components = {}
        self._initialize_components()
        
        # Operational state
        self.is_running = False
        self.last_update_time = None
        self.adaptation_count = 0
        self.performance_metrics = {
            'total_learning_events': 0,
            'successful_adaptations': 0,
            'constitutional_violations': 0,
            'collaboration_insights': 0,
            'meta_learning_episodes': 0
        }
        
        logger.info(f"ContinuousIntelligenceOrchestrator initialized in {config.mode.value} mode")
    
    def _initialize_components(self):
        """Initialize continuous intelligence components"""
        
        try:
            # Online Learning Manager
            if self.config.enable_online_learning:
                self.components['online_learning'] = OnlineLearningManager(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    learning_rate=0.001 * self.config.learning_rate_multiplier,
                    batch_size=self.config.batch_processing_size
                )
                logger.info("Initialized OnlineLearningManager")
            
            # Federated Learning Coordinator
            if self.config.enable_federated_learning:
                self.components['federated_learning'] = FederatedLearningCoordinator(
                    coordinator_id=f"{self.config.organization_name}_coordinator",
                    model=self.model
                )
                logger.info("Initialized FederatedLearningCoordinator")
            
            # Meta Learning Manager
            if self.config.enable_meta_learning:
                self.components['meta_learning'] = MetaLearningManager(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    strategy=MetaLearningStrategy.MAML,
                    meta_batch_size=self.config.meta_learning_batch_size
                )
                logger.info("Initialized MetaLearningManager")
            
            # Research Collaboration Manager
            if self.config.enable_research_collaboration:
                self.components['research_collaboration'] = ResearchCollaborationManager(
                    organization_name=self.config.organization_name
                )
                logger.info("Initialized ResearchCollaborationManager")
            
            # Constitutional AI Manager
            if self.config.enable_constitutional_ai:
                self.components['constitutional_ai'] = ConstitutionalAIManager()
                logger.info("Initialized ConstitutionalAIManager")
        
        except Exception as e:
            logger.error(f"Failed to initialize components: {str(e)}")
            raise
    
    async def start_continuous_intelligence(self):
        """Start the continuous intelligence system"""
        
        if self.is_running:
            logger.warning("Continuous intelligence system is already running")
            return
        
        self.is_running = True
        self.last_update_time = datetime.now()
        
        logger.info("Starting continuous intelligence system")
        
        # Start background tasks
        tasks = []
        
        if 'online_learning' in self.components:
            tasks.append(self._run_online_learning_loop())
        
        if 'meta_learning' in self.components:
            tasks.append(self._run_meta_learning_loop())
        
        if 'federated_learning' in self.components:
            tasks.append(self._run_federated_learning_loop())
        
        # Start monitoring task
        tasks.append(self._run_monitoring_loop())
        
        # Run all tasks concurrently
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"Error in continuous intelligence system: {str(e)}")
            self.is_running = False
    
    async def stop_continuous_intelligence(self):
        """Stop the continuous intelligence system"""
        
        self.is_running = False
        logger.info("Stopping continuous intelligence system")
        
        # Save current state
        await self._save_system_state()
    
    async def process_learning_event(self, event: LearningEvent) -> Dict[str, Any]:
        """Process a single learning event through all applicable components"""
        
        results = {}
        
        try:
            # Constitutional AI evaluation first
            if 'constitutional_ai' in self.components:
                constitutional_result = await self.components['constitutional_ai'].evaluate_and_enforce(
                    content=json.dumps(event.context),
                    context={'event_type': event.event_type.value}
                )
                
                if not constitutional_result['allowed']:
                    logger.warning(f"Learning event blocked by constitutional AI: {event.event_id}")
                    self.performance_metrics['constitutional_violations'] += 1
                    return {'blocked': True, 'reason': 'constitutional_violation'}
                
                results['constitutional_check'] = constitutional_result
            
            # Process through online learning
            if 'online_learning' in self.components:
                online_result = await self.components['online_learning'].process_learning_event(event)
                results['online_learning'] = online_result
            
            # Add to meta-learning episodes
            if 'meta_learning' in self.components:
                episodes_created = await self.components['meta_learning'].add_learning_episodes([event])
                results['meta_learning_episodes'] = episodes_created
                self.performance_metrics['meta_learning_episodes'] += episodes_created
            
            # Update performance metrics
            self.performance_metrics['total_learning_events'] += 1
            
            logger.debug(f"Processed learning event: {event.event_id}")
            
        except Exception as e:
            logger.error(f"Error processing learning event {event.event_id}: {str(e)}")
            results['error'] = str(e)
        
        return results
    
    async def trigger_adaptation_cycle(self) -> Dict[str, Any]:
        """Trigger a complete adaptation cycle across all components"""
        
        if self.adaptation_count >= self.config.max_autonomous_adaptations:
            logger.warning("Maximum autonomous adaptations reached, requiring human oversight")
            return {'blocked': True, 'reason': 'max_adaptations_reached'}
        
        adaptation_results = {}
        
        try:
            # Online learning adaptation
            if 'online_learning' in self.components:
                online_result = await self.components['online_learning'].apply_accumulated_updates()
                adaptation_results['online_learning'] = online_result
            
            # Meta-learning adaptation
            if 'meta_learning' in self.components:
                meta_result = await self.components['meta_learning'].meta_train_step()
                adaptation_results['meta_learning'] = meta_result
            
            # Update adaptation count
            if any(result.get('success') for result in adaptation_results.values()):
                self.adaptation_count += 1
                self.performance_metrics['successful_adaptations'] += 1
            
            logger.info(f"Completed adaptation cycle {self.adaptation_count}")
            
        except Exception as e:
            logger.error(f"Error in adaptation cycle: {str(e)}")
            adaptation_results['error'] = str(e)
        
        return adaptation_results
    
    async def _run_online_learning_loop(self):
        """Background loop for online learning"""
        
        logger.info("Starting online learning loop")
        
        while self.is_running:
            try:
                # Check if adaptation threshold is met
                if 'online_learning' in self.components:
                    stats = self.components['online_learning'].get_learning_statistics()
                    
                    if stats['pending_updates'] > 0 and stats['confidence_score'] >= self.config.adaptation_threshold:
                        await self.trigger_adaptation_cycle()
                
                await asyncio.sleep(self.config.update_frequency_minutes * 60)
                
            except Exception as e:
                logger.error(f"Error in online learning loop: {str(e)}")
                await asyncio.sleep(300)  # Wait 5 minutes on error
    
    async def _run_meta_learning_loop(self):
        """Background loop for meta-learning"""
        
        logger.info("Starting meta-learning loop")
        
        while self.is_running:
            try:
                if 'meta_learning' in self.components:
                    # Perform meta-training if enough episodes available
                    await self.components['meta_learning'].meta_train_step()
                
                await asyncio.sleep(self.config.update_frequency_minutes * 60 * 2)  # Less frequent
                
            except Exception as e:
                logger.error(f"Error in meta-learning loop: {str(e)}")
                await asyncio.sleep(600)  # Wait 10 minutes on error
    
    async def _run_federated_learning_loop(self):
        """Background loop for federated learning"""
        
        logger.info("Starting federated learning loop")
        
        while self.is_running:
            try:
                if 'federated_learning' in self.components:
                    # Check for federated learning opportunities
                    await self.components['federated_learning'].coordinate_federated_round()
                
                await asyncio.sleep(self.config.update_frequency_minutes * 60 * 4)  # Even less frequent
                
            except Exception as e:
                logger.error(f"Error in federated learning loop: {str(e)}")
                await asyncio.sleep(900)  # Wait 15 minutes on error
    
    async def _run_monitoring_loop(self):
        """Background loop for system monitoring"""
        
        logger.info("Starting monitoring loop")
        
        while self.is_running:
            try:
                # Log system status
                status = await self.get_system_status()
                
                # Check for any critical issues
                if status['memory_usage_gb'] > self.config.max_memory_usage_gb:
                    logger.warning(f"High memory usage: {status['memory_usage_gb']:.2f}GB")
                
                if status['constitutional_violations_rate'] > 0.1:
                    logger.warning(f"High constitutional violation rate: {status['constitutional_violations_rate']:.2f}")
                
                await asyncio.sleep(300)  # Monitor every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {str(e)}")
                await asyncio.sleep(300)
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        
        status = {
            'is_running': self.is_running,
            'mode': self.config.mode.value,
            'uptime_hours': (datetime.now() - (self.last_update_time or datetime.now())).total_seconds() / 3600,
            'adaptation_count': self.adaptation_count,
            'performance_metrics': self.performance_metrics.copy()
        }
        
        # Component-specific status
        component_status = {}
        
        if 'online_learning' in self.components:
            component_status['online_learning'] = self.components['online_learning'].get_learning_statistics()
        
        if 'meta_learning' in self.components:
            component_status['meta_learning'] = self.components['meta_learning'].get_meta_learning_statistics()
        
        if 'constitutional_ai' in self.components:
            component_status['constitutional_ai'] = self.components['constitutional_ai'].get_constitutional_statistics()
        
        if 'research_collaboration' in self.components:
            component_status['research_collaboration'] = self.components['research_collaboration'].get_collaboration_statistics()
        
        status['components'] = component_status
        
        # Calculate derived metrics
        total_events = self.performance_metrics['total_learning_events']
        if total_events > 0:
            status['constitutional_violations_rate'] = self.performance_metrics['constitutional_violations'] / total_events
            status['adaptation_success_rate'] = self.performance_metrics['successful_adaptations'] / self.adaptation_count if self.adaptation_count > 0 else 0.0
        else:
            status['constitutional_violations_rate'] = 0.0
            status['adaptation_success_rate'] = 0.0
        
        # Estimate memory usage (simplified)
        status['memory_usage_gb'] = 2.0  # Base estimate, would use actual monitoring in production
        
        return status
    
    async def _save_system_state(self):
        """Save current system state to disk"""
        
        try:
            state_data = {
                'config': self.config.to_dict(),
                'performance_metrics': self.performance_metrics,
                'adaptation_count': self.adaptation_count,
                'last_update_time': self.last_update_time.isoformat() if self.last_update_time else None,
                'timestamp': datetime.now().isoformat()
            }
            
            state_file = Path("data/system_state/continuous_intelligence_state.json")
            state_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(state_file, 'w') as f:
                json.dump(state_data, f, indent=2)
            
            logger.info(f"Saved system state to {state_file}")
            
        except Exception as e:
            logger.error(f"Failed to save system state: {str(e)}")
    
    async def load_system_state(self, state_file: Optional[str] = None) -> bool:
        """Load system state from disk"""
        
        try:
            if state_file is None:
                state_file = "data/system_state/continuous_intelligence_state.json"
            
            state_path = Path(state_file)
            
            if not state_path.exists():
                logger.info("No previous system state found")
                return False
            
            with open(state_path, 'r') as f:
                state_data = json.load(f)
            
            # Restore state
            self.performance_metrics = state_data.get('performance_metrics', {})
            self.adaptation_count = state_data.get('adaptation_count', 0)
            
            if state_data.get('last_update_time'):
                self.last_update_time = datetime.fromisoformat(state_data['last_update_time'])
            
            logger.info(f"Loaded system state from {state_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load system state: {str(e)}")
            return False
    
    def create_learning_event(self,
                            event_type: LearningEventType,
                            source: str,
                            context: Dict[str, Any],
                            confidence: float = 1.0,
                            priority: int = 1) -> LearningEvent:
        """Helper method to create learning events"""
        
        return LearningEvent(
            event_id=f"evt_{datetime.now().timestamp()}",
            event_type=event_type,
            source=source,
            timestamp=datetime.now(),
            context=context,
            confidence=confidence,
            priority=priority
        )

# Factory functions
def create_continuous_intelligence_config(mode: ContinuousIntelligenceMode = ContinuousIntelligenceMode.BALANCED,
                                        **kwargs) -> ContinuousIntelligenceConfig:
    """Create continuous intelligence configuration"""
    
    # Mode-specific defaults
    mode_defaults = {
        ContinuousIntelligenceMode.CONSERVATIVE: {
            'learning_rate_multiplier': 0.5,
            'constitutional_strict_mode': True,
            'human_oversight_threshold': 0.6,
            'max_autonomous_adaptations': 5,
            'update_frequency_minutes': 120
        },
        ContinuousIntelligenceMode.BALANCED: {
            'learning_rate_multiplier': 1.0,
            'constitutional_strict_mode': True,
            'human_oversight_threshold': 0.8,
            'max_autonomous_adaptations': 10,
            'update_frequency_minutes': 60
        },
        ContinuousIntelligenceMode.AGGRESSIVE: {
            'learning_rate_multiplier': 2.0,
            'constitutional_strict_mode': False,
            'human_oversight_threshold': 0.9,
            'max_autonomous_adaptations': 20,
            'update_frequency_minutes': 30
        },
        ContinuousIntelligenceMode.RESEARCH: {
            'enable_research_collaboration': True,
            'learning_rate_multiplier': 1.5,
            'constitutional_strict_mode': False,
            'collaboration_sensitivity_level': 'consortium',
            'update_frequency_minutes': 45
        },
        ContinuousIntelligenceMode.PRODUCTION: {
            'learning_rate_multiplier': 0.8,
            'constitutional_strict_mode': True,
            'human_oversight_threshold': 0.7,
            'max_autonomous_adaptations': 8,
            'update_frequency_minutes': 90
        }
    }
    
    # Start with mode defaults
    config_dict = mode_defaults.get(mode, {})
    
    # Override with user-provided kwargs
    config_dict.update(kwargs)
    config_dict['mode'] = mode
    
    return ContinuousIntelligenceConfig(**config_dict)

def create_continuous_intelligence_orchestrator(model,
                                              tokenizer, 
                                              config: Optional[ContinuousIntelligenceConfig] = None,
                                              **kwargs) -> ContinuousIntelligenceOrchestrator:
    """Create continuous intelligence orchestrator"""
    
    if config is None:
        config = create_continuous_intelligence_config(**kwargs)
    
    return ContinuousIntelligenceOrchestrator(config, model, tokenizer)
