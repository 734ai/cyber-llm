"""
Cyber-LLM Learning Module

This module implements advanced learning capabilities for continuous intelligence
and evolution of the Cyber-LLM system.

Components:
- online_learning: Real-time learning from operational feedback
- federated_learning: Secure multi-organization collaborative learning
- meta_learning: Rapid adaptation to new threats and attack patterns
- research_collaboration: Framework for sharing cybersecurity insights
- constitutional_ai: Ethical constraints and safety guardrails
- continuous_intelligence: Orchestration of all learning components

Author: Muzan Sano <sanosensei36@gmail.com>
"""

from .online_learning import (
    OnlineLearningManager,
    LearningEvent,
    LearningEventType,
    ThreatIntelligenceProcessor,
    create_online_learning_manager
)

from .federated_learning import (
    FederatedLearningCoordinator,
    FederatedLearningParticipant,
    FederatedLearningRound,
    SecureCommunicationManager,
    ModelAggregator,
    create_federated_learning_coordinator
)

from .meta_learning import (
    MetaLearningManager,
    MetaLearningStrategy,
    MetaTask,
    TaskType,
    CyberSecurityTaskGenerator,
    create_meta_learning_manager
)

from .research_collaboration import (
    ResearchCollaborationManager,
    CollaborationType,
    ResearchInsight,
    CollaborationParticipant,
    CollaborationProject,
    ParticipantRole,
    SensitivityLevel,
    create_research_collaboration_manager
)

from .constitutional_ai import (
    ConstitutionalAIManager,
    EthicalPrinciple,
    ViolationType,
    ActionType,
    ConstitutionalRule,
    ConstitutionalViolation,
    create_constitutional_ai_manager
)

from .continuous_intelligence import (
    ContinuousIntelligenceOrchestrator,
    ContinuousIntelligenceConfig,
    ContinuousIntelligenceMode,
    create_continuous_intelligence_config,
    create_continuous_intelligence_orchestrator
)

__all__ = [
    # Online Learning
    'OnlineLearningManager',
    'LearningEvent',
    'LearningEventType', 
    'ThreatIntelligenceProcessor',
    'create_online_learning_manager',
    
    # Federated Learning
    'FederatedLearningCoordinator',
    'FederatedLearningParticipant',
    'FederatedLearningRound',
    'SecureCommunicationManager',
    'ModelAggregator',
    'create_federated_learning_coordinator',
    
    # Meta Learning
    'MetaLearningManager',
    'MetaLearningStrategy',
    'MetaTask',
    'TaskType',
    'CyberSecurityTaskGenerator',
    'create_meta_learning_manager',
    
    # Research Collaboration
    'ResearchCollaborationManager',
    'CollaborationType',
    'ResearchInsight',
    'CollaborationParticipant',
    'CollaborationProject',
    'ParticipantRole',
    'SensitivityLevel',
    'create_research_collaboration_manager',
    
    # Constitutional AI
    'ConstitutionalAIManager',
    'EthicalPrinciple',
    'ViolationType',
    'ActionType',
    'ConstitutionalRule',
    'ConstitutionalViolation',
    'create_constitutional_ai_manager',
    
    # Continuous Intelligence
    'ContinuousIntelligenceOrchestrator',
    'ContinuousIntelligenceConfig',
    'ContinuousIntelligenceMode',
    'create_continuous_intelligence_config',
    'create_continuous_intelligence_orchestrator'
]

# Module metadata
__version__ = "1.0.0"
__author__ = "Muzan Sano <sanosensei36@gmail.com>"
__description__ = "Advanced learning and adaptation capabilities for Cyber-LLM"

# Convenience functions for common use cases
def create_full_continuous_intelligence_system(model,
                                             tokenizer,
                                             organization_name: str = "CyberLLM-Default",
                                             mode: ContinuousIntelligenceMode = ContinuousIntelligenceMode.BALANCED):
    """
    Create a complete continuous intelligence system with all components enabled.
    
    Args:
        model: The language model to enhance with continuous intelligence
        tokenizer: Tokenizer for the model
        organization_name: Name of the organization using the system
        mode: Operating mode for the continuous intelligence system
        
    Returns:
        ContinuousIntelligenceOrchestrator: Fully configured orchestrator
    """
    
    config = create_continuous_intelligence_config(
        mode=mode,
        organization_name=organization_name,
        enable_online_learning=True,
        enable_federated_learning=True,
        enable_meta_learning=True,
        enable_research_collaboration=True,
        enable_constitutional_ai=True
    )
    
    return create_continuous_intelligence_orchestrator(model, tokenizer, config)

def create_research_focused_system(model,
                                 tokenizer, 
                                 organization_name: str):
    """
    Create a research-focused continuous intelligence system optimized for 
    collaborative research and knowledge sharing.
    
    Args:
        model: The language model
        tokenizer: Tokenizer for the model
        organization_name: Name of the research organization
        
    Returns:
        ContinuousIntelligenceOrchestrator: Research-optimized orchestrator
    """
    
    return create_full_continuous_intelligence_system(
        model=model,
        tokenizer=tokenizer,
        organization_name=organization_name,
        mode=ContinuousIntelligenceMode.RESEARCH
    )

def create_production_system(model,
                           tokenizer,
                           organization_name: str):
    """
    Create a production-ready continuous intelligence system with conservative
    settings optimized for stability and safety.
    
    Args:
        model: The language model
        tokenizer: Tokenizer for the model
        organization_name: Name of the organization
        
    Returns:
        ContinuousIntelligenceOrchestrator: Production-optimized orchestrator
    """
    
    return create_full_continuous_intelligence_system(
        model=model,
        tokenizer=tokenizer,
        organization_name=organization_name,
        mode=ContinuousIntelligenceMode.PRODUCTION
    )
