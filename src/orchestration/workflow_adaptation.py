"""
Dynamic Workflow Adaptation System for Cyber-LLM
Provides context-aware workflow modification and intelligent decision making
"""

import asyncio
import json
import yaml
from typing import Dict, List, Any, Optional, Callable, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle
import os

from .multi_agent_scenarios import RedTeamScenario, ScenarioStep, ScenarioResult
from ..utils.logging_system import CyberLLMLogger, CyberLLMError, ErrorCategory

class AdaptationTrigger(Enum):
    """Triggers for workflow adaptation"""
    STEP_FAILURE = "step_failure"
    TIMEOUT = "timeout"
    CONTEXT_CHANGE = "context_change"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    SECURITY_ALERT = "security_alert"
    RESOURCE_CONSTRAINT = "resource_constraint"
    USER_INTERVENTION = "user_intervention"

class AdaptationStrategy(Enum):
    """Strategies for workflow adaptation"""
    RETRY_WITH_MODIFICATION = "retry_with_modification"
    SKIP_STEP = "skip_step"
    ALTERNATIVE_PATH = "alternative_path"
    PARAMETER_TUNING = "parameter_tuning"
    AGENT_SUBSTITUTION = "agent_substitution"
    PARALLEL_EXECUTION = "parallel_execution"
    ROLLBACK = "rollback"
    EMERGENCY_STOP = "emergency_stop"

@dataclass
class ContextFeatures:
    """Context features for adaptation decisions"""
    target_responsiveness: float  # 0-1 scale
    network_complexity: int
    security_level: str  # low, medium, high, critical
    time_constraints: int  # minutes remaining
    resource_availability: float  # 0-1 scale
    detection_probability: float  # 0-1 scale
    step_success_rate: float  # historical success rate
    agent_performance: Dict[str, float]  # agent-specific performance metrics

@dataclass
class AdaptationDecision:
    """Decision made by the adaptation system"""
    trigger: AdaptationTrigger
    strategy: AdaptationStrategy
    parameters: Dict[str, Any]
    confidence: float
    reasoning: str
    estimated_impact: Dict[str, float]

class WorkflowAdaptationEngine:
    """Engine for dynamic workflow adaptation based on context and performance"""
    
    def __init__(self, logger: Optional[CyberLLMLogger] = None):
        self.logger = logger or CyberLLMLogger(name="workflow_adaptation")
        
        # ML models for adaptation decisions
        self.adaptation_model = None
        self.feature_scaler = StandardScaler()
        self.model_trained = False
        
        # Historical data for learning
        self.execution_history = []
        self.adaptation_history = []
        
        # Adaptation rules and strategies
        self.adaptation_rules = self._initialize_adaptation_rules()
        self.strategy_implementations = self._initialize_strategies()
        
        # Performance thresholds
        self.performance_thresholds = {
            'success_rate_critical': 0.3,
            'success_rate_warning': 0.6,
            'timeout_threshold': 0.8,  # 80% of allocated time
            'detection_risk_critical': 0.8,
            'resource_critical': 0.2
        }
    
    def _initialize_adaptation_rules(self) -> Dict[AdaptationTrigger, List[Callable]]:
        """Initialize rule-based adaptation logic"""
        return {
            AdaptationTrigger.STEP_FAILURE: [
                self._handle_step_failure,
                self._check_alternative_paths,
                self._consider_agent_substitution
            ],
            AdaptationTrigger.TIMEOUT: [
                self._handle_timeout,
                self._optimize_parameters,
                self._consider_parallel_execution
            ],
            AdaptationTrigger.CONTEXT_CHANGE: [
                self._handle_context_change,
                self._update_strategy_priorities
            ],
            AdaptationTrigger.PERFORMANCE_DEGRADATION: [
                self._handle_performance_degradation,
                self._tune_parameters,
                self._consider_rollback
            ],
            AdaptationTrigger.SECURITY_ALERT: [
                self._handle_security_alert,
                self._assess_stealth_requirements,
                self._consider_emergency_stop
            ],
            AdaptationTrigger.RESOURCE_CONSTRAINT: [
                self._handle_resource_constraint,
                self._optimize_resource_usage,
                self._prioritize_critical_steps
            ]
        }
    
    def _initialize_strategies(self) -> Dict[AdaptationStrategy, Callable]:
        """Initialize strategy implementation functions"""
        return {
            AdaptationStrategy.RETRY_WITH_MODIFICATION: self._implement_retry_modification,
            AdaptationStrategy.SKIP_STEP: self._implement_skip_step,
            AdaptationStrategy.ALTERNATIVE_PATH: self._implement_alternative_path,
            AdaptationStrategy.PARAMETER_TUNING: self._implement_parameter_tuning,
            AdaptationStrategy.AGENT_SUBSTITUTION: self._implement_agent_substitution,
            AdaptationStrategy.PARALLEL_EXECUTION: self._implement_parallel_execution,
            AdaptationStrategy.ROLLBACK: self._implement_rollback,
            AdaptationStrategy.EMERGENCY_STOP: self._implement_emergency_stop
        }
    
    async def analyze_context(self, 
                            scenario: RedTeamScenario,
                            current_step: Optional[ScenarioStep],
                            execution_state: Dict[str, Any],
                            environment_data: Dict[str, Any]) -> ContextFeatures:
        """Analyze current execution context"""
        
        # Extract context features
        target_responsiveness = environment_data.get('target_response_time', 1.0)
        if target_responsiveness > 0:
            target_responsiveness = min(1.0, 1.0 / target_responsiveness)
        
        network_complexity = len(environment_data.get('discovered_hosts', [])) + \
                           len(environment_data.get('open_ports', [])) // 10
        
        security_level = environment_data.get('security_posture', 'medium')
        
        # Calculate time constraints
        elapsed_time = (datetime.now() - execution_state.get('start_time', datetime.now())).total_seconds() / 60
        estimated_duration = scenario.estimated_duration
        time_remaining = max(0, estimated_duration - elapsed_time)
        
        # Resource availability (simplified)
        cpu_usage = environment_data.get('cpu_usage', 0.5)
        memory_usage = environment_data.get('memory_usage', 0.5)
        resource_availability = 1.0 - max(cpu_usage, memory_usage)
        
        # Detection probability estimation
        stealth_actions = execution_state.get('stealth_actions', 0)
        total_actions = execution_state.get('total_actions', 1)
        detection_probability = max(0, min(1, (total_actions - stealth_actions) / total_actions))
        
        # Historical success rate
        completed_steps = len(execution_state.get('completed_steps', []))
        total_attempted = completed_steps + len(execution_state.get('failed_steps', []))
        step_success_rate = completed_steps / max(1, total_attempted)
        
        # Agent performance metrics
        agent_performance = {}
        for agent_name in ['recon', 'c2', 'post_exploit', 'safety', 'orchestrator']:
            success_count = execution_state.get(f'{agent_name}_successes', 0)
            total_count = execution_state.get(f'{agent_name}_attempts', 1)
            agent_performance[agent_name] = success_count / total_count
        
        context = ContextFeatures(
            target_responsiveness=target_responsiveness,
            network_complexity=network_complexity,
            security_level=security_level,
            time_constraints=int(time_remaining),
            resource_availability=resource_availability,
            detection_probability=detection_probability,
            step_success_rate=step_success_rate,
            agent_performance=agent_performance
        )
        
        self.logger.debug("Context analysis completed", 
                         context=asdict(context))
        
        return context
    
    async def detect_adaptation_trigger(self,
                                      scenario: RedTeamScenario,
                                      current_step: Optional[ScenarioStep],
                                      execution_state: Dict[str, Any],
                                      context: ContextFeatures) -> Optional[AdaptationTrigger]:
        """Detect if workflow adaptation is needed"""
        
        # Check for step failure
        if execution_state.get('last_step_failed', False):
            return AdaptationTrigger.STEP_FAILURE
        
        # Check for timeout risk
        if context.time_constraints < scenario.estimated_duration * self.performance_thresholds['timeout_threshold']:
            return AdaptationTrigger.TIMEOUT
        
        # Check for performance degradation
        if context.step_success_rate < self.performance_thresholds['success_rate_warning']:
            return AdaptationTrigger.PERFORMANCE_DEGRADATION
        
        # Check for security alert
        if context.detection_probability > self.performance_thresholds['detection_risk_critical']:
            return AdaptationTrigger.SECURITY_ALERT
        
        # Check for resource constraints
        if context.resource_availability < self.performance_thresholds['resource_critical']:
            return AdaptationTrigger.RESOURCE_CONSTRAINT
        
        # Check for significant context changes
        last_context = execution_state.get('last_context')
        if last_context and self._context_changed_significantly(context, last_context):
            return AdaptationTrigger.CONTEXT_CHANGE
        
        return None
    
    def _context_changed_significantly(self, 
                                     current: ContextFeatures,
                                     previous: ContextFeatures) -> bool:
        """Detect significant context changes"""
        
        thresholds = {
            'target_responsiveness': 0.3,
            'network_complexity': 5,
            'resource_availability': 0.2,
            'detection_probability': 0.2
        }
        
        changes = {
            'target_responsiveness': abs(current.target_responsiveness - previous.target_responsiveness),
            'network_complexity': abs(current.network_complexity - previous.network_complexity),
            'resource_availability': abs(current.resource_availability - previous.resource_availability),
            'detection_probability': abs(current.detection_probability - previous.detection_probability)
        }
        
        return any(change > thresholds[key] for key, change in changes.items())
    
    async def make_adaptation_decision(self,
                                     trigger: AdaptationTrigger,
                                     scenario: RedTeamScenario,
                                     current_step: Optional[ScenarioStep],
                                     context: ContextFeatures,
                                     execution_state: Dict[str, Any]) -> AdaptationDecision:
        """Make an adaptation decision based on context and trigger"""
        
        # Use ML model if trained, otherwise use rule-based approach
        if self.model_trained:
            decision = await self._ml_based_decision(trigger, context, execution_state)
        else:
            decision = await self._rule_based_decision(trigger, scenario, current_step, context, execution_state)
        
        # Log decision
        self.logger.info("Adaptation decision made",
                        trigger=trigger.value,
                        strategy=decision.strategy.value,
                        confidence=decision.confidence,
                        reasoning=decision.reasoning)
        
        # Store for learning
        self.adaptation_history.append({
            'trigger': trigger,
            'context': asdict(context),
            'decision': asdict(decision),
            'timestamp': datetime.now()
        })
        
        return decision
    
    async def _rule_based_decision(self,
                                 trigger: AdaptationTrigger,
                                 scenario: RedTeamScenario,
                                 current_step: Optional[ScenarioStep],
                                 context: ContextFeatures,
                                 execution_state: Dict[str, Any]) -> AdaptationDecision:
        """Make adaptation decision using rule-based logic"""
        
        # Get applicable rules for the trigger
        rules = self.adaptation_rules.get(trigger, [])
        
        # Apply rules and collect recommendations
        recommendations = []
        for rule in rules:
            try:
                recommendation = await rule(scenario, current_step, context, execution_state)
                if recommendation:
                    recommendations.append(recommendation)
            except Exception as e:
                self.logger.warning(f"Rule application failed: {rule.__name__}", error=str(e))
        
        # Select best recommendation
        if not recommendations:
            # Default fallback strategy
            return AdaptationDecision(
                trigger=trigger,
                strategy=AdaptationStrategy.RETRY_WITH_MODIFICATION,
                parameters={'retry_count': 1, 'modify_timeout': True},
                confidence=0.5,
                reasoning="Default fallback strategy",
                estimated_impact={'success_probability': 0.6, 'time_cost': 0.1}
            )
        
        # Score recommendations and select best
        best_recommendation = max(recommendations, key=lambda x: x.confidence)
        return best_recommendation
    
    async def _ml_based_decision(self,
                               trigger: AdaptationTrigger,
                               context: ContextFeatures,
                               execution_state: Dict[str, Any]) -> AdaptationDecision:
        """Make adaptation decision using ML model"""
        
        # Prepare features for ML model
        features = self._prepare_ml_features(trigger, context, execution_state)
        features_scaled = self.feature_scaler.transform([features])
        
        # Predict best strategy
        strategy_probs = self.adaptation_model.predict_proba(features_scaled)[0]
        best_strategy_idx = np.argmax(strategy_probs)
        confidence = strategy_probs[best_strategy_idx]
        
        # Map to strategy enum
        strategies = list(AdaptationStrategy)
        best_strategy = strategies[best_strategy_idx]
        
        # Generate parameters based on strategy and context
        parameters = self._generate_strategy_parameters(best_strategy, context)
        
        return AdaptationDecision(
            trigger=trigger,
            strategy=best_strategy,
            parameters=parameters,
            confidence=confidence,
            reasoning="ML model prediction",
            estimated_impact=self._estimate_strategy_impact(best_strategy, context)
        )
    
    def _prepare_ml_features(self, 
                           trigger: AdaptationTrigger,
                           context: ContextFeatures,
                           execution_state: Dict[str, Any]) -> List[float]:
        """Prepare features for ML model"""
        
        trigger_features = [0] * len(AdaptationTrigger)
        trigger_features[list(AdaptationTrigger).index(trigger)] = 1
        
        context_features = [
            context.target_responsiveness,
            context.network_complexity / 100.0,  # Normalize
            {'low': 0.25, 'medium': 0.5, 'high': 0.75, 'critical': 1.0}[context.security_level],
            context.time_constraints / 480.0,  # Normalize to 8 hours max
            context.resource_availability,
            context.detection_probability,
            context.step_success_rate,
            np.mean(list(context.agent_performance.values()))
        ]
        
        execution_features = [
            len(execution_state.get('completed_steps', [])) / 20.0,  # Normalize
            len(execution_state.get('failed_steps', [])) / 10.0,  # Normalize
            execution_state.get('retry_count', 0) / 5.0  # Normalize
        ]
        
        return trigger_features + context_features + execution_features
    
    async def apply_adaptation(self,
                             decision: AdaptationDecision,
                             scenario: RedTeamScenario,
                             current_step: Optional[ScenarioStep],
                             execution_state: Dict[str, Any]) -> Dict[str, Any]:
        """Apply the adaptation decision to the workflow"""
        
        self.logger.info(f"Applying adaptation strategy: {decision.strategy.value}",
                        confidence=decision.confidence,
                        parameters=decision.parameters)
        
        # Get strategy implementation
        strategy_impl = self.strategy_implementations.get(decision.strategy)
        if not strategy_impl:
            raise CyberLLMError(
                f"No implementation for strategy: {decision.strategy}",
                ErrorCategory.SYSTEM
            )
        
        # Apply strategy
        try:
            result = await strategy_impl(decision, scenario, current_step, execution_state)
            
            # Log success
            self.logger.info("Adaptation applied successfully",
                           strategy=decision.strategy.value,
                           result_type=type(result).__name__)
            
            return result
            
        except Exception as e:
            self.logger.error("Adaptation application failed",
                            strategy=decision.strategy.value,
                            error=str(e))
            raise
    
    # Rule implementations
    async def _handle_step_failure(self, 
                                 scenario: RedTeamScenario,
                                 current_step: Optional[ScenarioStep],
                                 context: ContextFeatures,
                                 execution_state: Dict[str, Any]) -> Optional[AdaptationDecision]:
        """Handle step failure"""
        
        if not current_step:
            return None
        
        retry_count = execution_state.get('retry_count', 0)
        
        if retry_count < current_step.retry_count:
            return AdaptationDecision(
                trigger=AdaptationTrigger.STEP_FAILURE,
                strategy=AdaptationStrategy.RETRY_WITH_MODIFICATION,
                parameters={
                    'retry_count': retry_count + 1,
                    'timeout_multiplier': 1.5,
                    'parameter_adjustment': True
                },
                confidence=0.7,
                reasoning="Step can be retried with modifications",
                estimated_impact={'success_probability': 0.6, 'time_cost': 0.2}
            )
        
        return AdaptationDecision(
            trigger=AdaptationTrigger.STEP_FAILURE,
            strategy=AdaptationStrategy.ALTERNATIVE_PATH,
            parameters={'skip_failed_step': True, 'find_alternative': True},
            confidence=0.6,
            reasoning="Max retries reached, seeking alternative path",
            estimated_impact={'success_probability': 0.5, 'time_cost': 0.1}
        )
    
    async def _handle_timeout(self,
                            scenario: RedTeamScenario,
                            current_step: Optional[ScenarioStep],
                            context: ContextFeatures,
                            execution_state: Dict[str, Any]) -> Optional[AdaptationDecision]:
        """Handle timeout scenarios"""
        
        if context.time_constraints < scenario.estimated_duration * 0.2:  # Less than 20% time remaining
            return AdaptationDecision(
                trigger=AdaptationTrigger.TIMEOUT,
                strategy=AdaptationStrategy.PARALLEL_EXECUTION,
                parameters={'max_parallel_agents': 3, 'prioritize_critical': True},
                confidence=0.8,
                reasoning="Critical time constraint, enabling parallel execution",
                estimated_impact={'success_probability': 0.7, 'time_cost': -0.3}
            )
        
        return AdaptationDecision(
            trigger=AdaptationTrigger.TIMEOUT,
            strategy=AdaptationStrategy.PARAMETER_TUNING,
            parameters={'reduce_timeout': 0.8, 'increase_aggressiveness': True},
            confidence=0.6,
            reasoning="Moderate time pressure, optimizing parameters",
            estimated_impact={'success_probability': 0.6, 'time_cost': -0.1}
        )
    
    # Strategy implementations
    async def _implement_retry_modification(self,
                                          decision: AdaptationDecision,
                                          scenario: RedTeamScenario,
                                          current_step: Optional[ScenarioStep],
                                          execution_state: Dict[str, Any]) -> Dict[str, Any]:
        """Implement retry with modification strategy"""
        
        if not current_step:
            return {'action': 'no_step_to_retry'}
        
        # Create modified step
        modified_step = ScenarioStep(
            id=f"{current_step.id}_retry_{decision.parameters.get('retry_count', 1)}",
            name=f"{current_step.name} (Retry)",
            description=current_step.description,
            agent_type=current_step.agent_type,
            dependencies=current_step.dependencies,
            parameters=current_step.parameters.copy(),
            timeout=int(current_step.timeout * decision.parameters.get('timeout_multiplier', 1.0)),
            retry_count=current_step.retry_count,
            critical=current_step.critical,
            parallel_group=current_step.parallel_group
        )
        
        # Adjust parameters if requested
        if decision.parameters.get('parameter_adjustment'):
            modified_step.parameters['retry_mode'] = True
            modified_step.parameters['increased_verbosity'] = True
        
        return {
            'action': 'retry_step',
            'modified_step': modified_step,
            'original_step': current_step
        }
    
    async def _implement_alternative_path(self,
                                        decision: AdaptationDecision,
                                        scenario: RedTeamScenario,
                                        current_step: Optional[ScenarioStep],
                                        execution_state: Dict[str, Any]) -> Dict[str, Any]:
        """Implement alternative path strategy"""
        
        # Find alternative steps that can achieve similar objectives
        alternative_steps = []
        
        if current_step and current_step.agent_type == 'recon':
            # Alternative reconnaissance methods
            alt_step = ScenarioStep(
                id=f"alt_recon_{uuid.uuid4().hex[:8]}",
                name="Alternative Reconnaissance",
                description="Alternative information gathering approach",
                agent_type="recon",
                dependencies=current_step.dependencies,
                parameters={
                    'alternative_method': True,
                    'stealth_priority': True,
                    'reduced_scope': True
                },
                timeout=current_step.timeout // 2
            )
            alternative_steps.append(alt_step)
        
        elif current_step and current_step.agent_type == 'c2':
            # Alternative C2 methods
            alt_step = ScenarioStep(
                id=f"alt_c2_{uuid.uuid4().hex[:8]}",
                name="Alternative Command & Control",
                description="Alternative C2 establishment method",
                agent_type="c2",
                dependencies=current_step.dependencies,
                parameters={
                    'backup_method': True,
                    'lower_profile': True
                },
                timeout=current_step.timeout
            )
            alternative_steps.append(alt_step)
        
        return {
            'action': 'alternative_path',
            'alternative_steps': alternative_steps,
            'skip_original': decision.parameters.get('skip_failed_step', False)
        }
    
    async def _implement_parameter_tuning(self,
                                        decision: AdaptationDecision,
                                        scenario: RedTeamScenario,
                                        current_step: Optional[ScenarioStep],
                                        execution_state: Dict[str, Any]) -> Dict[str, Any]:
        """Implement parameter tuning strategy"""
        
        tuning_recommendations = {}
        
        # Global parameter adjustments
        if decision.parameters.get('reduce_timeout'):
            tuning_recommendations['global_timeout_multiplier'] = decision.parameters['reduce_timeout']
        
        if decision.parameters.get('increase_aggressiveness'):
            tuning_recommendations['aggressiveness_level'] = 'high'
            tuning_recommendations['stealth_level'] = 'medium'
        
        # Step-specific adjustments
        if current_step:
            step_adjustments = {}
            
            if current_step.agent_type == 'recon':
                step_adjustments['scan_intensity'] = 'high'
                step_adjustments['parallel_scans'] = True
            
            elif current_step.agent_type == 'c2':
                step_adjustments['connection_attempts'] = 5
                step_adjustments['fallback_protocols'] = True
            
            tuning_recommendations['step_adjustments'] = step_adjustments
        
        return {
            'action': 'parameter_tuning',
            'tuning_recommendations': tuning_recommendations
        }
    
    async def _implement_parallel_execution(self,
                                          decision: AdaptationDecision,
                                          scenario: RedTeamScenario,
                                          current_step: Optional[ScenarioStep],
                                          execution_state: Dict[str, Any]) -> Dict[str, Any]:
        """Implement parallel execution strategy"""
        
        max_parallel = decision.parameters.get('max_parallel_agents', 3)
        prioritize_critical = decision.parameters.get('prioritize_critical', False)
        
        # Identify steps that can be run in parallel
        remaining_steps = [step for step in scenario.steps 
                          if step.id not in execution_state.get('completed_steps', [])]
        
        parallel_groups = {}
        for step in remaining_steps:
            # Group steps that can run in parallel
            if not step.dependencies or all(dep in execution_state.get('completed_steps', []) 
                                           for dep in step.dependencies):
                group_key = step.parallel_group or f"auto_parallel_{step.agent_type}"
                if group_key not in parallel_groups:
                    parallel_groups[group_key] = []
                parallel_groups[group_key].append(step)
        
        return {
            'action': 'enable_parallel_execution',
            'parallel_groups': parallel_groups,
            'max_parallel_agents': max_parallel,
            'prioritize_critical': prioritize_critical
        }
    
    async def _implement_emergency_stop(self,
                                      decision: AdaptationDecision,
                                      scenario: RedTeamScenario,
                                      current_step: Optional[ScenarioStep],
                                      execution_state: Dict[str, Any]) -> Dict[str, Any]:
        """Implement emergency stop strategy"""
        
        return {
            'action': 'emergency_stop',
            'reason': decision.reasoning,
            'safe_shutdown': True,
            'preserve_state': True
        }
    
    def train_adaptation_model(self, training_data: List[Dict[str, Any]]):
        """Train ML model for adaptation decisions"""
        
        if len(training_data) < 100:  # Need sufficient training data
            self.logger.warning("Insufficient training data for ML model")
            return
        
        # Prepare training data
        X = []
        y = []
        
        for record in training_data:
            features = self._prepare_ml_features(
                record['trigger'],
                ContextFeatures(**record['context']),
                record['execution_state']
            )
            X.append(features)
            y.append(list(AdaptationStrategy).index(record['best_strategy']))
        
        X = np.array(X)
        y = np.array(y)
        
        # Scale features
        X_scaled = self.feature_scaler.fit_transform(X)
        
        # Train model
        self.adaptation_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.adaptation_model.fit(X_scaled, y)
        
        self.model_trained = True
        self.logger.info("Adaptation model trained successfully",
                        training_samples=len(training_data),
                        accuracy=self.adaptation_model.score(X_scaled, y))
    
    def save_model(self, model_path: str):
        """Save trained model to disk"""
        if self.model_trained:
            model_data = {
                'model': self.adaptation_model,
                'scaler': self.feature_scaler,
                'history': self.adaptation_history[-1000:]  # Keep last 1000 records
            }
            with open(model_path, 'wb') as f:
                pickle.dump(model_data, f)
            self.logger.info(f"Model saved to {model_path}")
    
    def load_model(self, model_path: str):
        """Load trained model from disk"""
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.adaptation_model = model_data['model']
            self.feature_scaler = model_data['scaler']
            self.adaptation_history.extend(model_data.get('history', []))
            self.model_trained = True
            
            self.logger.info(f"Model loaded from {model_path}")

# Example usage
if __name__ == "__main__":
    import asyncio
    
    async def main():
        # Initialize adaptation engine
        engine = WorkflowAdaptationEngine()
        
        # Mock context and scenario for testing
        from .multi_agent_scenarios import RedTeamScenario, ScenarioType
        
        mock_scenario = RedTeamScenario(
            id="test_scenario",
            name="Test Scenario",
            description="Test scenario for adaptation",
            scenario_type=ScenarioType.RED_TEAM_EXERCISE,
            target_environment={},
            steps=[],
            success_criteria={},
            safety_constraints=[],
            estimated_duration=120,
            difficulty_level="intermediate"
        )
        
        mock_context = ContextFeatures(
            target_responsiveness=0.8,
            network_complexity=15,
            security_level="high",
            time_constraints=30,
            resource_availability=0.4,
            detection_probability=0.6,
            step_success_rate=0.4,
            agent_performance={'recon': 0.8, 'c2': 0.6}
        )
        
        # Detect adaptation trigger
        trigger = await engine.detect_adaptation_trigger(
            mock_scenario, None, {'last_step_failed': True}, mock_context
        )
        
        if trigger:
            print(f"Detected trigger: {trigger.value}")
            
            # Make adaptation decision
            decision = await engine.make_adaptation_decision(
                trigger, mock_scenario, None, mock_context, {}
            )
            
            print(f"Adaptation decision: {decision.strategy.value}")
            print(f"Confidence: {decision.confidence:.2f}")
            print(f"Reasoning: {decision.reasoning}")
    
    asyncio.run(main())
