"""
Meta-Cognitive Capabilities for Cyber-LLM
Self-reflection, adaptation, and cognitive load management

Author: Muzan Sano <sanosensei36@gmail.com>
"""

import asyncio
import json
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import torch
import torch.nn as nn
from collections import deque

from ..utils.logging_system import CyberLLMLogger, CyberLLMError, ErrorCategory
from ..memory.persistent_memory import PersistentMemoryManager
from ..memory.strategic_planning import StrategicPlanningEngine

class CognitiveState(Enum):
    """Cognitive processing states"""
    OPTIMAL = "optimal"
    MODERATE_LOAD = "moderate_load"
    HIGH_LOAD = "high_load"
    OVERLOADED = "overloaded"
    RECOVERING = "recovering"

class AdaptationStrategy(Enum):
    """Learning adaptation strategies"""
    AGGRESSIVE = "aggressive"
    MODERATE = "moderate"
    CONSERVATIVE = "conservative"
    CAUTIOUS = "cautious"

@dataclass
class CognitiveMetrics:
    """Cognitive performance metrics"""
    timestamp: datetime
    
    # Performance metrics
    task_completion_rate: float
    accuracy_score: float
    response_time: float
    resource_utilization: float
    
    # Cognitive load indicators
    attention_fragmentation: float  # 0-1, higher = more fragmented
    working_memory_usage: float    # 0-1, percentage used
    processing_complexity: float   # 0-1, task complexity measure
    
    # Adaptation metrics
    learning_rate: float
    confidence_level: float
    adaptation_success_rate: float
    
    # Error metrics
    error_count: int
    critical_errors: int
    recovery_time: Optional[float] = None

@dataclass
class SelfReflectionResult:
    """Results from self-reflection analysis"""
    reflection_id: str
    timestamp: datetime
    
    # Performance assessment
    strengths: List[str]
    weaknesses: List[str]
    improvement_areas: List[str]
    
    # Strategy effectiveness
    effective_strategies: List[str]
    ineffective_strategies: List[str]
    recommended_changes: List[str]
    
    # Cognitive insights
    cognitive_patterns: Dict[str, Any]
    load_management_insights: List[str]
    attention_allocation_insights: List[str]
    
    # Action items
    immediate_adjustments: List[str]
    medium_term_goals: List[str]
    long_term_objectives: List[str]

class MetaCognitiveEngine:
    """Advanced meta-cognitive capabilities for self-reflection and adaptation"""
    
    def __init__(self, 
                 memory_manager: PersistentMemoryManager,
                 strategic_planner: StrategicPlanningEngine,
                 logger: Optional[CyberLLMLogger] = None):
        
        self.memory_manager = memory_manager
        self.strategic_planner = strategic_planner
        self.logger = logger or CyberLLMLogger(name="meta_cognitive")
        
        # Cognitive state tracking
        self.current_state = CognitiveState.OPTIMAL
        self.state_history = deque(maxlen=1000)
        self.cognitive_metrics = deque(maxlen=10000)
        
        # Self-reflection system
        self.reflection_history = {}
        self.performance_baselines = {}
        self.adaptation_strategies = {}
        
        # Cognitive load management
        self.attention_allocator = AttentionAllocator()
        self.cognitive_load_monitor = CognitiveLoadMonitor()
        
        # Learning optimization
        self.learning_rate_optimizer = LearningRateOptimizer()
        self.strategy_evaluator = StrategyEvaluator()
        
        # Neural networks for meta-learning
        self.performance_predictor = self._build_performance_predictor()
        self.strategy_selector = self._build_strategy_selector()
        
        self.logger.info("Meta-Cognitive Engine initialized")
    
    async def conduct_self_reflection(self, 
                                    time_period: timedelta = timedelta(hours=1)) -> SelfReflectionResult:
        """Conduct comprehensive self-reflection analysis"""
        
        reflection_id = f"reflection_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            self.logger.info("Starting self-reflection analysis", reflection_id=reflection_id)
            
            # Gather performance data
            end_time = datetime.now()
            start_time = end_time - time_period
            
            performance_data = await self._gather_performance_data(start_time, end_time)
            cognitive_data = await self._gather_cognitive_data(start_time, end_time)
            strategy_data = await self._gather_strategy_data(start_time, end_time)
            
            # Analyze strengths and weaknesses
            strengths, weaknesses = await self._analyze_performance_patterns(performance_data)
            
            # Evaluate strategy effectiveness
            effective_strategies, ineffective_strategies = await self._evaluate_strategies(strategy_data)
            
            # Generate insights
            cognitive_patterns = await self._analyze_cognitive_patterns(cognitive_data)
            load_insights = await self._analyze_load_management(cognitive_data)
            attention_insights = await self._analyze_attention_allocation(cognitive_data)
            
            # Generate recommendations
            immediate_adjustments = await self._generate_immediate_adjustments(
                weaknesses, ineffective_strategies, cognitive_patterns)
            medium_term_goals = await self._generate_medium_term_goals(
                strengths, weaknesses, cognitive_patterns)
            long_term_objectives = await self._generate_long_term_objectives(
                performance_data, cognitive_patterns)
            
            # Create reflection result
            reflection_result = SelfReflectionResult(
                reflection_id=reflection_id,
                timestamp=datetime.now(),
                strengths=strengths,
                weaknesses=weaknesses,
                improvement_areas=list(set(weaknesses + [adj.split(':')[0] for adj in immediate_adjustments])),
                effective_strategies=effective_strategies,
                ineffective_strategies=ineffective_strategies,
                recommended_changes=immediate_adjustments + medium_term_goals,
                cognitive_patterns=cognitive_patterns,
                load_management_insights=load_insights,
                attention_allocation_insights=attention_insights,
                immediate_adjustments=immediate_adjustments,
                medium_term_goals=medium_term_goals,
                long_term_objectives=long_term_objectives
            )
            
            # Store reflection result
            self.reflection_history[reflection_id] = reflection_result
            
            # Store in persistent memory
            await self.memory_manager.store_reasoning_chain(
                chain_id=f"self_reflection_{reflection_id}",
                steps=[
                    f"Analyzed performance over {time_period}",
                    f"Identified {len(strengths)} strengths and {len(weaknesses)} weaknesses",
                    f"Evaluated {len(effective_strategies)} effective strategies",
                    f"Generated {len(immediate_adjustments)} immediate adjustments"
                ],
                conclusion=f"Self-reflection completed with actionable insights",
                confidence=0.85,
                metadata={
                    "reflection_type": "comprehensive_analysis",
                    "time_period": str(time_period),
                    "performance_score": np.mean([m.accuracy_score for m in self.cognitive_metrics if m.timestamp >= start_time])
                }
            )
            
            self.logger.info("Self-reflection analysis completed",
                           reflection_id=reflection_id,
                           strengths_count=len(strengths),
                           weaknesses_count=len(weaknesses),
                           recommendations_count=len(immediate_adjustments))
            
            return reflection_result
            
        except Exception as e:
            self.logger.error("Self-reflection analysis failed", error=str(e))
            raise CyberLLMError("Self-reflection failed", ErrorCategory.COGNITIVE_ERROR)
    
    async def optimize_learning_rate(self, 
                                   recent_performance: List[float],
                                   task_complexity: float) -> float:
        """Optimize learning rate based on recent performance and task complexity"""
        
        try:
            # Analyze performance trends
            performance_trend = self._calculate_performance_trend(recent_performance)
            performance_variance = np.var(recent_performance)
            
            # Current learning rate
            current_lr = self.learning_rate_optimizer.get_current_rate()
            
            # Adaptation strategy based on performance
            if performance_trend > 0.1 and performance_variance < 0.05:
                # Good performance, stable -> slightly increase learning rate
                adaptation_factor = 1.1
                strategy = AdaptationStrategy.AGGRESSIVE
            elif performance_trend > 0.05:
                # Moderate improvement -> maintain or slight increase
                adaptation_factor = 1.05
                strategy = AdaptationStrategy.MODERATE
            elif performance_trend < -0.1 or performance_variance > 0.2:
                # Poor performance or high variance -> decrease learning rate
                adaptation_factor = 0.8
                strategy = AdaptationStrategy.CAUTIOUS
            else:
                # Stable performance -> minor adjustment based on complexity
                adaptation_factor = 1.0 - (task_complexity - 0.5) * 0.1
                strategy = AdaptationStrategy.CONSERVATIVE
            
            # Apply complexity adjustment
            complexity_factor = 1.0 - (task_complexity * 0.3)
            final_factor = adaptation_factor * complexity_factor
            
            # Calculate new learning rate
            new_lr = current_lr * final_factor
            new_lr = np.clip(new_lr, 0.0001, 0.1)  # Keep within reasonable bounds
            
            # Update learning rate optimizer
            self.learning_rate_optimizer.update_rate(new_lr, strategy)
            
            self.logger.info("Learning rate optimized",
                           old_rate=current_lr,
                           new_rate=new_lr,
                           strategy=strategy.value,
                           performance_trend=performance_trend)
            
            return new_lr
            
        except Exception as e:
            self.logger.error("Learning rate optimization failed", error=str(e))
            return self.learning_rate_optimizer.get_current_rate()
    
    async def manage_cognitive_load(self, 
                                  current_tasks: List[Dict[str, Any]],
                                  available_resources: Dict[str, float]) -> Dict[str, Any]:
        """Manage cognitive load and optimize task allocation"""
        
        try:
            # Calculate current cognitive load
            current_load = await self._calculate_cognitive_load(current_tasks)
            
            # Determine cognitive state
            new_state = self._determine_cognitive_state(current_load, available_resources)
            
            # Update state if changed
            if new_state != self.current_state:
                self.logger.info("Cognitive state changed",
                               old_state=self.current_state.value,
                               new_state=new_state.value)
                self.current_state = new_state
                self.state_history.append((datetime.now(), new_state))
            
            # Generate load management strategy
            management_strategy = await self._generate_load_management_strategy(
                current_load, new_state, current_tasks, available_resources)
            
            # Apply attention allocation optimization
            attention_allocation = await self.attention_allocator.optimize_allocation(
                current_tasks, available_resources, new_state)
            
            # Generate recommendations
            recommendations = await self._generate_load_management_recommendations(
                current_load, new_state, management_strategy)
            
            result = {
                "cognitive_state": new_state.value,
                "cognitive_load": current_load,
                "management_strategy": management_strategy,
                "attention_allocation": attention_allocation,
                "recommendations": recommendations,
                "resource_adjustments": await self._calculate_resource_adjustments(
                    new_state, available_resources)
            }
            
            self.logger.info("Cognitive load management completed",
                           state=new_state.value,
                           load=current_load,
                           recommendations_count=len(recommendations))
            
            return result
            
        except Exception as e:
            self.logger.error("Cognitive load management failed", error=str(e))
            return {"error": str(e)}
    
    def _build_performance_predictor(self) -> nn.Module:
        """Build neural network for performance prediction"""
        
        class PerformancePredictor(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(20, 64)  # Input: various metrics
                self.fc2 = nn.Linear(64, 32)
                self.fc3 = nn.Linear(32, 16)
                self.fc4 = nn.Linear(16, 1)   # Output: predicted performance
                self.dropout = nn.Dropout(0.2)
                
            def forward(self, x):
                x = torch.relu(self.fc1(x))
                x = self.dropout(x)
                x = torch.relu(self.fc2(x))
                x = self.dropout(x)
                x = torch.relu(self.fc3(x))
                x = torch.sigmoid(self.fc4(x))
                return x
        
        return PerformancePredictor()
    
    def _build_strategy_selector(self) -> nn.Module:
        """Build neural network for strategy selection"""
        
        class StrategySelector(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(15, 48)  # Input: context features
                self.fc2 = nn.Linear(48, 24)
                self.fc3 = nn.Linear(24, 8)   # Output: strategy probabilities
                self.dropout = nn.Dropout(0.15)
                
            def forward(self, x):
                x = torch.relu(self.fc1(x))
                x = self.dropout(x)
                x = torch.relu(self.fc2(x))
                x = torch.softmax(self.fc3(x), dim=-1)
                return x
        
        return StrategySelector()

class AttentionAllocator:
    """Manages dynamic attention allocation across tasks"""
    
    def __init__(self):
        self.attention_weights = {}
        self.priority_scores = {}
        self.allocation_history = deque(maxlen=1000)
    
    async def optimize_allocation(self, 
                                tasks: List[Dict[str, Any]], 
                                resources: Dict[str, float],
                                cognitive_state: CognitiveState) -> Dict[str, float]:
        """Optimize attention allocation across tasks"""
        
        # Calculate base priority scores
        for task in tasks:
            task_id = task.get('id', str(hash(str(task))))
            priority = task.get('priority', 0.5)
            complexity = task.get('complexity', 0.5)
            deadline_pressure = task.get('deadline_pressure', 0.0)
            
            # Adjust priority based on cognitive state
            state_multiplier = {
                CognitiveState.OPTIMAL: 1.0,
                CognitiveState.MODERATE_LOAD: 0.9,
                CognitiveState.HIGH_LOAD: 0.7,
                CognitiveState.OVERLOADED: 0.5,
                CognitiveState.RECOVERING: 0.6
            }.get(cognitive_state, 1.0)
            
            adjusted_priority = (priority * 0.4 + 
                               deadline_pressure * 0.4 + 
                               (1.0 - complexity) * 0.2) * state_multiplier
            
            self.priority_scores[task_id] = adjusted_priority
        
        # Normalize allocation
        total_priority = sum(self.priority_scores.values())
        if total_priority > 0:
            allocation = {task_id: score / total_priority 
                         for task_id, score in self.priority_scores.items()}
        else:
            # Equal allocation if no priorities
            equal_weight = 1.0 / len(tasks) if tasks else 0.0
            allocation = {task.get('id', str(i)): equal_weight 
                         for i, task in enumerate(tasks)}
        
        # Store allocation history
        self.allocation_history.append((datetime.now(), allocation))
        
        return allocation

class CognitiveLoadMonitor:
    """Monitors and analyzes cognitive load patterns"""
    
    def __init__(self):
        self.load_history = deque(maxlen=10000)
        self.load_patterns = {}
    
    def calculate_load(self, 
                      active_tasks: int,
                      task_complexity: float,
                      resource_usage: float,
                      error_rate: float) -> float:
        """Calculate current cognitive load"""
        
        # Base load from task count (logarithmic scaling)
        task_load = min(np.log(active_tasks + 1) / np.log(10), 1.0)
        
        # Complexity contribution
        complexity_load = task_complexity * 0.3
        
        # Resource pressure
        resource_load = resource_usage * 0.25
        
        # Error pressure (exponential)
        error_load = min(error_rate ** 0.5, 1.0) * 0.2
        
        total_load = task_load + complexity_load + resource_load + error_load
        
        # Store in history
        self.load_history.append((datetime.now(), total_load))
        
        return min(total_load, 1.0)

class LearningRateOptimizer:
    """Optimizes learning rates based on performance feedback"""
    
    def __init__(self, initial_rate: float = 0.001):
        self.current_rate = initial_rate
        self.rate_history = deque(maxlen=1000)
        self.performance_history = deque(maxlen=1000)
        self.strategy_effectiveness = {}
    
    def get_current_rate(self) -> float:
        return self.current_rate
    
    def update_rate(self, new_rate: float, strategy: AdaptationStrategy):
        self.rate_history.append((datetime.now(), self.current_rate, new_rate, strategy))
        self.current_rate = new_rate

class StrategyEvaluator:
    """Evaluates effectiveness of different strategies"""
    
    def __init__(self):
        self.strategy_outcomes = {}
        self.strategy_scores = {}
    
    def record_strategy_outcome(self, strategy: str, outcome_score: float):
        if strategy not in self.strategy_outcomes:
            self.strategy_outcomes[strategy] = deque(maxlen=100)
        
        self.strategy_outcomes[strategy].append((datetime.now(), outcome_score))
        
        # Update average score
        scores = [score for _, score in self.strategy_outcomes[strategy]]
        self.strategy_scores[strategy] = np.mean(scores)

# Factory function
def create_meta_cognitive_engine(memory_manager: PersistentMemoryManager,
                               strategic_planner: StrategicPlanningEngine,
                               **kwargs) -> MetaCognitiveEngine:
    """Create meta-cognitive engine"""
    return MetaCognitiveEngine(memory_manager, strategic_planner, **kwargs)
