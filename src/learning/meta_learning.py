"""
Meta-Learning System for Cyber-LLM
Enables rapid adaptation to new attack patterns and defense strategies through meta-learning.

Author: Muzan Sano <sanosensei36@gmail.com>
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import defaultdict
import random

from ..utils.logging_system import CyberLLMLogger
from .online_learning import LearningEvent, LearningEventType

# Configure logging
logger = CyberLLMLogger(__name__).get_logger()

class MetaLearningStrategy(Enum):
    """Types of meta-learning strategies"""
    MAML = "model_agnostic_meta_learning"  # Model-Agnostic Meta-Learning
    REPTILE = "reptile"  # Reptile algorithm
    PROTOTYPICAL = "prototypical_networks"  # Prototype-based learning
    MEMORY_AUGMENTED = "memory_augmented"  # Memory-augmented networks
    GRADIENT_BASED = "gradient_based"  # Gradient-based meta-learning

class TaskType(Enum):
    """Types of cybersecurity tasks for meta-learning"""
    THREAT_CLASSIFICATION = "threat_classification"
    ATTACK_PREDICTION = "attack_prediction"
    IOC_DETECTION = "ioc_detection"
    VULNERABILITY_ASSESSMENT = "vulnerability_assessment"  
    INCIDENT_RESPONSE = "incident_response"
    OPSEC_EVALUATION = "opsec_evaluation"

@dataclass
class MetaTask:
    """Structure for meta-learning tasks"""
    task_id: str
    task_type: TaskType
    name: str
    description: str
    support_set: List[Dict[str, Any]]  # Few examples for learning
    query_set: List[Dict[str, Any]]    # Examples for evaluation
    domain: str  # Cybersecurity domain (malware, network, etc.)
    difficulty: float  # Task difficulty (0-1)
    created_at: datetime
    metadata: Dict[str, Any]
    
    def __len__(self) -> int:
        return len(self.support_set) + len(self.query_set)

class EpisodeBuffer:
    """Buffer for storing meta-learning episodes"""
    
    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self.episodes: List[MetaTask] = []
        self.episode_index = 0
    
    def add_episode(self, episode: MetaTask):
        """Add new episode to buffer"""
        if len(self.episodes) >= self.capacity:
            self.episodes[self.episode_index] = episode
            self.episode_index = (self.episode_index + 1) % self.capacity
        else:
            self.episodes.append(episode)
    
    def sample_episodes(self, batch_size: int) -> List[MetaTask]:
        """Sample batch of episodes for meta-training"""
        if len(self.episodes) < batch_size:
            return self.episodes.copy()
        return random.sample(self.episodes, batch_size)
    
    def get_episodes_by_domain(self, domain: str) -> List[MetaTask]:
        """Get episodes from specific domain"""
        return [ep for ep in self.episodes if ep.domain == domain]
    
    def get_episodes_by_type(self, task_type: TaskType) -> List[MetaTask]:
        """Get episodes of specific task type"""
        return [ep for ep in self.episodes if ep.task_type == task_type]

class MAMLOptimizer:
    """Model-Agnostic Meta-Learning optimizer"""
    
    def __init__(self, 
                 model: nn.Module,
                 meta_lr: float = 1e-3,
                 inner_lr: float = 1e-2,
                 inner_steps: int = 5):
        
        self.model = model
        self.meta_lr = meta_lr
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        
        # Meta-optimizer
        self.meta_optimizer = torch.optim.Adam(model.parameters(), lr=meta_lr)
        
    def meta_train_step(self, episode_batch: List[MetaTask]) -> Dict[str, float]:
        """Perform one meta-training step"""
        self.meta_optimizer.zero_grad()
        
        total_loss = 0.0
        total_accuracy = 0.0
        num_tasks = len(episode_batch)
        
        for task in episode_batch:
            # Clone model for inner loop
            model_copy = self._clone_model()
            
            # Inner loop adaptation
            adapted_model, adaptation_loss = self._inner_loop_adaptation(
                model_copy, task.support_set
            )
            
            # Evaluate on query set
            query_loss, query_accuracy = self._evaluate_on_query_set(
                adapted_model, task.query_set
            )
            
            total_loss += query_loss
            total_accuracy += query_accuracy
        
        # Meta-gradient update
        avg_loss = total_loss / num_tasks
        avg_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.meta_optimizer.step()
        
        return {
            'meta_loss': avg_loss.item(),
            'meta_accuracy': total_accuracy / num_tasks,
            'num_tasks': num_tasks
        }
    
    def _clone_model(self) -> nn.Module:
        """Create a copy of the model for inner loop"""
        model_copy = type(self.model)()
        model_copy.load_state_dict(self.model.state_dict())
        return model_copy
    
    def _inner_loop_adaptation(self, 
                             model: nn.Module, 
                             support_set: List[Dict[str, Any]]) -> Tuple[nn.Module, float]:
        """Perform inner loop adaptation on support set"""
        optimizer = torch.optim.SGD(model.parameters(), lr=self.inner_lr)
        
        total_loss = 0.0
        
        for step in range(self.inner_steps):
            optimizer.zero_grad()
            
            # Sample batch from support set
            batch = random.sample(support_set, min(4, len(support_set)))
            
            # Compute loss
            loss = self._compute_task_loss(model, batch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        return model, total_loss / self.inner_steps
    
    def _evaluate_on_query_set(self, 
                             model: nn.Module, 
                             query_set: List[Dict[str, Any]]) -> Tuple[torch.Tensor, float]:
        """Evaluate adapted model on query set"""
        model.eval()
        
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for query_example in query_set:
                loss = self._compute_task_loss(model, [query_example])
                total_loss += loss.item()
                
                # Compute accuracy (simplified)
                prediction = self._get_prediction(model, query_example)
                if prediction == query_example.get('label'):
                    correct_predictions += 1
                total_predictions += 1
        
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        return torch.tensor(total_loss / len(query_set)), accuracy
    
    def _compute_task_loss(self, model: nn.Module, batch: List[Dict[str, Any]]) -> torch.Tensor:
        """Compute loss for a task batch"""
        # Simplified loss computation - implement actual loss based on task type
        return torch.tensor(0.1, requires_grad=True)
    
    def _get_prediction(self, model: nn.Module, example: Dict[str, Any]) -> Any:
        """Get model prediction for an example"""
        # Simplified prediction - implement actual inference
        return "predicted_label"

class CyberSecurityTaskGenerator:
    """Generates meta-learning tasks from cybersecurity data"""
    
    def __init__(self, 
                 tokenizer,
                 min_support_size: int = 5,
                 max_support_size: int = 20,
                 min_query_size: int = 5,
                 max_query_size: int = 15):
        
        self.tokenizer = tokenizer
        self.min_support_size = min_support_size
        self.max_support_size = max_support_size
        self.min_query_size = min_query_size
        self.max_query_size = max_query_size
        
        # Task templates for different cybersecurity domains
        self.task_templates = {
            TaskType.THREAT_CLASSIFICATION: self._generate_threat_classification_task,
            TaskType.ATTACK_PREDICTION: self._generate_attack_prediction_task,
            TaskType.IOC_DETECTION: self._generate_ioc_detection_task,
            TaskType.VULNERABILITY_ASSESSMENT: self._generate_vuln_assessment_task,
            TaskType.INCIDENT_RESPONSE: self._generate_incident_response_task,
            TaskType.OPSEC_EVALUATION: self._generate_opsec_evaluation_task
        }
    
    def generate_task_from_events(self, 
                                events: List[LearningEvent],
                                task_type: TaskType,
                                domain: str = "general") -> Optional[MetaTask]:
        """Generate meta-learning task from learning events"""
        
        if len(events) < self.min_support_size + self.min_query_size:
            logger.warning(f"Insufficient events for task generation: {len(events)}")
            return None
        
        try:
            # Filter events by relevance to task type
            relevant_events = self._filter_events_by_task_type(events, task_type)
            
            if len(relevant_events) < self.min_support_size + self.min_query_size:
                return None
            
            # Split into support and query sets
            random.shuffle(relevant_events)
            support_size = random.randint(self.min_support_size, 
                                        min(self.max_support_size, len(relevant_events) // 2))
            
            support_events = relevant_events[:support_size]
            query_events = relevant_events[support_size:support_size + self.max_query_size]
            
            # Convert events to task format
            support_set = [self._event_to_task_example(event, task_type) for event in support_events]
            query_set = [self._event_to_task_example(event, task_type) for event in query_events]
            
            # Generate task using appropriate template
            generator_func = self.task_templates[task_type]
            return generator_func(support_set, query_set, domain)
            
        except Exception as e:
            logger.error(f"Error generating task: {str(e)}")
            return None
    
    def _filter_events_by_task_type(self, 
                                  events: List[LearningEvent], 
                                  task_type: TaskType) -> List[LearningEvent]:
        """Filter events relevant to specific task type"""
        
        relevant_event_types = {
            TaskType.THREAT_CLASSIFICATION: [
                LearningEventType.NEW_THREAT_INTELLIGENCE,
                LearningEventType.SECURITY_INCIDENT
            ],
            TaskType.ATTACK_PREDICTION: [
                LearningEventType.AGENT_SUCCESS,
                LearningEventType.AGENT_FAILURE,
                LearningEventType.SECURITY_INCIDENT
            ],
            TaskType.IOC_DETECTION: [
                LearningEventType.NEW_THREAT_INTELLIGENCE,
                LearningEventType.FALSE_POSITIVE
            ],
            TaskType.OPSEC_EVALUATION: [
                LearningEventType.OPSEC_VIOLATION,
                LearningEventType.AGENT_SUCCESS
            ]
        }
        
        target_types = relevant_event_types.get(task_type, [])
        return [event for event in events if event.event_type in target_types]
    
    def _event_to_task_example(self, 
                             event: LearningEvent, 
                             task_type: TaskType) -> Dict[str, Any]:
        """Convert learning event to task example"""
        
        base_example = {
            'id': event.event_id,
            'input': self._extract_input_from_event(event, task_type),
            'label': self._extract_label_from_event(event, task_type),
            'metadata': {
                'source': event.source,
                'timestamp': event.timestamp.isoformat(),
                'confidence': event.confidence,
                'priority': event.priority
            }
        }
        
        return base_example
    
    def _extract_input_from_event(self, event: LearningEvent, task_type: TaskType) -> str:
        """Extract input text from event based on task type"""
        
        if task_type == TaskType.THREAT_CLASSIFICATION:
            return event.context.get('threat_description', '')
        elif task_type == TaskType.ATTACK_PREDICTION:
            return f"Context: {event.context.get('context', '')} Previous actions: {event.context.get('actions', [])}"
        elif task_type == TaskType.IOC_DETECTION:
            return event.context.get('network_data', '') + " " + event.context.get('log_data', '')
        elif task_type == TaskType.OPSEC_EVALUATION:
            return f"Query: {event.context.get('query', '')} Response: {event.context.get('response', '')}"
        else:
            return json.dumps(event.context)
    
    def _extract_label_from_event(self, event: LearningEvent, task_type: TaskType) -> str:
        """Extract label from event based on task type"""
        
        if task_type == TaskType.THREAT_CLASSIFICATION:
            return event.context.get('threat_type', 'unknown')
        elif task_type == TaskType.ATTACK_PREDICTION:
            return "success" if event.event_type == LearningEventType.AGENT_SUCCESS else "failure"
        elif task_type == TaskType.IOC_DETECTION:
            return "positive" if event.event_type == LearningEventType.NEW_THREAT_INTELLIGENCE else "negative"
        elif task_type == TaskType.OPSEC_EVALUATION:
            return "violation" if event.event_type == LearningEventType.OPSEC_VIOLATION else "safe"
        else:
            return event.event_type.value
    
    def _generate_threat_classification_task(self, 
                                           support_set: List[Dict[str, Any]], 
                                           query_set: List[Dict[str, Any]], 
                                           domain: str) -> MetaTask:
        """Generate threat classification meta-task"""
        
        return MetaTask(
            task_id=f"threat_class_{datetime.now().timestamp()}",
            task_type=TaskType.THREAT_CLASSIFICATION,
            name="Threat Classification",
            description="Classify cybersecurity threats based on indicators and behavior",
            support_set=support_set,
            query_set=query_set,
            domain=domain,
            difficulty=0.7,
            created_at=datetime.now(),
            metadata={
                'threat_categories': list(set(ex['label'] for ex in support_set + query_set)),
                'num_classes': len(set(ex['label'] for ex in support_set + query_set))
            }
        )
    
    def _generate_attack_prediction_task(self, 
                                       support_set: List[Dict[str, Any]], 
                                       query_set: List[Dict[str, Any]], 
                                       domain: str) -> MetaTask:
        """Generate attack prediction meta-task"""
        
        return MetaTask(
            task_id=f"attack_pred_{datetime.now().timestamp()}",
            task_type=TaskType.ATTACK_PREDICTION,
            name="Attack Outcome Prediction",
            description="Predict the success/failure of attack strategies",
            support_set=support_set,
            query_set=query_set,
            domain=domain,
            difficulty=0.8,
            created_at=datetime.now(),
            metadata={
                'prediction_horizon': '1_step',
                'success_rate': len([ex for ex in support_set if ex['label'] == 'success']) / len(support_set)
            }
        )
    
    def _generate_ioc_detection_task(self, 
                                   support_set: List[Dict[str, Any]], 
                                   query_set: List[Dict[str, Any]], 
                                   domain: str) -> MetaTask:
        """Generate IoC detection meta-task"""
        
        return MetaTask(
            task_id=f"ioc_detect_{datetime.now().timestamp()}",
            task_type=TaskType.IOC_DETECTION,
            name="Indicator of Compromise Detection",
            description="Detect indicators of compromise in network/system data",
            support_set=support_set,
            query_set=query_set,
            domain=domain,
            difficulty=0.6,
            created_at=datetime.now(),
            metadata={
                'ioc_types': ['ip', 'domain', 'hash', 'registry', 'file_path'],
                'detection_accuracy_target': 0.95
            }
        )
    
    def _generate_vuln_assessment_task(self, 
                                     support_set: List[Dict[str, Any]], 
                                     query_set: List[Dict[str, Any]], 
                                     domain: str) -> MetaTask:
        """Generate vulnerability assessment meta-task"""
        
        return MetaTask(
            task_id=f"vuln_assess_{datetime.now().timestamp()}",
            task_type=TaskType.VULNERABILITY_ASSESSMENT,
            name="Vulnerability Assessment",
            description="Assess and prioritize system vulnerabilities",
            support_set=support_set,
            query_set=query_set,
            domain=domain,
            difficulty=0.75,
            created_at=datetime.now(),
            metadata={
                'severity_levels': ['low', 'medium', 'high', 'critical'],
                'assessment_framework': 'CVSS'
            }
        )
    
    def _generate_incident_response_task(self, 
                                       support_set: List[Dict[str, Any]], 
                                       query_set: List[Dict[str, Any]], 
                                       domain: str) -> MetaTask:
        """Generate incident response meta-task"""
        
        return MetaTask(
            task_id=f"incident_resp_{datetime.now().timestamp()}",
            task_type=TaskType.INCIDENT_RESPONSE,
            name="Incident Response Planning",
            description="Generate appropriate incident response strategies",
            support_set=support_set,
            query_set=query_set,
            domain=domain,
            difficulty=0.9,
            created_at=datetime.now(),
            metadata={
                'response_phases': ['preparation', 'identification', 'containment', 'eradication', 'recovery'],
                'incident_types': list(set(ex.get('metadata', {}).get('incident_type', 'unknown') 
                                         for ex in support_set + query_set))
            }
        )
    
    def _generate_opsec_evaluation_task(self, 
                                      support_set: List[Dict[str, Any]], 
                                      query_set: List[Dict[str, Any]], 
                                      domain: str) -> MetaTask:
        """Generate OPSEC evaluation meta-task"""
        
        return MetaTask(
            task_id=f"opsec_eval_{datetime.now().timestamp()}",
            task_type=TaskType.OPSEC_EVALUATION,
            name="OPSEC Violation Detection",
            description="Evaluate queries and responses for OPSEC violations",
            support_set=support_set,
            query_set=query_set,
            domain=domain,
            difficulty=0.85,
            created_at=datetime.now(),
            metadata={
                'violation_types': ['information_disclosure', 'attribution_risk', 'capability_exposure'],
                'stealth_score_threshold': 0.8
            }
        )

class MetaLearningManager:
    """Main manager for meta-learning in Cyber-LLM"""
    
    def __init__(self,
                 model,
                 tokenizer,
                 strategy: MetaLearningStrategy = MetaLearningStrategy.MAML,
                 episode_buffer_size: int = 1000,
                 meta_batch_size: int = 4):
        
        self.model = model
        self.tokenizer = tokenizer
        self.strategy = strategy
        self.meta_batch_size = meta_batch_size
        
        # Components
        self.episode_buffer = EpisodeBuffer(episode_buffer_size)
        self.task_generator = CyberSecurityTaskGenerator(tokenizer)
        
        # Strategy-specific optimizers
        if strategy == MetaLearningStrategy.MAML:
            self.optimizer = MAMLOptimizer(model)
        else:
            raise NotImplementedError(f"Strategy {strategy} not yet implemented")
        
        # Metrics tracking
        self.meta_learning_metrics = {
            'total_episodes': 0,
            'total_meta_updates': 0,
            'average_adaptation_time': 0.0,
            'task_performance': defaultdict(list),
            'domain_performance': defaultdict(list)
        }
        
        logger.info(f"MetaLearningManager initialized with strategy: {strategy.value}")
    
    async def add_learning_episodes(self, events: List[LearningEvent]) -> int:
        """Generate and add meta-learning episodes from events"""
        
        episodes_created = 0
        
        # Group events by potential task types
        for task_type in TaskType:
            try:
                task = self.task_generator.generate_task_from_events(
                    events, task_type, domain="cybersecurity"
                )
                
                if task:
                    self.episode_buffer.add_episode(task)
                    episodes_created += 1
                    
                    logger.info(f"Created meta-task: {task.name} ({task_type.value})")
                    
            except Exception as e:
                logger.error(f"Error creating task for {task_type}: {str(e)}")
        
        self.meta_learning_metrics['total_episodes'] += episodes_created
        return episodes_created
    
    async def meta_train_step(self) -> Dict[str, Any]:
        """Perform one meta-training step"""
        
        # Sample episode batch
        episode_batch = self.episode_buffer.sample_episodes(self.meta_batch_size)
        
        if len(episode_batch) < self.meta_batch_size:
            logger.warning(f"Insufficient episodes for meta-training: {len(episode_batch)}")
            return {'success': False, 'reason': 'insufficient_episodes'}
        
        try:
            # Perform meta-training step based on strategy
            if self.strategy == MetaLearningStrategy.MAML:
                results = self.optimizer.meta_train_step(episode_batch)
            else:
                raise NotImplementedError(f"Meta-training not implemented for {self.strategy}")
            
            # Update metrics
            self.meta_learning_metrics['total_meta_updates'] += 1
            
            # Track performance by task type and domain
            for episode in episode_batch:
                self.meta_learning_metrics['task_performance'][episode.task_type.value].append(
                    results.get('meta_accuracy', 0.0)
                )
                self.meta_learning_metrics['domain_performance'][episode.domain].append(
                    results.get('meta_accuracy', 0.0)
                )
            
            logger.info(f"Meta-training step completed. Meta-loss: {results.get('meta_loss', 0.0):.4f}")
            
            return {
                'success': True,
                'meta_loss': results.get('meta_loss', 0.0),
                'meta_accuracy': results.get('meta_accuracy', 0.0),
                'episodes_processed': len(episode_batch)
            }
            
        except Exception as e:
            logger.error(f"Meta-training step failed: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    async def rapid_adaptation(self, 
                             new_task_examples: List[Dict[str, Any]],
                             task_type: TaskType,
                             adaptation_steps: int = 5) -> Dict[str, Any]:
        """Rapidly adapt to new task with few examples"""
        
        try:
            start_time = datetime.now()
            
            # Create temporary task for adaptation
            adaptation_task = MetaTask(
                task_id=f"adapt_{datetime.now().timestamp()}",
                task_type=task_type,
                name=f"Rapid Adaptation - {task_type.value}",
                description="Rapid adaptation to new task",
                support_set=new_task_examples[:len(new_task_examples)//2],
                query_set=new_task_examples[len(new_task_examples)//2:],
                domain="adaptation",
                difficulty=0.8,
                created_at=datetime.now(),
                metadata={'adaptation_mode': True}
            )
            
            # Perform adaptation using inner loop
            if self.strategy == MetaLearningStrategy.MAML:
                adapted_model, adaptation_loss = self.optimizer._inner_loop_adaptation(
                    self.optimizer._clone_model(), 
                    adaptation_task.support_set
                )
                
                # Evaluate adaptation
                query_loss, query_accuracy = self.optimizer._evaluate_on_query_set(
                    adapted_model, adaptation_task.query_set
                )
                
                adaptation_time = (datetime.now() - start_time).total_seconds()
                
                # Update metrics
                self.meta_learning_metrics['average_adaptation_time'] = (
                    (self.meta_learning_metrics['average_adaptation_time'] * 
                     self.meta_learning_metrics['total_meta_updates'] + adaptation_time) /
                    (self.meta_learning_metrics['total_meta_updates'] + 1)
                )
                
                logger.info(f"Rapid adaptation completed in {adaptation_time:.2f}s. "
                           f"Query accuracy: {query_accuracy:.4f}")
                
                return {
                    'success': True,
                    'adaptation_time': adaptation_time,
                    'adaptation_loss': adaptation_loss,
                    'query_accuracy': query_accuracy,
                    'adapted_model': adapted_model
                }
            
        except Exception as e:
            logger.error(f"Rapid adaptation failed: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def get_meta_learning_statistics(self) -> Dict[str, Any]:
        """Get comprehensive meta-learning statistics"""
        
        task_performance_summary = {}
        for task_type, scores in self.meta_learning_metrics['task_performance'].items():
            if scores:
                task_performance_summary[task_type] = {
                    'average_performance': np.mean(scores),
                    'std_performance': np.std(scores),
                    'num_episodes': len(scores),
                    'best_performance': max(scores),
                    'worst_performance': min(scores)
                }
        
        domain_performance_summary = {}
        for domain, scores in self.meta_learning_metrics['domain_performance'].items():
            if scores:
                domain_performance_summary[domain] = {
                    'average_performance': np.mean(scores),
                    'std_performance': np.std(scores),
                    'num_episodes': len(scores)
                }
        
        return {
            'meta_learning_strategy': self.strategy.value,
            'total_episodes': self.meta_learning_metrics['total_episodes'],
            'total_meta_updates': self.meta_learning_metrics['total_meta_updates'],
            'average_adaptation_time': self.meta_learning_metrics['average_adaptation_time'],
            'episodes_in_buffer': len(self.episode_buffer.episodes),
            'task_performance': task_performance_summary,
            'domain_performance': domain_performance_summary,
            'buffer_capacity': self.episode_buffer.capacity
        }
    
    async def continuous_meta_learning_loop(self):
        """Continuous meta-learning loop"""
        
        logger.info("Starting continuous meta-learning loop")
        
        while True:
            try:
                # Perform meta-training if enough episodes available
                if len(self.episode_buffer.episodes) >= self.meta_batch_size:
                    await self.meta_train_step()
                
                # Wait before next iteration
                await asyncio.sleep(300)  # 5 minutes
                
            except Exception as e:
                logger.error(f"Error in meta-learning loop: {str(e)}")
                await asyncio.sleep(600)  # Wait longer on error

# Factory function
def create_meta_learning_manager(**kwargs) -> MetaLearningManager:
    """Create meta-learning manager with default configuration"""
    return MetaLearningManager(**kwargs)
