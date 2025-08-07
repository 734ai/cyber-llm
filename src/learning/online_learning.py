"""
Online Learning System for Cyber-LLM
Enables real-time model updates from operational feedback and new threat intelligence.

Author: Muzan Sano <sanosensei36@gmail.com>
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import redis
from pydantic import BaseModel, Field

from ..utils.logging_system import CyberLLMLogger
from ..utils.secrets_manager import SecretsManager

# Configure logging
logger = CyberLLMLogger(__name__).get_logger()

class LearningEventType(Enum):
    """Types of learning events that can trigger model updates"""
    FEEDBACK_POSITIVE = "feedback_positive"
    FEEDBACK_NEGATIVE = "feedback_negative"
    NEW_THREAT_INTELLIGENCE = "new_threat_intel"
    SECURITY_INCIDENT = "security_incident"
    AGENT_SUCCESS = "agent_success"
    AGENT_FAILURE = "agent_failure"
    OPSEC_VIOLATION = "opsec_violation"
    FALSE_POSITIVE = "false_positive"

@dataclass
class LearningEvent:
    """Structure for learning events"""
    event_id: str
    event_type: LearningEventType
    timestamp: datetime
    source: str  # Which agent or system generated this event
    context: Dict[str, Any]  # Relevant context for learning
    feedback_score: Optional[float] = None  # Human feedback score (0-1)
    confidence: float = 1.0  # Confidence in this event
    priority: int = 1  # Priority level (1=low, 5=critical)
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['event_type'] = self.event_type.value
        return data

class OnlineDataset(Dataset):
    """Dataset for online learning from streaming events"""
    
    def __init__(self, events: List[LearningEvent], tokenizer, max_length: int = 512):
        self.events = events
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self) -> int:
        return len(self.events)
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        event = self.events[idx]
        
        # Convert learning event to training sample
        context_text = self._event_to_text(event)
        
        # Tokenize
        encoding = self.tokenizer(
            context_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': encoding['input_ids'].squeeze(),
            'event_weight': torch.tensor(event.confidence * event.priority, dtype=torch.float32)
        }
    
    def _event_to_text(self, event: LearningEvent) -> str:
        """Convert learning event to training text"""
        if event.event_type == LearningEventType.FEEDBACK_POSITIVE:
            return f"[POSITIVE_FEEDBACK] Context: {event.context.get('query', '')} Response: {event.context.get('response', '')} Score: {event.feedback_score}"
        elif event.event_type == LearningEventType.FEEDBACK_NEGATIVE:
            return f"[NEGATIVE_FEEDBACK] Context: {event.context.get('query', '')} Response: {event.context.get('response', '')} Score: {event.feedback_score}"
        elif event.event_type == LearningEventType.NEW_THREAT_INTELLIGENCE:
            return f"[THREAT_INTEL] {event.context.get('threat_description', '')} TTPs: {event.context.get('ttps', [])}"
        elif event.event_type == LearningEventType.SECURITY_INCIDENT:
            return f"[INCIDENT] {event.context.get('incident_description', '')} Response: {event.context.get('response_actions', [])}"
        else:
            return f"[{event.event_type.value.upper()}] {json.dumps(event.context)}"

class OnlineLearningManager:
    """Manages online learning process for Cyber-LLM"""
    
    def __init__(self, 
                 model_name: str = "microsoft/DialoGPT-medium",
                 redis_host: str = "localhost",
                 redis_port: int = 6379,
                 learning_rate: float = 1e-5,
                 batch_size: int = 4,
                 update_frequency: int = 100,  # Update after N events
                 max_events_memory: int = 10000):
        
        self.model_name = model_name
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.update_frequency = update_frequency
        self.max_events_memory = max_events_memory
        
        # Initialize components
        self.secrets_manager = SecretsManager()
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Learning state
        self.learning_events: List[LearningEvent] = []
        self.total_events_processed = 0
        self.last_update_time = datetime.now()
        
        # Performance tracking
        self.learning_metrics = {
            'total_updates': 0,
            'successful_updates': 0,
            'failed_updates': 0,
            'average_loss': 0.0,
            'learning_rate_history': []
        }
        
        logger.info(f"OnlineLearningManager initialized with model: {model_name}")
    
    async def add_learning_event(self, event: LearningEvent) -> None:
        """Add a new learning event to the queue"""
        try:
            # Store event in memory
            self.learning_events.append(event)
            
            # Store event in Redis for persistence
            event_key = f"learning_event:{event.event_id}"
            await self._store_event_redis(event_key, event)
            
            # Maintain memory limit
            if len(self.learning_events) > self.max_events_memory:
                self.learning_events.pop(0)
            
            self.total_events_processed += 1
            
            logger.info(f"Added learning event: {event.event_type.value} from {event.source}")
            
            # Trigger update if threshold reached
            if len(self.learning_events) >= self.update_frequency:
                await self.trigger_model_update()
                
        except Exception as e:
            logger.error(f"Error adding learning event: {str(e)}")
    
    async def trigger_model_update(self) -> Dict[str, Any]:
        """Trigger an online model update based on accumulated events"""
        if not self.learning_events:
            logger.warning("No learning events available for model update")
            return {'success': False, 'reason': 'no_events'}
        
        try:
            logger.info(f"Starting online model update with {len(self.learning_events)} events")
            
            # Prepare dataset
            dataset = OnlineDataset(self.learning_events, self.tokenizer)
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            
            # Configure optimizer
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
            
            # Training loop
            self.model.train()
            total_loss = 0.0
            num_batches = 0
            
            for batch in dataloader:
                optimizer.zero_grad()
                
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels']
                )
                
                # Apply event weights to loss
                loss = outputs.loss * batch['event_weight'].mean()
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
            
            # Update metrics
            self.learning_metrics['total_updates'] += 1
            self.learning_metrics['successful_updates'] += 1
            self.learning_metrics['average_loss'] = avg_loss
            self.learning_metrics['learning_rate_history'].append(self.learning_rate)
            
            # Clear processed events
            self.learning_events.clear()
            self.last_update_time = datetime.now()
            
            logger.info(f"Online model update completed. Average loss: {avg_loss:.4f}")
            
            # Store updated model (in production, would save to model registry)
            await self._save_model_checkpoint()
            
            return {
                'success': True,
                'average_loss': avg_loss,
                'events_processed': num_batches * self.batch_size,
                'timestamp': self.last_update_time.isoformat()
            }
            
        except Exception as e:
            self.learning_metrics['failed_updates'] += 1
            logger.error(f"Online model update failed: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    async def process_feedback(self, 
                             query: str, 
                             response: str, 
                             feedback_score: float,
                             source: str = "human_feedback") -> None:
        """Process human feedback for online learning"""
        event_type = LearningEventType.FEEDBACK_POSITIVE if feedback_score > 0.5 else LearningEventType.FEEDBACK_NEGATIVE
        
        event = LearningEvent(
            event_id=f"feedback_{datetime.now().timestamp()}",
            event_type=event_type,
            timestamp=datetime.now(),
            source=source,
            context={
                'query': query,
                'response': response,
                'feedback_score': feedback_score
            },
            feedback_score=feedback_score,
            priority=3 if abs(feedback_score - 0.5) > 0.3 else 2  # Higher priority for strong feedback
        )
        
        await self.add_learning_event(event)
    
    async def process_threat_intelligence(self, 
                                        threat_data: Dict[str, Any],
                                        source: str = "threat_intel") -> None:
        """Process new threat intelligence for online learning"""
        event = LearningEvent(
            event_id=f"threat_{datetime.now().timestamp()}",
            event_type=LearningEventType.NEW_THREAT_INTELLIGENCE,
            timestamp=datetime.now(),
            source=source,
            context=threat_data,
            priority=4,  # High priority for new threats
            confidence=threat_data.get('confidence', 0.8)
        )
        
        await self.add_learning_event(event)
    
    async def process_agent_performance(self,
                                      agent_name: str,
                                      task: str,
                                      success: bool,
                                      performance_data: Dict[str, Any]) -> None:
        """Process agent performance data for online learning"""
        event_type = LearningEventType.AGENT_SUCCESS if success else LearningEventType.AGENT_FAILURE
        
        event = LearningEvent(
            event_id=f"agent_{agent_name}_{datetime.now().timestamp()}",
            event_type=event_type,
            timestamp=datetime.now(),
            source=agent_name,
            context={
                'task': task,
                'performance_data': performance_data,
                'success': success
            },
            priority=2 if success else 3,  # Higher priority for failures to learn from
            confidence=performance_data.get('confidence', 0.9)
        )
        
        await self.add_learning_event(event)
    
    async def get_learning_statistics(self) -> Dict[str, Any]:
        """Get comprehensive learning statistics"""
        return {
            'total_events_processed': self.total_events_processed,
            'current_events_in_memory': len(self.learning_events),
            'last_update_time': self.last_update_time.isoformat(),
            'metrics': self.learning_metrics,
            'event_type_distribution': self._get_event_type_distribution(),
            'learning_rate': self.learning_rate,
            'update_frequency': self.update_frequency
        }
    
    def _get_event_type_distribution(self) -> Dict[str, int]:
        """Get distribution of event types in current memory"""
        distribution = {}
        for event in self.learning_events:
            event_type = event.event_type.value
            distribution[event_type] = distribution.get(event_type, 0) + 1
        return distribution
    
    async def _store_event_redis(self, key: str, event: LearningEvent) -> None:
        """Store learning event in Redis for persistence"""
        try:
            event_data = json.dumps(event.to_dict())
            self.redis_client.setex(key, timedelta(days=7), event_data)
        except Exception as e:
            logger.warning(f"Failed to store event in Redis: {str(e)}")
    
    async def _save_model_checkpoint(self) -> None:
        """Save model checkpoint after online learning update"""
        try:
            checkpoint_path = f"models/online_learning_checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.model.save_pretrained(checkpoint_path)
            self.tokenizer.save_pretrained(checkpoint_path)
            logger.info(f"Model checkpoint saved to {checkpoint_path}")
        except Exception as e:
            logger.error(f"Failed to save model checkpoint: {str(e)}")

# Factory function for easy instantiation
def create_online_learning_manager(**kwargs) -> OnlineLearningManager:
    """Factory function to create OnlineLearningManager with default configuration"""
    return OnlineLearningManager(**kwargs)
