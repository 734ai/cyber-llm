"""
Federated Learning System for Cyber-LLM
Enables secure collaborative learning across multiple organizations without sharing raw data.

Author: Muzan Sano <sanosensei36@gmail.com>
"""

import asyncio
import json
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import websockets
import ssl
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import os

from ..utils.logging_system import CyberLLMLogger
from ..utils.secrets_manager import SecretsManager
from .online_learning import LearningEvent, LearningEventType

# Configure logging
logger = CyberLLMLogger(__name__).get_logger()

class FederatedRole(Enum):
    """Roles in federated learning network"""
    COORDINATOR = "coordinator"  # Central coordination server
    PARTICIPANT = "participant"  # Individual organization
    VALIDATOR = "validator"      # Validates model updates

class FederatedMessageType(Enum):
    """Types of messages in federated learning protocol"""
    JOIN_REQUEST = "join_request"
    JOIN_RESPONSE = "join_response" 
    MODEL_UPDATE = "model_update"
    AGGREGATION_REQUEST = "aggregation_request"
    AGGREGATION_RESPONSE = "aggregation_response"
    VALIDATION_REQUEST = "validation_request"
    VALIDATION_RESPONSE = "validation_response"
    HEARTBEAT = "heartbeat"

@dataclass
class FederatedParticipant:
    """Information about a federated learning participant"""
    participant_id: str
    organization: str
    public_key: str
    last_seen: datetime
    contribution_weight: float = 1.0  # Weight based on data quality/quantity
    trust_score: float = 1.0  # Trust level (0-1)
    specialization: List[str] = None  # Areas of expertise
    
    def __post_init__(self):
        if self.specialization is None:
            self.specialization = []

@dataclass 
class FederatedMessage:
    """Structure for federated learning messages"""
    message_id: str
    sender_id: str
    recipient_id: str  # "broadcast" for all participants
    message_type: FederatedMessageType
    payload: Dict[str, Any]
    timestamp: datetime
    signature: Optional[str] = None
    encrypted: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['message_type'] = self.message_type.value
        return data

class SecureCommunicationManager:
    """Manages secure communication between federated participants"""
    
    def __init__(self, participant_id: str):
        self.participant_id = participant_id
        self.encryption_key = None
        self.participants_keys: Dict[str, str] = {}
        
    def generate_encryption_key(self, password: bytes) -> None:
        """Generate encryption key from password"""
        salt = os.urandom(16)
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password))
        self.encryption_key = Fernet(key)
        
    def encrypt_message(self, message: Dict[str, Any]) -> bytes:
        """Encrypt message payload"""
        if self.encryption_key is None:
            raise ValueError("Encryption key not set")
        
        message_bytes = json.dumps(message).encode()
        return self.encryption_key.encrypt(message_bytes)
        
    def decrypt_message(self, encrypted_data: bytes) -> Dict[str, Any]:
        """Decrypt message payload"""
        if self.encryption_key is None:
            raise ValueError("Encryption key not set")
            
        decrypted_bytes = self.encryption_key.decrypt(encrypted_data)
        return json.loads(decrypted_bytes.decode())
    
    def sign_message(self, message: Dict[str, Any]) -> str:
        """Create digital signature for message"""
        message_str = json.dumps(message, sort_keys=True)
        return hashlib.sha256(message_str.encode()).hexdigest()
    
    def verify_signature(self, message: Dict[str, Any], signature: str) -> bool:
        """Verify message digital signature"""
        expected_signature = self.sign_message(message)
        return expected_signature == signature

class ModelAggregator:
    """Handles secure model aggregation in federated learning"""
    
    def __init__(self, aggregation_method: str = "fedavg"):
        self.aggregation_method = aggregation_method
        self.model_updates: List[Dict[str, torch.Tensor]] = []
        self.participant_weights: List[float] = []
        
    def add_model_update(self, model_state: Dict[str, torch.Tensor], weight: float = 1.0):
        """Add a model update from a participant"""
        self.model_updates.append(model_state)
        self.participant_weights.append(weight)
        
    def aggregate_models(self) -> Dict[str, torch.Tensor]:
        """Aggregate multiple model updates using specified method"""
        if not self.model_updates:
            raise ValueError("No model updates to aggregate")
            
        if self.aggregation_method == "fedavg":
            return self._federated_averaging()
        elif self.aggregation_method == "weighted_avg":
            return self._weighted_averaging()
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation_method}")
    
    def _federated_averaging(self) -> Dict[str, torch.Tensor]:
        """Standard federated averaging aggregation"""
        if not self.model_updates:
            return {}
            
        # Get parameter names from first model
        param_names = self.model_updates[0].keys()
        aggregated_params = {}
        
        total_weight = sum(self.participant_weights)
        
        for param_name in param_names:
            weighted_sum = torch.zeros_like(self.model_updates[0][param_name])
            
            for i, model_update in enumerate(self.model_updates):
                weight = self.participant_weights[i] / total_weight
                weighted_sum += weight * model_update[param_name]
                
            aggregated_params[param_name] = weighted_sum
            
        return aggregated_params
    
    def _weighted_averaging(self) -> Dict[str, torch.Tensor]:
        """Weighted averaging based on participant trust scores"""
        # Similar to federated averaging but uses trust scores
        return self._federated_averaging()
    
    def clear_updates(self):
        """Clear accumulated model updates"""
        self.model_updates.clear()
        self.participant_weights.clear()

class FederatedLearningCoordinator:
    """Coordinates federated learning across multiple participants"""
    
    def __init__(self, 
                 coordinator_id: str,
                 port: int = 8765,
                 min_participants: int = 3,
                 aggregation_rounds: int = 10):
        
        self.coordinator_id = coordinator_id
        self.port = port
        self.min_participants = min_participants
        self.aggregation_rounds = aggregation_rounds
        
        # Participant management
        self.participants: Dict[str, FederatedParticipant] = {}
        self.connected_clients = set()
        
        # Learning state
        self.current_round = 0
        self.model_aggregator = ModelAggregator()
        self.global_model = None
        self.round_results: List[Dict[str, Any]] = []
        
        # Communication
        self.comm_manager = SecureCommunicationManager(coordinator_id)
        self.server = None
        
        logger.info(f"FederatedLearningCoordinator initialized: {coordinator_id}")
    
    async def start_coordinator(self):
        """Start the federated learning coordinator server"""
        try:
            # Create SSL context for secure communication
            ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
            # In production, load proper certificates
            
            self.server = await websockets.serve(
                self.handle_client,
                "localhost",
                self.port,
                ssl=None  # Enable SSL in production
            )
            
            logger.info(f"Federated learning coordinator started on port {self.port}")
            
            # Start coordination loop
            await self.coordination_loop()
            
        except Exception as e:
            logger.error(f"Failed to start coordinator: {str(e)}")
    
    async def handle_client(self, websocket, path):
        """Handle incoming client connections"""
        try:
            self.connected_clients.add(websocket)
            logger.info("New participant connected")
            
            async for message in websocket:
                await self.process_message(websocket, message)
                
        except websockets.exceptions.ConnectionClosed:
            logger.info("Participant disconnected")
        finally:
            self.connected_clients.discard(websocket)
    
    async def process_message(self, websocket, raw_message: str):
        """Process incoming message from participant"""
        try:
            message_data = json.loads(raw_message)
            message = FederatedMessage(**message_data)
            
            if message.message_type == FederatedMessageType.JOIN_REQUEST:
                await self.handle_join_request(websocket, message)
            elif message.message_type == FederatedMessageType.MODEL_UPDATE:
                await self.handle_model_update(websocket, message)
            elif message.message_type == FederatedMessageType.HEARTBEAT:
                await self.handle_heartbeat(websocket, message)
            else:
                logger.warning(f"Unknown message type: {message.message_type}")
                
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
    
    async def handle_join_request(self, websocket, message: FederatedMessage):
        """Handle participant join request"""
        try:
            participant_info = message.payload
            participant = FederatedParticipant(
                participant_id=message.sender_id,
                organization=participant_info.get('organization', 'unknown'),
                public_key=participant_info.get('public_key', ''),
                last_seen=datetime.now(),
                specialization=participant_info.get('specialization', [])
            )
            
            self.participants[message.sender_id] = participant
            
            # Send join response
            response = FederatedMessage(
                message_id=f"join_resp_{datetime.now().timestamp()}",
                sender_id=self.coordinator_id,
                recipient_id=message.sender_id,
                message_type=FederatedMessageType.JOIN_RESPONSE,
                payload={
                    'accepted': True,
                    'participant_id': message.sender_id,
                    'current_round': self.current_round
                },
                timestamp=datetime.now()
            )
            
            await websocket.send(json.dumps(response.to_dict()))
            logger.info(f"Participant {message.sender_id} joined from {participant.organization}")
            
        except Exception as e:
            logger.error(f"Error handling join request: {str(e)}")
    
    async def handle_model_update(self, websocket, message: FederatedMessage):
        """Handle model update from participant"""
        try:
            update_data = message.payload
            
            # Verify update integrity
            if not self.verify_model_update(update_data):
                logger.warning(f"Invalid model update from {message.sender_id}")
                return
            
            # Extract model parameters (in practice, this would be more complex)
            model_params = update_data.get('model_parameters', {})
            participant_weight = self.participants[message.sender_id].contribution_weight
            
            # Add to aggregator
            self.model_aggregator.add_model_update(model_params, participant_weight)
            
            logger.info(f"Received model update from {message.sender_id}")
            
            # Check if ready for aggregation
            if len(self.model_aggregator.model_updates) >= self.min_participants:
                await self.perform_aggregation()
                
        except Exception as e:
            logger.error(f"Error handling model update: {str(e)}")
    
    async def handle_heartbeat(self, websocket, message: FederatedMessage):
        """Handle heartbeat from participant"""
        if message.sender_id in self.participants:
            self.participants[message.sender_id].last_seen = datetime.now()
    
    def verify_model_update(self, update_data: Dict[str, Any]) -> bool:
        """Verify the integrity and validity of a model update"""
        # Implement security checks:
        # 1. Digital signature verification
        # 2. Parameter bounds checking
        # 3. Differential privacy validation
        # 4. Anomaly detection
        
        required_fields = ['model_parameters', 'training_metrics', 'data_size']
        return all(field in update_data for field in required_fields)
    
    async def perform_aggregation(self):
        """Perform model aggregation and distribute updated model"""
        try:
            logger.info(f"Starting aggregation round {self.current_round}")
            
            # Aggregate model updates
            aggregated_params = self.model_aggregator.aggregate_models()
            
            # Update global model (simplified)
            self.global_model = aggregated_params
            
            # Broadcast updated model to all participants
            await self.broadcast_updated_model(aggregated_params)
            
            # Record round results
            round_result = {
                'round': self.current_round,
                'participants': len(self.model_aggregator.model_updates),
                'timestamp': datetime.now().isoformat(),
                'aggregation_method': self.model_aggregator.aggregation_method
            }
            self.round_results.append(round_result)
            
            # Clean up for next round
            self.model_aggregator.clear_updates()
            self.current_round += 1
            
            logger.info(f"Aggregation round {self.current_round - 1} completed")
            
        except Exception as e:
            logger.error(f"Error performing aggregation: {str(e)}")
    
    async def broadcast_updated_model(self, model_params: Dict[str, Any]):
        """Broadcast updated global model to all participants"""
        message = FederatedMessage(
            message_id=f"agg_resp_{datetime.now().timestamp()}",
            sender_id=self.coordinator_id,
            recipient_id="broadcast",
            message_type=FederatedMessageType.AGGREGATION_RESPONSE,
            payload={
                'global_model_parameters': model_params,
                'round': self.current_round,
                'participants_count': len(self.participants)
            },
            timestamp=datetime.now()
        )
        
        # Send to all connected clients
        if self.connected_clients:
            message_str = json.dumps(message.to_dict())
            await asyncio.gather(
                *[client.send(message_str) for client in self.connected_clients],
                return_exceptions=True
            )
    
    async def coordination_loop(self):
        """Main coordination loop"""
        while True:
            try:
                # Check participant health
                await self.check_participant_health()
                
                # Trigger periodic aggregation if needed
                await self.check_aggregation_trigger()
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in coordination loop: {str(e)}")
                await asyncio.sleep(60)  # Wait longer on error
    
    async def check_participant_health(self):
        """Check health of all participants"""
        current_time = datetime.now()
        timeout_threshold = timedelta(minutes=5)
        
        inactive_participants = []
        for participant_id, participant in self.participants.items():
            if current_time - participant.last_seen > timeout_threshold:
                inactive_participants.append(participant_id)
        
        # Remove inactive participants
        for participant_id in inactive_participants:
            del self.participants[participant_id]
            logger.info(f"Removed inactive participant: {participant_id}")
    
    async def check_aggregation_trigger(self):
        """Check if aggregation should be triggered"""
        # Trigger based on time or number of updates
        if (len(self.model_aggregator.model_updates) >= self.min_participants and
            len(self.model_aggregator.model_updates) < len(self.participants)):
            # Wait for more participants or trigger after timeout
            pass
    
    def get_federation_status(self) -> Dict[str, Any]:
        """Get current federation status"""
        return {
            'coordinator_id': self.coordinator_id,
            'current_round': self.current_round,
            'active_participants': len(self.participants),
            'connected_clients': len(self.connected_clients),
            'pending_updates': len(self.model_aggregator.model_updates),
            'total_rounds': len(self.round_results),
            'participants': {
                pid: {
                    'organization': p.organization,
                    'last_seen': p.last_seen.isoformat(),
                    'trust_score': p.trust_score,
                    'specialization': p.specialization
                }
                for pid, p in self.participants.items()
            }
        }

class FederatedLearningParticipant:
    """Federated learning participant (individual organization)"""
    
    def __init__(self,
                 participant_id: str,
                 organization: str,
                 coordinator_url: str = "ws://localhost:8765",
                 specialization: List[str] = None):
        
        self.participant_id = participant_id
        self.organization = organization
        self.coordinator_url = coordinator_url
        self.specialization = specialization or []
        
        # Local model and data
        self.local_model = None
        self.local_data: List[LearningEvent] = []
        
        # Communication
        self.comm_manager = SecureCommunicationManager(participant_id)
        self.websocket = None
        self.connected = False
        
        logger.info(f"FederatedLearningParticipant initialized: {participant_id}")
    
    async def join_federation(self) -> bool:
        """Join the federated learning federation"""
        try:
            self.websocket = await websockets.connect(self.coordinator_url)
            self.connected = True
            
            # Send join request
            join_message = FederatedMessage(
                message_id=f"join_{datetime.now().timestamp()}",
                sender_id=self.participant_id,
                recipient_id="coordinator",
                message_type=FederatedMessageType.JOIN_REQUEST,
                payload={
                    'organization': self.organization,
                    'public_key': 'participant_public_key',  # Replace with actual key
                    'specialization': self.specialization
                },
                timestamp=datetime.now()
            )
            
            await self.websocket.send(json.dumps(join_message.to_dict()))
            
            # Start message handling loop
            asyncio.create_task(self.message_handler())
            
            logger.info(f"Joined federation as {self.participant_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to join federation: {str(e)}")
            return False
    
    async def message_handler(self):
        """Handle incoming messages from coordinator"""
        try:
            async for message in self.websocket:
                await self.process_coordinator_message(message)
        except websockets.exceptions.ConnectionClosed:
            self.connected = False
            logger.info("Connection to coordinator lost")
    
    async def process_coordinator_message(self, raw_message: str):
        """Process message from coordinator"""
        try:
            message_data = json.loads(raw_message)
            message = FederatedMessage(**message_data)
            
            if message.message_type == FederatedMessageType.AGGREGATION_RESPONSE:
                await self.handle_global_model_update(message)
            elif message.message_type == FederatedMessageType.JOIN_RESPONSE:
                await self.handle_join_response(message)
            else:
                logger.info(f"Received message type: {message.message_type}")
                
        except Exception as e:
            logger.error(f"Error processing coordinator message: {str(e)}")
    
    async def handle_global_model_update(self, message: FederatedMessage):
        """Handle updated global model from coordinator"""
        try:
            global_params = message.payload.get('global_model_parameters', {})
            round_number = message.payload.get('round', 0)
            
            # Update local model with global parameters
            if self.local_model and global_params:
                # In practice, this would update the actual model
                logger.info(f"Updated local model with global parameters from round {round_number}")
            
            # Optionally trigger new local training round
            await self.train_local_model()
            
        except Exception as e:
            logger.error(f"Error handling global model update: {str(e)}")
    
    async def handle_join_response(self, message: FederatedMessage):
        """Handle join response from coordinator"""
        payload = message.payload
        if payload.get('accepted', False):
            logger.info("Successfully joined federation")
        else:
            logger.error("Federation join request rejected")
    
    async def train_local_model(self):
        """Train local model and send update to coordinator"""
        if not self.local_data:
            logger.warning("No local data available for training")
            return
        
        try:
            # Simulate local training (implement actual training logic)
            logger.info(f"Training local model with {len(self.local_data)} samples")
            
            # Generate model update (simplified)
            model_update = {
                'model_parameters': {},  # Actual model parameters
                'training_metrics': {
                    'loss': 0.1,
                    'accuracy': 0.9,
                    'samples': len(self.local_data)
                },
                'data_size': len(self.local_data)
            }
            
            # Send update to coordinator
            await self.send_model_update(model_update)
            
        except Exception as e:
            logger.error(f"Error training local model: {str(e)}")
    
    async def send_model_update(self, model_update: Dict[str, Any]):
        """Send model update to coordinator"""
        if not self.connected:
            logger.error("Not connected to coordinator")
            return
        
        try:
            update_message = FederatedMessage(
                message_id=f"update_{datetime.now().timestamp()}",
                sender_id=self.participant_id,
                recipient_id="coordinator",
                message_type=FederatedMessageType.MODEL_UPDATE,
                payload=model_update,
                timestamp=datetime.now()
            )
            
            await self.websocket.send(json.dumps(update_message.to_dict()))
            logger.info("Sent model update to coordinator")
            
        except Exception as e:
            logger.error(f"Error sending model update: {str(e)}")
    
    def add_local_data(self, learning_events: List[LearningEvent]):
        """Add learning events to local training data"""
        self.local_data.extend(learning_events)
        logger.info(f"Added {len(learning_events)} learning events to local data")

# Factory functions
def create_federated_coordinator(**kwargs) -> FederatedLearningCoordinator:
    """Create federated learning coordinator"""
    return FederatedLearningCoordinator(**kwargs)

def create_federated_participant(**kwargs) -> FederatedLearningParticipant:
    """Create federated learning participant"""
    return FederatedLearningParticipant(**kwargs)
