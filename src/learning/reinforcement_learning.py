"""
Reinforcement Learning for Adaptive Cyber Defense
Continuous learning and adaptation for cybersecurity strategies
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import random
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import logging
from abc import ABC, abstractmethod
from collections import deque, defaultdict
import sqlite3
import pickle
from enum import Enum
import gym
from gym import spaces
import asyncio

class ActionType(Enum):
    BLOCK_IP = "block_ip"
    ALLOW_IP = "allow_ip"
    QUARANTINE_HOST = "quarantine_host"
    PATCH_SYSTEM = "patch_system"
    UPDATE_RULES = "update_rules"
    SCAN_NETWORK = "scan_network"
    ISOLATE_SEGMENT = "isolate_segment"
    ESCALATE_ALERT = "escalate_alert"
    COLLECT_EVIDENCE = "collect_evidence"
    NO_ACTION = "no_action"

@dataclass
class CyberState:
    """State representation for cybersecurity environment"""
    timestamp: str
    network_traffic: Dict[str, float]
    active_connections: List[Dict[str, Any]]
    security_alerts: List[Dict[str, Any]]
    system_health: Dict[str, float]
    threat_indicators: Dict[str, float]
    previous_actions: List[str]
    environment_context: Dict[str, Any]

@dataclass
class CyberAction:
    """Action representation for cybersecurity decisions"""
    action_type: ActionType
    parameters: Dict[str, Any]
    confidence: float
    expected_impact: float
    resource_cost: float
    timestamp: str

@dataclass
class CyberReward:
    """Reward structure for cyber defense RL"""
    security_improvement: float
    false_positive_penalty: float
    resource_efficiency: float
    response_time_bonus: float
    total_reward: float
    detailed_breakdown: Dict[str, float]

class CyberDefenseEnvironment(gym.Env):
    """Gym environment for cybersecurity reinforcement learning"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__()
        
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Environment parameters
        self.max_timesteps = self.config.get('max_timesteps', 1000)
        self.attack_probability = self.config.get('attack_probability', 0.1)
        self.false_positive_rate = self.config.get('false_positive_rate', 0.05)
        
        # State space: network metrics, alerts, system health, etc.
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(50,), dtype=np.float32
        )
        
        # Action space: different cyber defense actions
        self.action_space = spaces.Discrete(len(ActionType))
        
        # Environment state
        self.current_state = None
        self.timestep = 0
        self.attack_in_progress = False
        self.attack_type = None
        self.network_state = self._initialize_network_state()
        
        # Metrics tracking
        self.episode_metrics = {
            'attacks_detected': 0,
            'attacks_blocked': 0,
            'false_positives': 0,
            'response_times': [],
            'resource_usage': 0.0,
            'total_reward': 0.0
        }
        
    def _initialize_network_state(self) -> Dict[str, Any]:
        """Initialize network state simulation"""
        return {
            'hosts': {f'host_{i}': {'status': 'normal', 'risk': 0.1} for i in range(20)},
            'services': {f'service_{i}': {'status': 'active', 'load': 0.3} for i in range(10)},
            'network_segments': {f'segment_{i}': {'traffic': 0.5, 'anomalies': 0.0} for i in range(5)},
            'security_controls': {
                'firewall': {'status': 'active', 'rules': 100},
                'ids': {'status': 'active', 'sensitivity': 0.7},
                'antivirus': {'status': 'active', 'definitions': 'updated'}
            }
        }
    
    def _generate_state_vector(self) -> np.ndarray:
        """Convert current environment state to observation vector"""
        state_vector = []
        
        # Network traffic metrics (10 features)
        traffic_metrics = [
            np.mean([self.network_state['network_segments'][seg]['traffic'] 
                    for seg in self.network_state['network_segments']]),
            np.max([self.network_state['network_segments'][seg]['traffic'] 
                   for seg in self.network_state['network_segments']]),
            np.std([self.network_state['network_segments'][seg]['traffic'] 
                   for seg in self.network_state['network_segments']]),
            np.mean([self.network_state['network_segments'][seg]['anomalies'] 
                    for seg in self.network_state['network_segments']]),
            np.sum([1 for host in self.network_state['hosts'].values() 
                   if host['status'] != 'normal']) / len(self.network_state['hosts']),
            np.mean([host['risk'] for host in self.network_state['hosts'].values()]),
            np.sum([1 for service in self.network_state['services'].values() 
                   if service['status'] == 'active']) / len(self.network_state['services']),
            np.mean([service['load'] for service in self.network_state['services'].values()]),
            1.0 if self.attack_in_progress else 0.0,
            self.timestep / self.max_timesteps
        ]
        state_vector.extend(traffic_metrics)
        
        # Security controls status (10 features)
        controls = self.network_state['security_controls']
        control_features = [
            1.0 if controls['firewall']['status'] == 'active' else 0.0,
            controls['firewall']['rules'] / 200.0,  # Normalize
            1.0 if controls['ids']['status'] == 'active' else 0.0,
            controls['ids']['sensitivity'],
            1.0 if controls['antivirus']['status'] == 'active' else 0.0,
            1.0 if controls['antivirus']['definitions'] == 'updated' else 0.0,
            # Additional derived features
            np.mean([1.0 if ctrl['status'] == 'active' else 0.0 
                    for ctrl in controls.values() if 'status' in ctrl]),
            self.episode_metrics['attacks_detected'] / max(1, self.timestep),
            self.episode_metrics['false_positives'] / max(1, self.timestep),
            self.episode_metrics['resource_usage'] / max(1, self.timestep)
        ]
        state_vector.extend(control_features)
        
        # Historical context (15 features)
        recent_actions = self.current_state.previous_actions[-10:] if self.current_state else []
        action_history = [0.0] * 10
        for i, action in enumerate(recent_actions):
            if i < len(action_history):
                action_history[i] = list(ActionType).index(ActionType(action)) / len(ActionType)
        
        context_features = action_history + [
            len(self.current_state.security_alerts) / 10.0 if self.current_state else 0.0,
            len(self.current_state.active_connections) / 100.0 if self.current_state else 0.0,
            np.mean(list(self.current_state.threat_indicators.values())) if self.current_state else 0.0,
            np.max(list(self.current_state.threat_indicators.values())) if self.current_state else 0.0,
            np.std(list(self.current_state.threat_indicators.values())) if self.current_state else 0.0
        ]
        state_vector.extend(context_features)
        
        # Threat landscape (15 features)
        threat_features = []
        if self.current_state:
            indicators = self.current_state.threat_indicators
            threat_features = [
                indicators.get('malware_probability', 0.0),
                indicators.get('intrusion_probability', 0.0),
                indicators.get('ddos_probability', 0.0),
                indicators.get('lateral_movement_probability', 0.0),
                indicators.get('data_exfiltration_probability', 0.0),
                indicators.get('credential_theft_probability', 0.0),
                indicators.get('ransomware_probability', 0.0),
                indicators.get('phishing_probability', 0.0),
                indicators.get('insider_threat_probability', 0.0),
                indicators.get('apt_probability', 0.0),
                # Derived features
                max(indicators.values()) if indicators else 0.0,
                min(indicators.values()) if indicators else 0.0,
                np.mean(list(indicators.values())) if indicators else 0.0,
                np.std(list(indicators.values())) if indicators else 0.0,
                len([v for v in indicators.values() if v > 0.5]) / max(1, len(indicators))
            ]
        else:
            threat_features = [0.0] * 15
        
        state_vector.extend(threat_features)
        
        # Ensure exactly 50 features
        while len(state_vector) < 50:
            state_vector.append(0.0)
        
        return np.array(state_vector[:50], dtype=np.float32)
    
    def _simulate_attack(self) -> Tuple[bool, str]:
        """Simulate potential cyber attacks"""
        if random.random() < self.attack_probability:
            attack_types = ['malware', 'intrusion', 'ddos', 'lateral_movement', 
                          'data_exfiltration', 'ransomware', 'phishing']
            attack_type = random.choice(attack_types)
            
            # Update network state based on attack
            if attack_type == 'malware':
                # Infect random hosts
                infected_hosts = random.sample(list(self.network_state['hosts'].keys()), 
                                             random.randint(1, 3))
                for host in infected_hosts:
                    self.network_state['hosts'][host]['status'] = 'infected'
                    self.network_state['hosts'][host]['risk'] = 0.9
            
            elif attack_type == 'ddos':
                # Increase traffic and service load
                for segment in self.network_state['network_segments'].values():
                    segment['traffic'] = min(1.0, segment['traffic'] + 0.3)
                for service in self.network_state['services'].values():
                    service['load'] = min(1.0, service['load'] + 0.4)
            
            elif attack_type == 'intrusion':
                # Compromise random host
                target_host = random.choice(list(self.network_state['hosts'].keys()))
                self.network_state['hosts'][target_host]['status'] = 'compromised'
                self.network_state['hosts'][target_host]['risk'] = 0.95
            
            return True, attack_type
        
        return False, None
    
    def _execute_action(self, action_idx: int) -> Dict[str, Any]:
        """Execute the chosen action and return its effects"""
        action_type = list(ActionType)[action_idx]
        action_effects = {
            'success': False,
            'impact': 0.0,
            'cost': 0.0,
            'side_effects': []
        }
        
        if action_type == ActionType.BLOCK_IP:
            # Block suspicious IP addresses
            action_effects['success'] = True
            action_effects['impact'] = 0.3 if self.attack_in_progress else -0.1  # False positive penalty
            action_effects['cost'] = 0.1
            
            if self.attack_in_progress and self.attack_type in ['intrusion', 'ddos']:
                # Effective against network-based attacks
                action_effects['impact'] = 0.6
                self.attack_in_progress = False
        
        elif action_type == ActionType.QUARANTINE_HOST:
            # Quarantine infected/suspicious hosts
            action_effects['success'] = True
            action_effects['cost'] = 0.3
            
            infected_hosts = [host for host, info in self.network_state['hosts'].items() 
                            if info['status'] in ['infected', 'compromised']]
            
            if infected_hosts:
                # Quarantine infected host
                target_host = random.choice(infected_hosts)
                self.network_state['hosts'][target_host]['status'] = 'quarantined'
                action_effects['impact'] = 0.7
                if self.attack_type == 'malware':
                    self.attack_in_progress = False
            else:
                # False positive
                action_effects['impact'] = -0.2
        
        elif action_type == ActionType.PATCH_SYSTEM:
            # Apply security patches
            action_effects['success'] = True
            action_effects['cost'] = 0.2
            action_effects['impact'] = 0.1  # Preventive measure
            
            # Reduce overall risk
            for host in self.network_state['hosts'].values():
                host['risk'] = max(0.1, host['risk'] - 0.1)
        
        elif action_type == ActionType.UPDATE_RULES:
            # Update firewall/IDS rules
            action_effects['success'] = True
            action_effects['cost'] = 0.1
            action_effects['impact'] = 0.2
            
            self.network_state['security_controls']['firewall']['rules'] += 10
            self.network_state['security_controls']['ids']['sensitivity'] = min(1.0, 
                self.network_state['security_controls']['ids']['sensitivity'] + 0.1)
        
        elif action_type == ActionType.SCAN_NETWORK:
            # Perform network security scan
            action_effects['success'] = True
            action_effects['cost'] = 0.2
            action_effects['impact'] = 0.15  # Information gathering
            
            # Detect hidden threats
            for segment in self.network_state['network_segments'].values():
                segment['anomalies'] = max(0.0, segment['anomalies'] - 0.2)
        
        elif action_type == ActionType.ISOLATE_SEGMENT:
            # Isolate network segment
            action_effects['success'] = True
            action_effects['cost'] = 0.4
            
            if self.attack_type == 'lateral_movement':
                action_effects['impact'] = 0.8
                self.attack_in_progress = False
            else:
                action_effects['impact'] = -0.1  # May affect normal operations
        
        elif action_type == ActionType.NO_ACTION:
            # Do nothing
            action_effects['success'] = True
            action_effects['cost'] = 0.0
            action_effects['impact'] = -0.1 if self.attack_in_progress else 0.0
        
        return action_effects
    
    def _calculate_reward(self, action_effects: Dict[str, Any]) -> CyberReward:
        """Calculate reward based on action outcomes and environment state"""
        
        # Security improvement component
        security_improvement = action_effects['impact']
        
        # False positive penalty
        false_positive_penalty = 0.0
        if not self.attack_in_progress and action_effects['impact'] < 0:
            false_positive_penalty = abs(action_effects['impact'])
            self.episode_metrics['false_positives'] += 1
        
        # Resource efficiency (favor low-cost effective actions)
        resource_efficiency = max(0, 0.1 - action_effects['cost'])
        
        # Response time bonus (quicker responses to attacks are better)
        response_time_bonus = 0.0
        if self.attack_in_progress and action_effects['impact'] > 0:
            response_time_bonus = 0.1
            self.episode_metrics['attacks_blocked'] += 1
        
        # Calculate total reward
        total_reward = (
            security_improvement + 
            resource_efficiency + 
            response_time_bonus - 
            false_positive_penalty
        )
        
        # Update metrics
        self.episode_metrics['resource_usage'] += action_effects['cost']
        self.episode_metrics['total_reward'] += total_reward
        
        return CyberReward(
            security_improvement=security_improvement,
            false_positive_penalty=false_positive_penalty,
            resource_efficiency=resource_efficiency,
            response_time_bonus=response_time_bonus,
            total_reward=total_reward,
            detailed_breakdown={
                'security_improvement': security_improvement,
                'resource_efficiency': resource_efficiency,
                'response_time_bonus': response_time_bonus,
                'false_positive_penalty': -false_positive_penalty
            }
        )
    
    def reset(self) -> np.ndarray:
        """Reset environment to initial state"""
        self.timestep = 0
        self.attack_in_progress = False
        self.attack_type = None
        self.network_state = self._initialize_network_state()
        
        # Reset metrics
        self.episode_metrics = {
            'attacks_detected': 0,
            'attacks_blocked': 0,
            'false_positives': 0,
            'response_times': [],
            'resource_usage': 0.0,
            'total_reward': 0.0
        }
        
        # Generate initial state
        self.current_state = CyberState(
            timestamp=datetime.now().isoformat(),
            network_traffic={'total': 0.3, 'suspicious': 0.1},
            active_connections=[],
            security_alerts=[],
            system_health={'cpu': 0.4, 'memory': 0.3, 'disk': 0.2},
            threat_indicators={
                'malware_probability': 0.1,
                'intrusion_probability': 0.1,
                'ddos_probability': 0.05,
                'lateral_movement_probability': 0.05,
                'data_exfiltration_probability': 0.05
            },
            previous_actions=[],
            environment_context={'time_of_day': 'business_hours'}
        )
        
        return self._generate_state_vector()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Execute one step in the environment"""
        self.timestep += 1
        
        # Simulate potential attacks
        attack_occurred, attack_type = self._simulate_attack()
        if attack_occurred:
            self.attack_in_progress = True
            self.attack_type = attack_type
            self.episode_metrics['attacks_detected'] += 1
        
        # Execute chosen action
        action_effects = self._execute_action(action)
        
        # Calculate reward
        reward_info = self._calculate_reward(action_effects)
        
        # Update state
        action_name = list(ActionType)[action].value
        if self.current_state:
            self.current_state.previous_actions.append(action_name)
            self.current_state.previous_actions = self.current_state.previous_actions[-10:]  # Keep last 10
        
        # Update threat indicators based on current situation
        if self.attack_in_progress:
            threat_boost = 0.3
            if self.attack_type in self.current_state.threat_indicators:
                self.current_state.threat_indicators[f"{self.attack_type}_probability"] = min(1.0,
                    self.current_state.threat_indicators.get(f"{self.attack_type}_probability", 0.1) + threat_boost)
        
        # Check if episode is done
        done = (
            self.timestep >= self.max_timesteps or
            self.episode_metrics['resource_usage'] > 5.0 or  # Resource limit
            self.episode_metrics['false_positives'] > 20   # Too many false positives
        )
        
        # Prepare info dictionary
        info = {
            'attack_in_progress': self.attack_in_progress,
            'attack_type': self.attack_type,
            'action_effects': action_effects,
            'reward_breakdown': asdict(reward_info),
            'episode_metrics': self.episode_metrics.copy(),
            'timestep': self.timestep
        }
        
        return self._generate_state_vector(), reward_info.total_reward, done, info

class DQNAgent(nn.Module):
    """Deep Q-Network agent for cyber defense"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Neural network layers
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # Dueling DQN components
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.advantage_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
        
        # Feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network"""
        # Extract features
        features = self.feature_extractor(state)
        
        # Dueling DQN: Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
        value = self.value_head(features)
        advantage = self.advantage_head(features)
        
        # Combine value and advantage
        q_values = value + (advantage - advantage.mean(dim=-1, keepdim=True))
        
        return q_values

class CyberDefenseRL:
    """Reinforcement Learning system for adaptive cyber defense"""
    
    def __init__(self, config: Dict[str, Any] = None, database_path: str = "cyber_rl.db"):
        self.config = config or {}
        self.database_path = database_path
        self.logger = logging.getLogger(__name__)
        
        # Initialize database
        self._init_database()
        
        # Environment
        self.env = CyberDefenseEnvironment(self.config.get('env_config', {}))
        
        # Agent configuration
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # DQN Agent
        self.q_network = DQNAgent(self.state_dim, self.action_dim).to(self.device)
        self.target_network = DQNAgent(self.state_dim, self.action_dim).to(self.device)
        
        # Copy parameters to target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Training parameters
        self.learning_rate = self.config.get('learning_rate', 1e-4)
        self.gamma = self.config.get('gamma', 0.99)
        self.epsilon = self.config.get('epsilon_start', 1.0)
        self.epsilon_min = self.config.get('epsilon_min', 0.01)
        self.epsilon_decay = self.config.get('epsilon_decay', 0.995)
        self.batch_size = self.config.get('batch_size', 32)
        self.memory_size = self.config.get('memory_size', 10000)
        self.target_update_freq = self.config.get('target_update_freq', 100)
        
        # Experience replay buffer
        self.memory = deque(maxlen=self.memory_size)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        
        # Training state
        self.total_steps = 0
        self.episode_count = 0
        self.training_metrics = defaultdict(list)
        
    def _init_database(self):
        """Initialize SQLite database for storing training data"""
        with sqlite3.connect(self.database_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS training_episodes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    episode_number INTEGER NOT NULL,
                    total_reward REAL NOT NULL,
                    episode_length INTEGER NOT NULL,
                    epsilon REAL NOT NULL,
                    metrics TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS experience_replay (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    state BLOB NOT NULL,
                    action INTEGER NOT NULL,
                    reward REAL NOT NULL,
                    next_state BLOB NOT NULL,
                    done INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS model_checkpoints (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    episode_number INTEGER NOT NULL,
                    model_state BLOB NOT NULL,
                    optimizer_state BLOB NOT NULL,
                    training_metrics BLOB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Select action using epsilon-greedy policy"""
        if training and random.random() < self.epsilon:
            return self.env.action_space.sample()
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
    
    def store_experience(self, state: np.ndarray, action: int, reward: float, 
                        next_state: np.ndarray, done: bool):
        """Store experience in replay buffer"""
        self.memory.append((state, action, reward, next_state, done))
        
        # Also store in database for persistence
        with sqlite3.connect(self.database_path) as conn:
            conn.execute(
                "INSERT INTO experience_replay (state, action, reward, next_state, done) VALUES (?, ?, ?, ?, ?)",
                (pickle.dumps(state), action, reward, pickle.dumps(next_state), int(done))
            )
    
    def train_step(self) -> Dict[str, float]:
        """Perform one training step"""
        if len(self.memory) < self.batch_size:
            return {}
        
        # Sample batch from memory
        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor([e[0] for e in batch]).to(self.device)
        actions = torch.LongTensor([e[1] for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e[2] for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e[3] for e in batch]).to(self.device)
        dones = torch.BoolTensor([e[4] for e in batch]).to(self.device)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values from target network
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
        self.optimizer.step()
        
        # Update target network
        if self.total_steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        return {
            'loss': loss.item(),
            'q_value_mean': current_q_values.mean().item(),
            'target_q_mean': target_q_values.mean().item()
        }
    
    def train_episode(self) -> Dict[str, Any]:
        """Train for one episode"""
        state = self.env.reset()
        total_reward = 0.0
        episode_length = 0
        episode_info = []
        
        while True:
            # Select action
            action = self.select_action(state, training=True)
            
            # Take step
            next_state, reward, done, info = self.env.step(action)
            
            # Store experience
            self.store_experience(state, action, reward, next_state, done)
            
            # Train
            train_metrics = self.train_step()
            
            # Update state
            state = next_state
            total_reward += reward
            episode_length += 1
            self.total_steps += 1
            
            # Store step info
            episode_info.append({
                'action': list(ActionType)[action].value,
                'reward': reward,
                'info': info
            })
            
            if done:
                break
        
        # Update epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.episode_count += 1
        
        # Prepare episode results
        episode_results = {
            'episode_number': self.episode_count,
            'total_reward': total_reward,
            'episode_length': episode_length,
            'epsilon': self.epsilon,
            'final_metrics': self.env.episode_metrics,
            'step_info': episode_info,
            'training_metrics': train_metrics
        }
        
        # Save episode to database
        self._save_episode(episode_results)
        
        return episode_results
    
    def _save_episode(self, episode_results: Dict[str, Any]):
        """Save episode results to database"""
        metrics_json = json.dumps(episode_results['final_metrics'])
        
        with sqlite3.connect(self.database_path) as conn:
            conn.execute(
                "INSERT INTO training_episodes (episode_number, total_reward, episode_length, epsilon, metrics) VALUES (?, ?, ?, ?, ?)",
                (episode_results['episode_number'], episode_results['total_reward'], 
                 episode_results['episode_length'], episode_results['epsilon'], metrics_json)
            )
    
    def save_model(self, filepath: str = None):
        """Save model checkpoint"""
        if filepath is None:
            filepath = f"cyber_defense_model_episode_{self.episode_count}.pth"
        
        checkpoint = {
            'episode_count': self.episode_count,
            'total_steps': self.total_steps,
            'q_network_state': self.q_network.state_dict(),
            'target_network_state': self.target_network.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'config': self.config,
            'training_metrics': dict(self.training_metrics)
        }
        
        torch.save(checkpoint, filepath)
        
        # Also save to database
        with sqlite3.connect(self.database_path) as conn:
            conn.execute(
                "INSERT INTO model_checkpoints (episode_number, model_state, optimizer_state, training_metrics) VALUES (?, ?, ?, ?)",
                (self.episode_count, pickle.dumps(checkpoint['q_network_state']),
                 pickle.dumps(checkpoint['optimizer_state']), pickle.dumps(checkpoint['training_metrics']))
            )
        
        self.logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.episode_count = checkpoint['episode_count']
        self.total_steps = checkpoint['total_steps']
        self.q_network.load_state_dict(checkpoint['q_network_state'])
        self.target_network.load_state_dict(checkpoint['target_network_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.epsilon = checkpoint['epsilon']
        self.training_metrics = defaultdict(list, checkpoint.get('training_metrics', {}))
        
        self.logger.info(f"Model loaded from {filepath}")
    
    def evaluate(self, num_episodes: int = 10) -> Dict[str, Any]:
        """Evaluate the trained agent"""
        evaluation_results = []
        
        for episode in range(num_episodes):
            state = self.env.reset()
            total_reward = 0.0
            episode_length = 0
            actions_taken = []
            
            while True:
                # Select action (no exploration)
                action = self.select_action(state, training=False)
                actions_taken.append(list(ActionType)[action].value)
                
                # Take step
                next_state, reward, done, info = self.env.step(action)
                
                state = next_state
                total_reward += reward
                episode_length += 1
                
                if done:
                    break
            
            evaluation_results.append({
                'episode': episode,
                'total_reward': total_reward,
                'episode_length': episode_length,
                'actions_taken': actions_taken,
                'final_metrics': self.env.episode_metrics.copy()
            })
        
        # Calculate aggregate statistics
        total_rewards = [r['total_reward'] for r in evaluation_results]
        episode_lengths = [r['episode_length'] for r in evaluation_results]
        
        aggregate_stats = {
            'num_episodes': num_episodes,
            'mean_reward': np.mean(total_rewards),
            'std_reward': np.std(total_rewards),
            'min_reward': min(total_rewards),
            'max_reward': max(total_rewards),
            'mean_episode_length': np.mean(episode_lengths),
            'success_rate': len([r for r in total_rewards if r > 0]) / num_episodes,
            'individual_episodes': evaluation_results
        }
        
        return aggregate_stats
    
    def get_action_recommendations(self, current_state: CyberState) -> List[Dict[str, Any]]:
        """Get action recommendations for a given state"""
        # Convert CyberState to observation vector
        self.env.current_state = current_state
        state_vector = self.env._generate_state_vector()
        
        # Get Q-values for all actions
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state_vector).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor).squeeze().cpu().numpy()
        
        # Create recommendations
        recommendations = []
        for i, q_value in enumerate(q_values):
            action_type = list(ActionType)[i]
            recommendations.append({
                'action': action_type.value,
                'q_value': float(q_value),
                'confidence': float(torch.softmax(torch.tensor(q_values), dim=0)[i]),
                'description': self._get_action_description(action_type)
            })
        
        # Sort by Q-value
        recommendations.sort(key=lambda x: x['q_value'], reverse=True)
        
        return recommendations
    
    def _get_action_description(self, action_type: ActionType) -> str:
        """Get human-readable description of action"""
        descriptions = {
            ActionType.BLOCK_IP: "Block suspicious IP addresses from accessing the network",
            ActionType.ALLOW_IP: "Allow blocked IP addresses to resume network access", 
            ActionType.QUARANTINE_HOST: "Isolate potentially compromised hosts from the network",
            ActionType.PATCH_SYSTEM: "Apply security patches to vulnerable systems",
            ActionType.UPDATE_RULES: "Update firewall and IDS rules to improve detection",
            ActionType.SCAN_NETWORK: "Perform comprehensive network security scan",
            ActionType.ISOLATE_SEGMENT: "Isolate network segment to contain potential threats",
            ActionType.ESCALATE_ALERT: "Escalate security alert to human analysts",
            ActionType.COLLECT_EVIDENCE: "Collect forensic evidence for incident analysis",
            ActionType.NO_ACTION: "Take no immediate action and continue monitoring"
        }
        return descriptions.get(action_type, "Unknown action")

# Example usage and testing
if __name__ == "__main__":
    print("🤖 Reinforcement Learning for Cyber Defense Testing:")
    print("=" * 60)
    
    # Initialize the RL system
    config = {
        'learning_rate': 1e-4,
        'gamma': 0.99,
        'epsilon_start': 1.0,
        'epsilon_min': 0.01,
        'epsilon_decay': 0.995,
        'batch_size': 32,
        'target_update_freq': 100,
        'env_config': {
            'max_timesteps': 200,
            'attack_probability': 0.15,
            'false_positive_rate': 0.05
        }
    }
    
    rl_system = CyberDefenseRL(config)
    print(f"  Initialized RL system with state dim: {rl_system.state_dim}, action dim: {rl_system.action_dim}")
    
    # Test environment
    print("\n🌍 Testing cyber defense environment...")
    state = rl_system.env.reset()
    print(f"  Initial state shape: {state.shape}")
    print(f"  Sample state values: {state[:10]}")
    
    # Test action selection
    print("\n🎯 Testing action selection...")
    for i in range(5):
        action = rl_system.select_action(state, training=True)
        next_state, reward, done, info = rl_system.env.step(action)
        action_name = list(ActionType)[action].value
        print(f"  Step {i+1}: Action={action_name}, Reward={reward:.3f}, Attack={info['attack_in_progress']}")
        state = next_state
        if done:
            break
    
    # Test short training run
    print("\n🏋️ Testing training episode...")
    episode_results = rl_system.train_episode()
    print(f"  Episode {episode_results['episode_number']}: Reward={episode_results['total_reward']:.2f}, Length={episode_results['episode_length']}")
    print(f"  Final metrics: {episode_results['final_metrics']}")
    print(f"  Epsilon: {episode_results['epsilon']:.3f}")
    
    # Test multiple episodes
    print("\n📊 Testing multiple training episodes...")
    for episode in range(3):
        episode_results = rl_system.train_episode()
        attacks_blocked = episode_results['final_metrics']['attacks_blocked']
        attacks_detected = episode_results['final_metrics']['attacks_detected']
        false_positives = episode_results['final_metrics']['false_positives']
        
        print(f"  Episode {episode_results['episode_number']}: "
              f"Reward={episode_results['total_reward']:.2f}, "
              f"Blocked={attacks_blocked}/{attacks_detected}, "
              f"FP={false_positives}")
    
    # Test action recommendations
    print("\n💡 Testing action recommendations...")
    sample_state = CyberState(
        timestamp=datetime.now().isoformat(),
        network_traffic={'total': 0.8, 'suspicious': 0.3},
        active_connections=[],
        security_alerts=[{'type': 'malware', 'severity': 'high'}],
        system_health={'cpu': 0.9, 'memory': 0.8, 'disk': 0.6},
        threat_indicators={
            'malware_probability': 0.8,
            'intrusion_probability': 0.3,
            'ddos_probability': 0.1
        },
        previous_actions=['scan_network', 'update_rules'],
        environment_context={'time_of_day': 'night'}
    )
    
    recommendations = rl_system.get_action_recommendations(sample_state)
    print(f"  Top 3 recommended actions:")
    for i, rec in enumerate(recommendations[:3]):
        print(f"    {i+1}. {rec['action']}: Q-value={rec['q_value']:.3f}, Confidence={rec['confidence']:.3f}")
        print(f"       Description: {rec['description']}")
    
    # Test evaluation
    print("\n🔍 Testing agent evaluation...")
    eval_results = rl_system.evaluate(num_episodes=3)
    print(f"  Evaluation over {eval_results['num_episodes']} episodes:")
    print(f"    Mean reward: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}")
    print(f"    Success rate: {eval_results['success_rate']:.2%}")
    print(f"    Mean episode length: {eval_results['mean_episode_length']:.1f}")
    
    # Test model saving/loading
    print("\n💾 Testing model persistence...")
    model_path = "test_cyber_defense_model.pth"
    rl_system.save_model(model_path)
    
    # Load model in new system
    rl_system_2 = CyberDefenseRL(config)
    rl_system_2.load_model(model_path)
    print(f"  Model loaded successfully, episode count: {rl_system_2.episode_count}")
    
    print("\n✅ Reinforcement Learning system implemented and tested")
    print(f"  Database: {rl_system.database_path}")
    print(f"  Action space: {len(ActionType)} actions")
    print(f"  State space: {rl_system.state_dim} dimensions")
    print(f"  Model: Deep Q-Network with Dueling architecture")
