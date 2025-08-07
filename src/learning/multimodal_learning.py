"""
Multimodal Learning System for Cybersecurity
Integration of text, network data, and visual security information
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import cv2
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import logging
from abc import ABC, abstractmethod
from PIL import Image
import base64
import io

@dataclass
class TextData:
    """Text-based security data"""
    content: str
    data_type: str  # log, alert, report, email, etc.
    metadata: Dict[str, Any]
    timestamp: str
    source: str

@dataclass
class NetworkData:
    """Network traffic data"""
    packet_data: bytes
    flow_features: Dict[str, float]
    protocol: str
    source_ip: str
    dest_ip: str
    timestamp: str
    metadata: Dict[str, Any]

@dataclass
class VisualData:
    """Visual security data"""
    image_data: np.ndarray
    image_type: str  # network_topology, malware_visualization, dashboard_screenshot
    features: Dict[str, Any]
    timestamp: str
    metadata: Dict[str, Any]

@dataclass
class MultimodalSample:
    """Combined multimodal sample"""
    sample_id: str
    text_data: Optional[TextData]
    network_data: Optional[NetworkData] 
    visual_data: Optional[VisualData]
    label: str
    confidence: float
    timestamp: str

class ModalityEncoder(nn.Module, ABC):
    """Abstract base class for modality encoders"""
    
    @abstractmethod
    def forward(self, data: Any) -> torch.Tensor:
        pass
    
    @abstractmethod
    def get_output_dim(self) -> int:
        pass

class TextEncoder(ModalityEncoder):
    """Encoder for text-based security data"""
    
    def __init__(self, vocab_size: int = 10000, embed_dim: int = 256, hidden_dim: int = 512):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        
        # Text processing layers
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attention = nn.MultiheadAttention(hidden_dim * 2, num_heads=8)
        self.output_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # Cybersecurity-specific text patterns
        self.threat_patterns = nn.Conv1d(hidden_dim * 2, 64, kernel_size=3, padding=1)
        self.temporal_patterns = nn.Conv1d(hidden_dim * 2, 64, kernel_size=5, padding=2)
        
    def forward(self, text_tokens: torch.Tensor) -> torch.Tensor:
        # text_tokens: [batch_size, seq_len]
        embedded = self.embedding(text_tokens)  # [batch_size, seq_len, embed_dim]
        
        # LSTM encoding
        lstm_out, (h_n, c_n) = self.lstm(embedded)  # [batch_size, seq_len, hidden_dim * 2]
        
        # Self-attention for important security keywords
        attn_out, _ = self.attention(
            lstm_out.transpose(0, 1),
            lstm_out.transpose(0, 1), 
            lstm_out.transpose(0, 1)
        )
        attn_out = attn_out.transpose(0, 1)  # [batch_size, seq_len, hidden_dim * 2]
        
        # Pattern detection
        lstm_transposed = lstm_out.transpose(1, 2)  # [batch_size, hidden_dim * 2, seq_len]
        threat_features = F.relu(self.threat_patterns(lstm_transposed))
        temporal_features = F.relu(self.temporal_patterns(lstm_transposed))
        
        # Global pooling
        threat_pooled = F.adaptive_avg_pool1d(threat_features, 1).squeeze(-1)
        temporal_pooled = F.adaptive_avg_pool1d(temporal_features, 1).squeeze(-1)
        
        # Combine features
        combined = torch.cat([
            attn_out.mean(dim=1),  # Attention-weighted average
            threat_pooled,
            temporal_pooled
        ], dim=1)
        
        output = self.output_proj(combined[:, :self.hidden_dim * 2])
        return F.relu(output)
    
    def get_output_dim(self) -> int:
        return self.hidden_dim

class NetworkEncoder(ModalityEncoder):
    """Encoder for network traffic data"""
    
    def __init__(self, flow_feature_dim: int = 50, packet_embed_dim: int = 128, hidden_dim: int = 512):
        super().__init__()
        self.flow_feature_dim = flow_feature_dim
        self.packet_embed_dim = packet_embed_dim
        self.hidden_dim = hidden_dim
        
        # Flow feature processing
        self.flow_encoder = nn.Sequential(
            nn.Linear(flow_feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        
        # Packet sequence processing (treat packets as sequences)
        self.packet_embedding = nn.Embedding(256, packet_embed_dim)  # For packet bytes
        self.packet_conv1d = nn.Conv1d(packet_embed_dim, 128, kernel_size=3, padding=1)
        self.packet_conv2d = nn.Conv1d(128, 64, kernel_size=5, padding=2)
        
        # Protocol-specific layers
        self.protocol_embedding = nn.Embedding(10, 32)  # Common protocols
        
        # Temporal analysis
        self.temporal_conv = nn.Conv1d(256 + 64 + 32, 128, kernel_size=3, padding=1)
        
        # Output projection
        self.output_proj = nn.Linear(128 + 256, hidden_dim)
        
    def forward(self, network_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Extract components
        flow_features = network_data['flow_features']  # [batch_size, flow_feature_dim]
        packet_bytes = network_data['packet_bytes']    # [batch_size, max_packet_len]
        protocol_ids = network_data['protocol_ids']    # [batch_size]
        
        # Process flow features
        flow_encoded = self.flow_encoder(flow_features)  # [batch_size, 256]
        
        # Process packet data
        packet_embedded = self.packet_embedding(packet_bytes)  # [batch_size, max_packet_len, packet_embed_dim]
        packet_transposed = packet_embedded.transpose(1, 2)    # [batch_size, packet_embed_dim, max_packet_len]
        
        packet_conv1 = F.relu(self.packet_conv1d(packet_transposed))
        packet_conv2 = F.relu(self.packet_conv2d(packet_conv1))
        packet_pooled = F.adaptive_avg_pool1d(packet_conv2, 1).squeeze(-1)  # [batch_size, 64]
        
        # Process protocol information
        protocol_embedded = self.protocol_embedding(protocol_ids)  # [batch_size, 32]
        
        # Combine features for temporal analysis
        combined_features = torch.cat([
            flow_encoded, packet_pooled, protocol_embedded
        ], dim=1).unsqueeze(-1)  # [batch_size, 256+64+32, 1]
        
        temporal_features = F.relu(self.temporal_conv(combined_features))
        temporal_pooled = temporal_features.squeeze(-1)  # [batch_size, 128]
        
        # Final combination
        final_features = torch.cat([temporal_pooled, flow_encoded], dim=1)
        output = self.output_proj(final_features)
        
        return F.relu(output)
    
    def get_output_dim(self) -> int:
        return self.hidden_dim

class VisualEncoder(ModalityEncoder):
    """Encoder for visual security data"""
    
    def __init__(self, hidden_dim: int = 512):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Convolutional layers for image processing
        self.conv_layers = nn.Sequential(
            # First block
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Second block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Third block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Fourth block
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7))
        )
        
        # Specialized layers for security visualization patterns
        self.topology_detector = nn.Conv2d(512, 64, kernel_size=1)
        self.anomaly_detector = nn.Conv2d(512, 64, kernel_size=1)
        self.threat_indicator_detector = nn.Conv2d(512, 64, kernel_size=1)
        
        # Final projection
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.output_proj = nn.Linear(512 + 64 * 3, hidden_dim)
        
    def forward(self, image_data: torch.Tensor) -> torch.Tensor:
        # image_data: [batch_size, 3, height, width]
        
        # Convolutional feature extraction
        conv_features = self.conv_layers(image_data)  # [batch_size, 512, 7, 7]
        
        # Security-specific pattern detection
        topology_features = F.relu(self.topology_detector(conv_features))
        anomaly_features = F.relu(self.anomaly_detector(conv_features))
        threat_features = F.relu(self.threat_indicator_detector(conv_features))
        
        # Global pooling for all features
        conv_pooled = self.global_pool(conv_features).view(conv_features.size(0), -1)
        topology_pooled = self.global_pool(topology_features).view(topology_features.size(0), -1)
        anomaly_pooled = self.global_pool(anomaly_features).view(anomaly_features.size(0), -1)
        threat_pooled = self.global_pool(threat_features).view(threat_features.size(0), -1)
        
        # Combine all features
        combined_features = torch.cat([
            conv_pooled, topology_pooled, anomaly_pooled, threat_pooled
        ], dim=1)
        
        output = self.output_proj(combined_features)
        return F.relu(output)
    
    def get_output_dim(self) -> int:
        return self.hidden_dim

class MultimodalFusionLayer(nn.Module):
    """Fusion layer for combining multimodal features"""
    
    def __init__(self, text_dim: int, network_dim: int, visual_dim: int, 
                 fusion_dim: int = 512, num_heads: int = 8):
        super().__init__()
        self.text_dim = text_dim
        self.network_dim = network_dim
        self.visual_dim = visual_dim
        self.fusion_dim = fusion_dim
        
        # Projection layers to common dimension
        self.text_proj = nn.Linear(text_dim, fusion_dim) if text_dim != fusion_dim else nn.Identity()
        self.network_proj = nn.Linear(network_dim, fusion_dim) if network_dim != fusion_dim else nn.Identity()
        self.visual_proj = nn.Linear(visual_dim, fusion_dim) if visual_dim != fusion_dim else nn.Identity()
        
        # Cross-modal attention
        self.cross_attention = nn.MultiheadAttention(fusion_dim, num_heads, batch_first=True)
        
        # Fusion strategies
        self.attention_weights = nn.Parameter(torch.ones(3) / 3)  # Learnable weights
        
        # Gate mechanisms
        self.text_gate = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 4),
            nn.ReLU(),
            nn.Linear(fusion_dim // 4, 1),
            nn.Sigmoid()
        )
        self.network_gate = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 4),
            nn.ReLU(),
            nn.Linear(fusion_dim // 4, 1),
            nn.Sigmoid()
        )
        self.visual_gate = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 4),
            nn.ReLU(),
            nn.Linear(fusion_dim // 4, 1),
            nn.Sigmoid()
        )
        
        # Final fusion
        self.fusion_network = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(fusion_dim, fusion_dim)
        )
        
    def forward(self, text_features: Optional[torch.Tensor] = None,
                network_features: Optional[torch.Tensor] = None,
                visual_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        available_modalities = []
        projected_features = []
        
        # Project features to common dimension
        if text_features is not None:
            text_proj = self.text_proj(text_features)
            available_modalities.append(('text', text_proj))
            projected_features.append(text_proj)
        
        if network_features is not None:
            network_proj = self.network_proj(network_features)
            available_modalities.append(('network', network_proj))
            projected_features.append(network_proj)
        
        if visual_features is not None:
            visual_proj = self.visual_proj(visual_features)
            available_modalities.append(('visual', visual_proj))
            projected_features.append(visual_proj)
        
        if not projected_features:
            raise ValueError("At least one modality must be provided")
        
        if len(projected_features) == 1:
            # Single modality
            return self.fusion_network(projected_features[0])
        
        # Stack features for cross-attention
        stacked_features = torch.stack(projected_features, dim=1)  # [batch_size, num_modalities, fusion_dim]
        
        # Cross-modal attention
        attended_features, attention_weights = self.cross_attention(
            stacked_features, stacked_features, stacked_features
        )
        
        # Apply modality-specific gates
        gated_features = []
        for i, (modality, features) in enumerate(available_modalities):
            if modality == 'text' and text_features is not None:
                gate = self.text_gate(features)
                gated_features.append(attended_features[:, i] * gate)
            elif modality == 'network' and network_features is not None:
                gate = self.network_gate(features)
                gated_features.append(attended_features[:, i] * gate)
            elif modality == 'visual' and visual_features is not None:
                gate = self.visual_gate(features)
                gated_features.append(attended_features[:, i] * gate)
        
        # Weighted fusion
        if len(gated_features) == 2:
            weights = F.softmax(self.attention_weights[:2], dim=0)
            fused = weights[0] * gated_features[0] + weights[1] * gated_features[1]
        elif len(gated_features) == 3:
            weights = F.softmax(self.attention_weights, dim=0)
            fused = (weights[0] * gated_features[0] + 
                    weights[1] * gated_features[1] + 
                    weights[2] * gated_features[2])
        else:
            fused = torch.stack(gated_features, dim=1).mean(dim=1)
        
        # Final processing
        output = self.fusion_network(fused)
        return output

class MultimodalSecurityClassifier(nn.Module):
    """Complete multimodal cybersecurity classifier"""
    
    def __init__(self, num_classes: int, vocab_size: int = 10000, 
                 flow_feature_dim: int = 50, fusion_dim: int = 512):
        super().__init__()
        self.num_classes = num_classes
        
        # Modality encoders
        self.text_encoder = TextEncoder(vocab_size=vocab_size, hidden_dim=fusion_dim)
        self.network_encoder = NetworkEncoder(flow_feature_dim=flow_feature_dim, hidden_dim=fusion_dim)
        self.visual_encoder = VisualEncoder(hidden_dim=fusion_dim)
        
        # Fusion layer
        self.fusion_layer = MultimodalFusionLayer(
            text_dim=fusion_dim,
            network_dim=fusion_dim, 
            visual_dim=fusion_dim,
            fusion_dim=fusion_dim
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(fusion_dim // 2, fusion_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(fusion_dim // 4, num_classes)
        )
        
        # Auxiliary classifiers for individual modalities (for training)
        self.text_classifier = nn.Linear(fusion_dim, num_classes)
        self.network_classifier = nn.Linear(fusion_dim, num_classes)
        self.visual_classifier = nn.Linear(fusion_dim, num_classes)
        
    def forward(self, text_tokens: Optional[torch.Tensor] = None,
                network_data: Optional[Dict[str, torch.Tensor]] = None,
                visual_data: Optional[torch.Tensor] = None,
                return_individual_outputs: bool = False) -> Dict[str, torch.Tensor]:
        
        outputs = {}
        
        # Encode individual modalities
        text_features = None
        network_features = None
        visual_features = None
        
        if text_tokens is not None:
            text_features = self.text_encoder(text_tokens)
            if return_individual_outputs:
                outputs['text_logits'] = self.text_classifier(text_features)
        
        if network_data is not None:
            network_features = self.network_encoder(network_data)
            if return_individual_outputs:
                outputs['network_logits'] = self.network_classifier(network_features)
        
        if visual_data is not None:
            visual_features = self.visual_encoder(visual_data)
            if return_individual_outputs:
                outputs['visual_logits'] = self.visual_classifier(visual_features)
        
        # Multimodal fusion
        if text_features is not None or network_features is not None or visual_features is not None:
            fused_features = self.fusion_layer(text_features, network_features, visual_features)
            outputs['fused_logits'] = self.classifier(fused_features)
        
        return outputs

class MultimodalSecuritySystem:
    """Complete multimodal learning system for cybersecurity"""
    
    def __init__(self, num_classes: int = 10, device: str = "cpu"):
        self.num_classes = num_classes
        self.device = device
        
        # Initialize model
        self.model = MultimodalSecurityClassifier(num_classes=num_classes)
        self.model.to(device)
        
        # Data processors
        self.text_processor = self._create_text_processor()
        self.network_processor = self._create_network_processor()
        self.visual_processor = self._create_visual_processor()
        
        # Training state
        self.optimizer = None
        self.criterion = nn.CrossEntropyLoss()
        self.logger = logging.getLogger(__name__)
        
    def _create_text_processor(self):
        """Create text data processor"""
        # Simple tokenizer (in production, use proper tokenizer)
        def process_text(text_data: TextData) -> torch.Tensor:
            # Simple word-based tokenization
            words = text_data.content.lower().split()
            # Convert to token IDs (simplified)
            token_ids = [hash(word) % 10000 for word in words[:512]]  # Max 512 tokens
            
            # Pad or truncate
            if len(token_ids) < 512:
                token_ids.extend([0] * (512 - len(token_ids)))
            else:
                token_ids = token_ids[:512]
            
            return torch.tensor(token_ids, dtype=torch.long)
        
        return process_text
    
    def _create_network_processor(self):
        """Create network data processor"""
        def process_network(network_data: NetworkData) -> Dict[str, torch.Tensor]:
            # Process flow features
            flow_features = torch.tensor([
                network_data.flow_features.get('packet_count', 0),
                network_data.flow_features.get('byte_count', 0),
                network_data.flow_features.get('duration', 0),
                network_data.flow_features.get('avg_packet_size', 0),
                network_data.flow_features.get('packets_per_second', 0)
            ] + [0] * 45, dtype=torch.float32)[:50]  # Ensure exactly 50 features
            
            # Process packet bytes (simplified)
            packet_bytes = list(network_data.packet_data[:1024])  # First 1024 bytes
            if len(packet_bytes) < 1024:
                packet_bytes.extend([0] * (1024 - len(packet_bytes)))
            packet_tensor = torch.tensor(packet_bytes, dtype=torch.long)
            
            # Protocol mapping (simplified)
            protocol_map = {'tcp': 0, 'udp': 1, 'icmp': 2, 'http': 3, 'https': 4}
            protocol_id = torch.tensor(
                protocol_map.get(network_data.protocol.lower(), 5), 
                dtype=torch.long
            )
            
            return {
                'flow_features': flow_features,
                'packet_bytes': packet_tensor,
                'protocol_ids': protocol_id
            }
        
        return process_network
    
    def _create_visual_processor(self):
        """Create visual data processor"""
        def process_visual(visual_data: VisualData) -> torch.Tensor:
            # Convert numpy array to tensor
            if visual_data.image_data.shape[-1] == 3:  # RGB
                image_tensor = torch.from_numpy(visual_data.image_data).float()
                image_tensor = image_tensor.permute(2, 0, 1)  # HWC to CHW
            else:
                # Handle grayscale or other formats
                image_tensor = torch.from_numpy(visual_data.image_data).float()
                if len(image_tensor.shape) == 2:
                    image_tensor = image_tensor.unsqueeze(0).repeat(3, 1, 1)  # Convert to RGB
            
            # Resize to standard size (simplified)
            if image_tensor.shape[1] != 224 or image_tensor.shape[2] != 224:
                image_tensor = F.interpolate(
                    image_tensor.unsqueeze(0), size=(224, 224), mode='bilinear'
                ).squeeze(0)
            
            # Normalize
            image_tensor = image_tensor / 255.0
            
            return image_tensor
        
        return process_visual
    
    def prepare_batch(self, samples: List[MultimodalSample]) -> Dict[str, torch.Tensor]:
        """Prepare a batch of multimodal samples"""
        batch = {
            'text_tokens': [],
            'network_data': {'flow_features': [], 'packet_bytes': [], 'protocol_ids': []},
            'visual_data': [],
            'labels': [],
            'sample_ids': []
        }
        
        for sample in samples:
            batch['sample_ids'].append(sample.sample_id)
            
            # Process text
            if sample.text_data:
                text_tokens = self.text_processor(sample.text_data)
                batch['text_tokens'].append(text_tokens)
            else:
                batch['text_tokens'].append(None)
            
            # Process network data
            if sample.network_data:
                network_processed = self.network_processor(sample.network_data)
                batch['network_data']['flow_features'].append(network_processed['flow_features'])
                batch['network_data']['packet_bytes'].append(network_processed['packet_bytes'])
                batch['network_data']['protocol_ids'].append(network_processed['protocol_ids'])
            else:
                batch['network_data']['flow_features'].append(None)
                batch['network_data']['packet_bytes'].append(None)
                batch['network_data']['protocol_ids'].append(None)
            
            # Process visual data
            if sample.visual_data:
                visual_processed = self.visual_processor(sample.visual_data)
                batch['visual_data'].append(visual_processed)
            else:
                batch['visual_data'].append(None)
            
            # Labels
            batch['labels'].append(sample.label)
        
        # Convert to tensors
        result = {}
        
        # Text tokens
        valid_text = [t for t in batch['text_tokens'] if t is not None]
        if valid_text:
            result['text_tokens'] = torch.stack(valid_text).to(self.device)
        
        # Network data
        valid_flow = [f for f in batch['network_data']['flow_features'] if f is not None]
        valid_packets = [p for p in batch['network_data']['packet_bytes'] if p is not None]
        valid_protocols = [p for p in batch['network_data']['protocol_ids'] if p is not None]
        
        if valid_flow:
            result['network_data'] = {
                'flow_features': torch.stack(valid_flow).to(self.device),
                'packet_bytes': torch.stack(valid_packets).to(self.device),
                'protocol_ids': torch.stack(valid_protocols).to(self.device)
            }
        
        # Visual data
        valid_visual = [v for v in batch['visual_data'] if v is not None]
        if valid_visual:
            result['visual_data'] = torch.stack(valid_visual).to(self.device)
        
        # Labels (convert string labels to indices)
        label_map = {
            'benign': 0, 'malware': 1, 'phishing': 2, 'ddos': 3, 'intrusion': 4,
            'lateral_movement': 5, 'data_exfiltration': 6, 'ransomware': 7,
            'insider_threat': 8, 'unknown': 9
        }
        label_indices = [label_map.get(label, 9) for label in batch['labels']]
        result['labels'] = torch.tensor(label_indices, dtype=torch.long).to(self.device)
        
        return result
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step"""
        if self.optimizer is None:
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        outputs = self.model(
            text_tokens=batch.get('text_tokens'),
            network_data=batch.get('network_data'),
            visual_data=batch.get('visual_data'),
            return_individual_outputs=True
        )
        
        # Calculate losses
        losses = {}
        total_loss = 0
        
        labels = batch['labels']
        
        # Main fusion loss
        if 'fused_logits' in outputs:
            fusion_loss = self.criterion(outputs['fused_logits'], labels)
            losses['fusion_loss'] = fusion_loss.item()
            total_loss += fusion_loss
        
        # Auxiliary losses for individual modalities
        aux_weight = 0.3
        if 'text_logits' in outputs:
            text_loss = self.criterion(outputs['text_logits'], labels)
            losses['text_loss'] = text_loss.item()
            total_loss += aux_weight * text_loss
        
        if 'network_logits' in outputs:
            network_loss = self.criterion(outputs['network_logits'], labels)
            losses['network_loss'] = network_loss.item()
            total_loss += aux_weight * network_loss
        
        if 'visual_logits' in outputs:
            visual_loss = self.criterion(outputs['visual_logits'], labels)
            losses['visual_loss'] = visual_loss.item()
            total_loss += aux_weight * visual_loss
        
        # Backward pass
        total_loss.backward()
        self.optimizer.step()
        
        losses['total_loss'] = total_loss.item()
        return losses
    
    def predict(self, samples: List[MultimodalSample]) -> List[Dict[str, Any]]:
        """Make predictions on multimodal samples"""
        self.model.eval()
        
        batch = self.prepare_batch(samples)
        predictions = []
        
        with torch.no_grad():
            outputs = self.model(
                text_tokens=batch.get('text_tokens'),
                network_data=batch.get('network_data'),
                visual_data=batch.get('visual_data'),
                return_individual_outputs=True
            )
            
            # Get predictions from fusion layer
            if 'fused_logits' in outputs:
                probs = F.softmax(outputs['fused_logits'], dim=1)
                pred_classes = torch.argmax(probs, dim=1)
                confidence_scores = torch.max(probs, dim=1)[0]
                
                # Class mapping
                class_names = [
                    'benign', 'malware', 'phishing', 'ddos', 'intrusion',
                    'lateral_movement', 'data_exfiltration', 'ransomware',
                    'insider_threat', 'unknown'
                ]
                
                for i, sample in enumerate(samples):
                    predictions.append({
                        'sample_id': sample.sample_id,
                        'predicted_class': class_names[pred_classes[i].item()],
                        'confidence': confidence_scores[i].item(),
                        'class_probabilities': {
                            class_names[j]: probs[i][j].item() 
                            for j in range(len(class_names))
                        }
                    })
        
        return predictions

# Example usage and testing
if __name__ == "__main__":
    print("🔀 Multimodal Learning System Testing:")
    print("=" * 50)
    
    # Initialize system
    multimodal_system = MultimodalSecuritySystem(num_classes=10, device="cpu")
    
    # Create sample multimodal data
    print("\n📊 Creating sample multimodal data...")
    
    # Text data sample
    text_sample = TextData(
        content="suspicious network activity detected from ip 192.168.1.100 attempting connection to external server",
        data_type="security_log",
        metadata={"source": "ids", "severity": "high"},
        timestamp=datetime.now().isoformat(),
        source="security_system"
    )
    
    # Network data sample
    network_sample = NetworkData(
        packet_data=b'\x45\x00\x00\x3c\x1c\x46\x40\x00\x40\x06\x00\x00\xc0\xa8\x01\x64' * 64,  # Sample packet
        flow_features={
            "packet_count": 150,
            "byte_count": 9600,
            "duration": 30.5,
            "avg_packet_size": 64,
            "packets_per_second": 4.9
        },
        protocol="tcp",
        source_ip="192.168.1.100",
        dest_ip="external_server",
        timestamp=datetime.now().isoformat(),
        metadata={"port": 443, "flags": ["SYN", "ACK"]}
    )
    
    # Visual data sample (synthetic network topology)
    visual_sample = VisualData(
        image_data=np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8),
        image_type="network_topology",
        features={"nodes": 15, "edges": 23, "anomalous_connections": 2},
        timestamp=datetime.now().isoformat(),
        metadata={"generated": True, "tool": "network_visualizer"}
    )
    
    # Create multimodal samples
    samples = [
        MultimodalSample(
            sample_id="sample_001",
            text_data=text_sample,
            network_data=network_sample,
            visual_data=visual_sample,
            label="intrusion",
            confidence=0.85,
            timestamp=datetime.now().isoformat()
        ),
        MultimodalSample(
            sample_id="sample_002",
            text_data=text_sample,
            network_data=None,  # Missing network data
            visual_data=visual_sample,
            label="malware",
            confidence=0.92,
            timestamp=datetime.now().isoformat()
        ),
        MultimodalSample(
            sample_id="sample_003",
            text_data=None,  # Missing text data
            network_data=network_sample,
            visual_data=None,  # Missing visual data
            label="benign",
            confidence=0.78,
            timestamp=datetime.now().isoformat()
        )
    ]
    
    # Test batch preparation
    print("🔧 Testing batch preparation...")
    batch = multimodal_system.prepare_batch(samples)
    print(f"  Batch components: {list(batch.keys())}")
    if 'text_tokens' in batch:
        print(f"  Text tokens shape: {batch['text_tokens'].shape}")
    if 'network_data' in batch:
        print(f"  Network flow features shape: {batch['network_data']['flow_features'].shape}")
    if 'visual_data' in batch:
        print(f"  Visual data shape: {batch['visual_data'].shape}")
    
    # Test inference
    print("\n🔮 Testing multimodal inference...")
    predictions = multimodal_system.predict(samples)
    
    for pred in predictions:
        print(f"\n  Sample: {pred['sample_id']}")
        print(f"    Predicted: {pred['predicted_class']}")
        print(f"    Confidence: {pred['confidence']:.3f}")
        print(f"    Top 3 probabilities:")
        sorted_probs = sorted(pred['class_probabilities'].items(), 
                            key=lambda x: x[1], reverse=True)[:3]
        for class_name, prob in sorted_probs:
            print(f"      {class_name}: {prob:.3f}")
    
    # Test training step
    print("\n🎓 Testing training step...")
    losses = multimodal_system.train_step(batch)
    print(f"  Training losses: {losses}")
    
    print("\n✅ Multimodal Learning System implemented and tested")
    print(f"  Model parameters: {sum(p.numel() for p in multimodal_system.model.parameters()):,}")
    print(f"  Supported modalities: Text, Network, Visual")
    print(f"  Fusion strategy: Cross-modal attention with learnable gates")
