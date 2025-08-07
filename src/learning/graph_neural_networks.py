"""
Graph Neural Networks for Cybersecurity
Advanced threat modeling using graph-based approaches
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import networkx as nx
from typing import Dict, List, Optional, Any, Tuple, Union, Set
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import logging
from abc import ABC, abstractmethod
from collections import defaultdict
import sqlite3
import pickle
from enum import Enum

class NodeType(Enum):
    HOST = "host"
    USER = "user" 
    PROCESS = "process"
    FILE = "file"
    NETWORK = "network"
    SERVICE = "service"
    VULNERABILITY = "vulnerability"
    THREAT = "threat"
    ASSET = "asset"
    DOMAIN = "domain"

class EdgeType(Enum):
    COMMUNICATES = "communicates"
    EXECUTES = "executes"
    ACCESSES = "accesses"
    CONTAINS = "contains"
    DEPENDS = "depends"
    EXPLOITS = "exploits"
    LATERAL_MOVE = "lateral_move"
    EXFILTRATES = "exfiltrates"
    CONTROLS = "controls"
    TRUSTS = "trusts"

@dataclass
class GraphNode:
    """Graph node representing a cybersecurity entity"""
    node_id: str
    node_type: NodeType
    properties: Dict[str, Any]
    risk_score: float
    timestamp: str
    metadata: Dict[str, Any]

@dataclass
class GraphEdge:
    """Graph edge representing a cybersecurity relationship"""
    edge_id: str
    source_id: str
    target_id: str
    edge_type: EdgeType
    properties: Dict[str, Any]
    weight: float
    confidence: float
    timestamp: str
    metadata: Dict[str, Any]

@dataclass
class SecurityGraph:
    """Complete security graph representation"""
    graph_id: str
    nodes: List[GraphNode]
    edges: List[GraphEdge]
    global_properties: Dict[str, Any]
    timestamp: str
    metadata: Dict[str, Any]

class GraphConvolutionalLayer(nn.Module):
    """Graph Convolutional Network layer for cybersecurity graphs"""
    
    def __init__(self, input_dim: int, output_dim: int, 
                 activation: Optional[nn.Module] = None, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Transformation matrices for different edge types
        self.edge_transforms = nn.ModuleDict({
            edge_type.value: nn.Linear(input_dim, output_dim, bias=False)
            for edge_type in EdgeType
        })
        
        # Self-loop transformation
        self.self_transform = nn.Linear(input_dim, output_dim, bias=False)
        
        # Bias term
        self.bias = nn.Parameter(torch.zeros(output_dim))
        
        # Normalization and regularization
        self.batch_norm = nn.BatchNorm1d(output_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation or nn.ReLU()
        
        # Edge attention mechanism
        self.edge_attention = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),
            nn.Tanh(),
            nn.Linear(output_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, node_features: torch.Tensor, adjacency_dict: Dict[str, torch.Tensor],
                edge_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass of GCN layer
        
        Args:
            node_features: [num_nodes, input_dim]
            adjacency_dict: Dict mapping edge types to adjacency matrices
            edge_features: Dict mapping edge types to edge feature matrices
        """
        batch_size, num_nodes, _ = node_features.shape
        
        # Self-loop transformation
        output = self.self_transform(node_features)
        
        # Process each edge type
        for edge_type_str, adj_matrix in adjacency_dict.items():
            if edge_type_str in self.edge_transforms and adj_matrix.sum() > 0:
                # Transform features for this edge type
                transformed = self.edge_transforms[edge_type_str](node_features)
                
                # Apply adjacency matrix (message passing)
                if len(adj_matrix.shape) == 2:
                    adj_matrix = adj_matrix.unsqueeze(0).expand(batch_size, -1, -1)
                
                # Compute attention weights for edges
                if edge_type_str in edge_features:
                    edge_feat = edge_features[edge_type_str]
                    # Create pairwise features for attention
                    source_features = transformed.unsqueeze(2).expand(-1, -1, num_nodes, -1)
                    target_features = transformed.unsqueeze(1).expand(-1, num_nodes, -1, -1)
                    pairwise_features = torch.cat([source_features, target_features], dim=-1)
                    
                    attention_weights = self.edge_attention(pairwise_features).squeeze(-1)
                    attention_weights = attention_weights * adj_matrix  # Mask non-edges
                    attention_weights = F.softmax(attention_weights, dim=-1)
                else:
                    attention_weights = adj_matrix
                
                # Message aggregation with attention
                aggregated = torch.bmm(attention_weights, transformed)
                output += aggregated
        
        # Apply bias, normalization, activation, and dropout
        output += self.bias
        
        # Reshape for batch normalization
        output_reshaped = output.view(-1, self.output_dim)
        output_reshaped = self.batch_norm(output_reshaped)
        output = output_reshaped.view(batch_size, num_nodes, -1)
        
        output = self.activation(output)
        output = self.dropout(output)
        
        return output

class GraphAttentionLayer(nn.Module):
    """Graph Attention Network layer for cybersecurity graphs"""
    
    def __init__(self, input_dim: int, output_dim: int, num_heads: int = 8,
                 dropout: float = 0.1, alpha: float = 0.2):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads
        self.alpha = alpha
        
        assert output_dim % num_heads == 0
        
        # Linear transformations for queries, keys, values
        self.query_transform = nn.Linear(input_dim, output_dim)
        self.key_transform = nn.Linear(input_dim, output_dim)
        self.value_transform = nn.Linear(input_dim, output_dim)
        
        # Attention mechanism
        self.attention = nn.Parameter(torch.randn(num_heads, 2 * self.head_dim))
        
        # Output projection
        self.output_proj = nn.Linear(output_dim, output_dim)
        
        # Regularization
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(output_dim)
        
        # Edge type embeddings
        self.edge_type_embeddings = nn.Embedding(len(EdgeType), self.head_dim)
        
    def forward(self, node_features: torch.Tensor, adjacency_dict: Dict[str, torch.Tensor],
                edge_types_matrix: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of GAT layer
        
        Args:
            node_features: [batch_size, num_nodes, input_dim]
            adjacency_dict: Dict mapping edge types to adjacency matrices
            edge_types_matrix: [batch_size, num_nodes, num_nodes] - edge type indices
        """
        batch_size, num_nodes, _ = node_features.shape
        
        # Linear transformations
        queries = self.query_transform(node_features)  # [batch_size, num_nodes, output_dim]
        keys = self.key_transform(node_features)
        values = self.value_transform(node_features)
        
        # Reshape for multi-head attention
        queries = queries.view(batch_size, num_nodes, self.num_heads, self.head_dim)
        keys = keys.view(batch_size, num_nodes, self.num_heads, self.head_dim)
        values = values.view(batch_size, num_nodes, self.num_heads, self.head_dim)
        
        # Compute attention scores
        attention_outputs = []
        
        for head in range(self.num_heads):
            q_h = queries[:, :, head, :]  # [batch_size, num_nodes, head_dim]
            k_h = keys[:, :, head, :]
            v_h = values[:, :, head, :]
            
            # Expand for pairwise computation
            q_expanded = q_h.unsqueeze(2).expand(-1, -1, num_nodes, -1)
            k_expanded = k_h.unsqueeze(1).expand(-1, num_nodes, -1, -1)
            
            # Concatenate for attention computation
            attention_input = torch.cat([q_expanded, k_expanded], dim=-1)  # [batch_size, num_nodes, num_nodes, 2*head_dim]
            
            # Compute attention scores
            attention_scores = torch.matmul(attention_input, self.attention[head])  # [batch_size, num_nodes, num_nodes]
            
            # Add edge type embeddings
            edge_type_embeds = self.edge_type_embeddings(edge_types_matrix)  # [batch_size, num_nodes, num_nodes, head_dim]
            edge_attention = torch.sum(edge_type_embeds * q_expanded.unsqueeze(-2), dim=-1)
            attention_scores += edge_attention
            
            # Apply leaky ReLU
            attention_scores = F.leaky_relu(attention_scores, self.alpha)
            
            # Create mask from adjacency matrices
            adjacency_mask = torch.zeros_like(attention_scores, dtype=torch.bool)
            for edge_type_str, adj_matrix in adjacency_dict.items():
                if len(adj_matrix.shape) == 2:
                    adj_matrix = adj_matrix.unsqueeze(0).expand(batch_size, -1, -1)
                adjacency_mask = adjacency_mask | (adj_matrix > 0)
            
            # Mask attention scores
            attention_scores = attention_scores.masked_fill(~adjacency_mask, float('-inf'))
            
            # Apply softmax
            attention_weights = F.softmax(attention_scores, dim=-1)
            attention_weights = self.dropout(attention_weights)
            
            # Apply attention to values
            attended_values = torch.bmm(attention_weights, v_h)  # [batch_size, num_nodes, head_dim]
            attention_outputs.append(attended_values)
        
        # Concatenate multi-head outputs
        multi_head_output = torch.cat(attention_outputs, dim=-1)  # [batch_size, num_nodes, output_dim]
        
        # Output projection and residual connection
        output = self.output_proj(multi_head_output)
        output = self.layer_norm(output + node_features if node_features.shape[-1] == output.shape[-1] else output)
        
        return output

class SecurityGraphEncoder(nn.Module):
    """Graph encoder for cybersecurity networks"""
    
    def __init__(self, node_feature_dim: int, edge_feature_dim: int, 
                 hidden_dim: int = 256, num_layers: int = 3,
                 use_attention: bool = True):
        super().__init__()
        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_attention = use_attention
        
        # Node feature embedding
        self.node_embedding = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Node type embeddings
        self.node_type_embedding = nn.Embedding(len(NodeType), hidden_dim // 4)
        
        # Graph layers
        self.graph_layers = nn.ModuleList()
        for i in range(num_layers):
            if use_attention:
                layer = GraphAttentionLayer(
                    input_dim=hidden_dim + (hidden_dim // 4 if i == 0 else 0),
                    output_dim=hidden_dim,
                    num_heads=8
                )
            else:
                layer = GraphConvolutionalLayer(
                    input_dim=hidden_dim + (hidden_dim // 4 if i == 0 else 0),
                    output_dim=hidden_dim
                )
            self.graph_layers.append(layer)
        
        # Global graph pooling
        self.global_attention = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        self.global_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, node_features: torch.Tensor, node_types: torch.Tensor,
                adjacency_dict: Dict[str, torch.Tensor], edge_features: Dict[str, torch.Tensor],
                edge_types_matrix: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Encode security graph
        
        Args:
            node_features: [batch_size, num_nodes, node_feature_dim]
            node_types: [batch_size, num_nodes] - node type indices
            adjacency_dict: Dict of adjacency matrices for each edge type
            edge_features: Dict of edge features for each edge type
            edge_types_matrix: [batch_size, num_nodes, num_nodes] - edge type indices
        """
        batch_size, num_nodes, _ = node_features.shape
        
        # Embed node features and types
        embedded_features = self.node_embedding(node_features)
        type_embeddings = self.node_type_embedding(node_types)
        
        # Combine node features and type embeddings for first layer
        x = torch.cat([embedded_features, type_embeddings], dim=-1)
        
        # Apply graph layers
        layer_outputs = [x]
        for i, layer in enumerate(self.graph_layers):
            if self.use_attention:
                x = layer(x, adjacency_dict, edge_types_matrix)
            else:
                x = layer(x, adjacency_dict, edge_features)
            layer_outputs.append(x)
        
        # Global graph representation
        # Use attention-based pooling
        graph_tokens = x  # [batch_size, num_nodes, hidden_dim]
        global_repr, attention_weights = self.global_attention(
            graph_tokens, graph_tokens, graph_tokens
        )
        
        # Average pooling for final graph representation
        global_repr = global_repr.mean(dim=1)  # [batch_size, hidden_dim]
        global_repr = self.global_mlp(global_repr)
        
        return {
            'node_embeddings': x,  # [batch_size, num_nodes, hidden_dim]
            'graph_embedding': global_repr,  # [batch_size, hidden_dim]
            'attention_weights': attention_weights,  # [batch_size, num_nodes, num_nodes]
            'layer_outputs': layer_outputs
        }

class ThreatPropagationGNN(nn.Module):
    """GNN for modeling threat propagation in security graphs"""
    
    def __init__(self, node_feature_dim: int, edge_feature_dim: int, 
                 num_threat_types: int = 10, hidden_dim: int = 256):
        super().__init__()
        self.num_threat_types = num_threat_types
        
        # Graph encoder
        self.graph_encoder = SecurityGraphEncoder(
            node_feature_dim=node_feature_dim,
            edge_feature_dim=edge_feature_dim,
            hidden_dim=hidden_dim,
            num_layers=4,
            use_attention=True
        )
        
        # Threat propagation layers
        self.propagation_layers = nn.ModuleList([
            GraphConvolutionalLayer(hidden_dim, hidden_dim) for _ in range(3)
        ])
        
        # Threat type classifier for each node
        self.threat_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_threat_types)
        )
        
        # Risk score predictor
        self.risk_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Propagation probability predictor
        self.propagation_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, node_features: torch.Tensor, node_types: torch.Tensor,
                adjacency_dict: Dict[str, torch.Tensor], edge_features: Dict[str, torch.Tensor],
                edge_types_matrix: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Predict threat propagation in security graph"""
        
        # Encode graph
        graph_repr = self.graph_encoder(
            node_features, node_types, adjacency_dict, edge_features, edge_types_matrix
        )
        
        node_embeddings = graph_repr['node_embeddings']
        
        # Apply propagation layers
        propagated_embeddings = node_embeddings
        for layer in self.propagation_layers:
            propagated_embeddings = layer(propagated_embeddings, adjacency_dict, edge_features)
        
        # Predict threat types for each node
        threat_logits = self.threat_classifier(propagated_embeddings)
        
        # Predict risk scores for each node
        risk_scores = self.risk_predictor(propagated_embeddings).squeeze(-1)
        
        # Predict propagation probabilities for each edge
        batch_size, num_nodes, hidden_dim = propagated_embeddings.shape
        
        # Create pairwise embeddings for edges
        source_embeddings = propagated_embeddings.unsqueeze(2).expand(-1, -1, num_nodes, -1)
        target_embeddings = propagated_embeddings.unsqueeze(1).expand(-1, num_nodes, -1, -1)
        edge_embeddings = torch.cat([source_embeddings, target_embeddings], dim=-1)
        
        propagation_probs = self.propagation_predictor(edge_embeddings).squeeze(-1)
        
        return {
            'threat_logits': threat_logits,      # [batch_size, num_nodes, num_threat_types]
            'risk_scores': risk_scores,          # [batch_size, num_nodes]
            'propagation_probs': propagation_probs,  # [batch_size, num_nodes, num_nodes]
            'node_embeddings': propagated_embeddings,
            'graph_embedding': graph_repr['graph_embedding']
        }

class SecurityGraphAnalyzer:
    """Complete security graph analysis system using GNNs"""
    
    def __init__(self, database_path: str = "security_graphs.db"):
        self.database_path = database_path
        self.logger = logging.getLogger(__name__)
        
        # Initialize database
        self._init_database()
        
        # Model components
        self.threat_propagation_model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Graph processing
        self.node_type_to_idx = {node_type: idx for idx, node_type in enumerate(NodeType)}
        self.edge_type_to_idx = {edge_type: idx for idx, edge_type in enumerate(EdgeType)}
        self.threat_types = [
            'malware', 'phishing', 'ddos', 'intrusion', 'lateral_movement',
            'data_exfiltration', 'ransomware', 'insider_threat', 'apt', 'benign'
        ]
        
    def _init_database(self):
        """Initialize SQLite database for graph storage"""
        with sqlite3.connect(self.database_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS security_graphs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    graph_id TEXT UNIQUE NOT NULL,
                    graph_data BLOB NOT NULL,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS threat_analyses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    graph_id TEXT NOT NULL,
                    analysis_type TEXT NOT NULL,
                    results BLOB NOT NULL,
                    confidence_score REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (graph_id) REFERENCES security_graphs (graph_id)
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS propagation_predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    graph_id TEXT NOT NULL,
                    source_node TEXT NOT NULL,
                    target_node TEXT NOT NULL,
                    propagation_prob REAL NOT NULL,
                    threat_type TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (graph_id) REFERENCES security_graphs (graph_id)
                )
            """)
            
    def create_graph_from_data(self, nodes: List[Dict], edges: List[Dict], 
                              graph_id: Optional[str] = None) -> SecurityGraph:
        """Create security graph from raw data"""
        if graph_id is None:
            graph_id = f"graph_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Process nodes
        graph_nodes = []
        for node_data in nodes:
            node = GraphNode(
                node_id=node_data['id'],
                node_type=NodeType(node_data['type']),
                properties=node_data.get('properties', {}),
                risk_score=node_data.get('risk_score', 0.5),
                timestamp=node_data.get('timestamp', datetime.now().isoformat()),
                metadata=node_data.get('metadata', {})
            )
            graph_nodes.append(node)
        
        # Process edges
        graph_edges = []
        for edge_data in edges:
            edge = GraphEdge(
                edge_id=edge_data.get('id', f"{edge_data['source']}_{edge_data['target']}"),
                source_id=edge_data['source'],
                target_id=edge_data['target'],
                edge_type=EdgeType(edge_data['type']),
                properties=edge_data.get('properties', {}),
                weight=edge_data.get('weight', 1.0),
                confidence=edge_data.get('confidence', 1.0),
                timestamp=edge_data.get('timestamp', datetime.now().isoformat()),
                metadata=edge_data.get('metadata', {})
            )
            graph_edges.append(edge)
        
        security_graph = SecurityGraph(
            graph_id=graph_id,
            nodes=graph_nodes,
            edges=graph_edges,
            global_properties={
                'num_nodes': len(graph_nodes),
                'num_edges': len(graph_edges),
                'node_types': list(set(node.node_type.value for node in graph_nodes)),
                'edge_types': list(set(edge.edge_type.value for edge in graph_edges))
            },
            timestamp=datetime.now().isoformat(),
            metadata={'created_by': 'SecurityGraphAnalyzer'}
        )
        
        # Save to database
        self.save_graph(security_graph)
        
        return security_graph
    
    def save_graph(self, graph: SecurityGraph):
        """Save security graph to database"""
        graph_data = pickle.dumps(asdict(graph))
        metadata = json.dumps(graph.metadata)
        
        with sqlite3.connect(self.database_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO security_graphs (graph_id, graph_data, metadata) VALUES (?, ?, ?)",
                (graph.graph_id, graph_data, metadata)
            )
    
    def load_graph(self, graph_id: str) -> Optional[SecurityGraph]:
        """Load security graph from database"""
        with sqlite3.connect(self.database_path) as conn:
            result = conn.execute(
                "SELECT graph_data FROM security_graphs WHERE graph_id = ?",
                (graph_id,)
            ).fetchone()
            
            if result:
                graph_dict = pickle.loads(result[0])
                # Reconstruct SecurityGraph object
                nodes = [GraphNode(**node) for node in graph_dict['nodes']]
                edges = [GraphEdge(**edge) for edge in graph_dict['edges']]
                
                # Convert string enums back to enum objects
                for node in nodes:
                    node.node_type = NodeType(node.node_type)
                for edge in edges:
                    edge.edge_type = EdgeType(edge.edge_type)
                
                return SecurityGraph(
                    graph_id=graph_dict['graph_id'],
                    nodes=nodes,
                    edges=edges,
                    global_properties=graph_dict['global_properties'],
                    timestamp=graph_dict['timestamp'],
                    metadata=graph_dict['metadata']
                )
        
        return None
    
    def convert_to_tensors(self, graph: SecurityGraph) -> Dict[str, torch.Tensor]:
        """Convert security graph to tensor representation"""
        node_id_to_idx = {node.node_id: idx for idx, node in enumerate(graph.nodes)}
        num_nodes = len(graph.nodes)
        
        # Node features (simplified feature extraction)
        node_features = []
        node_types = []
        
        for node in graph.nodes:
            # Create feature vector from node properties
            features = [
                node.risk_score,
                len(node.properties),
                hash(str(node.properties)) % 1000 / 1000.0,  # Property hash as feature
                1.0 if 'critical' in node.properties.get('tags', []) else 0.0,
                1.0 if 'external' in node.properties.get('tags', []) else 0.0
            ]
            
            # Pad to fixed size
            while len(features) < 20:
                features.append(0.0)
            
            node_features.append(features[:20])
            node_types.append(self.node_type_to_idx[node.node_type])
        
        # Adjacency matrices for each edge type
        adjacency_dict = {}
        edge_features_dict = {}
        edge_types_matrix = torch.zeros(num_nodes, num_nodes, dtype=torch.long)
        
        for edge_type in EdgeType:
            adjacency_dict[edge_type.value] = torch.zeros(num_nodes, num_nodes)
            edge_features_dict[edge_type.value] = torch.zeros(num_nodes, num_nodes, 5)
        
        for edge in graph.edges:
            if edge.source_id in node_id_to_idx and edge.target_id in node_id_to_idx:
                src_idx = node_id_to_idx[edge.source_id]
                tgt_idx = node_id_to_idx[edge.target_id]
                edge_type_str = edge.edge_type.value
                
                # Set adjacency
                adjacency_dict[edge_type_str][src_idx, tgt_idx] = 1.0
                
                # Edge features
                edge_feat = [
                    edge.weight,
                    edge.confidence,
                    len(edge.properties),
                    hash(str(edge.properties)) % 1000 / 1000.0,
                    1.0 if 'suspicious' in edge.properties.get('flags', []) else 0.0
                ]
                edge_features_dict[edge_type_str][src_idx, tgt_idx] = torch.tensor(edge_feat)
                
                # Edge type matrix
                edge_types_matrix[src_idx, tgt_idx] = self.edge_type_to_idx[edge.edge_type]
        
        return {
            'node_features': torch.tensor(node_features, dtype=torch.float32).unsqueeze(0),
            'node_types': torch.tensor(node_types, dtype=torch.long).unsqueeze(0),
            'adjacency_dict': {k: v.unsqueeze(0) for k, v in adjacency_dict.items()},
            'edge_features': {k: v.unsqueeze(0) for k, v in edge_features_dict.items()},
            'edge_types_matrix': edge_types_matrix.unsqueeze(0),
            'node_id_to_idx': node_id_to_idx
        }
    
    def analyze_threat_propagation(self, graph: SecurityGraph) -> Dict[str, Any]:
        """Analyze threat propagation patterns in the security graph"""
        if self.threat_propagation_model is None:
            # Initialize model
            self.threat_propagation_model = ThreatPropagationGNN(
                node_feature_dim=20,
                edge_feature_dim=5,
                num_threat_types=len(self.threat_types),
                hidden_dim=256
            ).to(self.device)
        
        # Convert graph to tensors
        tensor_data = self.convert_to_tensors(graph)
        
        # Move to device
        for key, value in tensor_data.items():
            if isinstance(value, torch.Tensor):
                tensor_data[key] = value.to(self.device)
            elif isinstance(value, dict):
                tensor_data[key] = {k: v.to(self.device) for k, v in value.items()}
        
        # Model inference
        self.threat_propagation_model.eval()
        with torch.no_grad():
            results = self.threat_propagation_model(
                tensor_data['node_features'],
                tensor_data['node_types'],
                tensor_data['adjacency_dict'],
                tensor_data['edge_features'],
                tensor_data['edge_types_matrix']
            )
        
        # Process results
        threat_probs = F.softmax(results['threat_logits'], dim=-1).cpu().numpy()[0]
        risk_scores = results['risk_scores'].cpu().numpy()[0]
        propagation_probs = results['propagation_probs'].cpu().numpy()[0]
        
        # Create analysis results
        node_analyses = []
        for i, node in enumerate(graph.nodes):
            node_analysis = {
                'node_id': node.node_id,
                'node_type': node.node_type.value,
                'risk_score': float(risk_scores[i]),
                'threat_probabilities': {
                    threat_type: float(prob) 
                    for threat_type, prob in zip(self.threat_types, threat_probs[i])
                },
                'top_threat': self.threat_types[np.argmax(threat_probs[i])],
                'threat_confidence': float(np.max(threat_probs[i]))
            }
            node_analyses.append(node_analysis)
        
        # Edge propagation analysis
        edge_analyses = []
        node_id_to_idx = tensor_data['node_id_to_idx']
        idx_to_node_id = {idx: node_id for node_id, idx in node_id_to_idx.items()}
        
        for i in range(len(graph.nodes)):
            for j in range(len(graph.nodes)):
                if i != j and propagation_probs[i, j] > 0.1:  # Threshold for significant propagation
                    edge_analysis = {
                        'source_node': idx_to_node_id[i],
                        'target_node': idx_to_node_id[j],
                        'propagation_probability': float(propagation_probs[i, j]),
                        'source_risk': float(risk_scores[i]),
                        'target_risk': float(risk_scores[j])
                    }
                    edge_analyses.append(edge_analysis)
        
        analysis_result = {
            'graph_id': graph.graph_id,
            'analysis_timestamp': datetime.now().isoformat(),
            'node_analyses': node_analyses,
            'edge_analyses': edge_analyses,
            'summary': {
                'total_nodes': len(graph.nodes),
                'high_risk_nodes': sum(1 for analysis in node_analyses if analysis['risk_score'] > 0.7),
                'critical_propagation_paths': len([e for e in edge_analyses if e['propagation_probability'] > 0.8]),
                'dominant_threat_type': max(self.threat_types, key=lambda t: sum(
                    analysis['threat_probabilities'][t] for analysis in node_analyses
                ))
            }
        }
        
        # Save results
        self._save_analysis_results(graph.graph_id, 'threat_propagation', analysis_result)
        
        return analysis_result
    
    def _save_analysis_results(self, graph_id: str, analysis_type: str, results: Dict[str, Any]):
        """Save analysis results to database"""
        results_data = pickle.dumps(results)
        confidence_score = results.get('summary', {}).get('confidence', 0.8)
        
        with sqlite3.connect(self.database_path) as conn:
            conn.execute(
                "INSERT INTO threat_analyses (graph_id, analysis_type, results, confidence_score) VALUES (?, ?, ?, ?)",
                (graph_id, analysis_type, results_data, confidence_score)
            )
    
    def detect_attack_paths(self, graph: SecurityGraph, start_nodes: List[str], 
                           target_nodes: List[str]) -> List[List[str]]:
        """Detect potential attack paths using graph analysis"""
        # Convert to NetworkX graph for path analysis
        nx_graph = nx.DiGraph()
        
        # Add nodes
        for node in graph.nodes:
            nx_graph.add_node(node.node_id, 
                             node_type=node.node_type.value,
                             risk_score=node.risk_score,
                             properties=node.properties)
        
        # Add edges
        for edge in graph.edges:
            nx_graph.add_edge(edge.source_id, edge.target_id,
                             edge_type=edge.edge_type.value,
                             weight=1.0 / edge.weight if edge.weight > 0 else 1.0,
                             confidence=edge.confidence)
        
        # Find paths between start and target nodes
        attack_paths = []
        for start_node in start_nodes:
            for target_node in target_nodes:
                if start_node in nx_graph and target_node in nx_graph:
                    try:
                        # Find shortest paths
                        paths = list(nx.all_shortest_paths(nx_graph, start_node, target_node))
                        attack_paths.extend(paths)
                    except nx.NetworkXNoPath:
                        continue
        
        return attack_paths
    
    def get_graph_statistics(self, graph: SecurityGraph) -> Dict[str, Any]:
        """Compute comprehensive graph statistics"""
        # Convert to NetworkX for analysis
        nx_graph = nx.Graph()  # Undirected for centrality measures
        
        for node in graph.nodes:
            nx_graph.add_node(node.node_id, node_type=node.node_type.value)
        
        for edge in graph.edges:
            nx_graph.add_edge(edge.source_id, edge.target_id, weight=edge.weight)
        
        # Compute statistics
        stats = {
            'basic_stats': {
                'num_nodes': len(graph.nodes),
                'num_edges': len(graph.edges),
                'density': nx.density(nx_graph),
                'is_connected': nx.is_connected(nx_graph),
                'num_components': nx.number_connected_components(nx_graph)
            },
            'centrality_measures': {
                'degree_centrality': dict(nx.degree_centrality(nx_graph)),
                'betweenness_centrality': dict(nx.betweenness_centrality(nx_graph)),
                'closeness_centrality': dict(nx.closeness_centrality(nx_graph)),
                'eigenvector_centrality': dict(nx.eigenvector_centrality(nx_graph, max_iter=1000))
            },
            'node_type_distribution': {},
            'edge_type_distribution': {},
            'risk_score_distribution': {
                'mean': np.mean([node.risk_score for node in graph.nodes]),
                'std': np.std([node.risk_score for node in graph.nodes]),
                'min': min([node.risk_score for node in graph.nodes]),
                'max': max([node.risk_score for node in graph.nodes])
            }
        }
        
        # Node type distribution
        for node_type in NodeType:
            count = sum(1 for node in graph.nodes if node.node_type == node_type)
            if count > 0:
                stats['node_type_distribution'][node_type.value] = count
        
        # Edge type distribution
        for edge_type in EdgeType:
            count = sum(1 for edge in graph.edges if edge.edge_type == edge_type)
            if count > 0:
                stats['edge_type_distribution'][edge_type.value] = count
        
        return stats

# Example usage and testing
if __name__ == "__main__":
    print("🕸️ Graph Neural Networks for Cybersecurity Testing:")
    print("=" * 60)
    
    # Initialize the system
    analyzer = SecurityGraphAnalyzer()
    
    # Create sample security graph
    print("\n📊 Creating sample security graph...")
    
    # Sample nodes representing a network environment
    sample_nodes = [
        {'id': 'host_001', 'type': 'host', 'risk_score': 0.3, 'properties': {'ip': '192.168.1.10', 'os': 'Windows 10'}},
        {'id': 'host_002', 'type': 'host', 'risk_score': 0.7, 'properties': {'ip': '192.168.1.20', 'os': 'Linux', 'tags': ['critical']}},
        {'id': 'user_admin', 'type': 'user', 'risk_score': 0.9, 'properties': {'username': 'admin', 'privileges': 'high'}},
        {'id': 'user_john', 'type': 'user', 'risk_score': 0.4, 'properties': {'username': 'john', 'department': 'finance'}},
        {'id': 'service_web', 'type': 'service', 'risk_score': 0.6, 'properties': {'port': 80, 'protocol': 'http'}},
        {'id': 'file_config', 'type': 'file', 'risk_score': 0.8, 'properties': {'path': '/etc/passwd', 'permissions': '644'}},
        {'id': 'vuln_001', 'type': 'vulnerability', 'risk_score': 0.95, 'properties': {'cve': 'CVE-2023-1234', 'severity': 'critical'}},
        {'id': 'external_ip', 'type': 'network', 'risk_score': 0.5, 'properties': {'ip': '8.8.8.8', 'tags': ['external']}}
    ]
    
    # Sample edges representing relationships
    sample_edges = [
        {'source': 'user_admin', 'target': 'host_001', 'type': 'accesses', 'weight': 1.0, 'confidence': 0.9},
        {'source': 'user_john', 'target': 'host_002', 'type': 'accesses', 'weight': 0.8, 'confidence': 0.85},
        {'source': 'host_001', 'target': 'service_web', 'type': 'contains', 'weight': 1.0, 'confidence': 1.0},
        {'source': 'host_002', 'target': 'file_config', 'type': 'contains', 'weight': 1.0, 'confidence': 1.0},
        {'source': 'service_web', 'target': 'vuln_001', 'type': 'exploits', 'weight': 0.9, 'confidence': 0.8},
        {'source': 'host_001', 'target': 'host_002', 'type': 'lateral_move', 'weight': 0.6, 'confidence': 0.7},
        {'source': 'host_002', 'target': 'external_ip', 'type': 'communicates', 'weight': 0.7, 'confidence': 0.6},
        {'source': 'user_admin', 'target': 'file_config', 'type': 'accesses', 'weight': 0.8, 'confidence': 0.9}
    ]
    
    # Create security graph
    security_graph = analyzer.create_graph_from_data(sample_nodes, sample_edges, "test_network_001")
    print(f"  Created graph with {len(security_graph.nodes)} nodes and {len(security_graph.edges)} edges")
    
    # Test tensor conversion
    print("\n🔢 Testing tensor conversion...")
    tensor_data = analyzer.convert_to_tensors(security_graph)
    print(f"  Node features shape: {tensor_data['node_features'].shape}")
    print(f"  Node types shape: {tensor_data['node_types'].shape}")
    print(f"  Adjacency matrices: {list(tensor_data['adjacency_dict'].keys())}")
    print(f"  Edge types matrix shape: {tensor_data['edge_types_matrix'].shape}")
    
    # Test threat propagation analysis
    print("\n🦠 Testing threat propagation analysis...")
    threat_analysis = analyzer.analyze_threat_propagation(security_graph)
    
    print(f"  Analysis timestamp: {threat_analysis['analysis_timestamp']}")
    print(f"  Total nodes analyzed: {threat_analysis['summary']['total_nodes']}")
    print(f"  High-risk nodes: {threat_analysis['summary']['high_risk_nodes']}")
    print(f"  Critical propagation paths: {threat_analysis['summary']['critical_propagation_paths']}")
    print(f"  Dominant threat type: {threat_analysis['summary']['dominant_threat_type']}")
    
    print("\n  Top 3 highest risk nodes:")
    sorted_nodes = sorted(threat_analysis['node_analyses'], 
                         key=lambda x: x['risk_score'], reverse=True)[:3]
    for node in sorted_nodes:
        print(f"    {node['node_id']}: Risk={node['risk_score']:.3f}, Top Threat={node['top_threat']}")
    
    # Test attack path detection
    print("\n🎯 Testing attack path detection...")
    start_nodes = ['external_ip']
    target_nodes = ['file_config', 'user_admin']
    attack_paths = analyzer.detect_attack_paths(security_graph, start_nodes, target_nodes)
    
    print(f"  Found {len(attack_paths)} potential attack paths:")
    for i, path in enumerate(attack_paths[:3]):  # Show first 3 paths
        print(f"    Path {i+1}: {' -> '.join(path)}")
    
    # Test graph statistics
    print("\n📈 Testing graph statistics...")
    stats = analyzer.get_graph_statistics(security_graph)
    
    print(f"  Graph density: {stats['basic_stats']['density']:.3f}")
    print(f"  Connected components: {stats['basic_stats']['num_components']}")
    print(f"  Average risk score: {stats['risk_score_distribution']['mean']:.3f}")
    
    print("\n  Node type distribution:")
    for node_type, count in stats['node_type_distribution'].items():
        print(f"    {node_type}: {count}")
    
    print("\n  Top 3 nodes by betweenness centrality:")
    sorted_centrality = sorted(stats['centrality_measures']['betweenness_centrality'].items(),
                              key=lambda x: x[1], reverse=True)[:3]
    for node_id, centrality in sorted_centrality:
        print(f"    {node_id}: {centrality:.3f}")
    
    print("\n✅ Graph Neural Networks system implemented and tested")
    print(f"  Database: {analyzer.database_path}")
    print(f"  Supported node types: {len(NodeType)} types")
    print(f"  Supported edge types: {len(EdgeType)} types")
    print(f"  GNN architecture: Multi-layer GAT with attention-based pooling")
