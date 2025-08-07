"""
Advanced Transfer Learning System for Cybersecurity AI
Implements domain adaptation, multi-task learning, and knowledge transfer capabilities
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import logging
from abc import ABC, abstractmethod
from collections import defaultdict

@dataclass
class TransferLearningConfig:
    """Configuration for transfer learning setup"""
    source_domain: str
    target_domain: str
    transfer_method: str
    freeze_layers: List[str]
    adaptation_layers: List[str]
    learning_rates: Dict[str, float]
    loss_weights: Dict[str, float]

@dataclass
class DomainAdaptationResult:
    """Results from domain adaptation"""
    source_loss: float
    target_loss: float
    domain_loss: float
    adaptation_score: float
    transfer_effectiveness: float

class FeatureExtractor(nn.Module):
    """Base feature extractor for transfer learning"""
    
    def __init__(self, input_size: int, hidden_sizes: List[int], output_size: int):
        super().__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

class DomainClassifier(nn.Module):
    """Domain classifier for adversarial domain adaptation"""
    
    def __init__(self, feature_size: int, hidden_size: int = 128):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(feature_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_size // 2, 2)  # Binary: source vs target domain
        )
    
    def forward(self, x):
        return self.classifier(x)

class TaskClassifier(nn.Module):
    """Task-specific classifier"""
    
    def __init__(self, feature_size: int, num_classes: int, hidden_size: int = 128):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(feature_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, num_classes)
        )
    
    def forward(self, x):
        return self.classifier(x)

class GradientReversalLayer(torch.autograd.Function):
    """Gradient reversal layer for adversarial training"""
    
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

class TransferLearningModel(nn.Module):
    """Multi-purpose transfer learning model for cybersecurity tasks"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        self.config = config
        self.feature_size = config.get('feature_size', 256)
        
        # Feature extractor (shared across domains/tasks)
        self.feature_extractor = FeatureExtractor(
            input_size=config['input_size'],
            hidden_sizes=config.get('hidden_sizes', [512, 256]),
            output_size=self.feature_size
        )
        
        # Task-specific classifiers
        self.task_classifiers = nn.ModuleDict()
        for task_name, num_classes in config.get('tasks', {}).items():
            self.task_classifiers[task_name] = TaskClassifier(
                self.feature_size, num_classes
            )
        
        # Domain classifier for domain adaptation
        self.domain_classifier = DomainClassifier(self.feature_size)
        
        # Layer freezing setup
        self.frozen_layers = set(config.get('frozen_layers', []))
        
    def forward(self, x, task_name: str = None, alpha: float = 0.0):
        # Extract features
        features = self.feature_extractor(x)
        
        results = {}
        
        # Task classification
        if task_name and task_name in self.task_classifiers:
            results['task_output'] = self.task_classifiers[task_name](features)
        
        # Domain classification (with gradient reversal)
        if alpha > 0:
            reversed_features = GradientReversalLayer.apply(features, alpha)
            results['domain_output'] = self.domain_classifier(reversed_features)
        
        results['features'] = features
        return results
    
    def freeze_layers(self, layer_names: List[str]):
        """Freeze specified layers"""
        for name, param in self.named_parameters():
            if any(layer_name in name for layer_name in layer_names):
                param.requires_grad = False
                self.frozen_layers.add(name)
    
    def unfreeze_layers(self, layer_names: List[str]):
        """Unfreeze specified layers"""
        for name, param in self.named_parameters():
            if any(layer_name in name for layer_name in layer_names):
                param.requires_grad = True
                self.frozen_layers.discard(name)

class CyberTransferLearning:
    """Advanced transfer learning system for cybersecurity applications"""
    
    def __init__(self, config_path: str = None):
        self.config = self._load_config(config_path)
        self.models = {}
        self.transfer_history = []
        self.logger = logging.getLogger(__name__)
        
        # Cybersecurity domain mappings
        self.domain_mappings = {
            'malware_detection': {
                'features': ['file_size', 'entropy', 'api_calls', 'strings'],
                'related_domains': ['threat_detection', 'anomaly_detection']
            },
            'network_intrusion': {
                'features': ['packet_size', 'flow_duration', 'protocol', 'ports'],
                'related_domains': ['anomaly_detection', 'traffic_analysis']
            },
            'phishing_detection': {
                'features': ['url_features', 'content_features', 'metadata'],
                'related_domains': ['malware_detection', 'fraud_detection']
            },
            'vulnerability_assessment': {
                'features': ['code_metrics', 'dependencies', 'patterns'],
                'related_domains': ['malware_detection', 'threat_detection']
            },
            'threat_intelligence': {
                'features': ['iocs', 'ttps', 'attribution', 'context'],
                'related_domains': ['malware_detection', 'network_intrusion']
            }
        }
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load transfer learning configuration"""
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return json.load(f)
        
        # Default configuration
        return {
            'model_configs': {
                'malware_detection': {
                    'input_size': 1024,
                    'hidden_sizes': [512, 256],
                    'tasks': {'malware_classification': 10, 'family_detection': 20}
                },
                'network_intrusion': {
                    'input_size': 128,
                    'hidden_sizes': [256, 128],
                    'tasks': {'intrusion_detection': 2, 'attack_type': 15}
                }
            },
            'transfer_strategies': {
                'fine_tuning': {'freeze_ratio': 0.5, 'learning_rate_ratio': 0.1},
                'domain_adaptation': {'alpha_schedule': 'progressive', 'max_alpha': 1.0},
                'multi_task': {'task_weights': 'adaptive', 'sharing_layers': ['feature_extractor']}
            }
        }
    
    def create_transfer_model(self, source_domain: str, target_domain: str,
                             transfer_method: str = 'fine_tuning') -> str:
        """Create a transfer learning model"""
        model_id = f"transfer_{source_domain}_to_{target_domain}_{transfer_method}"
        
        # Get model configuration
        if target_domain in self.config['model_configs']:
            model_config = self.config['model_configs'][target_domain].copy()
        else:
            # Use source domain config as base
            model_config = self.config['model_configs'].get(
                source_domain, self.config['model_configs']['malware_detection']
            ).copy()
        
        # Create model
        model = TransferLearningModel(model_config)
        
        # Load pre-trained source model if available
        source_model_path = f"/models/{source_domain}_pretrained.pth"
        if os.path.exists(source_model_path):
            self._load_pretrained_weights(model, source_model_path, source_domain, target_domain)
        
        # Apply transfer learning strategy
        if transfer_method == 'fine_tuning':
            self._setup_fine_tuning(model, source_domain, target_domain)
        elif transfer_method == 'domain_adaptation':
            self._setup_domain_adaptation(model, source_domain, target_domain)
        elif transfer_method == 'multi_task':
            self._setup_multi_task_learning(model, source_domain, target_domain)
        
        self.models[model_id] = {
            'model': model,
            'source_domain': source_domain,
            'target_domain': target_domain,
            'transfer_method': transfer_method,
            'created_at': datetime.now().isoformat()
        }
        
        self.logger.info(f"Created transfer model: {model_id}")
        return model_id
    
    def _load_pretrained_weights(self, model: TransferLearningModel, 
                                source_path: str, source_domain: str, target_domain: str):
        """Load pre-trained weights with domain adaptation"""
        try:
            if os.path.exists(source_path):
                checkpoint = torch.load(source_path, map_location='cpu')
                
                # Load compatible weights
                model_dict = model.state_dict()
                pretrained_dict = {
                    k: v for k, v in checkpoint.items() 
                    if k in model_dict and v.size() == model_dict[k].size()
                }
                
                model_dict.update(pretrained_dict)
                model.load_state_dict(model_dict)
                
                self.logger.info(f"Loaded {len(pretrained_dict)} pre-trained layers")
            else:
                self.logger.warning(f"Pre-trained model not found: {source_path}")
        except Exception as e:
            self.logger.error(f"Error loading pre-trained weights: {e}")
    
    def _setup_fine_tuning(self, model: TransferLearningModel, 
                          source_domain: str, target_domain: str):
        """Setup fine-tuning strategy"""
        strategy = self.config['transfer_strategies']['fine_tuning']
        
        # Freeze lower layers
        all_params = list(model.named_parameters())
        freeze_count = int(len(all_params) * strategy['freeze_ratio'])
        
        freeze_layers = []
        for i, (name, param) in enumerate(all_params):
            if i < freeze_count:
                freeze_layers.append(name.split('.')[0])
        
        model.freeze_layers(freeze_layers)
    
    def _setup_domain_adaptation(self, model: TransferLearningModel,
                                source_domain: str, target_domain: str):
        """Setup domain adaptation strategy"""
        # Domain adaptation typically doesn't freeze layers initially
        # but uses adversarial training instead
        pass
    
    def _setup_multi_task_learning(self, model: TransferLearningModel,
                                  source_domain: str, target_domain: str):
        """Setup multi-task learning strategy"""
        # Multi-task learning keeps all layers unfrozen
        # and learns multiple tasks simultaneously
        pass
    
    def adapt_domain(self, model_id: str, source_data: torch.Tensor, 
                    target_data: torch.Tensor, source_labels: torch.Tensor,
                    target_labels: torch.Tensor = None, 
                    epochs: int = 50) -> DomainAdaptationResult:
        """Perform domain adaptation training"""
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")
        
        model = self.models[model_id]['model']
        model.train()
        
        # Optimizers
        feature_optimizer = torch.optim.Adam(
            model.feature_extractor.parameters(), lr=1e-4
        )
        task_optimizer = torch.optim.Adam(
            model.task_classifiers.parameters(), lr=1e-4
        )
        domain_optimizer = torch.optim.Adam(
            model.domain_classifier.parameters(), lr=1e-3
        )
        
        # Loss functions
        task_criterion = nn.CrossEntropyLoss()
        domain_criterion = nn.CrossEntropyLoss()
        
        # Training metrics
        results = {
            'source_losses': [],
            'target_losses': [],
            'domain_losses': [],
            'adaptation_scores': []
        }
        
        for epoch in range(epochs):
            epoch_source_loss = 0.0
            epoch_target_loss = 0.0
            epoch_domain_loss = 0.0
            
            # Progressive alpha schedule for gradient reversal
            alpha = min(1.0, 2.0 / (1.0 + np.exp(-10 * epoch / epochs)) - 1.0)
            
            # Batch training
            batch_size = 32
            n_batches = max(len(source_data) // batch_size, len(target_data) // batch_size)
            
            for batch_idx in range(n_batches):
                # Get batches
                source_start = (batch_idx * batch_size) % len(source_data)
                target_start = (batch_idx * batch_size) % len(target_data)
                
                source_batch = source_data[source_start:source_start + batch_size]
                source_label_batch = source_labels[source_start:source_start + batch_size]
                target_batch = target_data[target_start:target_start + batch_size]
                
                # Create domain labels
                source_domain_labels = torch.zeros(len(source_batch), dtype=torch.long)
                target_domain_labels = torch.ones(len(target_batch), dtype=torch.long)
                
                # Forward pass - source data
                source_results = model(source_batch, task_name='task_classification', alpha=alpha)
                
                # Task loss on source data
                if 'task_classification' in model.task_classifiers:
                    task_loss = task_criterion(
                        source_results['task_output'], source_label_batch
                    )
                else:
                    # Use first available task
                    first_task = next(iter(model.task_classifiers.keys()))
                    source_results = model(source_batch, task_name=first_task, alpha=alpha)
                    task_loss = task_criterion(
                        source_results['task_output'], source_label_batch
                    )
                
                # Domain loss on both source and target
                combined_features = torch.cat([
                    source_results['features'], 
                    model(target_batch, alpha=alpha)['features']
                ])
                combined_domain_labels = torch.cat([
                    source_domain_labels, target_domain_labels
                ])
                
                reversed_features = GradientReversalLayer.apply(combined_features, alpha)
                domain_output = model.domain_classifier(reversed_features)
                domain_loss = domain_criterion(domain_output, combined_domain_labels)
                
                # Backward pass
                feature_optimizer.zero_grad()
                task_optimizer.zero_grad()
                domain_optimizer.zero_grad()
                
                total_loss = task_loss + domain_loss
                total_loss.backward()
                
                feature_optimizer.step()
                task_optimizer.step()
                domain_optimizer.step()
                
                epoch_source_loss += task_loss.item()
                epoch_domain_loss += domain_loss.item()
                
                # Target loss (if target labels available)
                if target_labels is not None:
                    target_start_label = (batch_idx * batch_size) % len(target_labels)
                    target_label_batch = target_labels[target_start_label:target_start_label + batch_size]
                    
                    target_results = model(target_batch, task_name=first_task)
                    target_loss = task_criterion(
                        target_results['task_output'], target_label_batch
                    )
                    epoch_target_loss += target_loss.item()
            
            # Record epoch metrics
            avg_source_loss = epoch_source_loss / n_batches
            avg_target_loss = epoch_target_loss / n_batches if target_labels is not None else 0.0
            avg_domain_loss = epoch_domain_loss / n_batches
            
            results['source_losses'].append(avg_source_loss)
            results['target_losses'].append(avg_target_loss)
            results['domain_losses'].append(avg_domain_loss)
            
            # Calculate adaptation score (domain confusion)
            adaptation_score = 1.0 - (avg_domain_loss / np.log(2))  # Normalized by random chance
            results['adaptation_scores'].append(adaptation_score)
            
            if epoch % 10 == 0:
                self.logger.info(
                    f"Epoch {epoch}: Source Loss {avg_source_loss:.4f}, "
                    f"Domain Loss {avg_domain_loss:.4f}, "
                    f"Adaptation Score {adaptation_score:.4f}"
                )
        
        # Calculate final results
        final_result = DomainAdaptationResult(
            source_loss=np.mean(results['source_losses'][-5:]),
            target_loss=np.mean(results['target_losses'][-5:]) if results['target_losses'] else 0.0,
            domain_loss=np.mean(results['domain_losses'][-5:]),
            adaptation_score=np.mean(results['adaptation_scores'][-5:]),
            transfer_effectiveness=self._calculate_transfer_effectiveness(results)
        )
        
        # Store adaptation history
        self.transfer_history.append({
            'model_id': model_id,
            'adaptation_result': asdict(final_result),
            'training_curves': results,
            'timestamp': datetime.now().isoformat()
        })
        
        return final_result
    
    def _calculate_transfer_effectiveness(self, results: Dict[str, List[float]]) -> float:
        """Calculate transfer learning effectiveness score"""
        if not results['source_losses']:
            return 0.0
        
        # Measure improvement over training
        initial_loss = np.mean(results['source_losses'][:5])
        final_loss = np.mean(results['source_losses'][-5:])
        
        # Measure domain adaptation quality
        final_adaptation = np.mean(results['adaptation_scores'][-5:])
        
        # Combine metrics (higher is better)
        improvement_ratio = max(0, (initial_loss - final_loss) / initial_loss)
        effectiveness = (improvement_ratio + final_adaptation) / 2.0
        
        return min(1.0, max(0.0, effectiveness))
    
    def get_transfer_recommendations(self, target_domain: str, 
                                   available_domains: List[str] = None) -> List[Dict[str, Any]]:
        """Get recommendations for transfer learning sources"""
        if available_domains is None:
            available_domains = list(self.domain_mappings.keys())
        
        recommendations = []
        
        if target_domain not in self.domain_mappings:
            return recommendations
        
        target_info = self.domain_mappings[target_domain]
        
        for source_domain in available_domains:
            if source_domain == target_domain:
                continue
            
            if source_domain not in self.domain_mappings:
                continue
            
            source_info = self.domain_mappings[source_domain]
            
            # Calculate similarity score
            feature_overlap = len(
                set(target_info['features']) & set(source_info['features'])
            ) / len(set(target_info['features']) | set(source_info['features']))
            
            domain_relatedness = 1.0 if source_domain in target_info['related_domains'] else 0.5
            
            similarity_score = (feature_overlap + domain_relatedness) / 2.0
            
            # Recommend transfer method based on similarity
            if similarity_score > 0.7:
                recommended_method = 'fine_tuning'
                expected_improvement = '20-40%'
            elif similarity_score > 0.4:
                recommended_method = 'domain_adaptation'
                expected_improvement = '10-25%'
            else:
                recommended_method = 'multi_task'
                expected_improvement = '5-15%'
            
            recommendations.append({
                'source_domain': source_domain,
                'similarity_score': similarity_score,
                'recommended_method': recommended_method,
                'expected_improvement': expected_improvement,
                'feature_overlap': feature_overlap,
                'domain_relatedness': domain_relatedness
            })
        
        # Sort by similarity score
        recommendations.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        return recommendations
    
    def evaluate_transfer_performance(self, model_id: str, test_data: torch.Tensor,
                                    test_labels: torch.Tensor, task_name: str = None) -> Dict[str, float]:
        """Evaluate transfer learning model performance"""
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")
        
        model = self.models[model_id]['model']
        model.eval()
        
        metrics = {}
        
        with torch.no_grad():
            # Get predictions
            if task_name is None:
                task_name = next(iter(model.task_classifiers.keys()))
            
            results = model(test_data, task_name=task_name)
            predictions = torch.argmax(results['task_output'], dim=1)
            
            # Calculate metrics
            accuracy = (predictions == test_labels).float().mean().item()
            
            # Convert to numpy for sklearn metrics
            predictions_np = predictions.cpu().numpy()
            labels_np = test_labels.cpu().numpy()
            
            # Calculate additional metrics
            from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
            
            precision = precision_score(labels_np, predictions_np, average='weighted', zero_division=0)
            recall = recall_score(labels_np, predictions_np, average='weighted', zero_division=0)
            f1 = f1_score(labels_np, predictions_np, average='weighted', zero_division=0)
            
            metrics = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'num_samples': len(test_labels)
            }
        
        return metrics
    
    def save_model(self, model_id: str, path: str):
        """Save transfer learning model"""
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")
        
        model_info = self.models[model_id]
        
        checkpoint = {
            'model_state_dict': model_info['model'].state_dict(),
            'model_config': model_info,
            'transfer_history': [
                h for h in self.transfer_history if h['model_id'] == model_id
            ]
        }
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(checkpoint, path)
        self.logger.info(f"Saved transfer model to {path}")
    
    def get_transfer_summary(self) -> Dict[str, Any]:
        """Get summary of transfer learning activities"""
        summary = {
            'total_models': len(self.models),
            'transfer_methods_used': set(),
            'domain_pairs': set(),
            'average_effectiveness': 0.0,
            'successful_transfers': 0,
            'generated_at': datetime.now().isoformat()
        }
        
        effectiveness_scores = []
        
        for model_id, model_info in self.models.items():
            summary['transfer_methods_used'].add(model_info['transfer_method'])
            summary['domain_pairs'].add(
                f"{model_info['source_domain']} → {model_info['target_domain']}"
            )
            
            # Find effectiveness scores from history
            for history in self.transfer_history:
                if history['model_id'] == model_id:
                    effectiveness = history['adaptation_result'].get('transfer_effectiveness', 0.0)
                    effectiveness_scores.append(effectiveness)
                    if effectiveness > 0.3:  # Threshold for success
                        summary['successful_transfers'] += 1
        
        if effectiveness_scores:
            summary['average_effectiveness'] = np.mean(effectiveness_scores)
        
        # Convert sets to lists for JSON serialization
        summary['transfer_methods_used'] = list(summary['transfer_methods_used'])
        summary['domain_pairs'] = list(summary['domain_pairs'])
        
        return summary

# Example usage and testing
if __name__ == "__main__":
    print("🔄 Transfer Learning System Testing:")
    print("=" * 50)
    
    # Initialize transfer learning system
    transfer_system = CyberTransferLearning()
    
    # Get transfer recommendations
    print("\n📋 Transfer Learning Recommendations:")
    recommendations = transfer_system.get_transfer_recommendations('phishing_detection')
    for rec in recommendations[:3]:
        print(f"  Source: {rec['source_domain']}")
        print(f"    Similarity: {rec['similarity_score']:.3f}")
        print(f"    Method: {rec['recommended_method']}")
        print(f"    Expected Improvement: {rec['expected_improvement']}")
        print()
    
    # Create transfer models
    print("🏗️ Creating transfer models...")
    model_id = transfer_system.create_transfer_model(
        'malware_detection', 
        'phishing_detection', 
        'domain_adaptation'
    )
    
    # Generate synthetic data for testing
    print("\n🧪 Testing with synthetic data...")
    source_data = torch.randn(1000, 1024)
    source_labels = torch.randint(0, 10, (1000,))
    target_data = torch.randn(500, 1024)
    target_labels = torch.randint(0, 10, (500,))
    
    # Perform domain adaptation
    print("🎯 Performing domain adaptation...")
    adaptation_result = transfer_system.adapt_domain(
        model_id, source_data, target_data, source_labels, target_labels, epochs=20
    )
    
    print(f"  Source Loss: {adaptation_result.source_loss:.4f}")
    print(f"  Domain Loss: {adaptation_result.domain_loss:.4f}")
    print(f"  Adaptation Score: {adaptation_result.adaptation_score:.4f}")
    print(f"  Transfer Effectiveness: {adaptation_result.transfer_effectiveness:.4f}")
    
    # Evaluate performance
    print("\n📊 Evaluating model performance...")
    test_data = torch.randn(200, 1024)
    test_labels = torch.randint(0, 10, (200,))
    
    performance = transfer_system.evaluate_transfer_performance(
        model_id, test_data, test_labels
    )
    
    print(f"  Accuracy: {performance['accuracy']:.3f}")
    print(f"  F1 Score: {performance['f1_score']:.3f}")
    print(f"  Precision: {performance['precision']:.3f}")
    print(f"  Recall: {performance['recall']:.3f}")
    
    # Get transfer summary
    print("\n📈 Transfer Learning Summary:")
    summary = transfer_system.get_transfer_summary()
    print(f"  Total Models: {summary['total_models']}")
    print(f"  Successful Transfers: {summary['successful_transfers']}")
    print(f"  Average Effectiveness: {summary['average_effectiveness']:.3f}")
    print(f"  Methods Used: {', '.join(summary['transfer_methods_used'])}")
    
    print("\n✅ Transfer Learning System implemented and tested")
