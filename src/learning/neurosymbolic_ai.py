"""
Neuro-Symbolic AI for Cybersecurity
Combining neural networks with symbolic reasoning for explainable AI
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from typing import Dict, List, Optional, Any, Tuple, Union, Set
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import logging
from abc import ABC, abstractmethod
from collections import defaultdict
import sqlite3
import pickle
from enum import Enum
import networkx as nx
import ast
import re
from sympy import symbols, And, Or, Not, Implies, simplify
from sympy.logic.boolalg import to_cnf, to_dnf

class SymbolicRule:
    """Symbolic rule for cybersecurity reasoning"""
    
    def __init__(self, rule_id: str, premise: str, conclusion: str, 
                 confidence: float = 1.0, priority: int = 1):
        self.rule_id = rule_id
        self.premise = premise  # Logical expression as string
        self.conclusion = conclusion
        self.confidence = confidence
        self.priority = priority
        self.usage_count = 0
        self.success_count = 0
        self.created_at = datetime.now().isoformat()
    
    def __str__(self):
        return f"Rule({self.rule_id}): IF {self.premise} THEN {self.conclusion} [conf={self.confidence:.2f}]"

@dataclass
class SymbolicFact:
    """Symbolic fact in the knowledge base"""
    fact_id: str
    predicate: str
    arguments: List[str]
    truth_value: bool
    confidence: float
    source: str
    timestamp: str
    metadata: Dict[str, Any]

@dataclass
class InferenceStep:
    """Single step in symbolic reasoning"""
    step_id: str
    rule_applied: str
    premises_used: List[str]
    conclusion_derived: str
    confidence: float
    timestamp: str

class SymbolicKnowledgeBase:
    """Knowledge base for symbolic reasoning"""
    
    def __init__(self):
        self.facts = {}  # fact_id -> SymbolicFact
        self.rules = {}  # rule_id -> SymbolicRule
        self.predicates = set()
        self.entities = set()
        
        # Initialize with cybersecurity domain knowledge
        self._init_cybersecurity_knowledge()
    
    def _init_cybersecurity_knowledge(self):
        """Initialize with domain-specific cybersecurity rules"""
        
        # Basic cybersecurity rules
        rules = [
            # Network security rules
            SymbolicRule("net_01", "external_connection(X) & suspicious_activity(X)", "potential_intrusion(X)", 0.8, 1),
            SymbolicRule("net_02", "port_scan(X) & failed_login(X)", "reconnaissance(X)", 0.9, 2),
            SymbolicRule("net_03", "large_data_transfer(X) & external_connection(X)", "potential_exfiltration(X)", 0.7, 2),
            
            # Malware detection rules
            SymbolicRule("mal_01", "unknown_process(X) & network_activity(X)", "suspicious_process(X)", 0.6, 1),
            SymbolicRule("mal_02", "file_modification(X) & system_file(X)", "potential_malware(X)", 0.8, 2),
            SymbolicRule("mal_03", "encrypted_communication(X) & c2_domain(X)", "malware_communication(X)", 0.9, 3),
            
            # User behavior rules
            SymbolicRule("usr_01", "off_hours_access(X) & privileged_account(X)", "suspicious_access(X)", 0.7, 2),
            SymbolicRule("usr_02", "multiple_failed_logins(X) & admin_account(X)", "brute_force_attempt(X)", 0.9, 3),
            SymbolicRule("usr_03", "data_access(X) & unusual_location(X)", "insider_threat(X)", 0.6, 2),
            
            # Attack progression rules
            SymbolicRule("att_01", "reconnaissance(X) & vulnerability_exploit(X)", "initial_compromise(X)", 0.8, 3),
            SymbolicRule("att_02", "initial_compromise(X) & credential_theft(X)", "lateral_movement(X)", 0.9, 3),
            SymbolicRule("att_03", "lateral_movement(X) & data_access(X)", "mission_completion(X)", 0.8, 3),
            
            # Response rules
            SymbolicRule("rsp_01", "potential_intrusion(X)", "alert_soc(X)", 1.0, 1),
            SymbolicRule("rsp_02", "malware_communication(X)", "block_traffic(X)", 1.0, 2),
            SymbolicRule("rsp_03", "brute_force_attempt(X)", "lock_account(X)", 1.0, 3),
        ]
        
        for rule in rules:
            self.add_rule(rule)
    
    def add_fact(self, fact: SymbolicFact):
        """Add a fact to the knowledge base"""
        self.facts[fact.fact_id] = fact
        self.predicates.add(fact.predicate)
        self.entities.update(fact.arguments)
    
    def add_rule(self, rule: SymbolicRule):
        """Add a rule to the knowledge base"""
        self.rules[rule.rule_id] = rule
    
    def get_facts_by_predicate(self, predicate: str) -> List[SymbolicFact]:
        """Get all facts with a specific predicate"""
        return [fact for fact in self.facts.values() if fact.predicate == predicate]
    
    def evaluate_premise(self, premise: str, variable_bindings: Dict[str, str]) -> Tuple[bool, float]:
        """Evaluate a logical premise given variable bindings"""
        try:
            # Simple evaluation - replace variables and check facts
            bound_premise = premise
            for var, value in variable_bindings.items():
                bound_premise = bound_premise.replace(var, f'"{value}"')
            
            # Parse logical expression
            terms = self._parse_logical_expression(bound_premise)
            
            # Evaluate each term
            term_results = []
            for term in terms:
                predicate, args, negated = term
                fact_exists = self._fact_exists(predicate, args)
                
                if negated:
                    term_results.append((not fact_exists, 1.0 if not fact_exists else 0.0))
                else:
                    confidence = self._get_fact_confidence(predicate, args) if fact_exists else 0.0
                    term_results.append((fact_exists, confidence))
            
            # Combine results (simplified - assume AND for now)
            all_true = all(result[0] for result in term_results)
            avg_confidence = np.mean([result[1] for result in term_results]) if term_results else 0.0
            
            return all_true, avg_confidence
            
        except Exception as e:
            logging.error(f"Error evaluating premise {premise}: {e}")
            return False, 0.0
    
    def _parse_logical_expression(self, expression: str) -> List[Tuple[str, List[str], bool]]:
        """Parse logical expression into terms"""
        # Simplified parsing - handles basic predicates with AND/OR
        terms = []
        
        # Split by & (AND) for now
        clauses = expression.split(" & ")
        
        for clause in clauses:
            clause = clause.strip()
            negated = clause.startswith("~") or clause.startswith("not ")
            
            if negated:
                clause = clause.replace("~", "").replace("not ", "").strip()
            
            # Extract predicate and arguments
            match = re.match(r'(\w+)\((.*)\)', clause)
            if match:
                predicate = match.group(1)
                args_str = match.group(2)
                args = [arg.strip().strip('"') for arg in args_str.split(",")]
                terms.append((predicate, args, negated))
        
        return terms
    
    def _fact_exists(self, predicate: str, args: List[str]) -> bool:
        """Check if a fact exists in the knowledge base"""
        for fact in self.facts.values():
            if fact.predicate == predicate and fact.arguments == args and fact.truth_value:
                return True
        return False
    
    def _get_fact_confidence(self, predicate: str, args: List[str]) -> float:
        """Get confidence of a fact"""
        for fact in self.facts.values():
            if fact.predicate == predicate and fact.arguments == args and fact.truth_value:
                return fact.confidence
        return 0.0

class NeuralSymbolicIntegrator(nn.Module):
    """Neural network component that interfaces with symbolic reasoning"""
    
    def __init__(self, input_dim: int, symbol_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.input_dim = input_dim
        self.symbol_dim = symbol_dim
        self.hidden_dim = hidden_dim
        
        # Neural feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Symbolic concept embeddings
        self.concept_embeddings = nn.Embedding(1000, symbol_dim)  # Assume up to 1000 concepts
        
        # Neural-symbolic fusion layers
        self.fusion_attention = nn.MultiheadAttention(
            embed_dim=symbol_dim, num_heads=8, batch_first=True
        )
        
        self.fusion_network = nn.Sequential(
            nn.Linear(hidden_dim + symbol_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Concept activation predictor
        self.concept_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 100),  # Predict activation of top 100 concepts
            nn.Sigmoid()
        )
        
        # Rule confidence predictor
        self.rule_confidence_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 50),  # Predict confidence for top 50 rules
            nn.Sigmoid()
        )
        
        # Explanation generator
        self.explanation_encoder = nn.LSTM(symbol_dim, hidden_dim // 2, batch_first=True)
        self.explanation_decoder = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, symbol_dim)
        )
    
    def forward(self, neural_input: torch.Tensor, 
                symbolic_concepts: torch.Tensor = None,
                concept_weights: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """Forward pass combining neural and symbolic processing"""
        
        # Extract neural features
        neural_features = self.feature_extractor(neural_input)
        
        # Get symbolic concept embeddings
        if symbolic_concepts is not None:
            concept_embeds = self.concept_embeddings(symbolic_concepts)
            
            # Apply attention between neural features and symbolic concepts
            if concept_weights is not None:
                # Weighted attention
                attended_concepts, attention_weights = self.fusion_attention(
                    concept_embeds, concept_embeds, concept_embeds,
                    key_padding_mask=(concept_weights == 0)
                )
            else:
                attended_concepts, attention_weights = self.fusion_attention(
                    concept_embeds, concept_embeds, concept_embeds
                )
            
            # Pool attended concepts
            pooled_concepts = attended_concepts.mean(dim=1)
            
            # Fuse neural and symbolic representations
            fused_features = torch.cat([neural_features, pooled_concepts], dim=-1)
            integrated_features = self.fusion_network(fused_features)
        else:
            integrated_features = neural_features
            pooled_concepts = torch.zeros(neural_features.shape[0], self.symbol_dim, 
                                        device=neural_features.device)
        
        # Predict concept activations
        concept_activations = self.concept_predictor(integrated_features)
        
        # Predict rule confidences
        rule_confidences = self.rule_confidence_predictor(integrated_features)
        
        # Generate explanation features
        explanation_input = pooled_concepts.unsqueeze(1)  # Add sequence dimension
        explanation_features, _ = self.explanation_encoder(explanation_input)
        explanation_output = self.explanation_decoder(explanation_features.squeeze(1))
        
        return {
            'neural_features': neural_features,
            'integrated_features': integrated_features,
            'concept_activations': concept_activations,
            'rule_confidences': rule_confidences,
            'explanation_features': explanation_output,
            'attention_weights': attention_weights if symbolic_concepts is not None else None
        }

class SymbolicReasoner:
    """Symbolic reasoning engine"""
    
    def __init__(self, knowledge_base: SymbolicKnowledgeBase):
        self.kb = knowledge_base
        self.inference_history = []
        self.logger = logging.getLogger(__name__)
    
    def forward_chaining(self, max_iterations: int = 100) -> List[InferenceStep]:
        """Perform forward chaining inference"""
        inference_steps = []
        iteration = 0
        
        while iteration < max_iterations:
            new_facts_derived = False
            iteration += 1
            
            # Try to apply each rule
            for rule in self.kb.rules.values():
                # Find variable bindings that satisfy the premise
                bindings = self._find_variable_bindings(rule.premise)
                
                for binding in bindings:
                    # Check if premise is satisfied
                    premise_satisfied, premise_confidence = self.kb.evaluate_premise(
                        rule.premise, binding
                    )
                    
                    if premise_satisfied and premise_confidence > 0.5:
                        # Derive conclusion
                        conclusion = self._apply_binding(rule.conclusion, binding)
                        
                        # Check if conclusion is already known
                        if not self._conclusion_exists(conclusion):
                            # Add new fact
                            new_fact = self._create_fact_from_conclusion(
                                conclusion, premise_confidence * rule.confidence, rule.rule_id
                            )
                            
                            if new_fact:
                                self.kb.add_fact(new_fact)
                                
                                # Record inference step
                                step = InferenceStep(
                                    step_id=f"step_{len(inference_steps)}",
                                    rule_applied=rule.rule_id,
                                    premises_used=[rule.premise],
                                    conclusion_derived=conclusion,
                                    confidence=premise_confidence * rule.confidence,
                                    timestamp=datetime.now().isoformat()
                                )
                                inference_steps.append(step)
                                
                                # Update rule usage
                                rule.usage_count += 1
                                rule.success_count += 1
                                
                                new_facts_derived = True
            
            # Stop if no new facts were derived
            if not new_facts_derived:
                break
        
        self.inference_history.extend(inference_steps)
        return inference_steps
    
    def _find_variable_bindings(self, premise: str) -> List[Dict[str, str]]:
        """Find possible variable bindings for a premise"""
        # Extract variables (uppercase single letters)
        variables = re.findall(r'\b[A-Z]\b', premise)
        
        if not variables:
            return [{}]  # No variables to bind
        
        # Generate possible bindings from entities in KB
        bindings = []
        entities_list = list(self.kb.entities)
        
        if len(variables) == 1:
            # Single variable
            var = variables[0]
            for entity in entities_list:
                bindings.append({var: entity})
        else:
            # Multiple variables - simplified approach
            for entity in entities_list:
                binding = {}
                for var in variables:
                    binding[var] = entity
                bindings.append(binding)
        
        return bindings[:100]  # Limit to prevent explosion
    
    def _apply_binding(self, expression: str, binding: Dict[str, str]) -> str:
        """Apply variable binding to an expression"""
        result = expression
        for var, value in binding.items():
            result = result.replace(var, value)
        return result
    
    def _conclusion_exists(self, conclusion: str) -> bool:
        """Check if a conclusion already exists as a fact"""
        # Parse conclusion to extract predicate and arguments
        match = re.match(r'(\w+)\((.*)\)', conclusion)
        if match:
            predicate = match.group(1)
            args_str = match.group(2)
            args = [arg.strip() for arg in args_str.split(",")]
            
            return self.kb._fact_exists(predicate, args)
        
        return False
    
    def _create_fact_from_conclusion(self, conclusion: str, confidence: float, source: str) -> Optional[SymbolicFact]:
        """Create a SymbolicFact from a conclusion string"""
        match = re.match(r'(\w+)\((.*)\)', conclusion)
        if match:
            predicate = match.group(1)
            args_str = match.group(2)
            args = [arg.strip() for arg in args_str.split(",")]
            
            fact_id = f"derived_{len(self.kb.facts)}"
            
            return SymbolicFact(
                fact_id=fact_id,
                predicate=predicate,
                arguments=args,
                truth_value=True,
                confidence=confidence,
                source=source,
                timestamp=datetime.now().isoformat(),
                metadata={'derived': True}
            )
        
        return None

class NeuroSymbolicCyberAI:
    """Complete Neuro-Symbolic AI system for cybersecurity"""
    
    def __init__(self, input_dim: int = 100, database_path: str = "neurosymbolic.db"):
        self.input_dim = input_dim
        self.database_path = database_path
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.knowledge_base = SymbolicKnowledgeBase()
        self.symbolic_reasoner = SymbolicReasoner(self.knowledge_base)
        self.neural_integrator = NeuralSymbolicIntegrator(input_dim)
        
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.neural_integrator.to(self.device)
        
        # Training setup
        self.optimizer = torch.optim.Adam(self.neural_integrator.parameters(), lr=1e-4)
        
        # Concept mapping
        self.concept_to_idx = {}
        self.idx_to_concept = {}
        self._build_concept_mapping()
        
        # Initialize database
        self._init_database()
    
    def _build_concept_mapping(self):
        """Build mapping between concepts and indices"""
        concepts = [
            # Network concepts
            'external_connection', 'port_scan', 'large_data_transfer', 'network_activity',
            'suspicious_activity', 'encrypted_communication', 'c2_domain',
            
            # System concepts
            'unknown_process', 'file_modification', 'system_file', 'privileged_account',
            'admin_account', 'failed_login', 'multiple_failed_logins',
            
            # Security events
            'potential_intrusion', 'reconnaissance', 'potential_exfiltration', 
            'suspicious_process', 'potential_malware', 'malware_communication',
            'suspicious_access', 'brute_force_attempt', 'insider_threat',
            
            # Attack stages
            'initial_compromise', 'lateral_movement', 'credential_theft', 
            'vulnerability_exploit', 'mission_completion', 'data_access',
            
            # Response actions
            'alert_soc', 'block_traffic', 'lock_account'
        ]
        
        for i, concept in enumerate(concepts):
            self.concept_to_idx[concept] = i
            self.idx_to_concept[i] = concept
    
    def _init_database(self):
        """Initialize SQLite database"""
        with sqlite3.connect(self.database_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS inference_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    neural_input BLOB,
                    symbolic_facts TEXT,
                    inference_steps TEXT,
                    conclusions TEXT,
                    explanation TEXT,
                    confidence_score REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS rule_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    rule_id TEXT NOT NULL,
                    usage_count INTEGER DEFAULT 0,
                    success_count INTEGER DEFAULT 0,
                    avg_confidence REAL DEFAULT 0.0,
                    last_used TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS concept_activations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    concept_name TEXT NOT NULL,
                    activation_score REAL NOT NULL,
                    neural_confidence REAL,
                    symbolic_confidence REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
    
    def add_observations(self, observations: List[Dict[str, Any]]):
        """Add observed facts to the knowledge base"""
        for obs in observations:
            fact = SymbolicFact(
                fact_id=f"obs_{len(self.knowledge_base.facts)}",
                predicate=obs['predicate'],
                arguments=obs['arguments'],
                truth_value=obs.get('truth_value', True),
                confidence=obs.get('confidence', 1.0),
                source="observation",
                timestamp=datetime.now().isoformat(),
                metadata=obs.get('metadata', {})
            )
            self.knowledge_base.add_fact(fact)
    
    def analyze_with_explanation(self, neural_input: np.ndarray, 
                               observations: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Perform neuro-symbolic analysis with detailed explanation"""
        
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        # Add new observations
        if observations:
            self.add_observations(observations)
        
        # Neural processing
        neural_tensor = torch.FloatTensor(neural_input).unsqueeze(0).to(self.device)
        
        # Extract symbolic concepts from current facts
        active_concepts = []
        for fact in self.knowledge_base.facts.values():
            if fact.predicate in self.concept_to_idx:
                active_concepts.append(self.concept_to_idx[fact.predicate])
        
        concept_tensor = torch.LongTensor(active_concepts).to(self.device) if active_concepts else None
        concept_weights = torch.ones(len(active_concepts)).to(self.device) if active_concepts else None
        
        if concept_tensor is not None:
            concept_tensor = concept_tensor.unsqueeze(0)
            concept_weights = concept_weights.unsqueeze(0)
        
        # Forward pass through neural integrator
        self.neural_integrator.eval()
        with torch.no_grad():
            neural_output = self.neural_integrator(
                neural_tensor, concept_tensor, concept_weights
            )
        
        # Symbolic reasoning
        inference_steps = self.symbolic_reasoner.forward_chaining()
        
        # Extract results
        concept_activations = neural_output['concept_activations'].cpu().numpy()[0]
        rule_confidences = neural_output['rule_confidences'].cpu().numpy()[0]
        
        # Get top activated concepts
        top_concepts = []
        for i, activation in enumerate(concept_activations):
            if i < len(self.idx_to_concept) and activation > 0.1:
                top_concepts.append({
                    'concept': self.idx_to_concept[i],
                    'activation': float(activation),
                    'source': 'neural'
                })
        
        top_concepts.sort(key=lambda x: x['activation'], reverse=True)
        
        # Get conclusions from symbolic reasoning
        conclusions = []
        for step in inference_steps:
            conclusions.append({
                'conclusion': step.conclusion_derived,
                'rule_used': step.rule_applied,
                'confidence': step.confidence,
                'source': 'symbolic'
            })
        
        # Generate explanation
        explanation = self._generate_explanation(
            neural_output, inference_steps, top_concepts, conclusions
        )
        
        # Calculate overall confidence
        neural_confidence = float(np.mean(concept_activations[concept_activations > 0.1])) if len(concept_activations[concept_activations > 0.1]) > 0 else 0.0
        symbolic_confidence = float(np.mean([step.confidence for step in inference_steps])) if inference_steps else 0.0
        overall_confidence = (neural_confidence + symbolic_confidence) / 2
        
        # Prepare results
        analysis_result = {
            'session_id': session_id,
            'timestamp': datetime.now().isoformat(),
            'neural_analysis': {
                'top_concepts': top_concepts[:10],
                'confidence': neural_confidence
            },
            'symbolic_analysis': {
                'inference_steps': len(inference_steps),
                'conclusions': conclusions,
                'confidence': symbolic_confidence
            },
            'integrated_analysis': {
                'overall_confidence': overall_confidence,
                'explanation': explanation,
                'recommendations': self._generate_recommendations(conclusions, top_concepts)
            },
            'metadata': {
                'facts_in_kb': len(self.knowledge_base.facts),
                'rules_applied': len(set(step.rule_applied for step in inference_steps)),
                'processing_time': 'simulated'
            }
        }
        
        # Save to database
        self._save_analysis(session_id, analysis_result, neural_input)
        
        return analysis_result
    
    def _generate_explanation(self, neural_output: Dict[str, torch.Tensor], 
                            inference_steps: List[InferenceStep],
                            top_concepts: List[Dict[str, Any]], 
                            conclusions: List[Dict[str, Any]]) -> str:
        """Generate human-readable explanation"""
        
        explanation_parts = []
        
        # Neural analysis explanation
        if top_concepts:
            explanation_parts.append("Neural Analysis:")
            explanation_parts.append(f"  The neural network identified {len(top_concepts)} relevant cybersecurity concepts:")
            
            for concept in top_concepts[:5]:
                explanation_parts.append(f"    - {concept['concept']}: {concept['activation']:.2f} confidence")
            
            explanation_parts.append("")
        
        # Symbolic reasoning explanation
        if inference_steps:
            explanation_parts.append("Symbolic Reasoning:")
            explanation_parts.append(f"  Applied {len(inference_steps)} inference rules to derive new knowledge:")
            
            for step in inference_steps[:3]:
                rule = self.knowledge_base.rules[step.rule_applied]
                explanation_parts.append(f"    - Applied rule {step.rule_applied}: {rule.premise} → {rule.conclusion}")
                explanation_parts.append(f"      Derived: {step.conclusion_derived} (confidence: {step.confidence:.2f})")
            
            if len(inference_steps) > 3:
                explanation_parts.append(f"    ... and {len(inference_steps) - 3} more inferences")
            
            explanation_parts.append("")
        
        # Conclusions explanation
        if conclusions:
            explanation_parts.append("Key Findings:")
            for conclusion in conclusions[:3]:
                explanation_parts.append(f"  - {conclusion['conclusion']} (confidence: {conclusion['confidence']:.2f})")
            
            explanation_parts.append("")
        
        # Integration explanation
        explanation_parts.append("Integration:")
        explanation_parts.append("  The neuro-symbolic system combined neural pattern recognition")
        explanation_parts.append("  with symbolic logical reasoning to provide explainable conclusions")
        explanation_parts.append("  based on both learned patterns and expert-defined rules.")
        
        return "\n".join(explanation_parts)
    
    def _generate_recommendations(self, conclusions: List[Dict[str, Any]], 
                                top_concepts: List[Dict[str, Any]]) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Based on conclusions
        threat_indicators = [c for c in conclusions if 'potential' in c['conclusion'] or 'suspicious' in c['conclusion']]
        
        if threat_indicators:
            recommendations.append("🚨 Potential security threats detected - immediate investigation recommended")
            
            for threat in threat_indicators[:3]:
                if 'intrusion' in threat['conclusion']:
                    recommendations.append("  - Implement network isolation measures")
                    recommendations.append("  - Review network access logs")
                elif 'malware' in threat['conclusion']:
                    recommendations.append("  - Perform full system scan")
                    recommendations.append("  - Isolate affected systems")
                elif 'exfiltration' in threat['conclusion']:
                    recommendations.append("  - Monitor outbound network traffic")
                    recommendations.append("  - Review data access patterns")
        
        # Based on neural concepts
        high_risk_concepts = [c for c in top_concepts if c['activation'] > 0.7]
        
        if high_risk_concepts:
            recommendations.append("🔍 High-confidence neural detections require attention:")
            for concept in high_risk_concepts[:2]:
                recommendations.append(f"  - Investigate {concept['concept']} (confidence: {concept['activation']:.2f})")
        
        # General recommendations
        if not recommendations:
            recommendations.append("✅ No immediate threats detected")
            recommendations.append("  - Continue normal monitoring")
            recommendations.append("  - Regular security updates recommended")
        
        return recommendations
    
    def _save_analysis(self, session_id: str, analysis_result: Dict[str, Any], neural_input: np.ndarray):
        """Save analysis results to database"""
        with sqlite3.connect(self.database_path) as conn:
            # Save main analysis
            conn.execute(
                """INSERT INTO inference_sessions 
                   (session_id, neural_input, symbolic_facts, inference_steps, conclusions, explanation, confidence_score) 
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (session_id, pickle.dumps(neural_input), 
                 json.dumps([asdict(fact) for fact in self.knowledge_base.facts.values()]),
                 json.dumps(analysis_result['symbolic_analysis']),
                 json.dumps(analysis_result['symbolic_analysis']['conclusions']),
                 analysis_result['integrated_analysis']['explanation'],
                 analysis_result['integrated_analysis']['overall_confidence'])
            )
            
            # Save concept activations
            for concept in analysis_result['neural_analysis']['top_concepts']:
                conn.execute(
                    """INSERT INTO concept_activations 
                       (session_id, concept_name, activation_score, neural_confidence) 
                       VALUES (?, ?, ?, ?)""",
                    (session_id, concept['concept'], concept['activation'], 
                     analysis_result['neural_analysis']['confidence'])
                )
    
    def get_knowledge_base_summary(self) -> Dict[str, Any]:
        """Get summary of current knowledge base"""
        
        # Fact statistics
        fact_stats = {
            'total_facts': len(self.knowledge_base.facts),
            'predicates': len(self.knowledge_base.predicates),
            'entities': len(self.knowledge_base.entities)
        }
        
        # Rule statistics
        rule_stats = {
            'total_rules': len(self.knowledge_base.rules),
            'rule_usage': {rule.rule_id: rule.usage_count for rule in self.knowledge_base.rules.values()},
            'rule_success': {rule.rule_id: rule.success_count for rule in self.knowledge_base.rules.values()}
        }
        
        # Recent inference history
        recent_inferences = self.symbolic_reasoner.inference_history[-10:]
        
        return {
            'fact_statistics': fact_stats,
            'rule_statistics': rule_stats,
            'recent_inferences': [asdict(inf) for inf in recent_inferences],
            'predicates': list(self.knowledge_base.predicates),
            'entities': list(self.knowledge_base.entities)
        }

# Example usage and testing
if __name__ == "__main__":
    print("🧠🔗 Neuro-Symbolic AI for Cybersecurity Testing:")
    print("=" * 60)
    
    # Initialize the system
    neurosymbolic_ai = NeuroSymbolicCyberAI(input_dim=50)
    print(f"  Initialized neuro-symbolic system")
    print(f"  Knowledge base: {len(neurosymbolic_ai.knowledge_base.facts)} facts, {len(neurosymbolic_ai.knowledge_base.rules)} rules")
    
    # Test knowledge base summary
    print("\n📚 Knowledge base summary:")
    kb_summary = neurosymbolic_ai.get_knowledge_base_summary()
    print(f"  Total facts: {kb_summary['fact_statistics']['total_facts']}")
    print(f"  Total rules: {kb_summary['rule_statistics']['total_rules']}")
    print(f"  Predicates: {kb_summary['fact_statistics']['predicates']}")
    print(f"  Sample predicates: {list(kb_summary['predicates'])[:5]}")
    
    # Add sample observations
    print("\n🔍 Adding sample cybersecurity observations...")
    sample_observations = [
        {
            'predicate': 'external_connection',
            'arguments': ['host_001'],
            'confidence': 0.9,
            'metadata': {'ip': '192.168.1.10', 'dest': '8.8.8.8'}
        },
        {
            'predicate': 'suspicious_activity',
            'arguments': ['host_001'],
            'confidence': 0.7,
            'metadata': {'activity_type': 'unusual_traffic'}
        },
        {
            'predicate': 'port_scan',
            'arguments': ['host_002'],
            'confidence': 0.8,
            'metadata': {'ports_scanned': [22, 80, 443]}
        },
        {
            'predicate': 'failed_login',
            'arguments': ['host_002'],
            'confidence': 0.9,
            'metadata': {'attempts': 5, 'user': 'admin'}
        },
        {
            'predicate': 'large_data_transfer',
            'arguments': ['host_003'],
            'confidence': 0.6,
            'metadata': {'bytes_transferred': 10485760}
        }
    ]
    
    neurosymbolic_ai.add_observations(sample_observations)
    print(f"  Added {len(sample_observations)} observations")
    
    # Generate sample neural input
    print("\n🧠 Generating sample neural input...")
    neural_input = np.random.rand(50)
    neural_input[:10] = np.array([0.8, 0.3, 0.9, 0.7, 0.2, 0.6, 0.4, 0.8, 0.1, 0.5])  # Simulate network features
    
    # Perform neuro-symbolic analysis
    print("\n🔄 Performing neuro-symbolic analysis...")
    analysis_result = neurosymbolic_ai.analyze_with_explanation(neural_input)
    
    # Display results
    print(f"\n📊 Analysis Results (Session: {analysis_result['session_id']}):")
    print(f"  Overall confidence: {analysis_result['integrated_analysis']['overall_confidence']:.3f}")
    
    print(f"\n🧠 Neural Analysis:")
    print(f"  Confidence: {analysis_result['neural_analysis']['confidence']:.3f}")
    print(f"  Top concepts detected:")
    for concept in analysis_result['neural_analysis']['top_concepts'][:5]:
        print(f"    - {concept['concept']}: {concept['activation']:.3f}")
    
    print(f"\n🔗 Symbolic Analysis:")
    print(f"  Inference steps: {analysis_result['symbolic_analysis']['inference_steps']}")
    print(f"  Confidence: {analysis_result['symbolic_analysis']['confidence']:.3f}")
    print(f"  Conclusions derived:")
    for conclusion in analysis_result['symbolic_analysis']['conclusions'][:3]:
        print(f"    - {conclusion['conclusion']} (conf: {conclusion['confidence']:.3f})")
    
    print(f"\n💡 Recommendations:")
    for rec in analysis_result['integrated_analysis']['recommendations'][:5]:
        print(f"  {rec}")
    
    print(f"\n📝 Explanation Preview:")
    explanation_lines = analysis_result['integrated_analysis']['explanation'].split('\n')[:10]
    for line in explanation_lines:
        print(f"  {line}")
    if len(analysis_result['integrated_analysis']['explanation'].split('\n')) > 10:
        print(f"  ... (full explanation available)")
    
    # Test another analysis with different observations
    print("\n🔄 Testing with different scenario...")
    additional_observations = [
        {
            'predicate': 'unknown_process',
            'arguments': ['host_004'],
            'confidence': 0.8,
            'metadata': {'process_name': 'suspicious.exe'}
        },
        {
            'predicate': 'network_activity',
            'arguments': ['host_004'],
            'confidence': 0.9,
            'metadata': {'connections': 15}
        },
        {
            'predicate': 'off_hours_access',
            'arguments': ['user_admin'],
            'confidence': 0.7,
            'metadata': {'time': '02:30:00'}
        },
        {
            'predicate': 'privileged_account',
            'arguments': ['user_admin'],
            'confidence': 1.0,
            'metadata': {'role': 'administrator'}
        }
    ]
    
    analysis_result_2 = neurosymbolic_ai.analyze_with_explanation(
        neural_input * 0.8, additional_observations
    )
    
    print(f"  New conclusions:")
    for conclusion in analysis_result_2['symbolic_analysis']['conclusions'][:3]:
        print(f"    - {conclusion['conclusion']} (conf: {conclusion['confidence']:.3f})")
    
    # Final knowledge base summary
    print("\n📚 Final knowledge base status:")
    final_summary = neurosymbolic_ai.get_knowledge_base_summary()
    print(f"  Total facts: {final_summary['fact_statistics']['total_facts']}")
    print(f"  Recent inferences: {len(final_summary['recent_inferences'])}")
    if final_summary['recent_inferences']:
        print(f"  Last inference: {final_summary['recent_inferences'][-1]['conclusion_derived']}")
    
    print("\n✅ Neuro-Symbolic AI system implemented and tested")
    print(f"  Database: {neurosymbolic_ai.database_path}")
    print(f"  Concept vocabulary: {len(neurosymbolic_ai.concept_to_idx)} concepts")
    print(f"  Architecture: Neural-Symbolic Integration with Attention-based Fusion")
    print(f"  Capabilities: Explainable AI with logical reasoning and neural pattern recognition")
