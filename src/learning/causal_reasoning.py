"""
Causal Reasoning System for Cybersecurity Events
Understanding cause-effect relationships in security incidents and attack chains
"""

import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
import networkx as nx
from enum import Enum
import logging

class CausalRelationType(Enum):
    """Types of causal relationships in cybersecurity events"""
    DIRECT_CAUSE = "direct_cause"
    INDIRECT_CAUSE = "indirect_cause"
    ENABLING_CONDITION = "enabling_condition"
    TEMPORAL_SEQUENCE = "temporal_sequence"
    CORRELATED = "correlated"
    SPURIOUS = "spurious"

class ConfidenceLevel(Enum):
    """Confidence levels for causal inferences"""
    HIGH = "high"          # > 0.8
    MEDIUM = "medium"      # 0.5 - 0.8
    LOW = "low"           # 0.2 - 0.5
    UNCERTAIN = "uncertain" # < 0.2

@dataclass
class SecurityEvent:
    """Represents a security event with temporal and contextual information"""
    event_id: str
    timestamp: datetime
    event_type: str
    source: str
    target: str
    severity: str
    attributes: Dict[str, Any]
    context: Dict[str, Any]

@dataclass
class CausalHypothesis:
    """A hypothesis about causal relationship between events"""
    hypothesis_id: str
    cause_event_id: str
    effect_event_id: str
    relationship_type: CausalRelationType
    confidence_score: float
    confidence_level: ConfidenceLevel
    evidence: Dict[str, Any]
    temporal_gap: float  # seconds
    mechanism: str
    created_at: str

@dataclass
class CausalChain:
    """A chain of causally linked security events"""
    chain_id: str
    events: List[SecurityEvent]
    causal_links: List[CausalHypothesis]
    root_cause: str
    final_effect: str
    attack_pattern: str
    mitigation_points: List[str]
    created_at: str

class CausalInferenceEngine:
    """Advanced causal inference engine for cybersecurity events"""
    
    def __init__(self, max_temporal_gap: int = 3600):
        self.max_temporal_gap = max_temporal_gap  # Maximum gap in seconds for causal consideration
        self.events = []
        self.causal_graph = nx.DiGraph()
        self.causal_patterns = self._load_causal_patterns()
        self.logger = logging.getLogger(__name__)
        
        # Causal inference models
        self.temporal_models = self._initialize_temporal_models()
        self.correlation_threshold = 0.7
        self.causation_threshold = 0.6
        
    def _load_causal_patterns(self) -> Dict[str, Any]:
        """Load known causal patterns in cybersecurity"""
        return {
            "attack_chains": {
                "lateral_movement": {
                    "pattern": ["reconnaissance", "initial_access", "persistence", "privilege_escalation", "lateral_movement"],
                    "temporal_constraints": [300, 1800, 900, 600],  # Max seconds between stages
                    "confidence_boost": 0.2
                },
                "data_exfiltration": {
                    "pattern": ["initial_access", "discovery", "collection", "exfiltration"],
                    "temporal_constraints": [1800, 3600, 1200],
                    "confidence_boost": 0.3
                },
                "ransomware": {
                    "pattern": ["initial_access", "persistence", "privilege_escalation", "lateral_movement", "encryption"],
                    "temporal_constraints": [600, 1200, 900, 300],
                    "confidence_boost": 0.25
                }
            },
            "vulnerability_exploitation": {
                "pattern": ["vulnerability_scan", "exploit_attempt", "successful_exploitation"],
                "temporal_constraints": [300, 60],
                "confidence_boost": 0.4
            },
            "insider_threat": {
                "pattern": ["anomalous_access", "data_access", "data_transfer"],
                "temporal_constraints": [1800, 900],
                "confidence_boost": 0.15
            }
        }
    
    def _initialize_temporal_models(self) -> Dict[str, Any]:
        """Initialize temporal causal inference models"""
        return {
            "granger_causality": {
                "window_size": 10,
                "max_lag": 5,
                "significance_level": 0.05
            },
            "transfer_entropy": {
                "k_history": 3,
                "embedding_dim": 2,
                "threshold": 0.1
            },
            "ccm": {  # Convergent Cross Mapping
                "embedding_dim": 3,
                "tau": 1,
                "library_size_range": (10, 100)
            }
        }
    
    def add_event(self, event: SecurityEvent) -> None:
        """Add a security event to the analysis"""
        self.events.append(event)
        self.causal_graph.add_node(event.event_id, event=event)
        
        # Update causal relationships
        self._update_causal_relationships(event)
    
    def _update_causal_relationships(self, new_event: SecurityEvent) -> None:
        """Update causal relationships when a new event is added"""
        # Look for potential causal relationships with recent events
        recent_events = [
            e for e in self.events 
            if abs((new_event.timestamp - e.timestamp).total_seconds()) <= self.max_temporal_gap
            and e.event_id != new_event.event_id
        ]
        
        for event in recent_events:
            hypothesis = self._generate_causal_hypothesis(event, new_event)
            if hypothesis and hypothesis.confidence_score >= 0.2:
                self._add_causal_edge(hypothesis)
    
    def _generate_causal_hypothesis(self, cause_event: SecurityEvent, 
                                  effect_event: SecurityEvent) -> Optional[CausalHypothesis]:
        """Generate a causal hypothesis between two events"""
        # Check temporal order
        if cause_event.timestamp >= effect_event.timestamp:
            return None
        
        temporal_gap = (effect_event.timestamp - cause_event.timestamp).total_seconds()
        if temporal_gap > self.max_temporal_gap:
            return None
        
        # Calculate confidence score based on multiple factors
        confidence_factors = {
            "temporal_proximity": self._calculate_temporal_confidence(temporal_gap),
            "semantic_similarity": self._calculate_semantic_similarity(cause_event, effect_event),
            "pattern_match": self._calculate_pattern_match(cause_event, effect_event),
            "contextual_similarity": self._calculate_contextual_similarity(cause_event, effect_event),
            "causal_mechanism": self._identify_causal_mechanism(cause_event, effect_event)
        }
        
        # Weighted confidence score
        weights = {
            "temporal_proximity": 0.2,
            "semantic_similarity": 0.25,
            "pattern_match": 0.3,
            "contextual_similarity": 0.15,
            "causal_mechanism": 0.1
        }
        
        confidence_score = sum(
            confidence_factors[factor] * weights[factor] 
            for factor in confidence_factors
        )
        
        # Determine relationship type
        relationship_type = self._determine_relationship_type(
            cause_event, effect_event, confidence_factors
        )
        
        # Determine confidence level
        if confidence_score > 0.8:
            confidence_level = ConfidenceLevel.HIGH
        elif confidence_score > 0.5:
            confidence_level = ConfidenceLevel.MEDIUM
        elif confidence_score > 0.2:
            confidence_level = ConfidenceLevel.LOW
        else:
            confidence_level = ConfidenceLevel.UNCERTAIN
        
        # Generate mechanism explanation
        mechanism = self._generate_mechanism_explanation(cause_event, effect_event, confidence_factors)
        
        hypothesis = CausalHypothesis(
            hypothesis_id=f"hyp_{cause_event.event_id}_{effect_event.event_id}",
            cause_event_id=cause_event.event_id,
            effect_event_id=effect_event.event_id,
            relationship_type=relationship_type,
            confidence_score=confidence_score,
            confidence_level=confidence_level,
            evidence=confidence_factors,
            temporal_gap=temporal_gap,
            mechanism=mechanism,
            created_at=datetime.now().isoformat()
        )
        
        return hypothesis
    
    def _calculate_temporal_confidence(self, temporal_gap: float) -> float:
        """Calculate confidence based on temporal proximity"""
        # Exponential decay function
        return np.exp(-temporal_gap / 600)  # 600 seconds half-life
    
    def _calculate_semantic_similarity(self, event1: SecurityEvent, event2: SecurityEvent) -> float:
        """Calculate semantic similarity between events"""
        # Simple keyword-based similarity (in production, use embeddings)
        keywords1 = set(event1.event_type.lower().split() + 
                        list(event1.attributes.get('keywords', [])))
        keywords2 = set(event2.event_type.lower().split() + 
                        list(event2.attributes.get('keywords', [])))
        
        if not keywords1 or not keywords2:
            return 0.0
        
        intersection = len(keywords1 & keywords2)
        union = len(keywords1 | keywords2)
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_pattern_match(self, cause_event: SecurityEvent, effect_event: SecurityEvent) -> float:
        """Calculate how well events match known causal patterns"""
        max_match = 0.0
        
        for pattern_name, pattern_info in self.causal_patterns.items():
            if isinstance(pattern_info, dict) and 'pattern' in pattern_info:
                pattern = pattern_info['pattern']
                confidence_boost = pattern_info.get('confidence_boost', 0.1)
                
                # Check if events match consecutive steps in pattern
                try:
                    cause_idx = pattern.index(cause_event.event_type.lower())
                    effect_idx = pattern.index(effect_event.event_type.lower())
                    
                    if effect_idx == cause_idx + 1:
                        match_score = 0.8 + confidence_boost
                        max_match = max(max_match, match_score)
                    elif effect_idx > cause_idx:
                        # Non-consecutive but in sequence
                        gap_penalty = (effect_idx - cause_idx - 1) * 0.1
                        match_score = max(0.3, 0.6 - gap_penalty + confidence_boost)
                        max_match = max(max_match, match_score)
                except ValueError:
                    continue
        
        return min(1.0, max_match)
    
    def _calculate_contextual_similarity(self, event1: SecurityEvent, event2: SecurityEvent) -> float:
        """Calculate contextual similarity (same host, network, user, etc.)"""
        context_matches = 0
        total_contexts = 0
        
        contexts_to_check = ['source', 'target', 'user', 'host', 'network', 'process']
        
        for context in contexts_to_check:
            val1 = getattr(event1, context, None) or event1.context.get(context)
            val2 = getattr(event2, context, None) or event2.context.get(context)
            
            if val1 is not None and val2 is not None:
                total_contexts += 1
                if val1 == val2:
                    context_matches += 1
        
        return context_matches / total_contexts if total_contexts > 0 else 0.0
    
    def _identify_causal_mechanism(self, cause_event: SecurityEvent, effect_event: SecurityEvent) -> float:
        """Identify potential causal mechanisms"""
        mechanisms = {
            "exploitation": ["exploit", "vulnerability", "compromise"],
            "lateral_movement": ["login", "access", "connection"],
            "persistence": ["install", "create", "modify"],
            "exfiltration": ["copy", "transfer", "download"]
        }
        
        cause_type = cause_event.event_type.lower()
        effect_type = effect_event.event_type.lower()
        
        for mechanism, keywords in mechanisms.items():
            if any(kw in cause_type for kw in keywords) and any(kw in effect_type for kw in keywords):
                return 0.7
        
        return 0.3
    
    def _determine_relationship_type(self, cause_event: SecurityEvent, 
                                   effect_event: SecurityEvent, 
                                   confidence_factors: Dict[str, float]) -> CausalRelationType:
        """Determine the type of causal relationship"""
        if confidence_factors["pattern_match"] > 0.7:
            return CausalRelationType.DIRECT_CAUSE
        elif confidence_factors["temporal_proximity"] > 0.8 and confidence_factors["contextual_similarity"] > 0.6:
            return CausalRelationType.DIRECT_CAUSE
        elif confidence_factors["semantic_similarity"] > 0.5:
            return CausalRelationType.INDIRECT_CAUSE
        elif confidence_factors["temporal_proximity"] > 0.5:
            return CausalRelationType.TEMPORAL_SEQUENCE
        else:
            return CausalRelationType.CORRELATED
    
    def _generate_mechanism_explanation(self, cause_event: SecurityEvent, 
                                      effect_event: SecurityEvent,
                                      confidence_factors: Dict[str, float]) -> str:
        """Generate human-readable explanation of causal mechanism"""
        cause_type = cause_event.event_type
        effect_type = effect_event.event_type
        temporal_gap = (effect_event.timestamp - cause_event.timestamp).total_seconds()
        
        if confidence_factors["pattern_match"] > 0.7:
            return f"'{cause_type}' directly enabled '{effect_type}' as part of a known attack pattern"
        elif confidence_factors["contextual_similarity"] > 0.6:
            return f"'{cause_type}' on same system/user likely caused '{effect_type}' ({temporal_gap:.0f}s later)"
        elif temporal_gap < 60:
            return f"'{cause_type}' immediately preceded '{effect_type}' ({temporal_gap:.0f}s gap)"
        else:
            return f"'{cause_type}' may have contributed to conditions enabling '{effect_type}'"
    
    def _add_causal_edge(self, hypothesis: CausalHypothesis) -> None:
        """Add a causal edge to the graph"""
        self.causal_graph.add_edge(
            hypothesis.cause_event_id,
            hypothesis.effect_event_id,
            hypothesis=hypothesis,
            weight=hypothesis.confidence_score
        )
    
    def identify_attack_chains(self, min_confidence: float = 0.4) -> List[CausalChain]:
        """Identify causal attack chains from the event graph"""
        chains = []
        
        # Find all paths in the causal graph
        for root_node in self.causal_graph.nodes():
            # Check if this could be a root cause (few incoming edges)
            if self.causal_graph.in_degree(root_node) <= 1:
                paths = self._find_causal_paths_from_root(root_node, min_confidence)
                
                for path in paths:
                    if len(path) >= 2:  # At least 2 events for a chain
                        chain = self._create_causal_chain(path)
                        if chain:
                            chains.append(chain)
        
        return chains
    
    def _find_causal_paths_from_root(self, root_node: str, min_confidence: float) -> List[List[str]]:
        """Find all causal paths starting from a root node"""
        paths = []
        
        def dfs(node: str, current_path: List[str], visited: Set[str]):
            if node in visited:
                return
            
            visited.add(node)
            current_path.append(node)
            
            # Get successors with sufficient confidence
            successors = []
            for successor in self.causal_graph.successors(node):
                edge_data = self.causal_graph.get_edge_data(node, successor)
                if edge_data and edge_data.get('weight', 0) >= min_confidence:
                    successors.append(successor)
            
            if not successors:
                # End of path
                if len(current_path) >= 2:
                    paths.append(current_path.copy())
            else:
                for successor in successors:
                    dfs(successor, current_path, visited.copy())
            
            current_path.pop()
        
        dfs(root_node, [], set())
        return paths
    
    def _create_causal_chain(self, event_path: List[str]) -> Optional[CausalChain]:
        """Create a causal chain from a path of events"""
        events = []
        causal_links = []
        
        # Get events for path
        event_dict = {event.event_id: event for event in self.events}
        
        for event_id in event_path:
            if event_id in event_dict:
                events.append(event_dict[event_id])
        
        if len(events) < 2:
            return None
        
        # Get causal links
        for i in range(len(event_path) - 1):
            edge_data = self.causal_graph.get_edge_data(event_path[i], event_path[i + 1])
            if edge_data and 'hypothesis' in edge_data:
                causal_links.append(edge_data['hypothesis'])
        
        # Identify attack pattern
        attack_pattern = self._identify_attack_pattern(events)
        
        # Identify mitigation points
        mitigation_points = self._identify_mitigation_points(events, causal_links)
        
        chain = CausalChain(
            chain_id=f"chain_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
            events=events,
            causal_links=causal_links,
            root_cause=events[0].event_id,
            final_effect=events[-1].event_id,
            attack_pattern=attack_pattern,
            mitigation_points=mitigation_points,
            created_at=datetime.now().isoformat()
        )
        
        return chain
    
    def _identify_attack_pattern(self, events: List[SecurityEvent]) -> str:
        """Identify the attack pattern represented by event sequence"""
        event_types = [event.event_type.lower() for event in events]
        
        # Check against known patterns
        for pattern_name, pattern_info in self.causal_patterns.items():
            if isinstance(pattern_info, dict) and 'pattern' in pattern_info:
                pattern = pattern_info['pattern']
                
                # Check if event sequence matches pattern
                if self._sequence_matches_pattern(event_types, pattern):
                    return pattern_name
        
        # Generic classification based on final effect
        final_event = events[-1].event_type.lower()
        if 'exfiltration' in final_event or 'download' in final_event:
            return "data_exfiltration"
        elif 'encryption' in final_event or 'ransomware' in final_event:
            return "ransomware"
        elif 'privilege' in final_event or 'escalation' in final_event:
            return "privilege_escalation"
        else:
            return "unknown_attack"
    
    def _sequence_matches_pattern(self, event_types: List[str], pattern: List[str]) -> bool:
        """Check if event sequence matches a known pattern"""
        if len(event_types) > len(pattern):
            return False
        
        pattern_index = 0
        for event_type in event_types:
            while pattern_index < len(pattern):
                if event_type in pattern[pattern_index] or pattern[pattern_index] in event_type:
                    pattern_index += 1
                    break
                pattern_index += 1
            else:
                return False
        
        return True
    
    def _identify_mitigation_points(self, events: List[SecurityEvent], 
                                  causal_links: List[CausalHypothesis]) -> List[str]:
        """Identify points where the attack chain could have been disrupted"""
        mitigation_points = []
        
        for i, event in enumerate(events[:-1]):  # Exclude final event
            event_type = event.event_type.lower()
            
            # Common mitigation points
            if 'reconnaissance' in event_type or 'scan' in event_type:
                mitigation_points.append(f"Detect and block reconnaissance at event {event.event_id}")
            elif 'initial_access' in event_type or 'exploit' in event_type:
                mitigation_points.append(f"Prevent initial access at event {event.event_id}")
            elif 'persistence' in event_type:
                mitigation_points.append(f"Detect persistence mechanisms at event {event.event_id}")
            elif 'lateral_movement' in event_type:
                mitigation_points.append(f"Segment network to prevent lateral movement at event {event.event_id}")
            
            # Check causal link strength
            if i < len(causal_links):
                link = causal_links[i]
                if link.confidence_score < 0.7:  # Weak causal link
                    mitigation_points.append(f"Strengthen monitoring between events {link.cause_event_id} and {link.effect_event_id}")
        
        return mitigation_points
    
    def analyze_root_causes(self, min_events: int = 3) -> List[Dict[str, Any]]:
        """Analyze root causes of security incidents"""
        root_causes = []
        
        # Find nodes with high out-degree (many effects) and low in-degree (few causes)
        for node in self.causal_graph.nodes():
            in_degree = self.causal_graph.in_degree(node)
            out_degree = self.causal_graph.out_degree(node)
            
            if in_degree <= 1 and out_degree >= min_events - 1:
                # Get all events caused by this root cause
                reachable_events = list(nx.descendants(self.causal_graph, node))
                
                if len(reachable_events) >= min_events - 1:
                    event = next((e for e in self.events if e.event_id == node), None)
                    if event:
                        root_causes.append({
                            "root_cause_event": event,
                            "affected_events": len(reachable_events),
                            "causal_impact_score": out_degree / len(self.events),
                            "downstream_events": [
                                next((e for e in self.events if e.event_id == eid), None)
                                for eid in reachable_events[:10]  # Top 10
                            ]
                        })
        
        # Sort by impact score
        root_causes.sort(key=lambda x: x["causal_impact_score"], reverse=True)
        
        return root_causes
    
    def get_causal_explanations(self, event_id: str) -> Dict[str, Any]:
        """Get causal explanations for a specific event"""
        if event_id not in self.causal_graph.nodes():
            return {"error": "Event not found"}
        
        # Get direct causes
        predecessors = list(self.causal_graph.predecessors(event_id))
        direct_causes = []
        
        for pred in predecessors:
            edge_data = self.causal_graph.get_edge_data(pred, event_id)
            if edge_data and 'hypothesis' in edge_data:
                hypothesis = edge_data['hypothesis']
                cause_event = next((e for e in self.events if e.event_id == pred), None)
                if cause_event:
                    direct_causes.append({
                        "cause_event": cause_event,
                        "mechanism": hypothesis.mechanism,
                        "confidence": hypothesis.confidence_score,
                        "relationship_type": hypothesis.relationship_type.value
                    })
        
        # Get effects
        successors = list(self.causal_graph.successors(event_id))
        effects = []
        
        for succ in successors:
            edge_data = self.causal_graph.get_edge_data(event_id, succ)
            if edge_data and 'hypothesis' in edge_data:
                hypothesis = edge_data['hypothesis']
                effect_event = next((e for e in self.events if e.event_id == succ), None)
                if effect_event:
                    effects.append({
                        "effect_event": effect_event,
                        "mechanism": hypothesis.mechanism,
                        "confidence": hypothesis.confidence_score,
                        "relationship_type": hypothesis.relationship_type.value
                    })
        
        # Get indirect causes (ancestors)
        ancestors = list(nx.ancestors(self.causal_graph, event_id))[:5]  # Limit to 5
        indirect_causes = [
            next((e for e in self.events if e.event_id == aid), None)
            for aid in ancestors if aid not in predecessors
        ]
        
        return {
            "target_event": next((e for e in self.events if e.event_id == event_id), None),
            "direct_causes": direct_causes,
            "effects": effects,
            "indirect_causes": [c for c in indirect_causes if c is not None],
            "causal_chain_length": len(list(nx.ancestors(self.causal_graph, event_id))) + 1
        }

# Example usage and testing
if __name__ == "__main__":
    print("🔍 Causal Reasoning System Testing:")
    print("=" * 50)
    
    # Initialize causal inference engine
    causal_engine = CausalInferenceEngine()
    
    # Create sample security events
    base_time = datetime.now()
    
    events = [
        SecurityEvent(
            event_id="evt_001",
            timestamp=base_time,
            event_type="vulnerability_scan",
            source="192.168.1.100",
            target="192.168.1.50",
            severity="low",
            attributes={"scanner": "nmap", "ports": "22,80,443"},
            context={"network": "internal", "user": "attacker"}
        ),
        SecurityEvent(
            event_id="evt_002", 
            timestamp=base_time + timedelta(minutes=5),
            event_type="exploit_attempt",
            source="192.168.1.100",
            target="192.168.1.50",
            severity="medium",
            attributes={"exploit": "ssh_brute_force", "port": "22"},
            context={"network": "internal", "user": "attacker"}
        ),
        SecurityEvent(
            event_id="evt_003",
            timestamp=base_time + timedelta(minutes=8),
            event_type="successful_exploitation",
            source="192.168.1.100",
            target="192.168.1.50",
            severity="high",
            attributes={"method": "credential_brute_force", "service": "ssh"},
            context={"network": "internal", "user": "attacker", "compromised_account": "admin"}
        ),
        SecurityEvent(
            event_id="evt_004",
            timestamp=base_time + timedelta(minutes=15),
            event_type="privilege_escalation",
            source="192.168.1.50",
            target="192.168.1.50",
            severity="high",
            attributes={"method": "sudo_exploit", "user": "admin"},
            context={"network": "internal", "user": "attacker", "host": "server1"}
        ),
        SecurityEvent(
            event_id="evt_005",
            timestamp=base_time + timedelta(minutes=25),
            event_type="lateral_movement",
            source="192.168.1.50",
            target="192.168.1.75",
            severity="high",
            attributes={"method": "ssh_key", "protocol": "ssh"},
            context={"network": "internal", "user": "attacker", "host": "server2"}
        ),
        SecurityEvent(
            event_id="evt_006",
            timestamp=base_time + timedelta(minutes=30),
            event_type="data_exfiltration",
            source="192.168.1.75",
            target="external_server",
            severity="critical",
            attributes={"method": "scp", "data_volume": "500MB"},
            context={"network": "external", "user": "attacker", "data_type": "sensitive"}
        )
    ]
    
    # Add events to causal engine
    print("\n📊 Adding security events...")
    for event in events:
        causal_engine.add_event(event)
        print(f"  Added: {event.event_type} at {event.timestamp.strftime('%H:%M:%S')}")
    
    # Identify attack chains
    print("\n🔗 Identifying causal attack chains...")
    attack_chains = causal_engine.identify_attack_chains(min_confidence=0.3)
    
    for i, chain in enumerate(attack_chains, 1):
        print(f"\n  Chain {i}: {chain.attack_pattern}")
        print(f"    Events: {len(chain.events)}")
        print(f"    Root Cause: {chain.root_cause}")
        print(f"    Final Effect: {chain.final_effect}")
        print(f"    Causal Links: {len(chain.causal_links)}")
        
        for link in chain.causal_links[:3]:  # Show first 3 links
            print(f"      {link.cause_event_id} → {link.effect_event_id} "
                  f"(confidence: {link.confidence_score:.3f}, {link.relationship_type.value})")
        
        if chain.mitigation_points:
            print(f"    Mitigation Points:")
            for point in chain.mitigation_points[:2]:  # Show first 2
                print(f"      - {point}")
    
    # Analyze root causes
    print("\n🎯 Root Cause Analysis:")
    root_causes = causal_engine.analyze_root_causes()
    
    for cause in root_causes[:3]:  # Show top 3
        event = cause["root_cause_event"]
        print(f"  Root Cause: {event.event_type}")
        print(f"    Impact Score: {cause['causal_impact_score']:.3f}")
        print(f"    Affected Events: {cause['affected_events']}")
    
    # Get causal explanations for specific event
    print("\n💡 Causal Explanation for Final Event:")
    explanation = causal_engine.get_causal_explanations("evt_006")
    
    print(f"  Target Event: {explanation['target_event'].event_type}")
    print(f"  Causal Chain Length: {explanation['causal_chain_length']}")
    
    if explanation['direct_causes']:
        print(f"  Direct Causes:")
        for cause in explanation['direct_causes'][:2]:
            print(f"    - {cause['cause_event'].event_type} "
                  f"(confidence: {cause['confidence']:.3f})")
            print(f"      Mechanism: {cause['mechanism']}")
    
    print("\n✅ Causal Reasoning System implemented and tested")
