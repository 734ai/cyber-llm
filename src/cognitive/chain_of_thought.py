"""
Chain-of-Thought Reasoning System for Multi-step Logical Inference
Implements advanced reasoning chains with step-by-step logical progression
"""
import sqlite3
import json
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import logging
from pathlib import Path
from enum import Enum

logger = logging.getLogger(__name__)

class ReasoningType(Enum):
    """Types of reasoning supported"""
    DEDUCTIVE = "deductive"           # General to specific
    INDUCTIVE = "inductive"           # Specific to general
    ABDUCTIVE = "abductive"           # Best explanation
    ANALOGICAL = "analogical"         # Pattern matching
    CAUSAL = "causal"                # Cause and effect
    COUNTERFACTUAL = "counterfactual" # What-if scenarios
    STRATEGIC = "strategic"           # Goal-oriented planning
    DIAGNOSTIC = "diagnostic"         # Problem identification

@dataclass
class ReasoningStep:
    """Individual step in a reasoning chain"""
    step_id: str
    step_number: int
    reasoning_type: ReasoningType
    premise: str
    inference_rule: str
    conclusion: str
    confidence: float
    evidence: List[str]
    assumptions: List[str]
    created_at: datetime

@dataclass
class ReasoningChain:
    """Complete chain of reasoning steps"""
    chain_id: str
    agent_id: str
    problem_statement: str
    reasoning_goal: str
    steps: List[ReasoningStep]
    final_conclusion: str
    overall_confidence: float
    created_at: datetime
    completed_at: Optional[datetime]
    metadata: Dict[str, Any]

class ChainOfThoughtReasoning:
    """Advanced chain-of-thought reasoning system"""
    
    def __init__(self, db_path: str = "data/cognitive/reasoning_chains.db"):
        """Initialize reasoning system"""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
        
        # Reasoning rules and patterns
        self._inference_rules = self._load_inference_rules()
        self._reasoning_patterns = self._load_reasoning_patterns()
    
    def _init_database(self):
        """Initialize database schemas"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS reasoning_chains (
                    chain_id TEXT PRIMARY KEY,
                    agent_id TEXT NOT NULL,
                    problem_statement TEXT NOT NULL,
                    reasoning_goal TEXT NOT NULL,
                    final_conclusion TEXT,
                    overall_confidence REAL,
                    created_at TEXT NOT NULL,
                    completed_at TEXT,
                    metadata TEXT,
                    status TEXT DEFAULT 'active'
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS reasoning_steps (
                    step_id TEXT PRIMARY KEY,
                    chain_id TEXT NOT NULL,
                    step_number INTEGER NOT NULL,
                    reasoning_type TEXT NOT NULL,
                    premise TEXT NOT NULL,
                    inference_rule TEXT NOT NULL,
                    conclusion TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    evidence TEXT,
                    assumptions TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (chain_id) REFERENCES reasoning_chains(chain_id)
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS inference_rules (
                    rule_id TEXT PRIMARY KEY,
                    rule_name TEXT NOT NULL,
                    rule_type TEXT NOT NULL,
                    rule_pattern TEXT NOT NULL,
                    confidence_modifier REAL DEFAULT 1.0,
                    usage_count INTEGER DEFAULT 0,
                    success_rate REAL DEFAULT 0.5,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS reasoning_evaluations (
                    evaluation_id TEXT PRIMARY KEY,
                    chain_id TEXT NOT NULL,
                    evaluation_type TEXT,
                    correctness_score REAL,
                    logical_validity REAL,
                    completeness_score REAL,
                    evaluator TEXT,
                    feedback TEXT,
                    timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (chain_id) REFERENCES reasoning_chains(chain_id)
                )
            """)
            
            # Create indices
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chains_agent ON reasoning_chains(agent_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_steps_chain ON reasoning_steps(chain_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_steps_type ON reasoning_steps(reasoning_type)")
    
    def start_reasoning_chain(self, agent_id: str, problem_statement: str,
                             reasoning_goal: str, initial_facts: List[str] = None) -> str:
        """Start a new chain of reasoning"""
        try:
            chain_id = str(uuid.uuid4())
            
            chain = ReasoningChain(
                chain_id=chain_id,
                agent_id=agent_id,
                problem_statement=problem_statement,
                reasoning_goal=reasoning_goal,
                steps=[],
                final_conclusion="",
                overall_confidence=0.0,
                created_at=datetime.now(),
                completed_at=None,
                metadata={
                    'initial_facts': initial_facts or [],
                    'reasoning_depth': 0,
                    'branch_count': 0
                }
            )
            
            # Store in database
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO reasoning_chains (
                        chain_id, agent_id, problem_statement, reasoning_goal,
                        created_at, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    chain.chain_id, chain.agent_id, chain.problem_statement,
                    chain.reasoning_goal, chain.created_at.isoformat(),
                    json.dumps(chain.metadata)
                ))
            
            logger.info(f"Started reasoning chain {chain_id} for problem: {problem_statement[:50]}...")
            return chain_id
            
        except Exception as e:
            logger.error(f"Error starting reasoning chain: {e}")
            return ""
    
    def add_reasoning_step(self, chain_id: str, reasoning_type: ReasoningType,
                          premise: str, inference_rule: str = "",
                          evidence: List[str] = None,
                          assumptions: List[str] = None) -> str:
        """Add a step to an existing reasoning chain"""
        try:
            step_id = str(uuid.uuid4())
            
            # Get current step count for this chain
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT COUNT(*) FROM reasoning_steps WHERE chain_id = ?
                """, (chain_id,))
                step_number = cursor.fetchone()[0] + 1
            
            # Apply reasoning to generate conclusion
            conclusion, confidence = self._apply_reasoning(
                reasoning_type, premise, inference_rule, evidence or []
            )
            
            step = ReasoningStep(
                step_id=step_id,
                step_number=step_number,
                reasoning_type=reasoning_type,
                premise=premise,
                inference_rule=inference_rule or self._select_inference_rule(reasoning_type),
                conclusion=conclusion,
                confidence=confidence,
                evidence=evidence or [],
                assumptions=assumptions or [],
                created_at=datetime.now()
            )
            
            # Store step in database
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO reasoning_steps (
                        step_id, chain_id, step_number, reasoning_type,
                        premise, inference_rule, conclusion, confidence,
                        evidence, assumptions, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    step.step_id, chain_id, step.step_number,
                    step.reasoning_type.value, step.premise,
                    step.inference_rule, step.conclusion,
                    step.confidence, json.dumps(step.evidence),
                    json.dumps(step.assumptions),
                    step.created_at.isoformat()
                ))
            
            logger.info(f"Added reasoning step {step_number} to chain {chain_id}")
            return step_id
            
        except Exception as e:
            logger.error(f"Error adding reasoning step: {e}")
            return ""
    
    def complete_reasoning_chain(self, chain_id: str) -> Dict[str, Any]:
        """Complete reasoning chain and generate final conclusion"""
        try:
            # Get all steps for this chain
            steps = self._get_chain_steps(chain_id)
            
            if not steps:
                return {'error': 'No reasoning steps found'}
            
            # Generate final conclusion by combining all steps
            final_conclusion, overall_confidence = self._synthesize_conclusion(steps)
            
            # Update chain in database
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    UPDATE reasoning_chains SET
                        final_conclusion = ?,
                        overall_confidence = ?,
                        completed_at = ?,
                        status = 'completed'
                    WHERE chain_id = ?
                """, (
                    final_conclusion, overall_confidence,
                    datetime.now().isoformat(), chain_id
                ))
            
            result = {
                'chain_id': chain_id,
                'final_conclusion': final_conclusion,
                'overall_confidence': overall_confidence,
                'step_count': len(steps),
                'reasoning_quality': self._assess_reasoning_quality(steps)
            }
            
            logger.info(f"Completed reasoning chain {chain_id}: {final_conclusion[:50]}...")
            return result
            
        except Exception as e:
            logger.error(f"Error completing reasoning chain: {e}")
            return {'error': str(e)}
    
    def reason_about_threat(self, threat_indicators: List[str], 
                           agent_id: str = "") -> Dict[str, Any]:
        """Perform comprehensive threat reasoning using multiple reasoning types"""
        try:
            problem = f"Analyze threat indicators: {', '.join(threat_indicators[:3])}..."
            
            # Start reasoning chain
            chain_id = self.start_reasoning_chain(
                agent_id, problem, "threat_assessment", threat_indicators
            )
            
            reasoning_results = {
                'chain_id': chain_id,
                'threat_indicators': threat_indicators,
                'reasoning_steps': [],
                'threat_assessment': {},
                'recommendations': []
            }
            
            # Step 1: Deductive reasoning - What do we know for certain?
            known_facts = f"Observed indicators: {', '.join(threat_indicators)}"
            step1_id = self.add_reasoning_step(
                chain_id, ReasoningType.DEDUCTIVE, known_facts,
                "indicator_classification",
                evidence=threat_indicators
            )
            
            # Step 2: Inductive reasoning - Pattern recognition
            pattern_premise = "Multiple indicators suggest coordinated activity"
            step2_id = self.add_reasoning_step(
                chain_id, ReasoningType.INDUCTIVE, pattern_premise,
                "pattern_generalization",
                evidence=[f"Indicator pattern analysis: {len(threat_indicators)} indicators"]
            )
            
            # Step 3: Abductive reasoning - Best explanation
            explanation_premise = "Finding most likely explanation for observed indicators"
            step3_id = self.add_reasoning_step(
                chain_id, ReasoningType.ABDUCTIVE, explanation_premise,
                "hypothesis_selection",
                assumptions=["Indicators represent malicious activity"]
            )
            
            # Step 4: Causal reasoning - Impact analysis
            impact_premise = "If threat is real, what are potential consequences?"
            step4_id = self.add_reasoning_step(
                chain_id, ReasoningType.CAUSAL, impact_premise,
                "impact_analysis",
                assumptions=["Current security controls", "System vulnerabilities"]
            )
            
            # Complete the reasoning chain
            completion_result = self.complete_reasoning_chain(chain_id)
            reasoning_results.update(completion_result)
            
            # Generate threat assessment based on reasoning
            steps = self._get_chain_steps(chain_id)
            avg_confidence = sum(step['confidence'] for step in steps) / len(steps) if steps else 0
            
            if avg_confidence > 0.8:
                threat_level = "HIGH"
                priority = "immediate"
            elif avg_confidence > 0.6:
                threat_level = "MEDIUM"
                priority = "elevated"
            else:
                threat_level = "LOW"
                priority = "monitor"
            
            reasoning_results['threat_assessment'] = {
                'threat_level': threat_level,
                'priority': priority,
                'confidence': avg_confidence,
                'reasoning_quality': completion_result.get('reasoning_quality', 0.5)
            }
            
            # Generate recommendations
            recommendations = [
                {
                    'action': 'investigate_indicators',
                    'priority': 'high' if avg_confidence > 0.7 else 'medium',
                    'rationale': 'Based on deductive analysis of indicators'
                },
                {
                    'action': 'monitor_systems',
                    'priority': 'medium',
                    'rationale': 'Based on causal impact analysis'
                }
            ]
            
            if threat_level == "HIGH":
                recommendations.insert(0, {
                    'action': 'activate_incident_response',
                    'priority': 'critical',
                    'rationale': 'High confidence threat detected through multi-step reasoning'
                })
            
            reasoning_results['recommendations'] = recommendations
            
            logger.info(f"Threat reasoning complete: {threat_level} threat (confidence: {avg_confidence:.3f})")
            return reasoning_results
            
        except Exception as e:
            logger.error(f"Error in threat reasoning: {e}")
            return {'error': str(e)}
    
    def _apply_reasoning(self, reasoning_type: ReasoningType, premise: str,
                        inference_rule: str, evidence: List[str]) -> Tuple[str, float]:
        """Apply specific reasoning type to generate conclusion"""
        try:
            base_confidence = 0.5
            
            if reasoning_type == ReasoningType.DEDUCTIVE:
                # Deductive: If premise is true and rule is valid, conclusion follows
                conclusion = f"Therefore: {self._apply_deductive_rule(premise, inference_rule)}"
                confidence = min(0.9, base_confidence + (len(evidence) * 0.1))
                
            elif reasoning_type == ReasoningType.INDUCTIVE:
                # Inductive: Generalize from specific observations
                conclusion = f"Pattern suggests: {self._apply_inductive_rule(premise, evidence)}"
                confidence = min(0.8, base_confidence + (len(evidence) * 0.05))
                
            elif reasoning_type == ReasoningType.ABDUCTIVE:
                # Abductive: Best explanation for observations
                conclusion = f"Most likely explanation: {self._apply_abductive_rule(premise, evidence)}"
                confidence = min(0.7, base_confidence + (len(evidence) * 0.08))
                
            elif reasoning_type == ReasoningType.CAUSAL:
                # Causal: Cause and effect relationships
                conclusion = f"Causal inference: {self._apply_causal_rule(premise, evidence)}"
                confidence = min(0.75, base_confidence + 0.2)
                
            elif reasoning_type == ReasoningType.STRATEGIC:
                # Strategic: Goal-oriented reasoning
                conclusion = f"Strategic conclusion: {self._apply_strategic_rule(premise)}"
                confidence = min(0.8, base_confidence + 0.25)
                
            else:
                # Default reasoning
                conclusion = f"Conclusion based on {reasoning_type.value}: {premise}"
                confidence = base_confidence
            
            return conclusion, confidence
            
        except Exception as e:
            logger.error(f"Error applying reasoning: {e}")
            return f"Unable to reason about: {premise}", 0.1
    
    def _apply_deductive_rule(self, premise: str, rule: str) -> str:
        """Apply deductive reasoning rule"""
        if "indicators" in premise.lower():
            return "specific threat types can be identified from these indicators"
        elif "malicious" in premise.lower():
            return "security response is warranted"
        else:
            return f"logical consequence follows from {premise[:30]}..."
    
    def _apply_inductive_rule(self, premise: str, evidence: List[str]) -> str:
        """Apply inductive reasoning rule"""
        if len(evidence) > 3:
            return "systematic attack pattern likely in progress"
        elif len(evidence) > 1:
            return "coordinated threat activity possible"
        else:
            return "isolated incident or false positive"
    
    def _apply_abductive_rule(self, premise: str, evidence: List[str]) -> str:
        """Apply abductive reasoning rule"""
        if any("network" in str(e).lower() for e in evidence):
            return "network-based attack scenario"
        elif any("file" in str(e).lower() for e in evidence):
            return "malware or file-based attack"
        else:
            return "unknown attack vector requiring investigation"
    
    def _apply_causal_rule(self, premise: str, evidence: List[str]) -> str:
        """Apply causal reasoning rule"""
        return "if threat is confirmed, system compromise and data exfiltration may occur"
    
    def _apply_strategic_rule(self, premise: str) -> str:
        """Apply strategic reasoning rule"""
        return "optimal response is to investigate thoroughly while maintaining operational security"
    
    def _select_inference_rule(self, reasoning_type: ReasoningType) -> str:
        """Select appropriate inference rule for reasoning type"""
        rule_map = {
            ReasoningType.DEDUCTIVE: "modus_ponens",
            ReasoningType.INDUCTIVE: "generalization",
            ReasoningType.ABDUCTIVE: "inference_to_best_explanation",
            ReasoningType.CAUSAL: "causal_inference",
            ReasoningType.STRATEGIC: "means_ends_analysis"
        }
        return rule_map.get(reasoning_type, "default_inference")
    
    def _get_chain_steps(self, chain_id: str) -> List[Dict[str, Any]]:
        """Get all steps for a reasoning chain"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT * FROM reasoning_steps 
                    WHERE chain_id = ? 
                    ORDER BY step_number
                """, (chain_id,))
                
                steps = []
                for row in cursor.fetchall():
                    step = {
                        'step_id': row[0],
                        'step_number': row[2],
                        'reasoning_type': row[3],
                        'premise': row[4],
                        'inference_rule': row[5],
                        'conclusion': row[6],
                        'confidence': row[7],
                        'evidence': json.loads(row[8]) if row[8] else [],
                        'assumptions': json.loads(row[9]) if row[9] else []
                    }
                    steps.append(step)
                
                return steps
                
        except Exception as e:
            logger.error(f"Error getting chain steps: {e}")
            return []
    
    def _synthesize_conclusion(self, steps: List[Dict[str, Any]]) -> Tuple[str, float]:
        """Synthesize final conclusion from reasoning steps"""
        if not steps:
            return "No conclusion reached", 0.0
        
        # Weight later steps more heavily
        weighted_confidence = 0.0
        total_weight = 0.0
        
        conclusions = []
        
        for i, step in enumerate(steps):
            weight = (i + 1) / len(steps)  # Later steps have higher weight
            weighted_confidence += step['confidence'] * weight
            total_weight += weight
            conclusions.append(step['conclusion'])
        
        final_confidence = weighted_confidence / total_weight if total_weight > 0 else 0.0
        
        # Create synthesized conclusion
        if len(conclusions) == 1:
            final_conclusion = conclusions[0]
        else:
            final_conclusion = f"Multi-step analysis concludes: {conclusions[-1]}"
        
        return final_conclusion, final_confidence
    
    def _assess_reasoning_quality(self, steps: List[Dict[str, Any]]) -> float:
        """Assess the quality of the reasoning chain"""
        if not steps:
            return 0.0
        
        quality_score = 0.0
        
        # Diversity of reasoning types (better)
        reasoning_types = set(step['reasoning_type'] for step in steps)
        diversity_score = min(len(reasoning_types) / 4.0, 1.0)  # Max 4 types
        
        # Logical progression (each step builds on previous)
        progression_score = 1.0  # Assume good progression
        
        # Evidence quality (more evidence is better)
        avg_evidence = sum(len(step['evidence']) for step in steps) / len(steps)
        evidence_score = min(avg_evidence / 3.0, 1.0)
        
        # Confidence consistency (not too variable)
        confidences = [step['confidence'] for step in steps]
        confidence_std = (max(confidences) - min(confidences)) if len(confidences) > 1 else 0
        consistency_score = max(0.0, 1.0 - confidence_std)
        
        quality_score = (
            diversity_score * 0.3 +
            progression_score * 0.3 +
            evidence_score * 0.2 +
            consistency_score * 0.2
        )
        
        return quality_score
    
    def _load_inference_rules(self) -> Dict[str, Any]:
        """Load available inference rules"""
        return {
            'modus_ponens': {'pattern': 'If P then Q; P; therefore Q', 'confidence': 0.9},
            'generalization': {'pattern': 'Multiple instances of X; therefore X is common', 'confidence': 0.7},
            'causal_inference': {'pattern': 'A precedes B; A and B correlated; A causes B', 'confidence': 0.6},
            'best_explanation': {'pattern': 'X explains Y better than alternatives', 'confidence': 0.8}
        }
    
    def _load_reasoning_patterns(self) -> Dict[str, Any]:
        """Load common reasoning patterns"""
        return {
            'threat_analysis': [
                ReasoningType.DEDUCTIVE,
                ReasoningType.INDUCTIVE,
                ReasoningType.ABDUCTIVE,
                ReasoningType.CAUSAL
            ],
            'vulnerability_assessment': [
                ReasoningType.DEDUCTIVE,
                ReasoningType.STRATEGIC,
                ReasoningType.CAUSAL
            ]
        }
    
    def get_reasoning_statistics(self) -> Dict[str, Any]:
        """Get comprehensive reasoning system statistics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                stats = {}
                
                # Basic counts
                cursor = conn.execute("SELECT COUNT(*) FROM reasoning_chains")
                stats['total_chains'] = cursor.fetchone()[0]
                
                cursor = conn.execute("SELECT COUNT(*) FROM reasoning_steps")
                stats['total_steps'] = cursor.fetchone()[0]
                
                # Reasoning type distribution
                cursor = conn.execute("""
                    SELECT reasoning_type, COUNT(*) 
                    FROM reasoning_steps 
                    GROUP BY reasoning_type
                """)
                stats['reasoning_types'] = dict(cursor.fetchall())
                
                # Average confidence by reasoning type
                cursor = conn.execute("""
                    SELECT reasoning_type, AVG(confidence) 
                    FROM reasoning_steps 
                    GROUP BY reasoning_type
                """)
                stats['avg_confidence_by_type'] = dict(cursor.fetchall())
                
                # Chain completion rate
                cursor = conn.execute("SELECT COUNT(*) FROM reasoning_chains WHERE status = 'completed'")
                completed = cursor.fetchone()[0]
                stats['completion_rate'] = completed / stats['total_chains'] if stats['total_chains'] > 0 else 0
                
                return stats
                
        except Exception as e:
            logger.error(f"Error getting reasoning statistics: {e}")
            return {'error': str(e)}

# Export the main classes
__all__ = ['ChainOfThoughtReasoning', 'ReasoningChain', 'ReasoningStep', 'ReasoningType']
