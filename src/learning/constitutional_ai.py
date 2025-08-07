"""
Constitutional AI Integration for Cyber-LLM
Implements ethical constraints and safety guardrails through constitutional AI principles.

Author: Muzan Sano <sanosensei36@gmail.com>
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import re
from pathlib import Path
import yaml
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from collections import defaultdict
import numpy as np

from ..utils.logging_system import CyberLLMLogger

# Configure logging
logger = CyberLLMLogger(__name__).get_logger()

class EthicalPrinciple(Enum):
    """Core ethical principles for cybersecurity AI"""
    HARM_PREVENTION = "harm_prevention"             # Don't cause harm
    LAWFULNESS = "lawfulness"                       # Operate within legal bounds
    TRANSPARENCY = "transparency"                   # Be transparent about capabilities
    ACCOUNTABILITY = "accountability"               # Take responsibility for actions
    PROPORTIONALITY = "proportionality"             # Response proportional to threat
    PRIVACY_PROTECTION = "privacy_protection"       # Protect user privacy
    DUAL_USE_AWARENESS = "dual_use_awareness"       # Recognize dual-use potential
    HUMAN_OVERSIGHT = "human_oversight"             # Maintain human oversight
    DEFENSIVE_ONLY = "defensive_only"               # Focus on defensive applications
    CONSENT_RESPECT = "consent_respect"             # Respect user consent

class ViolationType(Enum):
    """Types of constitutional violations"""
    ILLEGAL_ACTIVITY = "illegal_activity"
    HARMFUL_CONTENT = "harmful_content"
    PRIVACY_VIOLATION = "privacy_violation"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    MALICIOUS_INTENT = "malicious_intent"
    DISPROPORTIONATE_RESPONSE = "disproportionate_response"
    DUAL_USE_CONCERN = "dual_use_concern"
    LACK_OF_CONSENT = "lack_of_consent"
    TRANSPARENCY_FAILURE = "transparency_failure"
    HUMAN_OVERSIGHT_BYPASS = "human_oversight_bypass"

class ActionType(Enum):
    """Types of actions that can be taken"""
    BLOCK = "block"                   # Block the action completely
    MODIFY = "modify"                 # Modify to comply with principles
    WARN = "warn"                     # Issue warning but allow
    LOG = "log"                       # Log for review
    ESCALATE = "escalate"             # Escalate to human oversight
    REQUEST_PERMISSION = "request_permission"  # Request user permission

@dataclass
class ConstitutionalRule:
    """Individual constitutional rule"""
    rule_id: str
    name: str
    description: str
    principle: EthicalPrinciple
    violation_type: ViolationType
    
    # Rule logic
    triggers: List[str]  # Patterns or conditions that trigger this rule
    severity: float      # Severity score 0-1
    action: ActionType
    
    # Context
    applicable_contexts: List[str]  # Contexts where this rule applies
    exceptions: List[str]           # Exception conditions
    
    # Metadata
    created_at: datetime
    updated_at: datetime
    active: bool = True

@dataclass
class ConstitutionalViolation:
    """Constitutional violation record"""
    violation_id: str
    rule_id: str
    violation_type: ViolationType
    severity: float
    
    # Context
    input_text: str
    context: Dict[str, Any]
    timestamp: datetime
    
    # Details
    explanation: str
    suggested_modification: Optional[str]
    action_taken: ActionType
    
    # Resolution
    resolved: bool = False
    resolution_notes: Optional[str] = None
    human_review_required: bool = False

class EthicalClassifier:
    """Classifier for ethical content evaluation"""
    
    def __init__(self, model_name: str = "martin-ha/toxic-comment-model"):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.classifier = pipeline(
                "text-classification",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if torch.cuda.is_available() else -1
            )
            logger.info(f"Loaded ethical classifier: {model_name}")
        except Exception as e:
            logger.warning(f"Could not load classifier {model_name}: {str(e)}")
            self.classifier = None
    
    def classify_content(self, text: str) -> Dict[str, Any]:
        """Classify content for ethical concerns"""
        
        if not self.classifier:
            return {'toxic': False, 'confidence': 0.0, 'categories': []}
        
        try:
            results = self.classifier(text)
            
            # Process results based on model output format
            if isinstance(results, list) and results:
                result = results[0]
                label = result.get('label', '').lower()
                score = result.get('score', 0.0)
                
                is_toxic = 'toxic' in label or score > 0.7
                
                return {
                    'toxic': is_toxic,
                    'confidence': score,
                    'label': label,
                    'categories': [label] if is_toxic else []
                }
            
        except Exception as e:
            logger.error(f"Content classification failed: {str(e)}")
        
        return {'toxic': False, 'confidence': 0.0, 'categories': []}
    
    def detect_harmful_patterns(self, text: str) -> List[Dict[str, Any]]:
        """Detect specific harmful patterns in text"""
        
        harmful_patterns = {
            'illegal_hacking': [
                r'hack\s+into\s+\w+',
                r'break\s+into\s+system',
                r'unauthorized\s+access\s+to',
                r'steal\s+credentials',
                r'bypass\s+security'
            ],
            'malware_creation': [
                r'create\s+malware',
                r'develop\s+virus',
                r'write\s+trojan',
                r'backdoor\s+implementation',
                r'payload\s+execution'
            ],
            'privacy_violation': [
                r'extract\s+personal\s+data',
                r'collect\s+private\s+information',
                r'access\s+confidential',
                r'steal\s+identity',
                r'harvest\s+emails'
            ],
            'social_engineering': [
                r'phishing\s+campaign',
                r'deceive\s+users',
                r'manipulate\s+victims',
                r'pretend\s+to\s+be',
                r'social\s+engineer'
            ]
        }
        
        detected_patterns = []
        
        for category, patterns in harmful_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    detected_patterns.append({
                        'category': category,
                        'pattern': pattern,
                        'match': match.group(),
                        'start': match.start(),
                        'end': match.end(),
                        'severity': self._calculate_pattern_severity(category)
                    })
        
        return detected_patterns
    
    def _calculate_pattern_severity(self, category: str) -> float:
        """Calculate severity score for pattern category"""
        
        severity_map = {
            'illegal_hacking': 0.9,
            'malware_creation': 0.95,
            'privacy_violation': 0.8,
            'social_engineering': 0.7
        }
        
        return severity_map.get(category, 0.5)

class ConstitutionalRuleEngine:
    """Engine for managing and evaluating constitutional rules"""
    
    def __init__(self):
        self.rules: Dict[str, ConstitutionalRule] = {}
        self.violations_history: List[ConstitutionalViolation] = []
        self.ethical_classifier = EthicalClassifier()
        
        # Load default rules
        self._load_default_rules()
        
        # Statistics
        self.evaluation_stats = {
            'total_evaluations': 0,
            'violations_found': 0,
            'actions_taken': defaultdict(int),
            'rules_triggered': defaultdict(int)
        }
    
    def _load_default_rules(self):
        """Load default constitutional rules"""
        
        default_rules = [
            ConstitutionalRule(
                rule_id="no_illegal_hacking",
                name="No Illegal Hacking",
                description="Prevent assistance with illegal hacking activities",
                principle=EthicalPrinciple.LAWFULNESS,
                violation_type=ViolationType.ILLEGAL_ACTIVITY,
                triggers=[
                    "hack into", "break into system", "unauthorized access",
                    "steal credentials", "bypass security", "exploit vulnerability"
                ],
                severity=0.9,
                action=ActionType.BLOCK,
                applicable_contexts=["query_processing", "response_generation"],
                exceptions=["authorized penetration testing", "educational context"],
                created_at=datetime.now(),
                updated_at=datetime.now()
            ),
            
            ConstitutionalRule(
                rule_id="no_malware_creation",
                name="No Malware Creation",
                description="Prevent assistance with malware development",
                principle=EthicalPrinciple.HARM_PREVENTION,
                violation_type=ViolationType.HARMFUL_CONTENT,
                triggers=[
                    "create malware", "develop virus", "write trojan",
                    "backdoor implementation", "malicious payload"
                ],
                severity=0.95,
                action=ActionType.BLOCK,
                applicable_contexts=["code_generation", "query_processing"],
                exceptions=["malware analysis", "security research"],
                created_at=datetime.now(),
                updated_at=datetime.now()
            ),
            
            ConstitutionalRule(
                rule_id="privacy_protection",
                name="Privacy Protection",
                description="Protect user privacy and personal data",
                principle=EthicalPrinciple.PRIVACY_PROTECTION,
                violation_type=ViolationType.PRIVACY_VIOLATION,
                triggers=[
                    "extract personal data", "collect private information",
                    "access confidential", "harvest emails", "steal identity"
                ],
                severity=0.8,
                action=ActionType.MODIFY,
                applicable_contexts=["data_processing", "query_processing"],
                exceptions=["with explicit consent", "anonymized data"],
                created_at=datetime.now(),
                updated_at=datetime.now()
            ),
            
            ConstitutionalRule(
                rule_id="proportional_response",
                name="Proportional Response",
                description="Ensure responses are proportional to threats",
                principle=EthicalPrinciple.PROPORTIONALITY,
                violation_type=ViolationType.DISPROPORTIONATE_RESPONSE,
                triggers=[
                    "nuclear option", "destroy everything", "maximum damage",
                    "scorched earth", "overkill"
                ],
                severity=0.7,
                action=ActionType.WARN,
                applicable_contexts=["response_generation", "action_planning"],
                exceptions=["critical infrastructure protection"],
                created_at=datetime.now(),
                updated_at=datetime.now()
            ),
            
            ConstitutionalRule(
                rule_id="require_consent",
                name="Require Consent",
                description="Require explicit consent for sensitive operations",
                principle=EthicalPrinciple.CONSENT_RESPECT,
                violation_type=ViolationType.LACK_OF_CONSENT,
                triggers=[
                    "scan network", "access system", "modify configuration",
                    "deploy tool", "execute command"
                ],
                severity=0.6,
                action=ActionType.REQUEST_PERMISSION,
                applicable_contexts=["action_execution", "tool_deployment"],
                exceptions=["emergency response", "pre-authorized actions"],
                created_at=datetime.now(),
                updated_at=datetime.now()
            ),
            
            ConstitutionalRule(
                rule_id="human_oversight_required",
                name="Human Oversight Required",
                description="Require human oversight for high-risk operations",
                principle=EthicalPrinciple.HUMAN_OVERSIGHT,
                violation_type=ViolationType.HUMAN_OVERSIGHT_BYPASS,
                triggers=[
                    "autonomous operation", "unsupervised execution",
                    "critical system access", "irreversible action"
                ],
                severity=0.8,
                action=ActionType.ESCALATE,
                applicable_contexts=["autonomous_operations", "critical_actions"],
                exceptions=["emergency protocols", "pre-approved scenarios"],
                created_at=datetime.now(),
                updated_at=datetime.now()
            ),
            
            ConstitutionalRule(
                rule_id="dual_use_awareness",
                name="Dual Use Awareness",
                description="Be aware of dual-use potential of cybersecurity tools",
                principle=EthicalPrinciple.DUAL_USE_AWARENESS,
                violation_type=ViolationType.DUAL_USE_CONCERN,
                triggers=[
                    "offensive capability", "attack tool", "exploitation framework",
                    "weaponization", "dual use"
                ],
                severity=0.75,
                action=ActionType.WARN,
                applicable_contexts=["tool_recommendation", "capability_discussion"],
                exceptions=["defensive research", "authorized red team"],
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
        ]
        
        for rule in default_rules:
            self.rules[rule.rule_id] = rule
        
        logger.info(f"Loaded {len(default_rules)} default constitutional rules")
    
    def add_rule(self, rule: ConstitutionalRule):
        """Add new constitutional rule"""
        self.rules[rule.rule_id] = rule
        logger.info(f"Added constitutional rule: {rule.name}")
    
    def evaluate_content(self, 
                        content: str, 
                        context: Dict[str, Any]) -> List[ConstitutionalViolation]:
        """Evaluate content against constitutional rules"""
        
        self.evaluation_stats['total_evaluations'] += 1
        violations = []
        
        # Get applicable rules based on context
        applicable_rules = self._get_applicable_rules(context)
        
        for rule in applicable_rules:
            if not rule.active:
                continue
            
            # Check if rule is triggered
            if self._is_rule_triggered(rule, content, context):
                self.evaluation_stats['rules_triggered'][rule.rule_id] += 1
                
                # Create violation record
                violation = ConstitutionalViolation(
                    violation_id=f"violation_{datetime.now().timestamp()}",
                    rule_id=rule.rule_id,
                    violation_type=rule.violation_type,
                    severity=rule.severity,
                    input_text=content,
                    context=context,
                    timestamp=datetime.now(),
                    explanation=f"Triggered rule: {rule.name} - {rule.description}",
                    suggested_modification=self._suggest_modification(rule, content),
                    action_taken=rule.action,
                    human_review_required=rule.action in [ActionType.ESCALATE, ActionType.REQUEST_PERMISSION]
                )
                
                violations.append(violation)
                self.violations_history.append(violation)
                self.evaluation_stats['violations_found'] += 1
                self.evaluation_stats['actions_taken'][rule.action.value] += 1
        
        # Use ethical classifier for additional evaluation
        classification = self.ethical_classifier.classify_content(content)
        if classification['toxic'] and classification['confidence'] > 0.8:
            violation = ConstitutionalViolation(
                violation_id=f"violation_{datetime.now().timestamp()}",
                rule_id="ethical_classifier",
                violation_type=ViolationType.HARMFUL_CONTENT,
                severity=classification['confidence'],
                input_text=content,
                context=context,
                timestamp=datetime.now(),
                explanation=f"Ethical classifier detected harmful content: {classification['label']}",
                suggested_modification="Please rephrase to remove harmful content",
                action_taken=ActionType.MODIFY,
                human_review_required=classification['confidence'] > 0.9
            )
            violations.append(violation)
        
        # Detect harmful patterns
        harmful_patterns = self.ethical_classifier.detect_harmful_patterns(content)
        for pattern in harmful_patterns:
            if pattern['severity'] > 0.7:
                violation = ConstitutionalViolation(
                    violation_id=f"violation_{datetime.now().timestamp()}",
                    rule_id="pattern_detection",
                    violation_type=ViolationType.HARMFUL_CONTENT,
                    severity=pattern['severity'],
                    input_text=content,
                    context=context,
                    timestamp=datetime.now(),
                    explanation=f"Detected harmful pattern: {pattern['category']} - {pattern['match']}",
                    suggested_modification=f"Remove or rephrase: {pattern['match']}",
                    action_taken=ActionType.BLOCK if pattern['severity'] > 0.8 else ActionType.WARN
                )
                violations.append(violation)
        
        return violations
    
    def _get_applicable_rules(self, context: Dict[str, Any]) -> List[ConstitutionalRule]:
        """Get rules applicable to current context"""
        
        current_context = context.get('context_type', 'general')
        applicable_rules = []
        
        for rule in self.rules.values():
            if not rule.applicable_contexts or current_context in rule.applicable_contexts:
                applicable_rules.append(rule)
        
        return applicable_rules
    
    def _is_rule_triggered(self, 
                          rule: ConstitutionalRule, 
                          content: str, 
                          context: Dict[str, Any]) -> bool:
        """Check if rule is triggered by content"""
        
        # Check exceptions first
        for exception in rule.exceptions:
            if exception.lower() in content.lower() or exception.lower() in str(context).lower():
                return False
        
        # Check triggers
        for trigger in rule.triggers:
            if trigger.lower() in content.lower():
                return True
        
        return False
    
    def _suggest_modification(self, rule: ConstitutionalRule, content: str) -> Optional[str]:
        """Suggest modification to comply with rule"""
        
        modification_templates = {
            ViolationType.ILLEGAL_ACTIVITY: "Please rephrase to focus on authorized and legal cybersecurity practices.",
            ViolationType.HARMFUL_CONTENT: "Please modify to remove potentially harmful content.",
            ViolationType.PRIVACY_VIOLATION: "Please ensure explicit consent and privacy protection measures.",
            ViolationType.DISPROPORTIONATE_RESPONSE: "Please consider a more proportional approach to the threat level.",
            ViolationType.DUAL_USE_CONCERN: "Please clarify the defensive and ethical use of this capability.",
            ViolationType.LACK_OF_CONSENT: "Please ensure proper authorization before proceeding.",
        }
        
        return modification_templates.get(rule.violation_type)

class ConstitutionalAIManager:
    """Main manager for constitutional AI integration"""
    
    def __init__(self, config_path: str = "configs/constitutional_ai.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
        # Initialize components
        self.rule_engine = ConstitutionalRuleEngine()
        
        # Action handlers
        self.action_handlers = {
            ActionType.BLOCK: self._handle_block,
            ActionType.MODIFY: self._handle_modify,
            ActionType.WARN: self._handle_warn,
            ActionType.LOG: self._handle_log,
            ActionType.ESCALATE: self._handle_escalate,
            ActionType.REQUEST_PERMISSION: self._handle_request_permission
        }
        
        # Human oversight queue
        self.human_review_queue = []
        
        logger.info("ConstitutionalAIManager initialized")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load constitutional AI configuration"""
        
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            # Default configuration
            default_config = {
                'strict_mode': True,
                'auto_modify_enabled': True,
                'human_oversight_threshold': 0.8,
                'violation_reporting': True,
                'learning_from_violations': True,
                'transparency_level': 'high'
            }
            
            # Save default configuration
            self.config_path.parent.mkdir(exist_ok=True)
            with open(self.config_path, 'w') as f:
                yaml.dump(default_config, f)
            
            return default_config
    
    async def evaluate_and_enforce(self, 
                                 content: str, 
                                 context: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate content and enforce constitutional principles"""
        
        # Evaluate content against rules
        violations = self.rule_engine.evaluate_content(content, context)
        
        if not violations:
            return {
                'allowed': True,
                'content': content,
                'violations': [],
                'actions_taken': []
            }
        
        # Process violations
        actions_taken = []
        modified_content = content
        blocked = False
        
        for violation in violations:
            action_result = await self.action_handlers[violation.action_taken](
                violation, modified_content, context
            )
            
            actions_taken.append({
                'violation_id': violation.violation_id,
                'action': violation.action_taken.value,
                'result': action_result
            })
            
            # Update content based on action result
            if action_result.get('blocked'):
                blocked = True
                break
            elif action_result.get('modified_content'):
                modified_content = action_result['modified_content']
        
        return {
            'allowed': not blocked,
            'content': modified_content,
            'violations': [v.__dict__ for v in violations],
            'actions_taken': actions_taken,
            'human_review_required': any(v.human_review_required for v in violations)
        }
    
    async def _handle_block(self, 
                          violation: ConstitutionalViolation, 
                          content: str, 
                          context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle block action"""
        
        logger.warning(f"Blocked content due to violation: {violation.rule_id}")
        
        return {
            'blocked': True,
            'reason': violation.explanation,
            'severity': violation.severity
        }
    
    async def _handle_modify(self, 
                           violation: ConstitutionalViolation, 
                           content: str, 
                           context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle modify action"""
        
        if not self.config.get('auto_modify_enabled', True):
            return {'blocked': True, 'reason': 'Auto-modification disabled'}
        
        # Simple content modification (in practice, use more sophisticated methods)
        modified_content = self._auto_modify_content(content, violation)
        
        logger.info(f"Modified content due to violation: {violation.rule_id}")
        
        return {
            'blocked': False,
            'modified_content': modified_content,
            'modification_reason': violation.explanation
        }
    
    async def _handle_warn(self, 
                         violation: ConstitutionalViolation, 
                         content: str, 
                         context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle warn action"""
        
        logger.warning(f"Warning for potential violation: {violation.rule_id}")
        
        return {
            'blocked': False,
            'warning': violation.explanation,
            'suggested_modification': violation.suggested_modification
        }
    
    async def _handle_log(self, 
                        violation: ConstitutionalViolation, 
                        content: str, 
                        context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle log action"""
        
        logger.info(f"Logged violation: {violation.rule_id}")
        
        return {
            'blocked': False,
            'logged': True,
            'log_entry': violation.explanation
        }
    
    async def _handle_escalate(self, 
                             violation: ConstitutionalViolation, 
                             content: str, 
                             context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle escalate action"""
        
        # Add to human review queue
        self.human_review_queue.append({
            'violation': violation,
            'content': content,
            'context': context,
            'timestamp': datetime.now(),
            'status': 'pending'
        })
        
        logger.warning(f"Escalated to human oversight: {violation.rule_id}")
        
        if self.config.get('strict_mode', True):
            return {
                'blocked': True,
                'reason': 'Escalated to human oversight - awaiting approval',
                'escalation_id': violation.violation_id
            }
        else:
            return {
                'blocked': False,
                'escalated': True,
                'escalation_id': violation.violation_id
            }
    
    async def _handle_request_permission(self, 
                                       violation: ConstitutionalViolation, 
                                       content: str, 
                                       context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle request permission action"""
        
        # In a real implementation, this would integrate with a permission system
        logger.info(f"Permission requested for: {violation.rule_id}")
        
        return {
            'blocked': True,
            'reason': 'Explicit permission required',
            'permission_request': violation.explanation
        }
    
    def _auto_modify_content(self, content: str, violation: ConstitutionalViolation) -> str:
        """Automatically modify content to address violation"""
        
        # Simple modification strategies based on violation type
        if violation.violation_type == ViolationType.ILLEGAL_ACTIVITY:
            # Replace harmful terms with ethical alternatives
            harmful_terms = {
                'hack into': 'securely assess',
                'break into': 'authorized penetration test of',
                'steal': 'ethically collect',
                'exploit': 'responsibly disclose'
            }
            
            modified = content
            for harmful, ethical in harmful_terms.items():
                modified = modified.replace(harmful, ethical)
            
            return modified
        
        elif violation.violation_type == ViolationType.PRIVACY_VIOLATION:
            # Add privacy disclaimers
            return f"{content}\n\n[Note: Ensure proper consent and privacy protections are in place]"
        
        elif violation.violation_type == ViolationType.DUAL_USE_CONCERN:
            # Add ethical use disclaimer
            return f"{content}\n\n[Note: This information should only be used for defensive cybersecurity purposes]"
        
        return content
    
    def get_human_review_queue(self) -> List[Dict[str, Any]]:
        """Get pending human review items"""
        return [item for item in self.human_review_queue if item['status'] == 'pending']
    
    def resolve_human_review(self, escalation_id: str, decision: str, notes: str = ""):
        """Resolve human review item"""
        
        for item in self.human_review_queue:
            if item['violation'].violation_id == escalation_id:
                item['status'] = 'resolved'
                item['decision'] = decision
                item['resolution_notes'] = notes
                item['resolved_at'] = datetime.now()
                
                logger.info(f"Resolved human review: {escalation_id} - {decision}")
                break
    
    def get_constitutional_statistics(self) -> Dict[str, Any]:
        """Get constitutional AI statistics"""
        
        # Recent violations (last 24 hours)
        recent_violations = [
            v for v in self.rule_engine.violations_history
            if v.timestamp >= datetime.now() - timedelta(days=1)
        ]
        
        # Violation distribution by type
        violation_types = defaultdict(int)
        for violation in self.rule_engine.violations_history:
            violation_types[violation.violation_type.value] += 1
        
        # Rule effectiveness
        rule_effectiveness = {}
        for rule_id, count in self.rule_engine.evaluation_stats['rules_triggered'].items():
            rule = self.rule_engine.rules.get(rule_id)
            if rule:
                rule_effectiveness[rule.name] = {
                    'triggers': count,
                    'severity': rule.severity,
                    'action': rule.action.value
                }
        
        return {
            'evaluation_stats': self.rule_engine.evaluation_stats,
            'total_violations': len(self.rule_engine.violations_history),
            'recent_violations_24h': len(recent_violations),
            'violation_distribution': dict(violation_types),
            'active_rules': len([r for r in self.rule_engine.rules.values() if r.active]),
            'pending_human_reviews': len(self.get_human_review_queue()),
            'rule_effectiveness': rule_effectiveness
        }
    
    def update_rule(self, rule_id: str, updates: Dict[str, Any]):
        """Update constitutional rule"""
        
        if rule_id in self.rule_engine.rules:
            rule = self.rule_engine.rules[rule_id]
            
            for key, value in updates.items():
                if hasattr(rule, key):
                    setattr(rule, key, value)
            
            rule.updated_at = datetime.now()
            logger.info(f"Updated constitutional rule: {rule_id}")
        else:
            raise ValueError(f"Rule not found: {rule_id}")

# Factory function
def create_constitutional_ai_manager(**kwargs) -> ConstitutionalAIManager:
    """Create constitutional AI manager with configuration"""
    return ConstitutionalAIManager(**kwargs)
