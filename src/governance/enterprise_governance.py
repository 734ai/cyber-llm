"""
Enterprise Governance and AI Safety Framework for Cyber-LLM
Comprehensive model governance, regulatory compliance, and AI safety monitoring

Author: Muzan Sano <sanosensei36@gmail.com>
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import yaml
import pandas as pd
import numpy as np
from collections import defaultdict
import hashlib
import sqlite3
import aiofiles

from ..utils.logging_system import CyberLLMLogger, CyberLLMError, ErrorCategory
from ..utils.secrets_manager import get_secrets_manager
from ..learning.constitutional_ai import ConstitutionalAIManager

class ComplianceFramework(Enum):
    """Supported compliance frameworks"""
    GDPR = "gdpr"
    CCPA = "ccpa"
    SOC2 = "soc2"
    ISO27001 = "iso27001"
    NIST = "nist_cybersecurity_framework"
    PCI_DSS = "pci_dss"
    HIPAA = "hipaa"

class GovernancePolicy(Enum):
    """Types of governance policies"""
    DATA_HANDLING = "data_handling"
    MODEL_LIFECYCLE = "model_lifecycle"
    BIAS_DETECTION = "bias_detection"
    SAFETY_CONSTRAINTS = "safety_constraints"
    ACCESS_CONTROL = "access_control"
    AUDIT_LOGGING = "audit_logging"

class RiskLevel(Enum):
    """Risk assessment levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class GovernanceRule:
    """Individual governance rule"""
    rule_id: str
    name: str
    description: str
    policy_type: GovernancePolicy
    compliance_frameworks: List[ComplianceFramework]
    severity: RiskLevel
    
    # Rule logic
    conditions: List[str]
    actions: List[str]
    exceptions: List[str] = field(default_factory=list)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    active: bool = True

@dataclass
class ComplianceViolation:
    """Compliance violation record"""
    violation_id: str
    rule_id: str
    violation_type: str
    severity: RiskLevel
    
    # Context
    description: str
    evidence: Dict[str, Any]
    affected_systems: List[str]
    timestamp: datetime
    
    # Resolution
    status: str = "open"  # open, investigating, resolved, false_positive
    assigned_to: Optional[str] = None
    resolution_notes: Optional[str] = None
    resolved_at: Optional[datetime] = None

@dataclass
class ModelGovernanceRecord:
    """Model governance tracking record"""
    model_id: str
    model_name: str
    version: str
    
    # Lifecycle tracking
    created_at: datetime
    last_updated: datetime
    status: str  # development, testing, staging, production, retired
    
    # Governance metadata
    data_lineage: Dict[str, Any]
    training_metrics: Dict[str, float]
    validation_results: Dict[str, Any]
    bias_assessment: Dict[str, float]
    safety_assessment: Dict[str, float]
    
    # Compliance
    approved_by: List[str]
    compliance_status: Dict[str, bool]
    audit_trail: List[Dict[str, Any]]

class EnterpriseGovernanceManager:
    """Comprehensive enterprise governance and compliance management"""
    
    def __init__(self, 
                 config_path: str = "configs/governance_config.yaml",
                 logger: Optional[CyberLLMLogger] = None):
        
        self.logger = logger or CyberLLMLogger(name="enterprise_governance")
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
        # Initialize components
        self.constitutional_ai = ConstitutionalAIManager()
        self.governance_rules = {}
        self.compliance_violations = []
        self.model_registry = {}
        
        # Database for governance tracking
        self.db_path = Path("data/governance.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize governance framework
        asyncio.create_task(self._initialize_governance())
        
        self.logger.info("Enterprise governance manager initialized")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load governance configuration"""
        
        default_config = {
            "compliance_frameworks": ["GDPR", "SOC2", "NIST"],
            "governance_policies": {
                "data_retention_days": 90,
                "model_approval_required": True,
                "bias_threshold": 0.1,
                "safety_threshold": 0.9,
                "audit_log_retention": 365
            },
            "notification_settings": {
                "critical_violations": True,
                "compliance_reports": True,
                "model_lifecycle_events": True
            },
            "automated_remediation": {
                "enabled": True,
                "auto_quarantine_models": True,
                "auto_generate_reports": True
            }
        }
        
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                user_config = yaml.safe_load(f)
                default_config.update(user_config)
        else:
            # Save default config
            self.config_path.parent.mkdir(exist_ok=True, parents=True)
            with open(self.config_path, 'w') as f:
                yaml.dump(default_config, f)
        
        return default_config
    
    async def _initialize_governance(self):
        """Initialize governance framework and database"""
        
        try:
            # Initialize database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create governance tables
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS governance_rules (
                    rule_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    policy_type TEXT,
                    compliance_frameworks TEXT,  -- JSON
                    severity TEXT,
                    conditions TEXT,  -- JSON
                    actions TEXT,  -- JSON
                    exceptions TEXT,  -- JSON
                    created_at TIMESTAMP,
                    updated_at TIMESTAMP,
                    active BOOLEAN DEFAULT TRUE
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS compliance_violations (
                    violation_id TEXT PRIMARY KEY,
                    rule_id TEXT,
                    violation_type TEXT,
                    severity TEXT,
                    description TEXT,
                    evidence TEXT,  -- JSON
                    affected_systems TEXT,  -- JSON
                    timestamp TIMESTAMP,
                    status TEXT DEFAULT 'open',
                    assigned_to TEXT,
                    resolution_notes TEXT,
                    resolved_at TIMESTAMP
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_governance (
                    model_id TEXT PRIMARY KEY,
                    model_name TEXT NOT NULL,
                    version TEXT,
                    created_at TIMESTAMP,
                    last_updated TIMESTAMP,
                    status TEXT,
                    data_lineage TEXT,  -- JSON
                    training_metrics TEXT,  -- JSON
                    validation_results TEXT,  -- JSON
                    bias_assessment TEXT,  -- JSON
                    safety_assessment TEXT,  -- JSON
                    approved_by TEXT,  -- JSON
                    compliance_status TEXT,  -- JSON
                    audit_trail TEXT  -- JSON
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS audit_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    event_type TEXT NOT NULL,
                    user_id TEXT,
                    resource_id TEXT,
                    action TEXT,
                    details TEXT,  -- JSON
                    ip_address TEXT,
                    user_agent TEXT
                )
            """)
            
            conn.commit()
            conn.close()
            
            # Load default governance rules
            await self._load_default_governance_rules()
            
            self.logger.info("Governance database and rules initialized")
            
        except Exception as e:
            self.logger.error("Failed to initialize governance framework", error=str(e))
            raise CyberLLMError("Governance initialization failed", ErrorCategory.SYSTEM)
    
    async def _load_default_governance_rules(self):
        """Load default governance rules"""
        
        default_rules = [
            GovernanceRule(
                rule_id="data_privacy_gdpr_001",
                name="GDPR Data Processing Consent",
                description="Ensure explicit consent for personal data processing under GDPR",
                policy_type=GovernancePolicy.DATA_HANDLING,
                compliance_frameworks=[ComplianceFramework.GDPR],
                severity=RiskLevel.HIGH,
                conditions=[
                    "personal_data_detected == True",
                    "explicit_consent == False"
                ],
                actions=[
                    "block_processing",
                    "notify_data_protection_officer",
                    "log_violation"
                ]
            ),
            
            GovernanceRule(
                rule_id="model_bias_detection_001",
                name="Model Bias Assessment Required",
                description="Require bias assessment for all production models",
                policy_type=GovernancePolicy.BIAS_DETECTION,
                compliance_frameworks=[ComplianceFramework.SOC2, ComplianceFramework.ISO27001],
                severity=RiskLevel.MEDIUM,
                conditions=[
                    "model_status == 'production'",
                    "bias_assessment_completed == False"
                ],
                actions=[
                    "require_bias_assessment",
                    "prevent_deployment",
                    "notify_ml_governance_team"
                ]
            ),
            
            GovernanceRule(
                rule_id="safety_constitutional_001",
                name="Constitutional AI Safety Check",
                description="Apply constitutional AI safety constraints to all model outputs",
                policy_type=GovernancePolicy.SAFETY_CONSTRAINTS,
                compliance_frameworks=[ComplianceFramework.SOC2, ComplianceFramework.NIST],
                severity=RiskLevel.CRITICAL,
                conditions=[
                    "model_output_generated == True",
                    "constitutional_check_passed == False"
                ],
                actions=[
                    "block_output",
                    "apply_constitutional_constraints",
                    "escalate_to_safety_team"
                ]
            ),
            
            GovernanceRule(
                rule_id="audit_logging_001",
                name="Comprehensive Audit Logging",
                description="Log all model operations and access for compliance",
                policy_type=GovernancePolicy.AUDIT_LOGGING,
                compliance_frameworks=[
                    ComplianceFramework.SOC2, 
                    ComplianceFramework.ISO27001,
                    ComplianceFramework.PCI_DSS
                ],
                severity=RiskLevel.HIGH,
                conditions=[
                    "sensitive_operation_performed == True",
                    "audit_log_created == False"
                ],
                actions=[
                    "create_audit_log",
                    "ensure_log_integrity",
                    "notify_if_log_failure"
                ]
            ),
            
            GovernanceRule(
                rule_id="access_control_001",
                name="Role-Based Access Control",
                description="Enforce RBAC for all system access",
                policy_type=GovernancePolicy.ACCESS_CONTROL,
                compliance_frameworks=[
                    ComplianceFramework.SOC2,
                    ComplianceFramework.ISO27001,
                    ComplianceFramework.NIST
                ],
                severity=RiskLevel.HIGH,
                conditions=[
                    "user_access_requested == True",
                    "role_authorization_verified == False"
                ],
                actions=[
                    "verify_user_role",
                    "apply_principle_of_least_privilege",
                    "log_access_attempt"
                ]
            )
        ]
        
        for rule in default_rules:
            await self._register_governance_rule(rule)
        
        self.logger.info(f"Loaded {len(default_rules)} default governance rules")
    
    async def _register_governance_rule(self, rule: GovernanceRule):
        """Register a governance rule in the system"""
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO governance_rules
                (rule_id, name, description, policy_type, compliance_frameworks,
                 severity, conditions, actions, exceptions, created_at, updated_at, active)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                rule.rule_id,
                rule.name,
                rule.description,
                rule.policy_type.value,
                json.dumps([f.value for f in rule.compliance_frameworks]),
                rule.severity.value,
                json.dumps(rule.conditions),
                json.dumps(rule.actions),
                json.dumps(rule.exceptions),
                rule.created_at.isoformat(),
                rule.updated_at.isoformat(),
                rule.active
            ))
            
            conn.commit()
            conn.close()
            
            self.governance_rules[rule.rule_id] = rule
            
        except Exception as e:
            self.logger.error(f"Failed to register governance rule: {rule.rule_id}", error=str(e))
    
    async def register_model(self, 
                           model_id: str,
                           model_name: str,
                           version: str,
                           metadata: Dict[str, Any]) -> ModelGovernanceRecord:
        """Register a model in the governance system"""
        
        model_record = ModelGovernanceRecord(
            model_id=model_id,
            model_name=model_name,
            version=version,
            created_at=datetime.now(),
            last_updated=datetime.now(),
            status="development",
            data_lineage=metadata.get("data_lineage", {}),
            training_metrics=metadata.get("training_metrics", {}),
            validation_results=metadata.get("validation_results", {}),
            bias_assessment=metadata.get("bias_assessment", {}),
            safety_assessment=metadata.get("safety_assessment", {}),
            approved_by=[],
            compliance_status={},
            audit_trail=[]
        )
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO model_governance
                (model_id, model_name, version, created_at, last_updated, status,
                 data_lineage, training_metrics, validation_results, bias_assessment,
                 safety_assessment, approved_by, compliance_status, audit_trail)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                model_id, model_name, version,
                model_record.created_at.isoformat(),
                model_record.last_updated.isoformat(),
                model_record.status,
                json.dumps(model_record.data_lineage),
                json.dumps(model_record.training_metrics),
                json.dumps(model_record.validation_results),
                json.dumps(model_record.bias_assessment),
                json.dumps(model_record.safety_assessment),
                json.dumps(model_record.approved_by),
                json.dumps(model_record.compliance_status),
                json.dumps(model_record.audit_trail)
            ))
            
            conn.commit()
            conn.close()
            
            self.model_registry[model_id] = model_record
            
            await self._audit_log(
                event_type="model_registered",
                resource_id=model_id,
                action="register_model",
                details={"model_name": model_name, "version": version}
            )
            
            self.logger.info(f"Model registered in governance system: {model_id}")
            return model_record
            
        except Exception as e:
            self.logger.error(f"Failed to register model: {model_id}", error=str(e))
            raise CyberLLMError("Model registration failed", ErrorCategory.SYSTEM)
    
    async def assess_model_compliance(self, model_id: str) -> Dict[str, Any]:
        """Assess model compliance against governance rules"""
        
        if model_id not in self.model_registry:
            raise CyberLLMError(f"Model not found in registry: {model_id}", ErrorCategory.VALIDATION)
        
        model_record = self.model_registry[model_id]
        compliance_assessment = {
            "model_id": model_id,
            "assessment_timestamp": datetime.now().isoformat(),
            "framework_compliance": {},
            "violations": [],
            "overall_compliance": True,
            "recommendations": []
        }
        
        # Check compliance for each framework
        for framework in ComplianceFramework:
            framework_score = await self._assess_framework_compliance(model_record, framework)
            compliance_assessment["framework_compliance"][framework.value] = framework_score
            
            if framework_score["compliance_score"] < 0.8:  # 80% threshold
                compliance_assessment["overall_compliance"] = False
        
        # Check for governance rule violations
        violations = await self._check_governance_violations(model_record)
        compliance_assessment["violations"] = violations
        
        if violations:
            compliance_assessment["overall_compliance"] = False
        
        # Generate recommendations
        recommendations = await self._generate_compliance_recommendations(
            model_record, 
            compliance_assessment
        )
        compliance_assessment["recommendations"] = recommendations
        
        # Update model compliance status
        model_record.compliance_status = {
            f.value: compliance_assessment["framework_compliance"][f.value]["compliant"]
            for f in ComplianceFramework
        }
        
        await self._update_model_record(model_record)
        
        return compliance_assessment
    
    async def _assess_framework_compliance(self, 
                                         model_record: ModelGovernanceRecord,
                                         framework: ComplianceFramework) -> Dict[str, Any]:
        """Assess compliance for a specific framework"""
        
        framework_assessment = {
            "framework": framework.value,
            "compliant": True,
            "compliance_score": 1.0,
            "requirements_met": {},
            "violations": []
        }
        
        # Framework-specific compliance checks
        if framework == ComplianceFramework.GDPR:
            # GDPR compliance checks
            gdpr_checks = {
                "data_lineage_documented": bool(model_record.data_lineage),
                "consent_mechanism": "explicit_consent" in model_record.data_lineage,
                "right_to_explanation": bool(model_record.validation_results.get("explainability")),
                "data_minimization": "data_minimization_applied" in model_record.data_lineage
            }
            
            framework_assessment["requirements_met"] = gdpr_checks
            compliance_score = sum(gdpr_checks.values()) / len(gdpr_checks)
            framework_assessment["compliance_score"] = compliance_score
            framework_assessment["compliant"] = compliance_score >= 0.8
        
        elif framework == ComplianceFramework.SOC2:
            # SOC2 compliance checks
            soc2_checks = {
                "security_controls": bool(model_record.safety_assessment),
                "availability_monitoring": "uptime_monitoring" in model_record.validation_results,
                "processing_integrity": bool(model_record.validation_results),
                "confidentiality": "data_encryption" in model_record.data_lineage,
                "privacy": bool(model_record.bias_assessment)
            }
            
            framework_assessment["requirements_met"] = soc2_checks
            compliance_score = sum(soc2_checks.values()) / len(soc2_checks)
            framework_assessment["compliance_score"] = compliance_score
            framework_assessment["compliant"] = compliance_score >= 0.8
        
        elif framework == ComplianceFramework.NIST:
            # NIST Cybersecurity Framework checks
            nist_checks = {
                "identify_assets": bool(model_record.data_lineage),
                "protect_controls": bool(model_record.safety_assessment),
                "detect_anomalies": "anomaly_detection" in model_record.validation_results,
                "respond_procedures": bool(model_record.audit_trail),
                "recover_capabilities": "backup_procedures" in model_record.data_lineage
            }
            
            framework_assessment["requirements_met"] = nist_checks
            compliance_score = sum(nist_checks.values()) / len(nist_checks)
            framework_assessment["compliance_score"] = compliance_score
            framework_assessment["compliant"] = compliance_score >= 0.8
        
        return framework_assessment
    
    async def _check_governance_violations(self, 
                                         model_record: ModelGovernanceRecord) -> List[Dict[str, Any]]:
        """Check for governance rule violations"""
        
        violations = []
        
        # Create context for rule evaluation
        evaluation_context = {
            "model_status": model_record.status,
            "bias_assessment_completed": bool(model_record.bias_assessment),
            "safety_assessment_completed": bool(model_record.safety_assessment),
            "approval_required": self.config["governance_policies"]["model_approval_required"],
            "approved_by_count": len(model_record.approved_by)
        }
        
        for rule_id, rule in self.governance_rules.items():
            if not rule.active:
                continue
            
            # Evaluate rule conditions
            rule_violated = self._evaluate_rule_conditions(rule.conditions, evaluation_context)
            
            if rule_violated:
                violation = {
                    "rule_id": rule_id,
                    "rule_name": rule.name,
                    "severity": rule.severity.value,
                    "description": rule.description,
                    "suggested_actions": rule.actions
                }
                violations.append(violation)
        
        return violations
    
    def _evaluate_rule_conditions(self, 
                                conditions: List[str], 
                                context: Dict[str, Any]) -> bool:
        """Evaluate rule conditions against context"""
        
        try:
            for condition in conditions:
                # Simple condition evaluation (in production, use safe expression evaluator)
                if "==" in condition:
                    var, value = condition.split("==")
                    var = var.strip()
                    value = value.strip().strip('"\'')
                    
                    context_value = context.get(var)
                    if str(context_value) != value:
                        continue
                
                elif ">" in condition:
                    var, value = condition.split(">")
                    var = var.strip()
                    value = float(value.strip())
                    
                    context_value = context.get(var, 0)
                    if float(context_value) <= value:
                        continue
                
                elif "<" in condition:
                    var, value = condition.split("<")
                    var = var.strip()
                    value = float(value.strip())
                    
                    context_value = context.get(var, 0)
                    if float(context_value) >= value:
                        continue
                
                # If we reach here, all conditions are met
                return True
        
        except Exception as e:
            self.logger.warning(f"Failed to evaluate rule conditions: {conditions}", error=str(e))
            return False
        
        return False
    
    async def _generate_compliance_recommendations(self, 
                                                 model_record: ModelGovernanceRecord,
                                                 assessment: Dict[str, Any]) -> List[str]:
        """Generate compliance improvement recommendations"""
        
        recommendations = []
        
        # General recommendations based on compliance status
        if not assessment["overall_compliance"]:
            recommendations.append("Address identified compliance violations before deployment")
        
        # Framework-specific recommendations
        for framework, details in assessment["framework_compliance"].items():
            if not details["compliant"]:
                if framework == "gdpr":
                    recommendations.append("Implement explicit consent mechanism for data processing")
                    recommendations.append("Add explainability features for right to explanation")
                elif framework == "soc2":
                    recommendations.append("Implement comprehensive security controls")
                    recommendations.append("Add availability monitoring capabilities")
                elif framework == "nist":
                    recommendations.append("Document asset inventory and data lineage")
                    recommendations.append("Implement anomaly detection capabilities")
        
        # Model-specific recommendations
        if not model_record.bias_assessment:
            recommendations.append("Complete bias assessment before production deployment")
        
        if not model_record.safety_assessment:
            recommendations.append("Conduct comprehensive safety assessment")
        
        if model_record.status == "production" and not model_record.approved_by:
            recommendations.append("Obtain required approvals from governance team")
        
        return recommendations
    
    async def create_compliance_report(self, 
                                     framework: ComplianceFramework,
                                     time_period: Optional[Tuple[datetime, datetime]] = None) -> Dict[str, Any]:
        """Generate comprehensive compliance report"""
        
        if not time_period:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            time_period = (start_date, end_date)
        
        report = {
            "report_id": f"compliance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "framework": framework.value,
            "generated_at": datetime.now().isoformat(),
            "time_period": {
                "start": time_period[0].isoformat(),
                "end": time_period[1].isoformat()
            },
            "summary": {},
            "model_compliance": {},
            "violations": [],
            "recommendations": []
        }
        
        # Assess all models in registry
        total_models = len(self.model_registry)
        compliant_models = 0
        
        for model_id, model_record in self.model_registry.items():
            assessment = await self._assess_framework_compliance(model_record, framework)
            report["model_compliance"][model_id] = assessment
            
            if assessment["compliant"]:
                compliant_models += 1
        
        # Generate summary
        compliance_rate = compliant_models / total_models if total_models > 0 else 0
        report["summary"] = {
            "total_models": total_models,
            "compliant_models": compliant_models,
            "compliance_rate": compliance_rate,
            "non_compliant_models": total_models - compliant_models
        }
        
        # Get violations for time period
        violations = await self._get_violations_for_period(time_period[0], time_period[1])
        framework_violations = [
            v for v in violations 
            if framework in [ComplianceFramework(f) for f in self.governance_rules[v.rule_id].compliance_frameworks]
        ]
        report["violations"] = [asdict(v) for v in framework_violations]
        
        # Generate recommendations
        if compliance_rate < 0.8:
            report["recommendations"].append(f"Improve {framework.value} compliance rate (currently {compliance_rate:.1%})")
        
        if len(framework_violations) > 0:
            report["recommendations"].append("Address open compliance violations")
        
        self.logger.info(f"Generated {framework.value} compliance report",
                        total_models=total_models,
                        compliance_rate=compliance_rate)
        
        return report
    
    async def _get_violations_for_period(self, 
                                       start_date: datetime, 
                                       end_date: datetime) -> List[ComplianceViolation]:
        """Get compliance violations for time period"""
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM compliance_violations
                WHERE timestamp BETWEEN ? AND ?
                ORDER BY timestamp DESC
            """, (start_date.isoformat(), end_date.isoformat()))
            
            rows = cursor.fetchall()
            conn.close()
            
            violations = []
            for row in rows:
                violation = ComplianceViolation(
                    violation_id=row[0],
                    rule_id=row[1],
                    violation_type=row[2],
                    severity=RiskLevel(row[3]),
                    description=row[4],
                    evidence=json.loads(row[5]),
                    affected_systems=json.loads(row[6]),
                    timestamp=datetime.fromisoformat(row[7]),
                    status=row[8],
                    assigned_to=row[9],
                    resolution_notes=row[10],
                    resolved_at=datetime.fromisoformat(row[11]) if row[11] else None
                )
                violations.append(violation)
            
            return violations
            
        except Exception as e:
            self.logger.error("Failed to retrieve violations", error=str(e))
            return []
    
    async def _audit_log(self, 
                       event_type: str,
                       resource_id: str,
                       action: str,
                       details: Dict[str, Any],
                       user_id: Optional[str] = None,
                       ip_address: Optional[str] = None,
                       user_agent: Optional[str] = None):
        """Create audit log entry"""
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO audit_logs
                (timestamp, event_type, user_id, resource_id, action, details, ip_address, user_agent)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().isoformat(),
                event_type,
                user_id,
                resource_id,
                action,
                json.dumps(details),
                ip_address,
                user_agent
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error("Failed to create audit log", error=str(e))
    
    async def _update_model_record(self, model_record: ModelGovernanceRecord):
        """Update model record in database"""
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                UPDATE model_governance
                SET last_updated = ?, status = ?, compliance_status = ?, audit_trail = ?
                WHERE model_id = ?
            """, (
                datetime.now().isoformat(),
                model_record.status,
                json.dumps(model_record.compliance_status),
                json.dumps(model_record.audit_trail),
                model_record.model_id
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Failed to update model record: {model_record.model_id}", error=str(e))
    
    def get_governance_dashboard_data(self) -> Dict[str, Any]:
        """Get data for governance dashboard"""
        
        total_models = len(self.model_registry)
        compliant_models = sum(
            1 for model in self.model_registry.values()
            if all(model.compliance_status.values())
        )
        
        # Recent violations
        recent_violations = [
            v for v in self.compliance_violations
            if v.timestamp >= datetime.now() - timedelta(days=7)
        ]
        
        # Compliance by framework
        framework_compliance = {}
        for framework in ComplianceFramework:
            framework_compliant = sum(
                1 for model in self.model_registry.values()
                if model.compliance_status.get(framework.value, False)
            )
            framework_compliance[framework.value] = {
                "compliant": framework_compliant,
                "total": total_models,
                "rate": framework_compliant / total_models if total_models > 0 else 0
            }
        
        return {
            "overall_compliance": {
                "total_models": total_models,
                "compliant_models": compliant_models,
                "compliance_rate": compliant_models / total_models if total_models > 0 else 0
            },
            "framework_compliance": framework_compliance,
            "recent_violations": len(recent_violations),
            "governance_rules": len(self.governance_rules),
            "last_updated": datetime.now().isoformat()
        }

# Factory function
def create_enterprise_governance_manager(**kwargs) -> EnterpriseGovernanceManager:
    """Create enterprise governance manager with configuration"""
    return EnterpriseGovernanceManager(**kwargs)

# Example usage
if __name__ == "__main__":
    async def main():
        # Initialize governance manager
        governance = EnterpriseGovernanceManager()
        
        # Register a model
        model_record = await governance.register_model(
            model_id="cyber_llm_v1",
            model_name="Cyber-LLM",
            version="1.0.0",
            metadata={
                "training_metrics": {"accuracy": 0.95, "safety_score": 0.89},
                "bias_assessment": {"demographic_parity": 0.02},
                "data_lineage": {"explicit_consent": True}
            }
        )
        
        # Assess compliance
        compliance = await governance.assess_model_compliance("cyber_llm_v1")
        print(f"Model compliance: {compliance['overall_compliance']}")
        
        # Generate GDPR report
        gdpr_report = await governance.create_compliance_report(ComplianceFramework.GDPR)
        print(f"GDPR compliance rate: {gdpr_report['summary']['compliance_rate']:.1%}")
        
        # Get dashboard data
        dashboard = governance.get_governance_dashboard_data()
        print(f"Overall compliance rate: {dashboard['overall_compliance']['compliance_rate']:.1%}")
    
    asyncio.run(main())
