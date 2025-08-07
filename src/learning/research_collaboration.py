"""
Research Collaboration Framework for Cyber-LLM
Enables secure sharing of cybersecurity insights and collaborative research across organizations.

Author: Muzan Sano <sanosensei36@gmail.com>
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Set, Union, Callable
from dataclasses import dataclass, asdict, field
from enum import Enum
from abc import ABC, abstractmethod
import hashlib
import hmac
import base64
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import redis
import yaml
from pathlib import Path
import uuid

from ..utils.logging_system import CyberLLMLogger
from .online_learning import LearningEvent, LearningEventType

# Configure logging
logger = CyberLLMLogger(__name__).get_logger()

class CollaborationType(Enum):
    """Types of research collaboration"""
    THREAT_INTELLIGENCE_SHARING = "threat_intelligence_sharing"
    ATTACK_PATTERN_ANALYSIS = "attack_pattern_analysis" 
    DEFENSE_STRATEGY_DEVELOPMENT = "defense_strategy_development"
    VULNERABILITY_RESEARCH = "vulnerability_research"
    INCIDENT_CASE_STUDIES = "incident_case_studies"
    TOOL_BENCHMARKING = "tool_benchmarking"
    DATASET_SHARING = "dataset_sharing"

class ParticipantRole(Enum):
    """Roles in research collaboration"""
    COORDINATOR = "coordinator"          # Manages collaboration
    CONTRIBUTOR = "contributor"          # Contributes data/insights  
    VALIDATOR = "validator"             # Validates findings
    OBSERVER = "observer"               # Read-only access
    ANALYST = "analyst"                 # Analyzes shared data

class SensitivityLevel(Enum):
    """Data sensitivity levels for sharing"""
    PUBLIC = "public"                   # Publicly shareable
    CONSORTIUM = "consortium"           # Share within trusted consortium
    BILATERAL = "bilateral"             # Share between two organizations
    INTERNAL = "internal"               # Internal use only
    CLASSIFIED = "classified"           # Highly sensitive, restricted

@dataclass
class ResearchInsight:
    """Structure for research insights"""
    insight_id: str
    title: str
    description: str
    collaboration_type: CollaborationType
    sensitivity_level: SensitivityLevel
    
    # Content
    findings: Dict[str, Any]
    evidence: List[Dict[str, Any]]
    methodology: Dict[str, Any]
    
    # Metadata
    contributor_org: str
    contributors: List[str]
    created_at: datetime
    updated_at: datetime
    version: str
    
    # Validation
    validation_status: str = "pending"  # pending, validated, disputed
    validators: List[str] = field(default_factory=list)
    validation_feedback: List[Dict[str, Any]] = field(default_factory=list)
    
    # Privacy
    anonymized: bool = False
    data_retention_days: Optional[int] = None
    access_log: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'insight_id': self.insight_id,
            'title': self.title,
            'description': self.description,
            'collaboration_type': self.collaboration_type.value,
            'sensitivity_level': self.sensitivity_level.value,
            'findings': self.findings,
            'evidence': self.evidence,
            'methodology': self.methodology,
            'contributor_org': self.contributor_org,
            'contributors': self.contributors,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'version': self.version,
            'validation_status': self.validation_status,
            'validators': self.validators,
            'validation_feedback': self.validation_feedback,
            'anonymized': self.anonymized,
            'data_retention_days': self.data_retention_days,
            'access_log': self.access_log
        }

@dataclass
class CollaborationParticipant:
    """Research collaboration participant"""
    participant_id: str
    organization: str
    name: str
    email: str
    role: ParticipantRole
    public_key: str
    
    # Capabilities and interests
    expertise_areas: List[str]
    research_interests: List[CollaborationType]
    data_sharing_policy: Dict[str, Any]
    
    # Status
    status: str = "active"  # active, suspended, inactive
    joined_at: datetime = field(default_factory=datetime.now)
    last_active: Optional[datetime] = None
    
    # Metrics
    contributions_count: int = 0
    validations_count: int = 0
    reputation_score: float = 0.0

@dataclass  
class CollaborationProject:
    """Research collaboration project"""
    project_id: str
    name: str
    description: str
    collaboration_type: CollaborationType
    
    # Management
    coordinator: str  # participant_id
    participants: List[str]  # participant_ids
    created_at: datetime
    deadline: Optional[datetime]
    
    # Configuration
    sensitivity_level: SensitivityLevel
    data_sharing_rules: Dict[str, Any]
    validation_requirements: Dict[str, Any]
    
    # Status
    status: str = "active"  # active, completed, suspended
    progress: float = 0.0
    
    # Content
    insights: List[str] = field(default_factory=list)  # insight_ids
    deliverables: List[Dict[str, Any]] = field(default_factory=list)

class SecureCollaborationProtocol:
    """Secure communication protocol for research collaboration"""
    
    def __init__(self, private_key_path: str, public_key_path: str):
        self.private_key = self._load_private_key(private_key_path)
        self.public_key = self._load_public_key(public_key_path)
        
        # Key registry for participants
        self.participant_keys: Dict[str, Any] = {}
        
    def _load_private_key(self, key_path: str):
        """Load private key from file"""
        try:
            with open(key_path, 'rb') as f:
                return serialization.load_pem_private_key(
                    f.read(), password=None, backend=default_backend()
                )
        except FileNotFoundError:
            logger.warning(f"Private key not found at {key_path}, generating new key")
            return self._generate_key_pair(key_path)
    
    def _load_public_key(self, key_path: str):
        """Load public key from file"""
        try:
            with open(key_path, 'rb') as f:
                return serialization.load_pem_public_key(
                    f.read(), backend=default_backend()
                )
        except FileNotFoundError:
            return self.private_key.public_key()
    
    def _generate_key_pair(self, private_key_path: str):
        """Generate new RSA key pair"""
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=default_backend()
        )
        
        # Save private key
        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        
        with open(private_key_path, 'wb') as f:
            f.write(private_pem)
        
        # Save public key
        public_key = private_key.public_key()
        public_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        
        public_key_path = private_key_path.replace('private', 'public')
        with open(public_key_path, 'wb') as f:
            f.write(public_pem)
        
        logger.info(f"Generated new key pair: {private_key_path}")
        return private_key
    
    def encrypt_data(self, data: Dict[str, Any], recipient_public_key: str) -> str:
        """Encrypt data for specific recipient"""
        try:
            # Serialize data
            data_json = json.dumps(data, default=str).encode('utf-8')
            
            # Load recipient's public key
            recipient_key = serialization.load_pem_public_key(
                recipient_public_key.encode(), backend=default_backend()
            )
            
            # Encrypt with recipient's public key
            encrypted_data = recipient_key.encrypt(
                data_json,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            
            # Sign with our private key
            signature = self.private_key.sign(
                encrypted_data,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            
            # Combine encrypted data and signature
            payload = {
                'encrypted_data': base64.b64encode(encrypted_data).decode(),
                'signature': base64.b64encode(signature).decode(),
                'timestamp': datetime.now().isoformat()
            }
            
            return base64.b64encode(json.dumps(payload).encode()).decode()
            
        except Exception as e:
            logger.error(f"Encryption failed: {str(e)}")
            raise
    
    def decrypt_data(self, encrypted_payload: str, sender_public_key: str) -> Dict[str, Any]:
        """Decrypt data from sender"""
        try:
            # Decode payload
            payload = json.loads(base64.b64decode(encrypted_payload).decode())
            encrypted_data = base64.b64decode(payload['encrypted_data'])
            signature = base64.b64decode(payload['signature'])
            
            # Load sender's public key
            sender_key = serialization.load_pem_public_key(
                sender_public_key.encode(), backend=default_backend()
            )
            
            # Verify signature
            sender_key.verify(
                signature,
                encrypted_data,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            
            # Decrypt data with our private key
            decrypted_data = self.private_key.decrypt(
                encrypted_data,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            
            return json.loads(decrypted_data.decode('utf-8'))
            
        except Exception as e:
            logger.error(f"Decryption failed: {str(e)}")
            raise
    
    def register_participant_key(self, participant_id: str, public_key: str):
        """Register participant's public key"""
        self.participant_keys[participant_id] = public_key
        logger.info(f"Registered public key for participant: {participant_id}")

class PrivacyPreservingAnalytics:
    """Privacy-preserving analytics for collaborative research"""
    
    def __init__(self):
        self.anonymization_functions = {
            'k_anonymity': self._apply_k_anonymity,
            'differential_privacy': self._apply_differential_privacy,
            'homomorphic': self._apply_homomorphic_encryption
        }
    
    def anonymize_insight(self, insight: ResearchInsight, method: str = 'k_anonymity') -> ResearchInsight:
        """Anonymize research insight"""
        
        if method not in self.anonymization_functions:
            raise ValueError(f"Unsupported anonymization method: {method}")
        
        try:
            anonymized_insight = self.anonymization_functions[method](insight)
            anonymized_insight.anonymized = True
            
            logger.info(f"Applied {method} anonymization to insight: {insight.insight_id}")
            return anonymized_insight
            
        except Exception as e:
            logger.error(f"Anonymization failed: {str(e)}")
            raise
    
    def _apply_k_anonymity(self, insight: ResearchInsight, k: int = 5) -> ResearchInsight:
        """Apply k-anonymity to insight"""
        anonymized_insight = insight
        
        # Remove direct identifiers
        anonymized_insight.contributor_org = f"Organization_{hash(insight.contributor_org) % 1000}"
        anonymized_insight.contributors = [f"Researcher_{i}" for i in range(len(insight.contributors))]
        
        # Generalize sensitive fields in findings
        if 'ip_addresses' in insight.findings:
            ips = insight.findings['ip_addresses']
            anonymized_insight.findings['ip_addresses'] = [
                '.'.join(ip.split('.')[:2] + ['x', 'x']) for ip in ips
            ]
        
        if 'timestamps' in insight.findings:
            timestamps = insight.findings['timestamps']
            anonymized_insight.findings['timestamps'] = [
                ts[:10] for ts in timestamps  # Keep only date, remove time
            ]
        
        return anonymized_insight
    
    def _apply_differential_privacy(self, insight: ResearchInsight, epsilon: float = 1.0) -> ResearchInsight:
        """Apply differential privacy to insight"""
        import numpy as np
        
        anonymized_insight = insight
        
        # Add calibrated noise to numerical values
        for key, value in insight.findings.items():
            if isinstance(value, (int, float)):
                # Add Laplace noise
                sensitivity = 1.0  # Adjust based on data
                scale = sensitivity / epsilon
                noise = np.random.laplace(0, scale)
                anonymized_insight.findings[key] = max(0, value + noise)
        
        return anonymized_insight
    
    def _apply_homomorphic_encryption(self, insight: ResearchInsight) -> ResearchInsight:
        """Apply homomorphic encryption to insight"""
        # Simplified homomorphic encryption simulation
        # In production, use libraries like Microsoft SEAL or IBM HElib
        
        anonymized_insight = insight
        
        # Encrypt numerical values
        for key, value in insight.findings.items():
            if isinstance(value, (int, float)):
                # Simple encryption simulation (not real homomorphic encryption)
                encrypted_value = f"HE_encrypted_{hash(str(value)) % 10000}"
                anonymized_insight.findings[key] = encrypted_value
        
        return anonymized_insight
    
    def compute_privacy_risk_score(self, insight: ResearchInsight) -> float:
        """Compute privacy risk score for insight"""
        
        risk_score = 0.0
        
        # Check for direct identifiers
        if not insight.anonymized:
            risk_score += 0.3
        
        # Check sensitivity level
        sensitivity_risk = {
            SensitivityLevel.PUBLIC: 0.0,
            SensitivityLevel.CONSORTIUM: 0.1,
            SensitivityLevel.BILATERAL: 0.2,
            SensitivityLevel.INTERNAL: 0.4,
            SensitivityLevel.CLASSIFIED: 0.8
        }
        risk_score += sensitivity_risk.get(insight.sensitivity_level, 0.5)
        
        # Check for PII in findings
        pii_indicators = ['email', 'ip', 'username', 'id', 'address']
        for indicator in pii_indicators:
            if any(indicator in str(value).lower() for value in insight.findings.values()):
                risk_score += 0.1
        
        # Check data retention
        if insight.data_retention_days is None:
            risk_score += 0.1
        
        return min(1.0, risk_score)

class CollaborationRepository:
    """Repository for managing collaboration data"""
    
    def __init__(self, redis_host: str = "localhost", redis_port: int = 6379):
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        
        # Data structures
        self.participants: Dict[str, CollaborationParticipant] = {}
        self.projects: Dict[str, CollaborationProject] = {}
        self.insights: Dict[str, ResearchInsight] = {}
        
        # Load existing data
        self._load_data()
    
    def _load_data(self):
        """Load existing data from Redis"""
        try:
            # Load participants
            participant_ids = self.redis_client.smembers("collaboration:participants")
            for pid in participant_ids:
                data = self.redis_client.hget("collaboration:participant", pid)
                if data:
                    self.participants[pid] = CollaborationParticipant(**json.loads(data))
            
            # Load projects
            project_ids = self.redis_client.smembers("collaboration:projects")
            for proj_id in project_ids:
                data = self.redis_client.hget("collaboration:project", proj_id)
                if data:
                    self.projects[proj_id] = CollaborationProject(**json.loads(data))
            
            # Load insights
            insight_ids = self.redis_client.smembers("collaboration:insights")
            for insight_id in insight_ids:
                data = self.redis_client.hget("collaboration:insight", insight_id)
                if data:
                    self.insights[insight_id] = ResearchInsight(**json.loads(data))
            
            logger.info(f"Loaded {len(self.participants)} participants, "
                       f"{len(self.projects)} projects, {len(self.insights)} insights")
        
        except Exception as e:
            logger.error(f"Failed to load data from Redis: {str(e)}")
    
    def save_participant(self, participant: CollaborationParticipant):
        """Save participant to repository"""
        try:
            self.participants[participant.participant_id] = participant
            
            # Save to Redis
            self.redis_client.sadd("collaboration:participants", participant.participant_id)
            self.redis_client.hset(
                "collaboration:participant", 
                participant.participant_id,
                json.dumps(asdict(participant), default=str)
            )
            
            logger.info(f"Saved participant: {participant.participant_id}")
            
        except Exception as e:
            logger.error(f"Failed to save participant: {str(e)}")
            raise
    
    def save_project(self, project: CollaborationProject):
        """Save project to repository"""
        try:
            self.projects[project.project_id] = project
            
            # Save to Redis
            self.redis_client.sadd("collaboration:projects", project.project_id)
            self.redis_client.hset(
                "collaboration:project",
                project.project_id, 
                json.dumps(asdict(project), default=str)
            )
            
            logger.info(f"Saved project: {project.project_id}")
            
        except Exception as e:
            logger.error(f"Failed to save project: {str(e)}")
            raise
    
    def save_insight(self, insight: ResearchInsight):
        """Save insight to repository"""
        try:
            self.insights[insight.insight_id] = insight
            
            # Save to Redis
            self.redis_client.sadd("collaboration:insights", insight.insight_id)
            self.redis_client.hset(
                "collaboration:insight",
                insight.insight_id,
                json.dumps(insight.to_dict())
            )
            
            # Update access log
            access_entry = {
                'action': 'save',
                'timestamp': datetime.now().isoformat(),
                'user': 'system'
            }
            insight.access_log.append(access_entry)
            
            logger.info(f"Saved insight: {insight.insight_id}")
            
        except Exception as e:
            logger.error(f"Failed to save insight: {str(e)}")
            raise
    
    def get_participant(self, participant_id: str) -> Optional[CollaborationParticipant]:
        """Get participant by ID"""
        return self.participants.get(participant_id)
    
    def get_project(self, project_id: str) -> Optional[CollaborationProject]:
        """Get project by ID"""
        return self.projects.get(project_id)
    
    def get_insight(self, insight_id: str) -> Optional[ResearchInsight]:
        """Get insight by ID"""
        insight = self.insights.get(insight_id)
        
        if insight:
            # Log access
            access_entry = {
                'action': 'access',
                'timestamp': datetime.now().isoformat(),
                'user': 'system'
            }
            insight.access_log.append(access_entry)
        
        return insight
    
    def search_insights(self, 
                       collaboration_type: Optional[CollaborationType] = None,
                       sensitivity_level: Optional[SensitivityLevel] = None,
                       contributor_org: Optional[str] = None) -> List[ResearchInsight]:
        """Search insights by criteria"""
        
        results = []
        
        for insight in self.insights.values():
            if (collaboration_type is None or insight.collaboration_type == collaboration_type) and \
               (sensitivity_level is None or insight.sensitivity_level == sensitivity_level) and \
               (contributor_org is None or insight.contributor_org == contributor_org):
                results.append(insight)
        
        return results

class ResearchCollaborationManager:
    """Main manager for research collaboration"""
    
    def __init__(self, 
                 organization_name: str,
                 private_key_path: str = "keys/collaboration_private.pem",
                 public_key_path: str = "keys/collaboration_public.pem"):
        
        self.organization_name = organization_name
        
        # Initialize components
        self.security_protocol = SecureCollaborationProtocol(private_key_path, public_key_path)
        self.privacy_analytics = PrivacyPreservingAnalytics()
        self.repository = CollaborationRepository()
        
        # Configuration
        self.collaboration_config = self._load_collaboration_config()
        
        logger.info(f"ResearchCollaborationManager initialized for: {organization_name}")
    
    def _load_collaboration_config(self) -> Dict[str, Any]:
        """Load collaboration configuration"""
        
        config_path = Path("configs/collaboration.yaml")
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            # Default configuration
            default_config = {
                'default_sensitivity_level': SensitivityLevel.CONSORTIUM.value,
                'auto_validation_enabled': True,
                'data_retention_days': 365,
                'privacy_method': 'k_anonymity',
                'min_validation_score': 0.8,
                'collaboration_timeout_hours': 72
            }
            
            # Save default configuration
            config_path.parent.mkdir(exist_ok=True)
            with open(config_path, 'w') as f:
                yaml.dump(default_config, f)
            
            return default_config
    
    async def create_collaboration_project(self,
                                         name: str,
                                         description: str,
                                         collaboration_type: CollaborationType,
                                         coordinator_id: str,
                                         participants: List[str],
                                         sensitivity_level: SensitivityLevel = SensitivityLevel.CONSORTIUM,
                                         deadline: Optional[datetime] = None) -> str:
        """Create new collaboration project"""
        
        project_id = f"proj_{uuid.uuid4().hex[:8]}"
        
        project = CollaborationProject(
            project_id=project_id,
            name=name,
            description=description,
            collaboration_type=collaboration_type,
            coordinator=coordinator_id,
            participants=participants,
            created_at=datetime.now(),
            deadline=deadline,
            sensitivity_level=sensitivity_level,
            data_sharing_rules={
                'anonymization_required': sensitivity_level != SensitivityLevel.PUBLIC,
                'validation_required': True,
                'retention_days': self.collaboration_config.get('data_retention_days', 365)
            },
            validation_requirements={
                'min_validators': 2,
                'min_score': self.collaboration_config.get('min_validation_score', 0.8)
            }
        )
        
        self.repository.save_project(project)
        
        logger.info(f"Created collaboration project: {project_id} - {name}")
        return project_id
    
    async def contribute_insight(self,
                               project_id: str,
                               title: str,
                               description: str,
                               findings: Dict[str, Any],
                               evidence: List[Dict[str, Any]],
                               methodology: Dict[str, Any],
                               contributor_id: str) -> str:
        """Contribute research insight to project"""
        
        project = self.repository.get_project(project_id)
        if not project:
            raise ValueError(f"Project not found: {project_id}")
        
        contributor = self.repository.get_participant(contributor_id)
        if not contributor:
            raise ValueError(f"Contributor not found: {contributor_id}")
        
        insight_id = f"insight_{uuid.uuid4().hex[:8]}"
        
        insight = ResearchInsight(
            insight_id=insight_id,
            title=title,
            description=description,
            collaboration_type=project.collaboration_type,
            sensitivity_level=project.sensitivity_level,
            findings=findings,
            evidence=evidence,
            methodology=methodology,
            contributor_org=contributor.organization,
            contributors=[contributor.name],
            created_at=datetime.now(),
            updated_at=datetime.now(),
            version="1.0",
            data_retention_days=project.data_sharing_rules.get('retention_days')
        )
        
        # Apply privacy protection if required
        if project.data_sharing_rules.get('anonymization_required', False):
            privacy_method = self.collaboration_config.get('privacy_method', 'k_anonymity')
            insight = self.privacy_analytics.anonymize_insight(insight, privacy_method)
        
        # Compute privacy risk
        privacy_risk = self.privacy_analytics.compute_privacy_risk_score(insight)
        if privacy_risk > 0.7:
            logger.warning(f"High privacy risk detected for insight: {insight_id} (risk: {privacy_risk:.2f})")
        
        self.repository.save_insight(insight)
        
        # Add insight to project
        project.insights.append(insight_id)
        self.repository.save_project(project)
        
        # Update contributor metrics
        contributor.contributions_count += 1
        contributor.last_active = datetime.now()
        self.repository.save_participant(contributor)
        
        logger.info(f"Contributed insight: {insight_id} to project: {project_id}")
        return insight_id
    
    async def validate_insight(self,
                             insight_id: str,
                             validator_id: str,
                             validation_score: float,
                             feedback: str) -> bool:
        """Validate research insight"""
        
        insight = self.repository.get_insight(insight_id)
        if not insight:
            raise ValueError(f"Insight not found: {insight_id}")
        
        validator = self.repository.get_participant(validator_id)
        if not validator:
            raise ValueError(f"Validator not found: {validator_id}")
        
        # Add validation feedback
        validation_feedback = {
            'validator_id': validator_id,
            'validator_name': validator.name,
            'score': validation_score,
            'feedback': feedback,
            'timestamp': datetime.now().isoformat()
        }
        
        insight.validation_feedback.append(validation_feedback)
        insight.validators.append(validator_id)
        
        # Update validation status
        if len(insight.validators) >= 2:  # Minimum validators met
            avg_score = sum(vf['score'] for vf in insight.validation_feedback) / len(insight.validation_feedback)
            min_score = self.collaboration_config.get('min_validation_score', 0.8)
            
            if avg_score >= min_score:
                insight.validation_status = "validated"
                logger.info(f"Insight {insight_id} validated with score: {avg_score:.2f}")
            else:
                insight.validation_status = "disputed"
                logger.warning(f"Insight {insight_id} disputed with score: {avg_score:.2f}")
        
        self.repository.save_insight(insight)
        
        # Update validator metrics
        validator.validations_count += 1
        validator.last_active = datetime.now()
        self.repository.save_participant(validator)
        
        return insight.validation_status == "validated"
    
    async def share_insight_securely(self,
                                   insight_id: str,
                                   recipient_ids: List[str]) -> Dict[str, str]:
        """Share insight securely with specific recipients"""
        
        insight = self.repository.get_insight(insight_id)
        if not insight:
            raise ValueError(f"Insight not found: {insight_id}")
        
        shared_data = {}
        
        for recipient_id in recipient_ids:
            recipient = self.repository.get_participant(recipient_id)
            if not recipient:
                logger.warning(f"Recipient not found: {recipient_id}")
                continue
            
            try:
                # Encrypt insight for recipient
                encrypted_payload = self.security_protocol.encrypt_data(
                    insight.to_dict(),
                    recipient.public_key
                )
                
                shared_data[recipient_id] = encrypted_payload
                
                logger.info(f"Encrypted insight {insight_id} for recipient: {recipient_id}")
                
            except Exception as e:
                logger.error(f"Failed to encrypt for {recipient_id}: {str(e)}")
        
        return shared_data
    
    def generate_collaboration_report(self, project_id: str) -> Dict[str, Any]:
        """Generate comprehensive collaboration report"""
        
        project = self.repository.get_project(project_id)
        if not project:
            raise ValueError(f"Project not found: {project_id}")
        
        # Collect project insights
        project_insights = []
        for insight_id in project.insights:
            insight = self.repository.get_insight(insight_id)
            if insight:
                project_insights.append(insight)
        
        # Calculate metrics
        total_insights = len(project_insights)
        validated_insights = len([i for i in project_insights if i.validation_status == "validated"])
        disputed_insights = len([i for i in project_insights if i.validation_status == "disputed"])
        pending_insights = total_insights - validated_insights - disputed_insights
        
        # Participant statistics
        participant_contributions = {}
        for insight in project_insights:
            org = insight.contributor_org
            participant_contributions[org] = participant_contributions.get(org, 0) + 1
        
        # Validation statistics
        validation_scores = []
        for insight in project_insights:
            if insight.validation_feedback:
                avg_score = sum(vf['score'] for vf in insight.validation_feedback) / len(insight.validation_feedback)
                validation_scores.append(avg_score)
        
        avg_validation_score = sum(validation_scores) / len(validation_scores) if validation_scores else 0.0
        
        return {
            'project_info': {
                'project_id': project.project_id,
                'name': project.name,
                'collaboration_type': project.collaboration_type.value,
                'status': project.status,
                'created_at': project.created_at.isoformat(),
                'participants_count': len(project.participants)
            },
            'insight_statistics': {
                'total_insights': total_insights,
                'validated_insights': validated_insights,
                'disputed_insights': disputed_insights,
                'pending_insights': pending_insights,
                'validation_rate': validated_insights / total_insights if total_insights > 0 else 0.0
            },
            'validation_metrics': {
                'average_validation_score': avg_validation_score,
                'total_validations': sum(len(i.validators) for i in project_insights),
                'unique_validators': len(set(v for i in project_insights for v in i.validators))
            },
            'participant_contributions': participant_contributions,
            'collaboration_effectiveness': {
                'insights_per_participant': total_insights / len(project.participants) if project.participants else 0.0,
                'validation_coverage': len([i for i in project_insights if i.validators]) / total_insights if total_insights > 0 else 0.0
            }
        }
    
    def get_collaboration_statistics(self) -> Dict[str, Any]:
        """Get overall collaboration statistics"""
        
        total_participants = len(self.repository.participants)
        total_projects = len(self.repository.projects)
        total_insights = len(self.repository.insights)
        
        # Active projects
        active_projects = len([p for p in self.repository.projects.values() if p.status == "active"])
        
        # Recent activity (last 30 days)
        thirty_days_ago = datetime.now() - timedelta(days=30)
        recent_insights = len([
            i for i in self.repository.insights.values() 
            if i.created_at >= thirty_days_ago
        ])
        
        # Collaboration types distribution
        collab_type_dist = {}
        for project in self.repository.projects.values():
            ct = project.collaboration_type.value
            collab_type_dist[ct] = collab_type_dist.get(ct, 0) + 1
        
        return {
            'overview': {
                'total_participants': total_participants,
                'total_projects': total_projects,
                'total_insights': total_insights,
                'active_projects': active_projects
            },
            'recent_activity': {
                'insights_last_30_days': recent_insights,
                'activity_rate': recent_insights / 30.0
            },
            'collaboration_distribution': collab_type_dist,
            'organization': self.organization_name
        }

# Factory function
def create_research_collaboration_manager(organization_name: str, **kwargs) -> ResearchCollaborationManager:
    """Create research collaboration manager with configuration"""
    return ResearchCollaborationManager(organization_name, **kwargs)
