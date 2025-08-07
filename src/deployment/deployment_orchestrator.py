"""
Project Deployment Orchestrator for Cyber-LLM
Complete deployment automation across cloud platforms with enterprise features

Author: Muzan Sano <sanosensei36@gmail.com>
"""

import asyncio
import json
import logging
import subprocess
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import yaml
import boto3
import kubernetes
from azure.identity import DefaultAzureCredential
from azure.mgmt.containerservice import ContainerServiceClient
from google.cloud import container_v1

from ..utils.logging_system import CyberLLMLogger, CyberLLMError, ErrorCategory
from ..monitoring.prometheus import PrometheusMonitoring
from ..governance.enterprise_governance import EnterpriseGovernanceManager

class DeploymentPlatform(Enum):
    """Supported deployment platforms"""
    AWS_EKS = "aws_eks"
    AZURE_AKS = "azure_aks"
    GCP_GKE = "gcp_gke"
    ON_PREMISE = "on_premise"
    HYBRID_CLOUD = "hybrid_cloud"
    MULTI_CLOUD = "multi_cloud"

class DeploymentEnvironment(Enum):
    """Deployment environments"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    DISASTER_RECOVERY = "disaster_recovery"

class DeploymentStatus(Enum):
    """Deployment status"""
    PENDING = "pending"
    DEPLOYING = "deploying"
    DEPLOYED = "deployed"
    FAILED = "failed"
    ROLLING_BACK = "rolling_back"
    ROLLED_BACK = "rolled_back"
    UPDATING = "updating"

@dataclass
class DeploymentConfiguration:
    """Deployment configuration"""
    platform: DeploymentPlatform
    environment: DeploymentEnvironment
    
    # Resource configuration
    cpu_requests: str = "1000m"
    memory_requests: str = "2Gi"
    cpu_limits: str = "2000m"
    memory_limits: str = "4Gi"
    
    # Scaling configuration
    min_replicas: int = 2
    max_replicas: int = 10
    target_cpu_utilization: int = 70
    
    # Security configuration
    enable_security_policies: bool = True
    enable_network_policies: bool = True
    enable_pod_security_policies: bool = True
    
    # Storage configuration
    storage_class: str = "fast-ssd"
    persistent_volume_size: str = "100Gi"
    
    # Monitoring configuration
    enable_monitoring: bool = True
    monitoring_namespace: str = "monitoring"
    
    # Additional configuration
    custom_annotations: Dict[str, str] = field(default_factory=dict)
    custom_labels: Dict[str, str] = field(default_factory=dict)
    environment_variables: Dict[str, str] = field(default_factory=dict)

@dataclass
class DeploymentResult:
    """Deployment result"""
    deployment_id: str
    status: DeploymentStatus
    platform: DeploymentPlatform
    environment: DeploymentEnvironment
    
    # Deployment details
    deployed_at: Optional[datetime] = None
    deployment_duration: Optional[timedelta] = None
    
    # Resources created
    services_created: List[str] = field(default_factory=list)
    deployments_created: List[str] = field(default_factory=list)
    configmaps_created: List[str] = field(default_factory=list)
    secrets_created: List[str] = field(default_factory=list)
    
    # Access information
    external_endpoints: List[str] = field(default_factory=list)
    internal_endpoints: List[str] = field(default_factory=list)
    
    # Monitoring information
    monitoring_dashboard_url: Optional[str] = None
    health_check_endpoint: Optional[str] = None
    
    # Error information
    error_message: Optional[str] = None
    rollback_available: bool = False

class ProjectDeploymentOrchestrator:
    """Complete project deployment orchestration system"""
    
    def __init__(self, 
                 governance_manager: EnterpriseGovernanceManager,
                 monitoring: PrometheusMonitoring,
                 logger: Optional[CyberLLMLogger] = None):
        
        self.governance_manager = governance_manager
        self.monitoring = monitoring
        self.logger = logger or CyberLLMLogger(name="deployment_orchestrator")
        
        # Deployment tracking
        self.active_deployments = {}
        self.deployment_history = {}
        
        # Platform clients
        self._aws_client = None
        self._azure_client = None
        self._gcp_client = None
        self._k8s_client = None
        
        # Deployment templates
        self.deployment_templates = {}
        
        self.logger.info("Project Deployment Orchestrator initialized")
    
    async def deploy_complete_project(self, 
                                    platform: DeploymentPlatform,
                                    environment: DeploymentEnvironment,
                                    config: Optional[DeploymentConfiguration] = None) -> DeploymentResult:
        """Deploy complete Cyber-LLM project"""
        
        deployment_id = f"cyber_llm_{environment.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            self.logger.info("Starting complete project deployment",
                           deployment_id=deployment_id,
                           platform=platform.value,
                           environment=environment.value)
            
            # Initialize deployment configuration
            if not config:
                config = self._get_default_configuration(platform, environment)
            
            # Mark deployment as starting
            deployment_result = DeploymentResult(
                deployment_id=deployment_id,
                status=DeploymentStatus.DEPLOYING,
                platform=platform,
                environment=environment
            )
            self.active_deployments[deployment_id] = deployment_result
            
            start_time = datetime.now()
            
            # Phase 1: Infrastructure setup
            await self._setup_infrastructure(deployment_id, platform, environment, config)
            
            # Phase 2: Deploy core services
            await self._deploy_core_services(deployment_id, platform, environment, config)
            
            # Phase 3: Deploy AI agents
            await self._deploy_ai_agents(deployment_id, platform, environment, config)
            
            # Phase 4: Deploy orchestration layer
            await self._deploy_orchestration_layer(deployment_id, platform, environment, config)
            
            # Phase 5: Deploy API gateway and web interface
            await self._deploy_api_gateway(deployment_id, platform, environment, config)
            
            # Phase 6: Setup monitoring and observability
            await self._setup_monitoring(deployment_id, platform, environment, config)
            
            # Phase 7: Configure security and compliance
            await self._configure_security(deployment_id, platform, environment, config)
            
            # Phase 8: Run deployment validation
            await self._validate_deployment(deployment_id, platform, environment, config)
            
            # Update deployment result
            end_time = datetime.now()
            deployment_result.status = DeploymentStatus.DEPLOYED
            deployment_result.deployed_at = end_time
            deployment_result.deployment_duration = end_time - start_time
            
            # Move to history
            self.deployment_history[deployment_id] = deployment_result
            del self.active_deployments[deployment_id]
            
            self.logger.info("Project deployment completed successfully",
                           deployment_id=deployment_id,
                           duration=deployment_result.deployment_duration)
            
            return deployment_result
            
        except Exception as e:
            self.logger.error("Project deployment failed",
                            deployment_id=deployment_id,
                            error=str(e))
            
            # Update deployment result with failure
            deployment_result.status = DeploymentStatus.FAILED
            deployment_result.error_message = str(e)
            deployment_result.rollback_available = True
            
            # Attempt rollback
            await self._rollback_deployment(deployment_id)
            
            return deployment_result
    
    async def _setup_infrastructure(self, deployment_id: str, 
                                  platform: DeploymentPlatform,
                                  environment: DeploymentEnvironment,
                                  config: DeploymentConfiguration):
        """Setup underlying infrastructure"""
        
        self.logger.info("Setting up infrastructure", deployment_id=deployment_id)
        
        if platform == DeploymentPlatform.AWS_EKS:
            await self._setup_aws_infrastructure(deployment_id, environment, config)
        elif platform == DeploymentPlatform.AZURE_AKS:
            await self._setup_azure_infrastructure(deployment_id, environment, config)
        elif platform == DeploymentPlatform.GCP_GKE:
            await self._setup_gcp_infrastructure(deployment_id, environment, config)
        elif platform == DeploymentPlatform.ON_PREMISE:
            await self._setup_onpremise_infrastructure(deployment_id, environment, config)
        
        # Create namespace
        await self._create_namespace(deployment_id, environment)
        
        # Setup RBAC
        await self._setup_rbac(deployment_id, environment, config)
        
        # Create secrets and configmaps
        await self._create_secrets_configmaps(deployment_id, environment, config)
    
    async def _deploy_core_services(self, deployment_id: str,
                                   platform: DeploymentPlatform,
                                   environment: DeploymentEnvironment,
                                   config: DeploymentConfiguration):
        """Deploy core services"""
        
        self.logger.info("Deploying core services", deployment_id=deployment_id)
        
        # Deploy databases
        await self._deploy_databases(deployment_id, environment, config)
        
        # Deploy message queues
        await self._deploy_message_queues(deployment_id, environment, config)
        
        # Deploy caching layer
        await self._deploy_caching_layer(deployment_id, environment, config)
        
        # Deploy logging and metrics collection
        await self._deploy_logging_metrics(deployment_id, environment, config)
    
    async def _deploy_ai_agents(self, deployment_id: str,
                               platform: DeploymentPlatform,
                               environment: DeploymentEnvironment,
                               config: DeploymentConfiguration):
        """Deploy AI agent services"""
        
        self.logger.info("Deploying AI agents", deployment_id=deployment_id)
        
        # Deploy reconnaissance agent
        await self._deploy_service("recon-agent", deployment_id, environment, config)
        
        # Deploy C2 agent
        await self._deploy_service("c2-agent", deployment_id, environment, config)
        
        # Deploy post-exploit agent
        await self._deploy_service("post-exploit-agent", deployment_id, environment, config)
        
        # Deploy explainability agent
        await self._deploy_service("explainability-agent", deployment_id, environment, config)
        
        # Deploy safety agent
        await self._deploy_service("safety-agent", deployment_id, environment, config)
    
    async def _deploy_orchestration_layer(self, deployment_id: str,
                                        platform: DeploymentPlatform,
                                        environment: DeploymentEnvironment,
                                        config: DeploymentConfiguration):
        """Deploy orchestration layer"""
        
        self.logger.info("Deploying orchestration layer", deployment_id=deployment_id)
        
        # Deploy main orchestrator
        await self._deploy_service("orchestrator", deployment_id, environment, config)
        
        # Deploy workflow engine
        await self._deploy_service("workflow-engine", deployment_id, environment, config)
        
        # Deploy external tool integration
        await self._deploy_service("tool-integration", deployment_id, environment, config)
        
        # Deploy learning systems
        await self._deploy_service("learning-system", deployment_id, environment, config)
    
    async def _deploy_api_gateway(self, deployment_id: str,
                                platform: DeploymentPlatform,
                                environment: DeploymentEnvironment,
                                config: DeploymentConfiguration):
        """Deploy API gateway and web interface"""
        
        self.logger.info("Deploying API gateway", deployment_id=deployment_id)
        
        # Deploy API gateway
        await self._deploy_service("api-gateway", deployment_id, environment, config)
        
        # Deploy web interface
        await self._deploy_service("web-interface", deployment_id, environment, config)
        
        # Deploy CLI interface
        await self._deploy_service("cli-interface", deployment_id, environment, config)
        
        # Setup ingress/load balancer
        await self._setup_ingress(deployment_id, environment, config)
    
    async def _setup_monitoring(self, deployment_id: str,
                              platform: DeploymentPlatform,
                              environment: DeploymentEnvironment,
                              config: DeploymentConfiguration):
        """Setup monitoring and observability"""
        
        if not config.enable_monitoring:
            return
        
        self.logger.info("Setting up monitoring", deployment_id=deployment_id)
        
        # Deploy Prometheus
        await self._deploy_prometheus(deployment_id, environment, config)
        
        # Deploy Grafana
        await self._deploy_grafana(deployment_id, environment, config)
        
        # Deploy alerting
        await self._deploy_alertmanager(deployment_id, environment, config)
        
        # Deploy distributed tracing
        await self._deploy_jaeger(deployment_id, environment, config)
        
        # Setup custom dashboards
        await self._setup_custom_dashboards(deployment_id, environment, config)
    
    async def _configure_security(self, deployment_id: str,
                                platform: DeploymentPlatform,
                                environment: DeploymentEnvironment,
                                config: DeploymentConfiguration):
        """Configure security and compliance"""
        
        self.logger.info("Configuring security", deployment_id=deployment_id)
        
        # Setup network policies
        if config.enable_network_policies:
            await self._setup_network_policies(deployment_id, environment, config)
        
        # Setup security policies
        if config.enable_security_policies:
            await self._setup_security_policies(deployment_id, environment, config)
        
        # Setup pod security policies
        if config.enable_pod_security_policies:
            await self._setup_pod_security_policies(deployment_id, environment, config)
        
        # Setup certificate management
        await self._setup_certificate_management(deployment_id, environment, config)
        
        # Setup secrets management
        await self._setup_secrets_management(deployment_id, environment, config)
    
    async def _validate_deployment(self, deployment_id: str,
                                 platform: DeploymentPlatform,
                                 environment: DeploymentEnvironment,
                                 config: DeploymentConfiguration):
        """Validate deployment"""
        
        self.logger.info("Validating deployment", deployment_id=deployment_id)
        
        # Health checks
        await self._run_health_checks(deployment_id, environment)
        
        # Connectivity tests
        await self._run_connectivity_tests(deployment_id, environment)
        
        # Performance tests
        await self._run_performance_tests(deployment_id, environment)
        
        # Security validation
        await self._run_security_validation(deployment_id, environment)
        
        # Compliance validation
        await self._run_compliance_validation(deployment_id, environment)
    
    async def _deploy_service(self, service_name: str,
                            deployment_id: str,
                            environment: DeploymentEnvironment,
                            config: DeploymentConfiguration):
        """Deploy a specific service"""
        
        self.logger.info(f"Deploying {service_name}", deployment_id=deployment_id)
        
        # Generate Kubernetes manifests
        manifests = self._generate_service_manifests(service_name, environment, config)
        
        # Apply manifests
        for manifest in manifests:
            await self._apply_k8s_manifest(manifest)
        
        # Wait for deployment to be ready
        await self._wait_for_deployment_ready(service_name, environment)
        
        # Update deployment result
        if deployment_id in self.active_deployments:
            self.active_deployments[deployment_id].deployments_created.append(service_name)
    
    def _generate_service_manifests(self, service_name: str,
                                  environment: DeploymentEnvironment,
                                  config: DeploymentConfiguration) -> List[Dict[str, Any]]:
        """Generate Kubernetes manifests for a service"""
        
        namespace = f"cyber-llm-{environment.value}"
        
        # Deployment manifest
        deployment = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": service_name,
                "namespace": namespace,
                "labels": {
                    "app": service_name,
                    "version": "v1.0.0",
                    "environment": environment.value,
                    **config.custom_labels
                },
                "annotations": config.custom_annotations
            },
            "spec": {
                "replicas": config.min_replicas,
                "selector": {
                    "matchLabels": {
                        "app": service_name
                    }
                },
                "template": {
                    "metadata": {
                        "labels": {
                            "app": service_name,
                            "version": "v1.0.0"
                        }
                    },
                    "spec": {
                        "containers": [{
                            "name": service_name,
                            "image": f"cyber-llm/{service_name}:latest",
                            "ports": [{
                                "containerPort": 8080,
                                "name": "http"
                            }],
                            "resources": {
                                "requests": {
                                    "cpu": config.cpu_requests,
                                    "memory": config.memory_requests
                                },
                                "limits": {
                                    "cpu": config.cpu_limits,
                                    "memory": config.memory_limits
                                }
                            },
                            "env": [
                                {"name": k, "value": v} 
                                for k, v in config.environment_variables.items()
                            ],
                            "livenessProbe": {
                                "httpGet": {
                                    "path": "/health",
                                    "port": 8080
                                },
                                "initialDelaySeconds": 30,
                                "periodSeconds": 10
                            },
                            "readinessProbe": {
                                "httpGet": {
                                    "path": "/ready",
                                    "port": 8080
                                },
                                "initialDelaySeconds": 5,
                                "periodSeconds": 5
                            }
                        }]
                    }
                }
            }
        }
        
        # Service manifest
        service = {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {
                "name": service_name,
                "namespace": namespace,
                "labels": {
                    "app": service_name
                }
            },
            "spec": {
                "selector": {
                    "app": service_name
                },
                "ports": [{
                    "port": 80,
                    "targetPort": 8080,
                    "name": "http"
                }],
                "type": "ClusterIP"
            }
        }
        
        # HPA manifest
        hpa = {
            "apiVersion": "autoscaling/v2",
            "kind": "HorizontalPodAutoscaler",
            "metadata": {
                "name": f"{service_name}-hpa",
                "namespace": namespace
            },
            "spec": {
                "scaleTargetRef": {
                    "apiVersion": "apps/v1",
                    "kind": "Deployment",
                    "name": service_name
                },
                "minReplicas": config.min_replicas,
                "maxReplicas": config.max_replicas,
                "metrics": [{
                    "type": "Resource",
                    "resource": {
                        "name": "cpu",
                        "target": {
                            "type": "Utilization",
                            "averageUtilization": config.target_cpu_utilization
                        }
                    }
                }]
            }
        }
        
        return [deployment, service, hpa]
    
    def _get_default_configuration(self, platform: DeploymentPlatform, 
                                 environment: DeploymentEnvironment) -> DeploymentConfiguration:
        """Get default deployment configuration"""
        
        # Adjust resources based on environment
        if environment == DeploymentEnvironment.PRODUCTION:
            return DeploymentConfiguration(
                platform=platform,
                environment=environment,
                cpu_requests="2000m",
                memory_requests="4Gi",
                cpu_limits="4000m",
                memory_limits="8Gi",
                min_replicas=3,
                max_replicas=20,
                target_cpu_utilization=70
            )
        elif environment == DeploymentEnvironment.STAGING:
            return DeploymentConfiguration(
                platform=platform,
                environment=environment,
                cpu_requests="1000m",
                memory_requests="2Gi",
                cpu_limits="2000m",
                memory_limits="4Gi",
                min_replicas=2,
                max_replicas=10,
                target_cpu_utilization=75
            )
        else:
            return DeploymentConfiguration(
                platform=platform,
                environment=environment,
                cpu_requests="500m",
                memory_requests="1Gi",
                cpu_limits="1000m",
                memory_limits="2Gi",
                min_replicas=1,
                max_replicas=5,
                target_cpu_utilization=80
            )

# Factory function
def create_deployment_orchestrator(governance_manager: EnterpriseGovernanceManager,
                                 monitoring: PrometheusMonitoring,
                                 **kwargs) -> ProjectDeploymentOrchestrator:
    """Create project deployment orchestrator"""
    return ProjectDeploymentOrchestrator(governance_manager, monitoring, **kwargs)
