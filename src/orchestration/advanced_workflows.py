"""
Advanced Workflow Orchestration System for Cyber-LLM
Implements complex multi-agent scenarios, dynamic adaptation, and external tool integration
"""

import os
import json
import yaml
import asyncio
from typing import Dict, List, Any, Optional, Callable, Union
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict, field
from enum import Enum
import uuid
import subprocess
from concurrent.futures import ThreadPoolExecutor
import aiohttp
import redis

from ..utils.logging_system import CyberLLMLogger, CyberLLMError, ErrorCategory, retry_with_backoff
from ..utils.secrets_manager import get_secrets_manager
from ..agents.orchestrator_agent import OrchestratorAgent

class WorkflowStatus(Enum):
    """Workflow execution status"""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class TaskPriority(Enum):
    """Task priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class WorkflowContext:
    """Dynamic workflow context with adaptive capabilities"""
    workflow_id: str
    current_stage: str
    variables: Dict[str, Any] = field(default_factory=dict)
    agent_states: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    execution_history: List[Dict[str, Any]] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    external_tool_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    adaptation_rules: List[Dict[str, Any]] = field(default_factory=list)
    rollback_points: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class ExternalToolConfig:
    """Configuration for external tool integration"""
    tool_name: str
    tool_type: str  # metasploit, burp_suite, nmap, etc.
    endpoint: str
    api_key: Optional[str] = None
    credentials: Optional[Dict[str, str]] = None
    timeout: int = 300
    retry_attempts: int = 3
    custom_headers: Dict[str, str] = field(default_factory=dict)

class AdvancedWorkflowEngine:
    """Advanced workflow orchestration engine with dynamic adaptation"""
    
    def __init__(self, 
                 redis_url: str = "redis://localhost:6379",
                 max_concurrent_workflows: int = 10,
                 logger: Optional[CyberLLMLogger] = None):
        
        self.logger = logger or CyberLLMLogger(name="advanced_workflow_engine")
        self.max_concurrent_workflows = max_concurrent_workflows
        
        # Initialize Redis for state management
        try:
            self.redis_client = redis.from_url(redis_url, decode_responses=True)
            self.redis_client.ping()
        except Exception as e:
            self.logger.warning(f"Redis not available, using in-memory storage: {e}")
            self.redis_client = None
        
        # Workflow state management
        self.active_workflows: Dict[str, WorkflowContext] = {}
        self.workflow_templates: Dict[str, Dict[str, Any]] = {}
        self.external_tools: Dict[str, ExternalToolConfig] = {}
        
        # Execution control
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent_workflows)
        self.is_running = False
        
        # Load workflow templates and external tool configs
        self._load_workflow_templates()
        self._load_external_tool_configs()
    
    def _load_workflow_templates(self):
        """Load advanced workflow templates"""
        templates_dir = Path(__file__).parent / "templates"
        
        if not templates_dir.exists():
            templates_dir.mkdir(parents=True)
            self._create_default_templates(templates_dir)
        
        for template_file in templates_dir.glob("*.yaml"):
            try:
                with open(template_file, 'r') as f:
                    template = yaml.safe_load(f)
                
                template_name = template_file.stem
                self.workflow_templates[template_name] = template
                
                self.logger.info(f"Loaded workflow template: {template_name}")
                
            except Exception as e:
                self.logger.error(f"Failed to load template {template_file}", error=str(e))
    
    def _create_default_templates(self, templates_dir: Path):
        """Create default advanced workflow templates"""
        
        # Advanced Red Team Exercise Template
        red_team_template = {
            "name": "advanced_red_team_exercise",
            "description": "Comprehensive red team exercise with dynamic adaptation",
            "version": "2.0",
            "timeout": 3600,  # 1 hour
            "parallel_execution": True,
            "dynamic_adaptation": True,
            "rollback_enabled": True,
            
            "stages": {
                "reconnaissance": {
                    "type": "parallel",
                    "agents": ["recon_agent"],
                    "external_tools": ["nmap", "shodan", "amass"],
                    "timeout": 600,
                    "adaptation_rules": [
                        {
                            "condition": "target_ports_found > 50",
                            "action": "increase_scan_depth",
                            "parameters": {"depth": 2}
                        },
                        {
                            "condition": "stealth_score < 0.7",
                            "action": "reduce_scan_frequency",
                            "parameters": {"delay": 5}
                        }
                    ],
                    "tasks": [
                        {
                            "name": "network_discovery",
                            "agent": "recon_agent",
                            "action": "discover_network",
                            "parameters": {
                                "target": "${workflow.target}",
                                "scan_type": "stealth",
                                "timeout": 300
                            },
                            "success_criteria": {
                                "min_hosts_found": 1,
                                "stealth_score": 0.7
                            }
                        },
                        {
                            "name": "service_enumeration",
                            "agent": "recon_agent",
                            "action": "enumerate_services",
                            "dependencies": ["network_discovery"],
                            "parameters": {
                                "targets": "${reconnaissance.network_discovery.hosts}",
                                "depth": "moderate"
                            }
                        },
                        {
                            "name": "vulnerability_assessment",
                            "external_tool": "nessus",
                            "parameters": {
                                "targets": "${reconnaissance.service_enumeration.services}",
                                "scan_template": "advanced"
                            },
                            "parallel": True
                        }
                    ]
                },
                
                "initial_access": {
                    "type": "sequential",
                    "agents": ["c2_agent"],
                    "external_tools": ["metasploit", "burp_suite"],
                    "dependencies": ["reconnaissance"],
                    "timeout": 900,
                    "rollback_point": True,
                    "adaptation_rules": [
                        {
                            "condition": "exploit_attempts > 3 and success_rate < 0.3",
                            "action": "switch_strategy",
                            "parameters": {"strategy": "social_engineering"}
                        }
                    ],
                    "tasks": [
                        {
                            "name": "exploit_selection",
                            "agent": "c2_agent",
                            "action": "select_exploits",
                            "parameters": {
                                "vulnerabilities": "${reconnaissance.vulnerability_assessment.findings}",
                                "target_os": "${reconnaissance.network_discovery.os_info}",
                                "stealth_required": True
                            }
                        },
                        {
                            "name": "payload_generation",
                            "external_tool": "metasploit",
                            "action": "generate_payload",
                            "parameters": {
                                "exploit": "${initial_access.exploit_selection.chosen_exploit}",
                                "target": "${workflow.target}",
                                "avoid_detection": True
                            }
                        },
                        {
                            "name": "exploitation",
                            "agent": "c2_agent",
                            "action": "execute_exploit",
                            "parameters": {
                                "payload": "${initial_access.payload_generation.payload}",
                                "target": "${workflow.target}",
                                "max_attempts": 3
                            },
                            "success_criteria": {
                                "shell_obtained": True,
                                "detection_avoided": True
                            }
                        }
                    ]
                },
                
                "post_exploitation": {
                    "type": "parallel",
                    "agents": ["post_exploit_agent"],
                    "dependencies": ["initial_access"],
                    "timeout": 1200,
                    "conditional": {
                        "condition": "initial_access.exploitation.success == True",
                        "else_action": "skip_stage"
                    },
                    "tasks": [
                        {
                            "name": "privilege_escalation",
                            "agent": "post_exploit_agent",
                            "action": "escalate_privileges",
                            "parameters": {
                                "session": "${initial_access.exploitation.session}",
                                "target_privilege": "system"
                            },
                            "priority": "high"
                        },
                        {
                            "name": "persistence",
                            "agent": "post_exploit_agent",
                            "action": "establish_persistence",
                            "parameters": {
                                "session": "${initial_access.exploitation.session}",
                                "method": "service",
                                "stealth": True
                            },
                            "priority": "normal"
                        },
                        {
                            "name": "lateral_movement",
                            "agent": "post_exploit_agent",
                            "action": "move_laterally",
                            "dependencies": ["privilege_escalation"],
                            "parameters": {
                                "session": "${initial_access.exploitation.session}",
                                "discovery_method": "active_directory"
                            },
                            "priority": "high"
                        }
                    ]
                },
                
                "cleanup": {
                    "type": "sequential",
                    "agents": ["post_exploit_agent", "safety_agent"],
                    "always_execute": True,
                    "timeout": 300,
                    "tasks": [
                        {
                            "name": "remove_persistence",
                            "agent": "post_exploit_agent",
                            "action": "cleanup_persistence",
                            "parameters": {
                                "persistence_info": "${post_exploitation.persistence.info}"
                            }
                        },
                        {
                            "name": "close_sessions",
                            "agent": "post_exploit_agent",
                            "action": "cleanup_sessions",
                            "parameters": {
                                "sessions": "${workflow.active_sessions}"
                            }
                        },
                        {
                            "name": "safety_verification",
                            "agent": "safety_agent",
                            "action": "verify_cleanup",
                            "parameters": {
                                "target": "${workflow.target}",
                                "cleanup_actions": "${cleanup.remove_persistence.actions}"
                            }
                        }
                    ]
                }
            },
            
            "success_criteria": {
                "overall": {
                    "min_stages_completed": 3,
                    "stealth_score": 0.8,
                    "safety_compliance": 0.95
                }
            },
            
            "failure_handling": {
                "max_retries": 2,
                "rollback_on_failure": True,
                "escalation_rules": [
                    {
                        "condition": "detection_probability > 0.8",
                        "action": "immediate_cleanup"
                    }
                ]
            }
        }
        
        # Web Application Security Assessment Template
        web_app_template = {
            "name": "web_app_security_assessment",
            "description": "Comprehensive web application security testing",
            "version": "2.0",
            "timeout": 2400,  # 40 minutes
            "parallel_execution": True,
            
            "stages": {
                "reconnaissance": {
                    "type": "parallel",
                    "agents": ["recon_agent"],
                    "external_tools": ["burp_suite", "dirb", "nikto"],
                    "tasks": [
                        {
                            "name": "web_discovery",
                            "agent": "recon_agent",
                            "action": "discover_web_assets",
                            "parameters": {
                                "target": "${workflow.target}",
                                "depth": 3,
                                "follow_redirects": True
                            }
                        },
                        {
                            "name": "technology_fingerprinting",
                            "external_tool": "whatweb",
                            "parameters": {
                                "target": "${workflow.target}",
                                "aggression": 3
                            }
                        }
                    ]
                },
                
                "vulnerability_scanning": {
                    "type": "parallel",
                    "external_tools": ["burp_suite", "owasp_zap", "sqlmap"],
                    "dependencies": ["reconnaissance"],
                    "tasks": [
                        {
                            "name": "automated_scan",
                            "external_tool": "owasp_zap",
                            "parameters": {
                                "target": "${workflow.target}",
                                "scan_policy": "full",
                                "spider_depth": 5
                            }
                        },
                        {
                            "name": "sql_injection_test",
                            "external_tool": "sqlmap",
                            "parameters": {
                                "urls": "${reconnaissance.web_discovery.forms}",
                                "risk": 2,
                                "level": 3
                            }
                        }
                    ]
                },
                
                "manual_testing": {
                    "type": "sequential",
                    "agents": ["recon_agent", "c2_agent"],
                    "dependencies": ["vulnerability_scanning"],
                    "tasks": [
                        {
                            "name": "business_logic_testing",
                            "agent": "recon_agent",
                            "action": "test_business_logic",
                            "parameters": {
                                "application_map": "${reconnaissance.web_discovery.map}",
                                "user_roles": "${workflow.user_roles}"
                            }
                        },
                        {
                            "name": "authentication_bypass",
                            "agent": "c2_agent",
                            "action": "test_auth_bypass",
                            "parameters": {
                                "auth_endpoints": "${reconnaissance.web_discovery.auth_endpoints}",
                                "methods": ["parameter_pollution", "race_condition", "jwt_manipulation"]
                            }
                        }
                    ]
                }
            }
        }
        
        # Save templates
        with open(templates_dir / "advanced_red_team_exercise.yaml", 'w') as f:
            yaml.dump(red_team_template, f, default_flow_style=False)
        
        with open(templates_dir / "web_app_security_assessment.yaml", 'w') as f:
            yaml.dump(web_app_template, f, default_flow_style=False)
        
        self.logger.info("Created default advanced workflow templates")
    
    def _load_external_tool_configs(self):
        """Load external tool configurations"""
        secrets_manager = get_secrets_manager()
        
        # Common external tools configuration
        tool_configs = {
            "metasploit": ExternalToolConfig(
                tool_name="metasploit",
                tool_type="exploitation_framework",
                endpoint="http://localhost:55553",
                api_key=None,  # Will be loaded from secrets
                timeout=300
            ),
            "burp_suite": ExternalToolConfig(
                tool_name="burp_suite",
                tool_type="web_security",
                endpoint="http://localhost:1337",
                timeout=600
            ),
            "nmap": ExternalToolConfig(
                tool_name="nmap",
                tool_type="network_scanner",
                endpoint="localhost",  # CLI tool
                timeout=300
            ),
            "owasp_zap": ExternalToolConfig(
                tool_name="owasp_zap",
                tool_type="web_security",
                endpoint="http://localhost:8080",
                timeout=900
            ),
            "sqlmap": ExternalToolConfig(
                tool_name="sqlmap",
                tool_type="sql_injection",
                endpoint="localhost",  # CLI tool
                timeout=600
            ),
            "nessus": ExternalToolConfig(
                tool_name="nessus",
                tool_type="vulnerability_scanner",
                endpoint="https://localhost:8834",
                timeout=1800
            )
        }
        
        # Load API keys from secrets manager
        for tool_name, config in tool_configs.items():
            try:
                api_key_name = f"{tool_name}_api_key"
                # This would normally load from secrets manager
                # config.api_key = asyncio.run(secrets_manager.get_secret(api_key_name))
                self.logger.debug(f"External tool configured: {tool_name}")
            except Exception as e:
                self.logger.warning(f"Failed to load API key for {tool_name}", error=str(e))
        
        self.external_tools = tool_configs
        self.logger.info(f"Configured {len(tool_configs)} external tools")
    
    async def create_workflow(self, 
                            template_name: str,
                            parameters: Dict[str, Any],
                            workflow_id: Optional[str] = None) -> str:
        """Create a new workflow instance"""
        
        if template_name not in self.workflow_templates:
            raise CyberLLMError(f"Workflow template not found: {template_name}", ErrorCategory.VALIDATION)
        
        workflow_id = workflow_id or str(uuid.uuid4())
        
        # Create workflow context
        context = WorkflowContext(
            workflow_id=workflow_id,
            current_stage="pending",
            variables=parameters.copy()
        )
        
        # Add template-specific configuration
        template = self.workflow_templates[template_name]
        context.variables.update({
            "template_name": template_name,
            "template_version": template.get("version", "1.0"),
            "created_at": datetime.now().isoformat(),
            "timeout": template.get("timeout", 1800)
        })
        
        # Load adaptation rules from template
        if template.get("dynamic_adaptation"):
            context.adaptation_rules = self._extract_adaptation_rules(template)
        
        # Store workflow context
        self.active_workflows[workflow_id] = context
        
        if self.redis_client:
            await self._save_workflow_state(workflow_id, context)
        
        self.logger.info(f"Created workflow: {workflow_id} from template: {template_name}")
        return workflow_id
    
    def _extract_adaptation_rules(self, template: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract adaptation rules from workflow template"""
        rules = []
        
        for stage_name, stage_config in template.get("stages", {}).items():
            stage_rules = stage_config.get("adaptation_rules", [])
            for rule in stage_rules:
                rule["stage"] = stage_name
                rules.append(rule)
        
        return rules
    
    async def execute_workflow(self, workflow_id: str) -> Dict[str, Any]:
        """Execute workflow with advanced orchestration"""
        
        if workflow_id not in self.active_workflows:
            raise CyberLLMError(f"Workflow not found: {workflow_id}", ErrorCategory.VALIDATION)
        
        context = self.active_workflows[workflow_id]
        template = self.workflow_templates[context.variables["template_name"]]
        
        try:
            context.current_stage = "running"
            context.variables["started_at"] = datetime.now().isoformat()
            
            self.logger.info(f"Starting workflow execution: {workflow_id}")
            
            # Execute stages
            stages = template["stages"]
            execution_results = {}
            
            for stage_name, stage_config in stages.items():
                self.logger.info(f"Executing stage: {stage_name}")
                
                # Check dependencies
                dependencies = stage_config.get("dependencies", [])
                if not self._check_dependencies(dependencies, execution_results):
                    self.logger.warning(f"Dependencies not met for stage: {stage_name}")
                    continue
                
                # Check conditional execution
                if not self._check_conditional(stage_config.get("conditional"), context, execution_results):
                    self.logger.info(f"Skipping stage due to condition: {stage_name}")
                    continue
                
                # Create rollback point if specified
                if stage_config.get("rollback_point", False):
                    await self._create_rollback_point(context, stage_name)
                
                # Execute stage
                stage_result = await self._execute_stage(
                    stage_name, stage_config, context, execution_results
                )
                
                execution_results[stage_name] = stage_result
                
                # Apply dynamic adaptations
                if template.get("dynamic_adaptation"):
                    await self._apply_adaptations(context, stage_name, stage_result)
                
                # Check for failure and handle rollback
                if not stage_result.get("success", False):
                    failure_config = template.get("failure_handling", {})
                    if failure_config.get("rollback_on_failure", False):
                        await self._handle_failure_rollback(context, stage_name)
                        break
            
            # Calculate overall success
            success_criteria = template.get("success_criteria", {}).get("overall", {})
            overall_success = self._evaluate_success_criteria(success_criteria, execution_results, context)
            
            # Update workflow status
            context.current_stage = "completed" if overall_success else "failed"
            context.variables["completed_at"] = datetime.now().isoformat()
            context.variables["success"] = overall_success
            
            # Generate execution report
            execution_report = {
                "workflow_id": workflow_id,
                "template": context.variables["template_name"],
                "status": context.current_stage,
                "duration": self._calculate_duration(context),
                "stages_executed": len(execution_results),
                "overall_success": overall_success,
                "results": execution_results,
                "metrics": context.metrics,
                "adaptation_actions": len(context.execution_history)
            }
            
            self.logger.info(f"Workflow execution completed: {workflow_id}, Success: {overall_success}")
            return execution_report
            
        except Exception as e:
            context.current_stage = "failed"
            self.logger.error(f"Workflow execution failed: {workflow_id}", error=str(e))
            raise CyberLLMError(f"Workflow execution failed: {str(e)}", ErrorCategory.SYSTEM)
    
    async def _execute_stage(self, 
                           stage_name: str,
                           stage_config: Dict[str, Any],
                           context: WorkflowContext,
                           previous_results: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a workflow stage with parallel/sequential support"""
        
        stage_type = stage_config.get("type", "sequential")
        tasks = stage_config.get("tasks", [])
        timeout = stage_config.get("timeout", 300)
        
        stage_result = {
            "stage": stage_name,
            "type": stage_type,
            "started_at": datetime.now().isoformat(),
            "tasks": {},
            "success": True,
            "errors": []
        }
        
        try:
            if stage_type == "parallel":
                # Execute tasks in parallel
                task_futures = []
                
                for task in tasks:
                    if self._check_task_dependencies(task, stage_result["tasks"]):
                        future = asyncio.create_task(
                            self._execute_task(task, context, previous_results, stage_result)
                        )
                        task_futures.append((task["name"], future))
                
                # Wait for all tasks with timeout
                try:
                    results = await asyncio.wait_for(
                        asyncio.gather(*[future for _, future in task_futures], return_exceptions=True),
                        timeout=timeout
                    )
                    
                    for (task_name, _), result in zip(task_futures, results):
                        if isinstance(result, Exception):
                            stage_result["errors"].append(f"Task {task_name} failed: {str(result)}")
                            stage_result["success"] = False
                        else:
                            stage_result["tasks"][task_name] = result
                            
                except asyncio.TimeoutError:
                    stage_result["errors"].append(f"Stage timeout after {timeout} seconds")
                    stage_result["success"] = False
            
            else:  # sequential execution
                for task in tasks:
                    if not self._check_task_dependencies(task, stage_result["tasks"]):
                        continue
                    
                    try:
                        task_result = await asyncio.wait_for(
                            self._execute_task(task, context, previous_results, stage_result),
                            timeout=task.get("timeout", timeout)
                        )
                        
                        stage_result["tasks"][task["name"]] = task_result
                        
                        # Check if task failed and should stop stage
                        if not task_result.get("success", False) and task.get("critical", False):
                            stage_result["success"] = False
                            break
                            
                    except asyncio.TimeoutError:
                        error_msg = f"Task {task['name']} timeout"
                        stage_result["errors"].append(error_msg)
                        if task.get("critical", False):
                            stage_result["success"] = False
                            break
                    except Exception as e:
                        error_msg = f"Task {task['name']} failed: {str(e)}"
                        stage_result["errors"].append(error_msg)
                        if task.get("critical", False):
                            stage_result["success"] = False
                            break
        
        except Exception as e:
            stage_result["errors"].append(f"Stage execution error: {str(e)}")
            stage_result["success"] = False
        
        stage_result["completed_at"] = datetime.now().isoformat()
        stage_result["duration"] = (
            datetime.fromisoformat(stage_result["completed_at"]) - 
            datetime.fromisoformat(stage_result["started_at"])
        ).total_seconds()
        
        return stage_result
    
    async def _execute_task(self, 
                          task: Dict[str, Any],
                          context: WorkflowContext,
                          previous_results: Dict[str, Any],
                          stage_result: Dict[str, Any]) -> Dict[str, Any]:
        """Execute individual task (agent action or external tool)"""
        
        task_name = task["name"]
        task_result = {
            "task": task_name,
            "started_at": datetime.now().isoformat(),
            "success": False,
            "output": {},
            "errors": []
        }
        
        try:
            # Resolve parameters with variable substitution
            resolved_params = self._resolve_parameters(
                task.get("parameters", {}), 
                context, 
                previous_results, 
                stage_result
            )
            
            if "agent" in task:
                # Execute agent task
                task_result = await self._execute_agent_task(
                    task["agent"], 
                    task["action"], 
                    resolved_params,
                    task_result
                )
            
            elif "external_tool" in task:
                # Execute external tool task
                task_result = await self._execute_external_tool_task(
                    task["external_tool"],
                    task.get("action"),
                    resolved_params,
                    task_result
                )
            
            else:
                raise CyberLLMError(f"Task type not specified: {task_name}", ErrorCategory.VALIDATION)
            
            # Validate success criteria
            success_criteria = task.get("success_criteria", {})
            if success_criteria and not self._validate_success_criteria(success_criteria, task_result["output"]):
                task_result["success"] = False
                task_result["errors"].append("Success criteria not met")
        
        except Exception as e:
            task_result["errors"].append(f"Task execution error: {str(e)}")
            task_result["success"] = False
        
        task_result["completed_at"] = datetime.now().isoformat()
        return task_result
    
    async def _execute_agent_task(self, 
                                agent_name: str,
                                action: str,
                                parameters: Dict[str, Any],
                                task_result: Dict[str, Any]) -> Dict[str, Any]:
        """Execute task using cyber-llm agent"""
        
        # This would integrate with actual agent implementations
        # For now, we'll simulate the execution
        
        self.logger.info(f"Executing agent task: {agent_name}.{action}")
        
        # Simulate agent execution
        await asyncio.sleep(0.1)  # Simulate processing time
        
        task_result["output"] = {
            "agent": agent_name,
            "action": action,
            "parameters": parameters,
            "simulated": True,
            "result": f"Simulated execution of {action} by {agent_name}"
        }
        task_result["success"] = True
        
        return task_result
    
    async def _execute_external_tool_task(self,
                                        tool_name: str,
                                        action: Optional[str],
                                        parameters: Dict[str, Any],
                                        task_result: Dict[str, Any]) -> Dict[str, Any]:
        """Execute task using external security tool"""
        
        if tool_name not in self.external_tools:
            raise CyberLLMError(f"External tool not configured: {tool_name}", ErrorCategory.CONFIGURATION)
        
        tool_config = self.external_tools[tool_name]
        
        self.logger.info(f"Executing external tool task: {tool_name}")
        
        try:
            if tool_config.tool_type in ["network_scanner", "sql_injection"]:
                # CLI-based tools
                result = await self._execute_cli_tool(tool_name, parameters)
            else:
                # API-based tools
                result = await self._execute_api_tool(tool_config, action, parameters)
            
            task_result["output"] = result
            task_result["success"] = True
            
        except Exception as e:
            self.logger.error(f"External tool execution failed: {tool_name}", error=str(e))
            task_result["errors"].append(f"External tool error: {str(e)}")
            task_result["success"] = False
        
        return task_result
    
    async def _execute_cli_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute CLI-based security tool"""
        
        if tool_name == "nmap":
            target = parameters.get("target", "localhost")
            scan_type = parameters.get("scan_type", "stealth")
            
            cmd = ["nmap", "-sS", "-O", target] if scan_type == "stealth" else ["nmap", "-sV", target]
            
        elif tool_name == "sqlmap":
            url = parameters.get("url", "")
            cmd = ["sqlmap", "-u", url, "--batch", "--risk=2", "--level=3"]
            
        else:
            raise CyberLLMError(f"Unsupported CLI tool: {tool_name}", ErrorCategory.VALIDATION)
        
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            return {
                "tool": tool_name,
                "command": " ".join(cmd),
                "exit_code": process.returncode,
                "stdout": stdout.decode() if stdout else "",
                "stderr": stderr.decode() if stderr else "",
                "success": process.returncode == 0
            }
            
        except Exception as e:
            raise CyberLLMError(f"CLI tool execution failed: {str(e)}", ErrorCategory.SYSTEM)
    
    async def _execute_api_tool(self, 
                              tool_config: ExternalToolConfig,
                              action: Optional[str],
                              parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute API-based security tool"""
        
        headers = {"Content-Type": "application/json"}
        headers.update(tool_config.custom_headers)
        
        if tool_config.api_key:
            headers["Authorization"] = f"Bearer {tool_config.api_key}"
        
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=tool_config.timeout)) as session:
            
            if tool_config.tool_name == "metasploit":
                # Metasploit RPC API simulation
                endpoint = f"{tool_config.endpoint}/api/v1/exploit"
                payload = {
                    "exploit": parameters.get("exploit", ""),
                    "target": parameters.get("target", ""),
                    "payload_type": parameters.get("payload_type", "reverse_tcp")
                }
                
            elif tool_config.tool_name == "burp_suite":
                # Burp Suite API simulation
                endpoint = f"{tool_config.endpoint}/burp/scanner/scans"
                payload = {
                    "scan_type": "active",
                    "target": parameters.get("target", "")
                }
                
            elif tool_config.tool_name == "owasp_zap":
                # OWASP ZAP API simulation
                endpoint = f"{tool_config.endpoint}/JSON/ascan/action/scan/"
                payload = {
                    "url": parameters.get("target", ""),
                    "policy": parameters.get("scan_policy", "default")
                }
                
            else:
                raise CyberLLMError(f"Unsupported API tool: {tool_config.tool_name}", ErrorCategory.VALIDATION)
            
            try:
                async with session.post(endpoint, json=payload, headers=headers) as response:
                    result_data = await response.json() if response.content_type == 'application/json' else await response.text()
                    
                    return {
                        "tool": tool_config.tool_name,
                        "endpoint": endpoint,
                        "status_code": response.status,
                        "response": result_data,
                        "success": response.status < 400
                    }
                    
            except aiohttp.ClientError as e:
                raise CyberLLMError(f"API tool request failed: {str(e)}", ErrorCategory.NETWORK)
    
    def _resolve_parameters(self, 
                          parameters: Dict[str, Any],
                          context: WorkflowContext,
                          previous_results: Dict[str, Any],
                          stage_result: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve parameter variables with workflow context"""
        
        resolved = {}
        
        for key, value in parameters.items():
            if isinstance(value, str) and value.startswith("${"):
                # Variable substitution
                var_path = value[2:-1]  # Remove ${ and }
                resolved_value = self._resolve_variable(var_path, context, previous_results, stage_result)
                resolved[key] = resolved_value
            else:
                resolved[key] = value
        
        return resolved
    
    def _resolve_variable(self, 
                        var_path: str,
                        context: WorkflowContext,
                        previous_results: Dict[str, Any],
                        stage_result: Dict[str, Any]) -> Any:
        """Resolve variable from context"""
        
        parts = var_path.split(".")
        
        if parts[0] == "workflow":
            # Workflow variables
            current = context.variables
            for part in parts[1:]:
                current = current.get(part, "")
            return current
        
        elif parts[0] in previous_results:
            # Previous stage results
            current = previous_results[parts[0]]
            for part in parts[1:]:
                if isinstance(current, dict):
                    current = current.get(part, "")
                else:
                    current = ""
            return current
        
        else:
            self.logger.warning(f"Unable to resolve variable: {var_path}")
            return ""
    
    async def _apply_adaptations(self, 
                               context: WorkflowContext,
                               stage_name: str,
                               stage_result: Dict[str, Any]):
        """Apply dynamic workflow adaptations based on results"""
        
        for rule in context.adaptation_rules:
            if rule.get("stage") != stage_name:
                continue
            
            condition = rule.get("condition", "")
            if self._evaluate_condition(condition, context, stage_result):
                action = rule.get("action", "")
                parameters = rule.get("parameters", {})
                
                await self._execute_adaptation_action(action, parameters, context, stage_result)
                
                # Log adaptation
                adaptation_record = {
                    "timestamp": datetime.now().isoformat(),
                    "stage": stage_name,
                    "condition": condition,
                    "action": action,
                    "parameters": parameters
                }
                context.execution_history.append(adaptation_record)
                
                self.logger.info(f"Applied adaptation: {action} in stage {stage_name}")
    
    def _evaluate_condition(self, 
                          condition: str,
                          context: WorkflowContext,
                          stage_result: Dict[str, Any]) -> bool:
        """Evaluate adaptation condition"""
        
        # Simple condition evaluation (in production, use a proper expression evaluator)
        # Examples: "target_ports_found > 50", "stealth_score < 0.7"
        
        try:
            # This is a simplified implementation
            # In production, use a safe expression evaluator
            
            if ">" in condition:
                left, right = condition.split(">")
                left_val = self._get_condition_value(left.strip(), context, stage_result)
                right_val = float(right.strip())
                return float(left_val) > right_val
            
            elif "<" in condition:
                left, right = condition.split("<")
                left_val = self._get_condition_value(left.strip(), context, stage_result)
                right_val = float(right.strip())
                return float(left_val) < right_val
            
            elif "==" in condition:
                left, right = condition.split("==")
                left_val = self._get_condition_value(left.strip(), context, stage_result)
                right_val = right.strip().strip('"\'')
                return str(left_val) == right_val
            
            return False
            
        except Exception as e:
            self.logger.warning(f"Failed to evaluate condition: {condition}", error=str(e))
            return False
    
    def _get_condition_value(self, 
                           variable: str,
                           context: WorkflowContext,
                           stage_result: Dict[str, Any]) -> Any:
        """Get value for condition evaluation"""
        
        if variable in context.metrics:
            return context.metrics[variable]
        elif variable in context.variables:
            return context.variables[variable]
        else:
            # Try to extract from stage result
            return 0  # Default value
    
    async def _execute_adaptation_action(self, 
                                       action: str,
                                       parameters: Dict[str, Any],
                                       context: WorkflowContext,
                                       stage_result: Dict[str, Any]):
        """Execute adaptation action"""
        
        if action == "increase_scan_depth":
            depth = parameters.get("depth", 1)
            context.variables["scan_depth"] = context.variables.get("scan_depth", 1) + depth
            
        elif action == "reduce_scan_frequency":
            delay = parameters.get("delay", 1)
            context.variables["scan_delay"] = delay
            
        elif action == "switch_strategy":
            new_strategy = parameters.get("strategy", "default")
            context.variables["current_strategy"] = new_strategy
            
        elif action == "immediate_cleanup":
            context.variables["emergency_cleanup"] = True
            
        else:
            self.logger.warning(f"Unknown adaptation action: {action}")
    
    async def _create_rollback_point(self, context: WorkflowContext, stage_name: str):
        """Create rollback point for workflow recovery"""
        
        rollback_point = {
            "stage": stage_name,
            "timestamp": datetime.now().isoformat(),
            "context_snapshot": {
                "variables": context.variables.copy(),
                "agent_states": context.agent_states.copy(),
                "metrics": context.metrics.copy()
            }
        }
        
        context.rollback_points.append(rollback_point)
        self.logger.info(f"Created rollback point at stage: {stage_name}")
    
    async def _handle_failure_rollback(self, context: WorkflowContext, failed_stage: str):
        """Handle workflow rollback on failure"""
        
        if not context.rollback_points:
            self.logger.warning("No rollback points available")
            return
        
        # Find the most recent rollback point before the failed stage
        rollback_point = context.rollback_points[-1]
        
        # Restore context
        context.variables.update(rollback_point["context_snapshot"]["variables"])
        context.agent_states.update(rollback_point["context_snapshot"]["agent_states"])
        context.metrics.update(rollback_point["context_snapshot"]["metrics"])
        
        self.logger.info(f"Rolled back workflow to stage: {rollback_point['stage']}")
    
    # Additional utility methods would continue here...
    
    async def _save_workflow_state(self, workflow_id: str, context: WorkflowContext):
        """Save workflow state to Redis"""
        if self.redis_client:
            try:
                state_data = asdict(context)
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    self.redis_client.setex,
                    f"workflow:{workflow_id}",
                    3600,  # 1 hour TTL
                    json.dumps(state_data, default=str)
                )
            except Exception as e:
                self.logger.warning(f"Failed to save workflow state: {workflow_id}", error=str(e))
    
    def _check_dependencies(self, dependencies: List[str], results: Dict[str, Any]) -> bool:
        """Check if stage dependencies are satisfied"""
        return all(dep in results and results[dep].get("success", False) for dep in dependencies)
    
    def _check_conditional(self, 
                         conditional: Optional[Dict[str, Any]],
                         context: WorkflowContext,
                         results: Dict[str, Any]) -> bool:
        """Check conditional execution requirements"""
        if not conditional:
            return True
        
        condition = conditional.get("condition", "")
        return self._evaluate_condition(condition, context, {"tasks": results})
    
    def _check_task_dependencies(self, task: Dict[str, Any], completed_tasks: Dict[str, Any]) -> bool:
        """Check if task dependencies are satisfied"""
        dependencies = task.get("dependencies", [])
        return all(dep in completed_tasks and completed_tasks[dep].get("success", False) for dep in dependencies)
    
    def _validate_success_criteria(self, criteria: Dict[str, Any], output: Dict[str, Any]) -> bool:
        """Validate task success criteria"""
        for key, expected_value in criteria.items():
            if key not in output:
                return False
            
            actual_value = output[key]
            if isinstance(expected_value, (int, float)):
                if actual_value < expected_value:
                    return False
            elif actual_value != expected_value:
                return False
        
        return True
    
    def _evaluate_success_criteria(self, 
                                 criteria: Dict[str, Any],
                                 results: Dict[str, Any],
                                 context: WorkflowContext) -> bool:
        """Evaluate overall workflow success criteria"""
        
        min_stages = criteria.get("min_stages_completed", 0)
        completed_stages = sum(1 for r in results.values() if r.get("success", False))
        
        if completed_stages < min_stages:
            return False
        
        # Check metric thresholds
        for metric, threshold in criteria.items():
            if metric.startswith("min_"):
                continue
            
            if metric in context.metrics:
                if context.metrics[metric] < threshold:
                    return False
        
        return True
    
    def _calculate_duration(self, context: WorkflowContext) -> float:
        """Calculate workflow execution duration"""
        if "started_at" in context.variables and "completed_at" in context.variables:
            start = datetime.fromisoformat(context.variables["started_at"])
            end = datetime.fromisoformat(context.variables["completed_at"])
            return (end - start).total_seconds()
        return 0.0

# Example usage and testing
if __name__ == "__main__":
    import asyncio
    
    async def test_advanced_workflow():
        # Initialize advanced workflow engine
        engine = AdvancedWorkflowEngine()
        
        # Create red team exercise workflow
        workflow_id = await engine.create_workflow(
            "advanced_red_team_exercise",
            {
                "target": "192.168.1.100",
                "stealth_required": True,
                "max_duration": 3600
            }
        )
        
        print(f"Created workflow: {workflow_id}")
        
        # Execute workflow
        result = await engine.execute_workflow(workflow_id)
        
        print("\nWorkflow Results:")
        print(f"Status: {result['status']}")
        print(f"Duration: {result['duration']}s")
        print(f"Stages Executed: {result['stages_executed']}")
        print(f"Overall Success: {result['overall_success']}")
    
    asyncio.run(test_advanced_workflow())
