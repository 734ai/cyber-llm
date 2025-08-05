"""
Advanced Multi-Agent Scenario Orchestration for Cyber-LLM
Handles complex red team exercises and coordinated agent operations
"""

import asyncio
import json
import yaml
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
import networkx as nx
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..agents.orchestrator_agent import OrchestratorAgent
from ..agents.recon_agent import ReconnaissanceAgent
from ..agents.c2_agent import CommandControlAgent
from ..agents.post_exploit_agent import PostExploitAgent
from ..agents.safety_agent import SafetyAgent
from ..agents.explainability_agent import ExplainabilityAgent
from ..utils.logging_system import CyberLLMLogger, CyberLLMError, ErrorCategory, retry_with_backoff

class ScenarioType(Enum):
    """Types of security scenarios"""
    RED_TEAM_EXERCISE = "red_team_exercise"
    PENETRATION_TEST = "penetration_test"
    THREAT_HUNTING = "threat_hunting"
    INCIDENT_RESPONSE = "incident_response"
    VULNERABILITY_ASSESSMENT = "vulnerability_assessment"
    SECURITY_AUDIT = "security_audit"

class AgentRole(Enum):
    """Agent roles in scenarios"""
    LEADER = "leader"
    SPECIALIST = "specialist"
    SUPPORT = "support"
    OBSERVER = "observer"

@dataclass
class ScenarioStep:
    """Individual step in a security scenario"""
    id: str
    name: str
    description: str
    agent_type: str
    dependencies: List[str]
    parameters: Dict[str, Any]
    timeout: int = 300  # 5 minutes default
    retry_count: int = 3
    critical: bool = False
    parallel_group: Optional[str] = None

@dataclass
class ScenarioResult:
    """Result from executing a scenario step"""
    step_id: str
    success: bool
    output: Any
    error: Optional[str]
    execution_time: float
    timestamp: datetime

@dataclass
class RedTeamScenario:
    """Complete red team exercise scenario"""
    id: str
    name: str
    description: str
    scenario_type: ScenarioType
    target_environment: Dict[str, Any]
    steps: List[ScenarioStep]
    success_criteria: Dict[str, Any]
    safety_constraints: List[str]
    estimated_duration: int  # minutes
    difficulty_level: str  # beginner, intermediate, advanced, expert

class MultiAgentOrchestrator:
    """Advanced orchestrator for complex multi-agent scenarios"""
    
    def __init__(self, 
                 logger: Optional[CyberLLMLogger] = None,
                 max_concurrent_agents: int = 5):
        
        self.logger = logger or CyberLLMLogger(name="multi_agent_orchestrator")
        self.max_concurrent_agents = max_concurrent_agents
        
        # Initialize agents
        self.agents = {
            'orchestrator': OrchestratorAgent(logger=self.logger),
            'recon': ReconnaissanceAgent(logger=self.logger),
            'c2': CommandControlAgent(logger=self.logger),
            'post_exploit': PostExploitAgent(logger=self.logger),
            'safety': SafetyAgent(logger=self.logger),
            'explainability': ExplainabilityAgent(logger=self.logger)
        }
        
        # Execution state
        self.active_scenarios = {}
        self.scenario_results = {}
        self.agent_status = {name: "idle" for name in self.agents.keys()}
        
        # Scenario templates
        self.scenario_templates = self._load_scenario_templates()
    
    def _load_scenario_templates(self) -> Dict[str, RedTeamScenario]:
        """Load predefined scenario templates"""
        templates = {}
        
        # Advanced Persistent Threat (APT) Simulation
        apt_scenario = RedTeamScenario(
            id="apt_simulation_001",
            name="Advanced Persistent Threat Simulation",
            description="Multi-stage APT attack simulation with stealth focus",
            scenario_type=ScenarioType.RED_TEAM_EXERCISE,
            target_environment={
                "network_range": "10.0.0.0/24",
                "domain": "target.local",
                "critical_assets": ["domain_controller", "file_server", "database"]
            },
            steps=[
                ScenarioStep(
                    id="recon_phase",
                    name="Reconnaissance",
                    description="Gather intelligence on target environment",
                    agent_type="recon",
                    dependencies=[],
                    parameters={
                        "target": "target.local",
                        "scope": "external",
                        "stealth_level": "high"
                    },
                    timeout=600,
                    critical=True
                ),
                ScenarioStep(
                    id="initial_access",
                    name="Initial Access",
                    description="Gain initial foothold in target network",
                    agent_type="c2",
                    dependencies=["recon_phase"],
                    parameters={
                        "attack_vector": "spear_phishing",
                        "payload_type": "custom_backdoor"
                    },
                    timeout=900,
                    critical=True
                ),
                ScenarioStep(
                    id="privilege_escalation",
                    name="Privilege Escalation",
                    description="Escalate privileges on compromised system",
                    agent_type="post_exploit",
                    dependencies=["initial_access"],
                    parameters={
                        "technique": "token_manipulation",
                        "target_privilege": "SYSTEM"
                    },
                    timeout=300
                ),
                ScenarioStep(
                    id="lateral_movement",
                    name="Lateral Movement",
                    description="Move laterally through the network",
                    agent_type="post_exploit",
                    dependencies=["privilege_escalation"],
                    parameters={
                        "technique": "pass_the_hash",
                        "targets": ["file_server", "database"]
                    },
                    timeout=600,
                    parallel_group="expansion"
                ),
                ScenarioStep(
                    id="data_exfiltration",
                    name="Data Exfiltration",
                    description="Exfiltrate sensitive data",
                    agent_type="post_exploit",
                    dependencies=["lateral_movement"],
                    parameters={
                        "data_types": ["financial", "pii", "intellectual_property"],
                        "exfil_method": "dns_tunneling"
                    },
                    timeout=450,
                    parallel_group="expansion"
                ),
                ScenarioStep(
                    id="persistence",
                    name="Establish Persistence",
                    description="Maintain long-term access",
                    agent_type="c2",
                    dependencies=["privilege_escalation"],
                    parameters={
                        "persistence_type": "service_persistence",
                        "stealth_level": "high"
                    },
                    timeout=300
                )
            ],
            success_criteria={
                "min_steps_completed": 4,
                "critical_steps_required": ["recon_phase", "initial_access"],
                "stealth_score_threshold": 0.7,
                "detection_threshold": 0.3
            },
            safety_constraints=[
                "no_actual_data_modification",
                "no_service_disruption",
                "maintain_system_stability",
                "document_all_actions"
            ],
            estimated_duration=180,  # 3 hours
            difficulty_level="advanced"
        )
        templates["apt_simulation"] = apt_scenario
        
        # Vulnerability Assessment Scenario
        vuln_assessment = RedTeamScenario(
            id="vuln_assessment_001",
            name="Comprehensive Vulnerability Assessment",
            description="Multi-vector vulnerability discovery and assessment",
            scenario_type=ScenarioType.VULNERABILITY_ASSESSMENT,
            target_environment={
                "targets": ["web_app", "network_services", "host_systems"],
                "assessment_scope": "comprehensive"
            },
            steps=[
                ScenarioStep(
                    id="network_discovery",
                    name="Network Discovery",
                    description="Discover network topology and services",
                    agent_type="recon",
                    dependencies=[],
                    parameters={
                        "scan_type": "comprehensive",
                        "port_range": "1-65535"
                    },
                    timeout=1800,
                    parallel_group="discovery"
                ),
                ScenarioStep(
                    id="service_enumeration",
                    name="Service Enumeration",
                    description="Enumerate discovered services",
                    agent_type="recon",
                    dependencies=[],
                    parameters={
                        "service_types": ["web", "database", "file_sharing"],
                        "deep_scan": True
                    },
                    timeout=1200,
                    parallel_group="discovery"
                ),
                ScenarioStep(
                    id="vulnerability_scanning",
                    name="Vulnerability Scanning",
                    description="Scan for known vulnerabilities",
                    agent_type="recon",
                    dependencies=["network_discovery", "service_enumeration"],
                    parameters={
                        "scanner_types": ["nessus", "openvas", "custom"],
                        "authenticated": False
                    },
                    timeout=2400,
                    critical=True
                ),
                ScenarioStep(
                    id="web_app_testing",
                    name="Web Application Testing",
                    description="Test web applications for vulnerabilities",
                    agent_type="recon",
                    dependencies=["service_enumeration"],
                    parameters={
                        "test_types": ["owasp_top10", "custom_checks"],
                        "authentication_bypass": True
                    },
                    timeout=1800,
                    parallel_group="testing"
                ),
                ScenarioStep(
                    id="exploit_validation",
                    name="Exploit Validation",
                    description="Validate critical vulnerabilities",
                    agent_type="c2",
                    dependencies=["vulnerability_scanning"],
                    parameters={
                        "exploit_types": ["proof_of_concept"],
                        "severity_threshold": "high"
                    },
                    timeout=900
                )
            ],
            success_criteria={
                "vulnerability_discovery_rate": 0.8,
                "false_positive_rate": 0.1,
                "coverage_percentage": 0.9
            },
            safety_constraints=[
                "read_only_operations",
                "no_system_modification",
                "minimal_service_impact"
            ],
            estimated_duration=360,  # 6 hours
            difficulty_level="intermediate"
        )
        templates["vuln_assessment"] = vuln_assessment
        
        return templates
    
    async def execute_scenario(self, 
                             scenario: RedTeamScenario,
                             context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute a complete multi-agent scenario"""
        
        scenario_id = f"{scenario.id}_{uuid.uuid4().hex[:8]}"
        start_time = datetime.now()
        
        self.logger.info(f"Starting scenario execution",
                        scenario_id=scenario_id,
                        scenario_name=scenario.name,
                        scenario_type=scenario.scenario_type.value,
                        estimated_duration=scenario.estimated_duration)
        
        # Initialize scenario state
        self.active_scenarios[scenario_id] = {
            'scenario': scenario,
            'context': context or {},
            'start_time': start_time,
            'status': 'running',
            'completed_steps': [],
            'failed_steps': [],
            'step_results': {}
        }
        
        try:
            # Safety check
            safety_approval = await self._safety_check(scenario)
            if not safety_approval['approved']:
                raise CyberLLMError(
                    f"Scenario failed safety check: {safety_approval['reason']}",
                    ErrorCategory.SAFETY
                )
            
            # Build execution graph
            execution_graph = self._build_execution_graph(scenario.steps)
            
            # Execute scenario steps
            results = await self._execute_scenario_graph(
                scenario_id, 
                execution_graph,
                scenario.success_criteria
            )
            
            # Evaluate results
            evaluation = self._evaluate_scenario_results(scenario, results)
            
            # Generate report
            report = await self._generate_scenario_report(
                scenario_id, 
                scenario, 
                results, 
                evaluation
            )
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            self.active_scenarios[scenario_id]['status'] = 'completed'
            self.scenario_results[scenario_id] = {
                'scenario': scenario,
                'results': results,
                'evaluation': evaluation,
                'report': report,
                'execution_time': execution_time,
                'completed_at': datetime.now()
            }
            
            self.logger.info(f"Scenario execution completed",
                           scenario_id=scenario_id,
                           success=evaluation['overall_success'],
                           execution_time=execution_time,
                           steps_completed=len(results['completed']),
                           steps_failed=len(results['failed']))
            
            return self.scenario_results[scenario_id]
            
        except Exception as e:
            self.active_scenarios[scenario_id]['status'] = 'failed'
            self.logger.error(f"Scenario execution failed",
                            scenario_id=scenario_id,
                            error=str(e))
            raise
        
        finally:
            # Cleanup
            if scenario_id in self.active_scenarios:
                del self.active_scenarios[scenario_id]
    
    async def _safety_check(self, scenario: RedTeamScenario) -> Dict[str, Any]:
        """Perform safety check on scenario"""
        
        safety_agent = self.agents['safety']
        
        # Check scenario against safety constraints
        check_request = {
            'scenario_type': scenario.scenario_type.value,
            'target_environment': scenario.target_environment,
            'steps': [asdict(step) for step in scenario.steps],
            'safety_constraints': scenario.safety_constraints
        }
        
        try:
            safety_result = await safety_agent.evaluate_scenario_safety(check_request)
            
            return {
                'approved': safety_result.get('approved', False),
                'reason': safety_result.get('reason', ''),
                'risk_level': safety_result.get('risk_level', 'unknown'),
                'recommendations': safety_result.get('recommendations', [])
            }
            
        except Exception as e:
            self.logger.error("Safety check failed", error=str(e))
            return {
                'approved': False,
                'reason': f"Safety check error: {str(e)}",
                'risk_level': 'critical'
            }
    
    def _build_execution_graph(self, steps: List[ScenarioStep]) -> nx.DiGraph:
        """Build directed graph for scenario execution"""
        
        graph = nx.DiGraph()
        
        # Add nodes
        for step in steps:
            graph.add_node(step.id, step=step)
        
        # Add dependency edges
        for step in steps:
            for dependency in step.dependencies:
                if dependency in [s.id for s in steps]:
                    graph.add_edge(dependency, step.id)
        
        # Verify graph is acyclic
        if not nx.is_directed_acyclic_graph(graph):
            raise CyberLLMError(
                "Scenario contains circular dependencies",
                ErrorCategory.VALIDATION
            )
        
        return graph
    
    async def _execute_scenario_graph(self, 
                                    scenario_id: str,
                                    graph: nx.DiGraph,
                                    success_criteria: Dict[str, Any]) -> Dict[str, Any]:
        """Execute scenario steps according to dependency graph"""
        
        completed_steps = set()
        failed_steps = set()
        step_results = {}
        parallel_groups = {}
        
        # Group steps by parallel execution groups
        for node_id in graph.nodes():
            step = graph.nodes[node_id]['step']
            if step.parallel_group:
                if step.parallel_group not in parallel_groups:
                    parallel_groups[step.parallel_group] = []
                parallel_groups[step.parallel_group].append(step)
        
        # Execute steps in topological order
        execution_order = list(nx.topological_sort(graph))
        
        for step_id in execution_order:
            step = graph.nodes[step_id]['step']
            
            # Check if dependencies are satisfied
            dependencies_met = all(
                dep in completed_steps for dep in step.dependencies
            )
            
            if not dependencies_met:
                failed_steps.add(step_id)
                self.logger.warning(f"Step dependencies not met: {step_id}")
                continue
            
            # Execute step
            try:
                if step.parallel_group and step.parallel_group in parallel_groups:
                    # Execute parallel group
                    group_steps = parallel_groups[step.parallel_group]
                    group_results = await self._execute_parallel_steps(group_steps)
                    
                    for group_step, result in group_results.items():
                        step_results[group_step.id] = result
                        if result.success:
                            completed_steps.add(group_step.id)
                        else:
                            failed_steps.add(group_step.id)
                    
                    # Remove processed group
                    del parallel_groups[step.parallel_group]
                
                else:
                    # Execute single step
                    result = await self._execute_single_step(step)
                    step_results[step_id] = result
                    
                    if result.success:
                        completed_steps.add(step_id)
                    else:
                        failed_steps.add(step_id)
                        
                        # Check if critical step failed
                        if step.critical:
                            self.logger.error(f"Critical step failed: {step_id}")
                            break
                
            except Exception as e:
                self.logger.error(f"Step execution error: {step_id}", error=str(e))
                failed_steps.add(step_id)
                
                if step.critical:
                    break
        
        return {
            'completed': completed_steps,
            'failed': failed_steps,
            'results': step_results,
            'success_rate': len(completed_steps) / len(graph.nodes()) if graph.nodes() else 0
        }
    
    async def _execute_parallel_steps(self, 
                                    steps: List[ScenarioStep]) -> Dict[ScenarioStep, ScenarioResult]:
        """Execute multiple steps in parallel"""
        
        tasks = []
        for step in steps:
            task = asyncio.create_task(self._execute_single_step(step))
            tasks.append((step, task))
        
        results = {}
        completed_tasks = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)
        
        for (step, _), result in zip(tasks, completed_tasks):
            if isinstance(result, Exception):
                results[step] = ScenarioResult(
                    step_id=step.id,
                    success=False,
                    output=None,
                    error=str(result),
                    execution_time=0,
                    timestamp=datetime.now()
                )
            else:
                results[step] = result
        
        return results
    
    @retry_with_backoff(max_retries=3)
    async def _execute_single_step(self, step: ScenarioStep) -> ScenarioResult:
        """Execute a single scenario step"""
        
        start_time = datetime.now()
        
        self.logger.info(f"Executing step: {step.name}",
                        step_id=step.id,
                        agent_type=step.agent_type)
        
        try:
            # Get appropriate agent
            agent = self.agents.get(step.agent_type)
            if not agent:
                raise CyberLLMError(
                    f"Unknown agent type: {step.agent_type}",
                    ErrorCategory.VALIDATION
                )
            
            # Update agent status
            self.agent_status[step.agent_type] = "busy"
            
            # Execute step with timeout
            result = await asyncio.wait_for(
                agent.execute_task(step.parameters),
                timeout=step.timeout
            )
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            step_result = ScenarioResult(
                step_id=step.id,
                success=True,
                output=result,
                error=None,
                execution_time=execution_time,
                timestamp=datetime.now()
            )
            
            self.logger.info(f"Step completed successfully: {step.name}",
                           step_id=step.id,
                           execution_time=execution_time)
            
            return step_result
            
        except asyncio.TimeoutError:
            error_msg = f"Step timed out after {step.timeout} seconds"
            self.logger.error(f"Step timeout: {step.name}", step_id=step.id)
            
            return ScenarioResult(
                step_id=step.id,
                success=False,
                output=None,
                error=error_msg,
                execution_time=step.timeout,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            error_msg = str(e)
            
            self.logger.error(f"Step execution failed: {step.name}",
                            step_id=step.id,
                            error=error_msg)
            
            return ScenarioResult(
                step_id=step.id,
                success=False,
                output=None,
                error=error_msg,
                execution_time=execution_time,
                timestamp=datetime.now()
            )
        
        finally:
            # Reset agent status
            self.agent_status[step.agent_type] = "idle"
    
    def _evaluate_scenario_results(self, 
                                 scenario: RedTeamScenario,
                                 results: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate scenario results against success criteria"""
        
        success_criteria = scenario.success_criteria
        evaluation = {
            'overall_success': False,
            'criteria_met': {},
            'score': 0.0,
            'recommendations': []
        }
        
        # Check minimum steps completed
        min_steps = success_criteria.get('min_steps_completed', 0)
        steps_completed = len(results['completed'])
        evaluation['criteria_met']['min_steps_completed'] = steps_completed >= min_steps
        
        # Check critical steps
        critical_steps = success_criteria.get('critical_steps_required', [])
        critical_met = all(step in results['completed'] for step in critical_steps)
        evaluation['criteria_met']['critical_steps_completed'] = critical_met
        
        # Calculate success score
        total_steps = len(results['completed']) + len(results['failed'])
        if total_steps > 0:
            success_rate = len(results['completed']) / total_steps
            evaluation['score'] = success_rate
        
        # Overall success determination
        evaluation['overall_success'] = (
            evaluation['criteria_met'].get('min_steps_completed', False) and
            evaluation['criteria_met'].get('critical_steps_completed', False) and
            evaluation['score'] >= 0.7  # 70% success threshold
        )
        
        # Generate recommendations
        if not evaluation['overall_success']:
            if not critical_met:
                evaluation['recommendations'].append("Complete all critical steps")
            if evaluation['score'] < 0.7:
                evaluation['recommendations'].append("Improve step success rate")
        
        return evaluation
    
    async def _generate_scenario_report(self,
                                      scenario_id: str,
                                      scenario: RedTeamScenario,
                                      results: Dict[str, Any],
                                      evaluation: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive scenario execution report"""
        
        # Get explainability analysis
        explainability_agent = self.agents['explainability']
        
        analysis_request = {
            'scenario': asdict(scenario),
            'results': results,
            'evaluation': evaluation
        }
        
        try:
            explanation = await explainability_agent.analyze_scenario_execution(analysis_request)
        except Exception as e:
            self.logger.warning("Failed to generate explanation", error=str(e))
            explanation = {"analysis": "Analysis unavailable", "insights": []}
        
        report = {
            'scenario_id': scenario_id,
            'scenario_summary': {
                'name': scenario.name,
                'type': scenario.scenario_type.value,
                'difficulty': scenario.difficulty_level,
                'estimated_duration': scenario.estimated_duration
            },
            'execution_summary': {
                'total_steps': len(scenario.steps),
                'completed_steps': len(results['completed']),
                'failed_steps': len(results['failed']),
                'success_rate': results['success_rate'],
                'overall_success': evaluation['overall_success']
            },
            'detailed_results': results['results'],
            'evaluation': evaluation,
            'explanation': explanation,
            'generated_at': datetime.now().isoformat()
        }
        
        return report
    
    def get_scenario_template(self, template_name: str) -> Optional[RedTeamScenario]:
        """Get a scenario template by name"""
        return self.scenario_templates.get(template_name)
    
    def list_scenario_templates(self) -> List[str]:
        """List available scenario templates"""
        return list(self.scenario_templates.keys())
    
    def get_active_scenarios(self) -> Dict[str, Any]:
        """Get currently active scenarios"""
        return self.active_scenarios.copy()
    
    def get_agent_status(self) -> Dict[str, str]:
        """Get current status of all agents"""
        return self.agent_status.copy()

# Convenience functions
async def execute_red_team_scenario(scenario_name: str = "apt_simulation") -> Dict[str, Any]:
    """Execute a predefined red team scenario"""
    orchestrator = MultiAgentOrchestrator()
    template = orchestrator.get_scenario_template(scenario_name)
    
    if not template:
        raise ValueError(f"Unknown scenario template: {scenario_name}")
    
    return await orchestrator.execute_scenario(template)

async def execute_vulnerability_assessment(targets: List[str]) -> Dict[str, Any]:
    """Execute vulnerability assessment scenario"""
    orchestrator = MultiAgentOrchestrator()
    template = orchestrator.get_scenario_template("vuln_assessment")
    
    if not template:
        raise ValueError("Vulnerability assessment template not found")
    
    # Customize template with specific targets
    context = {"custom_targets": targets}
    
    return await orchestrator.execute_scenario(template, context)

# Example usage
if __name__ == "__main__":
    async def main():
        # Initialize orchestrator
        orchestrator = MultiAgentOrchestrator()
        
        # List available templates
        templates = orchestrator.list_scenario_templates()
        print(f"Available scenario templates: {templates}")
        
        # Execute APT simulation
        result = await execute_red_team_scenario("apt_simulation")
        
        print(f"Scenario completed: {result['evaluation']['overall_success']}")
        print(f"Success rate: {result['evaluation']['score']:.2%}")
    
    asyncio.run(main())
