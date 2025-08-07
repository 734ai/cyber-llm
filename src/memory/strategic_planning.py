"""
Advanced Strategic Planning Engine for Cyber-LLM
Long-term goal decomposition, execution planning, and adaptive strategy

Author: Muzan Sano <sanosensei36@gmail.com>
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import uuid
import numpy as np
from pathlib import Path

from .persistent_memory import PersistentMemoryManager, MemoryType, ReasoningType
from ..utils.logging_system import CyberLLMLogger, CyberLLMError, ErrorCategory

class StrategicObjective(Enum):
    """Types of strategic objectives"""
    THREAT_HUNTING = "threat_hunting"
    VULNERABILITY_ASSESSMENT = "vulnerability_assessment"
    INCIDENT_RESPONSE = "incident_response"
    DEFENSE_OPTIMIZATION = "defense_optimization"
    ATTACK_SIMULATION = "attack_simulation"
    COMPLIANCE_ASSURANCE = "compliance_assurance"
    KNOWLEDGE_ACQUISITION = "knowledge_acquisition"

class PlanStatus(Enum):
    """Strategic plan execution status"""
    DRAFT = "draft"
    APPROVED = "approved"
    EXECUTING = "executing"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    ADAPTIVE = "adaptive"

class DecisionNode(Enum):
    """Types of decision nodes in strategic plans"""
    CONDITIONAL = "conditional"      # If-then decisions
    PARALLEL = "parallel"           # Execute multiple paths
    SEQUENTIAL = "sequential"       # Step-by-step execution
    CHOICE = "choice"              # Select best option
    LOOP = "loop"                  # Iterative processes
    MERGE = "merge"                # Combine results

@dataclass
class StrategicPhase:
    """Individual phase in strategic plan"""
    phase_id: str
    name: str
    description: str
    
    # Execution details
    estimated_duration: timedelta
    dependencies: List[str] = field(default_factory=list)
    prerequisites: List[str] = field(default_factory=list)
    
    # Resource requirements
    resource_requirements: Dict[str, Any] = field(default_factory=dict)
    agent_assignments: List[str] = field(default_factory=list)
    
    # Success criteria
    success_criteria: List[str] = field(default_factory=list)
    completion_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Execution tracking
    status: PlanStatus = PlanStatus.DRAFT
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    progress: float = 0.0
    
    # Adaptation and learning
    execution_notes: List[str] = field(default_factory=list)
    lessons_learned: List[str] = field(default_factory=list)

@dataclass
class StrategicMilestone:
    """Strategic milestone within a plan"""
    milestone_id: str
    name: str
    description: str
    target_date: datetime
    
    # Dependencies and prerequisites
    dependent_phases: List[str] = field(default_factory=list)
    success_conditions: List[str] = field(default_factory=list)
    
    # Tracking
    achieved: bool = False
    achieved_date: Optional[datetime] = None
    completion_percentage: float = 0.0
    
    # Risk assessment
    risk_factors: List[str] = field(default_factory=list)
    mitigation_strategies: List[str] = field(default_factory=list)

@dataclass
class DecisionPoint:
    """Strategic decision point in plan execution"""
    decision_id: str
    node_type: DecisionNode
    description: str
    
    # Decision logic
    conditions: Dict[str, Any] = field(default_factory=dict)
    options: List[Dict[str, Any]] = field(default_factory=list)
    decision_criteria: List[str] = field(default_factory=list)
    
    # Execution tracking
    decision_time: Optional[datetime] = None
    selected_option: Optional[str] = None
    reasoning: Optional[str] = None
    confidence: float = 0.0

class StrategicPlanningEngine:
    """Advanced strategic planning engine with adaptive capabilities"""
    
    def __init__(self, 
                 memory_manager: PersistentMemoryManager,
                 logger: Optional[CyberLLMLogger] = None):
        
        self.memory_manager = memory_manager
        self.logger = logger or CyberLLMLogger(name="strategic_planning")
        
        # Planning state
        self.active_plans = {}
        self.plan_templates = {}
        self.execution_context = {}
        
        # Decision making
        self.decision_history = {}
        self.strategy_patterns = {}
        
        # Performance tracking
        self.plan_performance_metrics = {}
        
        self.logger.info("Strategic Planning Engine initialized")
    
    async def create_strategic_plan(self, 
                                  objective: StrategicObjective,
                                  target_outcomes: List[str],
                                  constraints: Dict[str, Any],
                                  timeline: timedelta,
                                  priority_level: int = 5) -> str:
        """Create a comprehensive strategic plan"""
        
        plan_id = f"strategic_{objective.value}_{uuid.uuid4().hex[:8]}"
        
        try:
            # Analyze historical patterns for similar objectives
            historical_context = await self._analyze_historical_patterns(objective)
            
            # Generate phases using strategic decomposition
            phases = await self._decompose_strategic_objective(
                objective, target_outcomes, constraints, timeline
            )
            
            # Create milestones and decision points
            milestones = await self._generate_milestones(phases, timeline)
            decision_points = await self._identify_decision_points(phases)
            
            # Risk assessment and mitigation planning
            risk_assessment = await self._assess_strategic_risks(
                objective, phases, constraints
            )
            
            # Resource allocation planning
            resource_plan = await self._plan_resource_allocation(phases, constraints)
            
            # Create the strategic plan
            strategic_plan = {
                "plan_id": plan_id,
                "objective": objective.value,
                "target_outcomes": target_outcomes,
                "constraints": constraints,
                "timeline": timeline.total_seconds(),
                "priority_level": priority_level,
                
                # Plan structure
                "phases": [phase.__dict__ for phase in phases],
                "milestones": [milestone.__dict__ for milestone in milestones],
                "decision_points": [dp.__dict__ for dp in decision_points],
                
                # Analysis and planning
                "historical_context": historical_context,
                "risk_assessment": risk_assessment,
                "resource_plan": resource_plan,
                
                # Execution tracking
                "created_at": datetime.now().isoformat(),
                "status": PlanStatus.DRAFT.value,
                "progress": 0.0,
                "current_phase": 0,
                "execution_log": [],
                
                # Adaptation tracking
                "adaptations": [],
                "performance_metrics": {},
                "lessons_learned": []
            }
            
            # Store plan in memory system
            await self.memory_manager.store_memory(
                memory_type=MemoryType.STRATEGIC,
                content=strategic_plan,
                importance=0.8 + (priority_level / 10),
                context_tags=[objective.value, "strategic_plan", "long_term"],
                agent_id="strategic_planning_engine"
            )
            
            self.active_plans[plan_id] = strategic_plan
            
            # Create reasoning chain for plan execution
            reasoning_chain_id = await self.memory_manager.create_reasoning_chain(
                reasoning_type=ReasoningType.STRATEGIC,
                goal=f"Execute strategic plan for {objective.value}",
                premises=[f"Objective: {obj}" for obj in target_outcomes],
                agent_id="strategic_planning_engine"
            )
            
            strategic_plan["reasoning_chain_id"] = reasoning_chain_id
            
            self.logger.info(f"Created strategic plan: {plan_id}",
                           objective=objective.value,
                           phases=len(phases),
                           timeline_days=timeline.days)
            
            return plan_id
            
        except Exception as e:
            self.logger.error(f"Failed to create strategic plan", error=str(e))
            raise CyberLLMError("Strategic plan creation failed", ErrorCategory.PLANNING)
    
    async def execute_strategic_plan(self, plan_id: str) -> bool:
        """Execute a strategic plan with adaptive monitoring"""
        
        if plan_id not in self.active_plans:
            raise CyberLLMError(f"Strategic plan not found: {plan_id}", ErrorCategory.VALIDATION)
        
        plan = self.active_plans[plan_id]
        
        try:
            plan["status"] = PlanStatus.EXECUTING.value
            plan["execution_started_at"] = datetime.now().isoformat()
            
            # Execute phases sequentially with adaptive monitoring
            for phase_index, phase_data in enumerate(plan["phases"]):
                phase = StrategicPhase(**phase_data)
                
                # Pre-phase analysis and adaptation
                adaptation_needed = await self._assess_adaptation_need(plan, phase)
                if adaptation_needed:
                    await self._adapt_strategic_plan(plan_id, phase.phase_id)
                
                # Execute phase
                success = await self._execute_strategic_phase(plan_id, phase)
                
                if not success:
                    plan["status"] = PlanStatus.FAILED.value
                    return False
                
                # Update plan progress
                plan["current_phase"] = phase_index + 1
                plan["progress"] = (phase_index + 1) / len(plan["phases"])
                
                # Check milestones
                await self._check_milestone_completion(plan_id)
            
            # Plan completion
            plan["status"] = PlanStatus.COMPLETED.value
            plan["execution_completed_at"] = datetime.now().isoformat()
            
            # Generate final performance report
            await self._generate_plan_performance_report(plan_id)
            
            self.logger.info(f"Strategic plan completed successfully: {plan_id}")
            return True
            
        except Exception as e:
            plan["status"] = PlanStatus.FAILED.value
            plan["failure_reason"] = str(e)
            self.logger.error(f"Strategic plan execution failed: {plan_id}", error=str(e))
            return False
    
    async def _decompose_strategic_objective(self, 
                                          objective: StrategicObjective,
                                          outcomes: List[str],
                                          constraints: Dict[str, Any],
                                          timeline: timedelta) -> List[StrategicPhase]:
        """Decompose strategic objective into executable phases"""
        
        phases = []
        
        # Objective-specific decomposition
        if objective == StrategicObjective.THREAT_HUNTING:
            phases = await self._decompose_threat_hunting(outcomes, constraints, timeline)
        elif objective == StrategicObjective.VULNERABILITY_ASSESSMENT:
            phases = await self._decompose_vulnerability_assessment(outcomes, constraints, timeline)
        elif objective == StrategicObjective.INCIDENT_RESPONSE:
            phases = await self._decompose_incident_response(outcomes, constraints, timeline)
        elif objective == StrategicObjective.DEFENSE_OPTIMIZATION:
            phases = await self._decompose_defense_optimization(outcomes, constraints, timeline)
        elif objective == StrategicObjective.ATTACK_SIMULATION:
            phases = await self._decompose_attack_simulation(outcomes, constraints, timeline)
        else:
            phases = await self._decompose_generic_objective(outcomes, constraints, timeline)
        
        return phases
    
    async def _decompose_threat_hunting(self, 
                                      outcomes: List[str],
                                      constraints: Dict[str, Any],
                                      timeline: timedelta) -> List[StrategicPhase]:
        """Decompose threat hunting objective into phases"""
        
        phase_duration = timeline / 4  # Divide into 4 main phases
        
        phases = [
            StrategicPhase(
                phase_id="threat_intel_gathering",
                name="Threat Intelligence Gathering",
                description="Collect and analyze current threat intelligence",
                estimated_duration=phase_duration * 0.3,
                resource_requirements={"cpu": 2, "memory": "4GB", "storage": "10GB"},
                agent_assignments=["recon_agent", "intelligence_agent"],
                success_criteria=[
                    "Threat intelligence database populated",
                    "IOCs identified and categorized",
                    "Threat landscape analysis completed"
                ]
            ),
            
            StrategicPhase(
                phase_id="hunting_hypothesis_formation",
                name="Hunting Hypothesis Formation",
                description="Develop testable hypotheses about potential threats",
                estimated_duration=phase_duration * 0.2,
                dependencies=["threat_intel_gathering"],
                resource_requirements={"cpu": 1, "memory": "2GB"},
                agent_assignments=["analysis_agent"],
                success_criteria=[
                    "Hunting hypotheses documented",
                    "Detection logic defined",
                    "Search queries prepared"
                ]
            ),
            
            StrategicPhase(
                phase_id="active_hunting_execution",
                name="Active Hunting Execution",
                description="Execute threat hunting operations",
                estimated_duration=phase_duration * 0.4,
                dependencies=["hunting_hypothesis_formation"],
                resource_requirements={"cpu": 4, "memory": "8GB", "storage": "50GB"},
                agent_assignments=["hunting_agent", "analysis_agent"],
                success_criteria=[
                    "All hunting queries executed",
                    "Potential threats investigated",
                    "Evidence collected and documented"
                ]
            ),
            
            StrategicPhase(
                phase_id="results_analysis_reporting",
                name="Results Analysis and Reporting",
                description="Analyze findings and generate comprehensive report",
                estimated_duration=phase_duration * 0.1,
                dependencies=["active_hunting_execution"],
                resource_requirements={"cpu": 1, "memory": "2GB"},
                agent_assignments=["reporting_agent"],
                success_criteria=[
                    "Threat hunting report generated",
                    "Recommendations documented",
                    "Follow-up actions identified"
                ]
            )
        ]
        
        return phases
    
    async def _decompose_vulnerability_assessment(self, 
                                                outcomes: List[str],
                                                constraints: Dict[str, Any],
                                                timeline: timedelta) -> List[StrategicPhase]:
        """Decompose vulnerability assessment into phases"""
        
        phase_duration = timeline / 5
        
        phases = [
            StrategicPhase(
                phase_id="asset_discovery",
                name="Asset Discovery and Inventory",
                description="Discover and catalog all assets in scope",
                estimated_duration=phase_duration,
                resource_requirements={"cpu": 2, "memory": "4GB"},
                agent_assignments=["recon_agent"],
                success_criteria=[
                    "Asset inventory completed",
                    "Network topology mapped",
                    "Service enumeration finished"
                ]
            ),
            
            StrategicPhase(
                phase_id="vulnerability_scanning",
                name="Automated Vulnerability Scanning",
                description="Execute comprehensive vulnerability scans",
                estimated_duration=phase_duration * 2,
                dependencies=["asset_discovery"],
                resource_requirements={"cpu": 4, "memory": "8GB"},
                agent_assignments=["scanning_agent"],
                success_criteria=[
                    "All assets scanned",
                    "Vulnerabilities identified",
                    "False positives filtered"
                ]
            ),
            
            StrategicPhase(
                phase_id="manual_validation",
                name="Manual Validation and Testing",
                description="Manually validate critical vulnerabilities",
                estimated_duration=phase_duration,
                dependencies=["vulnerability_scanning"],
                resource_requirements={"cpu": 2, "memory": "4GB"},
                agent_assignments=["validation_agent"],
                success_criteria=[
                    "Critical vulnerabilities validated",
                    "Exploitability confirmed",
                    "Impact assessment completed"
                ]
            ),
            
            StrategicPhase(
                phase_id="risk_analysis",
                name="Risk Analysis and Prioritization",
                description="Analyze and prioritize identified risks",
                estimated_duration=phase_duration * 0.5,
                dependencies=["manual_validation"],
                resource_requirements={"cpu": 1, "memory": "2GB"},
                agent_assignments=["analysis_agent"],
                success_criteria=[
                    "Risk scores calculated",
                    "Vulnerabilities prioritized",
                    "Remediation timeline proposed"
                ]
            ),
            
            StrategicPhase(
                phase_id="reporting_recommendations",
                name="Reporting and Recommendations",
                description="Generate comprehensive assessment report",
                estimated_duration=phase_duration * 0.5,
                dependencies=["risk_analysis"],
                resource_requirements={"cpu": 1, "memory": "2GB"},
                agent_assignments=["reporting_agent"],
                success_criteria=[
                    "Assessment report completed",
                    "Executive summary prepared",
                    "Remediation plan documented"
                ]
            )
        ]
        
        return phases
    
    async def _generate_milestones(self, 
                                 phases: List[StrategicPhase],
                                 timeline: timedelta) -> List[StrategicMilestone]:
        """Generate strategic milestones based on phases"""
        
        milestones = []
        cumulative_duration = timedelta()
        start_date = datetime.now()
        
        for i, phase in enumerate(phases):
            cumulative_duration += phase.estimated_duration
            
            milestone = StrategicMilestone(
                milestone_id=f"milestone_{i+1}",
                name=f"Phase {i+1} Completion: {phase.name}",
                description=f"Successful completion of {phase.name} phase",
                target_date=start_date + cumulative_duration,
                dependent_phases=[phase.phase_id],
                success_conditions=phase.success_criteria,
                risk_factors=[
                    "Resource availability",
                    "Technical complexity",
                    "External dependencies"
                ],
                mitigation_strategies=[
                    "Regular progress monitoring",
                    "Adaptive resource allocation",
                    "Early risk identification"
                ]
            )
            
            milestones.append(milestone)
        
        return milestones
    
    async def _identify_decision_points(self, 
                                      phases: List[StrategicPhase]) -> List[DecisionPoint]:
        """Identify key decision points in the strategic plan"""
        
        decision_points = []
        
        for i, phase in enumerate(phases):
            # Phase transition decision point
            decision_point = DecisionPoint(
                decision_id=f"phase_transition_{i}",
                node_type=DecisionNode.CONDITIONAL,
                description=f"Decision to proceed from {phase.name} to next phase",
                conditions={
                    "success_criteria_met": phase.success_criteria,
                    "resource_availability": True,
                    "timeline_adherence": True
                },
                options=[
                    {"action": "proceed", "description": "Continue to next phase"},
                    {"action": "adapt", "description": "Adapt plan before proceeding"},
                    {"action": "pause", "description": "Pause execution for review"},
                    {"action": "abort", "description": "Abort plan execution"}
                ],
                decision_criteria=[
                    "Phase completion status",
                    "Resource constraints",
                    "Timeline adherence",
                    "Risk level assessment"
                ]
            )
            
            decision_points.append(decision_point)
        
        return decision_points
    
    async def _execute_strategic_phase(self, plan_id: str, phase: StrategicPhase) -> bool:
        """Execute a single strategic phase"""
        
        try:
            phase.status = PlanStatus.EXECUTING
            phase.start_time = datetime.now()
            
            # Create reasoning chain for phase execution
            reasoning_chain_id = await self.memory_manager.create_reasoning_chain(
                reasoning_type=ReasoningType.STRATEGIC,
                goal=f"Execute phase: {phase.name}",
                premises=phase.success_criteria,
                agent_id="strategic_planning_engine"
            )
            
            # Execute phase logic based on phase type
            success = await self._execute_phase_logic(plan_id, phase)
            
            # Update phase completion
            phase.end_time = datetime.now()
            phase.status = PlanStatus.COMPLETED if success else PlanStatus.FAILED
            phase.progress = 1.0 if success else 0.0
            
            # Store execution results in memory
            execution_result = {
                "phase_id": phase.phase_id,
                "success": success,
                "execution_time": (phase.end_time - phase.start_time).total_seconds(),
                "lessons_learned": phase.lessons_learned
            }
            
            await self.memory_manager.store_memory(
                memory_type=MemoryType.EPISODIC,
                content=execution_result,
                importance=0.7,
                context_tags=["phase_execution", phase.phase_id, plan_id],
                agent_id="strategic_planning_engine"
            )
            
            return success
            
        except Exception as e:
            phase.status = PlanStatus.FAILED
            phase.execution_notes.append(f"Execution failed: {str(e)}")
            self.logger.error(f"Phase execution failed: {phase.phase_id}", error=str(e))
            return False
    
    async def _execute_phase_logic(self, plan_id: str, phase: StrategicPhase) -> bool:
        """Execute the core logic for a strategic phase"""
        
        # Simulate phase execution (in production, would delegate to appropriate agents)
        execution_steps = []
        
        for criterion in phase.success_criteria:
            # Simulate work on each success criterion
            step_result = {
                "criterion": criterion,
                "started_at": datetime.now().isoformat(),
                "success": True,  # Simulate successful execution
                "notes": f"Successfully completed: {criterion}"
            }
            execution_steps.append(step_result)
            
            # Add some realistic delay
            await asyncio.sleep(0.1)
        
        phase.execution_notes.extend([step["notes"] for step in execution_steps])
        
        # All steps succeeded
        return all(step["success"] for step in execution_steps)
    
    def get_plan_status(self, plan_id: str) -> Dict[str, Any]:
        """Get current status of a strategic plan"""
        
        if plan_id not in self.active_plans:
            return {"error": "Plan not found"}
        
        plan = self.active_plans[plan_id]
        
        return {
            "plan_id": plan_id,
            "objective": plan["objective"],
            "status": plan["status"],
            "progress": plan["progress"],
            "current_phase": plan["current_phase"],
            "total_phases": len(plan["phases"]),
            "created_at": plan["created_at"],
            "execution_time": self._calculate_execution_time(plan),
            "milestones_achieved": self._count_achieved_milestones(plan),
            "total_milestones": len(plan["milestones"])
        }
    
    def _calculate_execution_time(self, plan: Dict[str, Any]) -> float:
        """Calculate total execution time for plan"""
        
        if "execution_started_at" not in plan:
            return 0.0
        
        start_time = datetime.fromisoformat(plan["execution_started_at"])
        
        if plan["status"] == PlanStatus.COMPLETED.value and "execution_completed_at" in plan:
            end_time = datetime.fromisoformat(plan["execution_completed_at"])
        else:
            end_time = datetime.now()
        
        return (end_time - start_time).total_seconds()
    
    def _count_achieved_milestones(self, plan: Dict[str, Any]) -> int:
        """Count achieved milestones in plan"""
        
        return sum(1 for milestone in plan["milestones"] if milestone.get("achieved", False))
    
    async def _assess_adaptation_need(self, plan: Dict[str, Any], phase: StrategicPhase) -> bool:
        """Assess if strategic plan needs adaptation"""
        
        # Check for adaptation triggers
        triggers = [
            self._check_timeline_deviation(plan),
            self._check_resource_constraints(plan, phase),
            self._check_external_changes(plan),
            self._check_performance_degradation(plan)
        ]
        
        return any(await asyncio.gather(*triggers))
    
    async def _check_timeline_deviation(self, plan: Dict[str, Any]) -> bool:
        """Check if plan is deviating from timeline"""
        
        # Simple timeline check (would be more sophisticated in production)
        expected_progress = min(1.0, self._calculate_execution_time(plan) / plan["timeline"])
        actual_progress = plan["progress"]
        
        return abs(expected_progress - actual_progress) > 0.2  # 20% deviation threshold
    
    async def _check_resource_constraints(self, plan: Dict[str, Any], phase: StrategicPhase) -> bool:
        """Check if resource constraints require adaptation"""
        
        # Simulate resource constraint checking
        return False  # No constraints for simulation
    
    async def _check_external_changes(self, plan: Dict[str, Any]) -> bool:
        """Check for external changes that might affect plan"""
        
        # Simulate external change detection
        return False  # No external changes for simulation
    
    async def _check_performance_degradation(self, plan: Dict[str, Any]) -> bool:
        """Check for performance degradation"""
        
        # Simulate performance checking
        return False  # No performance issues for simulation

# Factory function
def create_strategic_planning_engine(memory_manager: PersistentMemoryManager, **kwargs) -> StrategicPlanningEngine:
    """Create strategic planning engine with memory manager"""
    return StrategicPlanningEngine(memory_manager, **kwargs)
