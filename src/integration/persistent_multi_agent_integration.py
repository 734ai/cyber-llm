"""
Persistent Multi-Agent Integration
Integrates persistent cognitive system with existing agent framework

Author: Cyber-LLM Development Team  
Date: August 6, 2025
Version: 2.0.0
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Union
from dataclasses import dataclass, field
from pathlib import Path
import uuid

# Import existing agents
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from agents.recon_agent import ReconAgent
from agents.c2_agent import C2Agent
from agents.post_exploit_agent import PostExploitAgent
from agents.safety_agent import SafetyAgent
from agents.explainability_agent import ExplainabilityAgent
from agents.orchestrator import Orchestrator

# Import our persistent systems
from cognitive.persistent_reasoning_system import (
    PersistentCognitiveSystem, MemoryEntry, MemoryType, 
    ReasoningType, StrategicPlan, ReasoningChain
)
from server.persistent_agent_server import PersistentAgentServer, create_server_config

@dataclass
class AgentMemoryProfile:
    """Memory profile for each agent type"""
    agent_id: str
    agent_type: str
    primary_memory_types: List[MemoryType] = field(default_factory=list)
    reasoning_preferences: List[ReasoningType] = field(default_factory=list)
    strategic_capabilities: List[str] = field(default_factory=list)
    memory_retention_policy: Dict[str, Any] = field(default_factory=dict)
    knowledge_domains: Set[str] = field(default_factory=set)

@dataclass
class CognitiveAgentState:
    """Enhanced agent state with cognitive capabilities"""
    agent_id: str
    base_agent: Any
    memory_profile: AgentMemoryProfile
    cognitive_session_id: str
    active_reasoning_chains: List[str] = field(default_factory=list)
    strategic_plans: List[str] = field(default_factory=list)
    memory_consolidation_schedule: Optional[datetime] = None
    last_cognitive_update: datetime = field(default_factory=datetime.now)
    cognitive_metrics: Dict[str, Any] = field(default_factory=dict)

class PersistentMultiAgentSystem:
    """
    Enhanced multi-agent system with persistent cognitive capabilities
    """
    
    def __init__(self, 
                 cognitive_db_path: str = "data/cognitive_system.db",
                 server_config: Optional[Any] = None):
        
        self.cognitive_system = PersistentCognitiveSystem(cognitive_db_path)
        self.logger = logging.getLogger("persistent_multi_agent")
        
        # Agent registry with cognitive enhancement
        self.cognitive_agents: Dict[str, CognitiveAgentState] = {}
        self.agent_profiles: Dict[str, AgentMemoryProfile] = {}
        
        # Original orchestrator integration
        self.orchestrator = None
        self.base_agents = {}
        
        # Server integration
        self.server = None
        if server_config:
            self.server = PersistentAgentServer(server_config)
        
        # Cognitive coordination
        self.global_memory_graph = {}
        self.inter_agent_reasoning_chains = {}
        self.collaborative_strategic_plans = {}
        
        # Initialize system
        self._initialize_agent_profiles()
        
        # Background processes
        self.cognitive_tasks = []
        self.system_running = False
    
    def _initialize_agent_profiles(self):
        """Initialize memory profiles for each agent type"""
        
        # Reconnaissance Agent Profile
        self.agent_profiles["recon"] = AgentMemoryProfile(
            agent_id="recon",
            agent_type="reconnaissance",
            primary_memory_types=[
                MemoryType.EPISODIC,    # Target discovery events
                MemoryType.SEMANTIC,    # Network knowledge
                MemoryType.PROCEDURAL   # Scanning techniques
            ],
            reasoning_preferences=[
                ReasoningType.DEDUCTIVE,    # Network analysis
                ReasoningType.INDUCTIVE,    # Pattern recognition
                ReasoningType.ANALOGICAL    # Similar network patterns
            ],
            strategic_capabilities=[
                "network_mapping",
                "vulnerability_discovery", 
                "target_prioritization"
            ],
            memory_retention_policy={
                "target_info_retention": 90,  # days
                "scan_results_retention": 30,
                "technique_learning_retention": 365
            },
            knowledge_domains={
                "network_protocols", "vulnerability_databases",
                "scanning_techniques", "target_profiling"
            }
        )
        
        # Command & Control Agent Profile  
        self.agent_profiles["c2"] = AgentMemoryProfile(
            agent_id="c2",
            agent_type="command_control",
            primary_memory_types=[
                MemoryType.EPISODIC,    # Command history
                MemoryType.WORKING,     # Active sessions
                MemoryType.PROCEDURAL   # C2 techniques
            ],
            reasoning_preferences=[
                ReasoningType.STRATEGIC,     # Mission planning
                ReasoningType.CAUSAL,        # Command effects
                ReasoningType.COUNTERFACTUAL # Alternative approaches
            ],
            strategic_capabilities=[
                "session_management",
                "payload_delivery",
                "persistence_mechanisms"
            ],
            memory_retention_policy={
                "session_logs_retention": 365,
                "command_history_retention": 180,
                "technique_effectiveness_retention": 730
            },
            knowledge_domains={
                "c2_protocols", "payload_types", "persistence_methods",
                "evasion_techniques", "communication_channels"
            }
        )
        
        # Post-Exploitation Agent Profile
        self.agent_profiles["post_exploit"] = AgentMemoryProfile(
            agent_id="post_exploit", 
            agent_type="post_exploitation",
            primary_memory_types=[
                MemoryType.EPISODIC,    # Exploitation events
                MemoryType.SEMANTIC,    # System knowledge
                MemoryType.STRATEGIC    # Long-term objectives
            ],
            reasoning_preferences=[
                ReasoningType.DEDUCTIVE,     # System analysis
                ReasoningType.STRATEGIC,     # Privilege escalation planning
                ReasoningType.META_COGNITIVE # Technique adaptation
            ],
            strategic_capabilities=[
                "privilege_escalation",
                "lateral_movement", 
                "data_extraction"
            ],
            memory_retention_policy={
                "system_mapping_retention": 180,
                "credential_retention": 365,
                "technique_success_retention": 730
            },
            knowledge_domains={
                "operating_systems", "privilege_escalation", 
                "lateral_movement", "data_exfiltration", "steganography"
            }
        )
        
        # Safety Agent Profile
        self.agent_profiles["safety"] = AgentMemoryProfile(
            agent_id="safety",
            agent_type="safety_monitor",
            primary_memory_types=[
                MemoryType.EPISODIC,    # Safety violations
                MemoryType.SEMANTIC,    # Safety rules
                MemoryType.WORKING      # Active monitoring
            ],
            reasoning_preferences=[
                ReasoningType.DEDUCTIVE,     # Rule application
                ReasoningType.CAUSAL,        # Impact analysis  
                ReasoningType.COUNTERFACTUAL # Risk scenarios
            ],
            strategic_capabilities=[
                "risk_assessment",
                "intervention_planning",
                "compliance_monitoring"
            ],
            memory_retention_policy={
                "safety_violations_retention": 2555,  # 7 years
                "rule_updates_retention": 1825,       # 5 years
                "intervention_logs_retention": 730    # 2 years
            },
            knowledge_domains={
                "safety_regulations", "risk_assessment", "compliance_frameworks",
                "incident_response", "legal_requirements"
            }
        )
        
        # Explainability Agent Profile
        self.agent_profiles["explainability"] = AgentMemoryProfile(
            agent_id="explainability",
            agent_type="explainability",
            primary_memory_types=[
                MemoryType.EPISODIC,    # Decision events
                MemoryType.SEMANTIC,    # Explanation patterns
                MemoryType.META_MEMORY  # Reasoning about reasoning
            ],
            reasoning_preferences=[
                ReasoningType.META_COGNITIVE, # Reasoning analysis
                ReasoningType.ANALOGICAL,     # Example-based explanations
                ReasoningType.ABDUCTIVE       # Best explanation inference
            ],
            strategic_capabilities=[
                "decision_tracing",
                "explanation_generation",
                "transparency_reporting"
            ],
            memory_retention_policy={
                "decision_traces_retention": 365,
                "explanation_templates_retention": 730,
                "transparency_logs_retention": 1095
            },
            knowledge_domains={
                "decision_analysis", "explanation_theory", "transparency_methods",
                "audit_trails", "interpretability_techniques"
            }
        )
        
        self.logger.info(f"Initialized {len(self.agent_profiles)} agent profiles")
    
    async def initialize_system(self):
        """Initialize the persistent multi-agent system"""
        
        try:
            self.system_running = True
            
            # Initialize cognitive system
            await self.cognitive_system.initialize()
            
            # Create base agents
            await self._create_base_agents()
            
            # Enhance agents with cognitive capabilities
            await self._enhance_agents_with_cognition()
            
            # Start cognitive background processes
            self._start_cognitive_processes()
            
            # Initialize orchestrator with cognitive enhancement
            await self._initialize_cognitive_orchestrator()
            
            # Start server if configured
            if self.server:
                asyncio.create_task(self.server.start_server())
            
            self.logger.info("Persistent multi-agent system initialized")
            
        except Exception as e:
            self.logger.error(f"Error initializing system: {e}")
            raise
    
    async def _create_base_agents(self):
        """Create base agent instances"""
        
        try:
            # Create standard agents
            self.base_agents["recon"] = ReconAgent()
            self.base_agents["c2"] = C2Agent()  
            self.base_agents["post_exploit"] = PostExploitAgent()
            self.base_agents["safety"] = SafetyAgent()
            self.base_agents["explainability"] = ExplainabilityAgent()
            
            self.logger.info(f"Created {len(self.base_agents)} base agents")
            
        except Exception as e:
            self.logger.error(f"Error creating base agents: {e}")
            raise
    
    async def _enhance_agents_with_cognition(self):
        """Enhance base agents with persistent cognitive capabilities"""
        
        try:
            for agent_id, base_agent in self.base_agents.items():
                
                # Get memory profile
                profile = self.agent_profiles.get(agent_id)
                if not profile:
                    self.logger.warning(f"No memory profile found for {agent_id}")
                    continue
                
                # Create cognitive session
                cognitive_session_id = str(uuid.uuid4())
                
                # Initialize agent memories
                await self._initialize_agent_memories(profile)
                
                # Create cognitive agent state
                cognitive_state = CognitiveAgentState(
                    agent_id=agent_id,
                    base_agent=base_agent,
                    memory_profile=profile,
                    cognitive_session_id=cognitive_session_id,
                    memory_consolidation_schedule=datetime.now() + timedelta(hours=6)
                )
                
                self.cognitive_agents[agent_id] = cognitive_state
                
                # Enhance base agent with cognitive methods
                await self._inject_cognitive_methods(base_agent, cognitive_state)
                
                self.logger.info(f"Enhanced {agent_id} agent with cognitive capabilities")
            
        except Exception as e:
            self.logger.error(f"Error enhancing agents: {e}")
            raise
    
    async def _initialize_agent_memories(self, profile: AgentMemoryProfile):
        """Initialize memories for an agent based on its profile"""
        
        try:
            # Create initial semantic memories for knowledge domains
            for domain in profile.knowledge_domains:
                memory_entry = MemoryEntry(
                    memory_type=MemoryType.SEMANTIC,
                    content={
                        "domain": domain,
                        "agent_id": profile.agent_id,
                        "knowledge_base": f"Initialized knowledge base for {domain}",
                        "expertise_level": "basic",
                        "last_updated": datetime.now().isoformat()
                    },
                    importance=0.8,
                    tags={f"agent:{profile.agent_id}", f"domain:{domain}", "initialization"}
                )
                
                await self.cognitive_system.memory_manager.store_memory(memory_entry)
            
            # Create procedural memories for capabilities
            for capability in profile.strategic_capabilities:
                memory_entry = MemoryEntry(
                    memory_type=MemoryType.PROCEDURAL,
                    content={
                        "capability": capability,
                        "agent_id": profile.agent_id,
                        "procedure_steps": f"Standard procedure for {capability}",
                        "success_rate": 0.0,
                        "last_used": None
                    },
                    importance=0.7,
                    tags={f"agent:{profile.agent_id}", f"capability:{capability}", "procedure"}
                )
                
                await self.cognitive_system.memory_manager.store_memory(memory_entry)
            
            self.logger.debug(f"Initialized memories for agent {profile.agent_id}")
            
        except Exception as e:
            self.logger.error(f"Error initializing memories for {profile.agent_id}: {e}")
    
    async def _inject_cognitive_methods(self, base_agent: Any, cognitive_state: CognitiveAgentState):
        """Inject cognitive methods into base agent"""
        
        try:
            agent_id = cognitive_state.agent_id
            
            # Inject memory methods
            async def remember(content: Dict[str, Any], 
                             memory_type: MemoryType = MemoryType.EPISODIC,
                             importance: float = 0.5,
                             tags: Set[str] = None) -> str:
                """Enhanced memory storage method"""
                
                if tags is None:
                    tags = set()
                tags.add(f"agent:{agent_id}")
                
                memory_entry = MemoryEntry(
                    memory_type=memory_type,
                    content=content,
                    importance=importance,
                    tags=tags
                )
                
                memory_id = await self.cognitive_system.memory_manager.store_memory(memory_entry)
                
                # Update agent metrics
                cognitive_state.cognitive_metrics.setdefault("memories_created", 0)
                cognitive_state.cognitive_metrics["memories_created"] += 1
                cognitive_state.last_cognitive_update = datetime.now()
                
                return memory_id
            
            async def recall(query: str,
                           memory_types: List[MemoryType] = None,
                           limit: int = 10) -> List[MemoryEntry]:
                """Enhanced memory recall method"""
                
                if memory_types is None:
                    memory_types = cognitive_state.memory_profile.primary_memory_types
                
                # Add agent-specific query filter
                agent_query = f"{query} agent:{agent_id}"
                
                memories = await self.cognitive_system.memory_manager.search_memories(
                    agent_query, memory_types, limit
                )
                
                # Update agent metrics
                cognitive_state.cognitive_metrics.setdefault("memory_recalls", 0)
                cognitive_state.cognitive_metrics["memory_recalls"] += 1
                
                return memories
            
            async def reason(topic: str, 
                           goal: str,
                           reasoning_type: ReasoningType = None) -> str:
                """Enhanced reasoning method"""
                
                if reasoning_type is None:
                    reasoning_type = cognitive_state.memory_profile.reasoning_preferences[0]
                
                chain_id = await self.cognitive_system.reasoning_engine.start_reasoning_chain(
                    topic=f"{topic} (Agent: {agent_id})",
                    goal=goal,
                    reasoning_type=reasoning_type
                )
                
                # Track reasoning chain
                cognitive_state.active_reasoning_chains.append(chain_id)
                
                # Update metrics
                cognitive_state.cognitive_metrics.setdefault("reasoning_chains_started", 0) 
                cognitive_state.cognitive_metrics["reasoning_chains_started"] += 1
                
                return chain_id
            
            async def plan_strategically(title: str,
                                       primary_goal: str, 
                                       template_type: str = "cybersecurity_assessment") -> str:
                """Enhanced strategic planning method"""
                
                plan_id = await self.cognitive_system.strategic_planner.create_strategic_plan(
                    title=f"{title} (Agent: {agent_id})",
                    primary_goal=primary_goal,
                    template_type=template_type
                )
                
                # Track strategic plan
                cognitive_state.strategic_plans.append(plan_id)
                
                # Update metrics
                cognitive_state.cognitive_metrics.setdefault("strategic_plans_created", 0)
                cognitive_state.cognitive_metrics["strategic_plans_created"] += 1
                
                return plan_id
            
            async def get_cognitive_status() -> Dict[str, Any]:
                """Get cognitive status for agent"""
                
                return {
                    "agent_id": agent_id,
                    "active_reasoning_chains": len(cognitive_state.active_reasoning_chains),
                    "strategic_plans": len(cognitive_state.strategic_plans),
                    "last_cognitive_update": cognitive_state.last_cognitive_update.isoformat(),
                    "cognitive_metrics": cognitive_state.cognitive_metrics,
                    "memory_profile": {
                        "primary_memory_types": [mt.value for mt in cognitive_state.memory_profile.primary_memory_types],
                        "reasoning_preferences": [rt.value for rt in cognitive_state.memory_profile.reasoning_preferences],
                        "knowledge_domains": list(cognitive_state.memory_profile.knowledge_domains)
                    }
                }
            
            # Inject methods into base agent
            setattr(base_agent, 'remember', remember)
            setattr(base_agent, 'recall', recall)
            setattr(base_agent, 'reason', reason)
            setattr(base_agent, 'plan_strategically', plan_strategically)
            setattr(base_agent, 'get_cognitive_status', get_cognitive_status)
            setattr(base_agent, 'cognitive_state', cognitive_state)
            
            self.logger.debug(f"Injected cognitive methods into {agent_id} agent")
            
        except Exception as e:
            self.logger.error(f"Error injecting cognitive methods: {e}")
    
    def _start_cognitive_processes(self):
        """Start background cognitive processes"""
        
        # Memory consolidation process
        async def memory_consolidation_worker():
            while self.system_running:
                try:
                    await asyncio.sleep(3600)  # Every hour
                    await self._consolidate_agent_memories()
                except Exception as e:
                    self.logger.error(f"Memory consolidation error: {e}")
        
        # Inter-agent reasoning coordination
        async def inter_agent_reasoning_coordinator():
            while self.system_running:
                try:
                    await asyncio.sleep(1800)  # Every 30 minutes
                    await self._coordinate_inter_agent_reasoning()
                except Exception as e:
                    self.logger.error(f"Inter-agent reasoning error: {e}")
        
        # Strategic planning synchronization
        async def strategic_plan_synchronizer():
            while self.system_running:
                try:
                    await asyncio.sleep(7200)  # Every 2 hours  
                    await self._synchronize_strategic_plans()
                except Exception as e:
                    self.logger.error(f"Strategic plan sync error: {e}")
        
        # Global memory graph maintenance
        async def global_memory_maintenance():
            while self.system_running:
                try:
                    await asyncio.sleep(10800)  # Every 3 hours
                    await self._maintain_global_memory_graph()
                except Exception as e:
                    self.logger.error(f"Global memory maintenance error: {e}")
        
        # Start background tasks
        self.cognitive_tasks = [
            asyncio.create_task(memory_consolidation_worker()),
            asyncio.create_task(inter_agent_reasoning_coordinator()),
            asyncio.create_task(strategic_plan_synchronizer()),
            asyncio.create_task(global_memory_maintenance())
        ]
        
        self.logger.info("Started cognitive background processes")
    
    async def _consolidate_agent_memories(self):
        """Consolidate memories for all agents"""
        
        try:
            for agent_id, cognitive_state in self.cognitive_agents.items():
                
                # Check if consolidation is due
                if (cognitive_state.memory_consolidation_schedule and 
                    datetime.now() >= cognitive_state.memory_consolidation_schedule):
                    
                    # Perform memory consolidation
                    await self.cognitive_system.memory_manager.consolidate_memories()
                    
                    # Schedule next consolidation
                    cognitive_state.memory_consolidation_schedule = datetime.now() + timedelta(hours=6)
                    
                    # Update metrics
                    cognitive_state.cognitive_metrics.setdefault("memory_consolidations", 0)
                    cognitive_state.cognitive_metrics["memory_consolidations"] += 1
                    
                    self.logger.debug(f"Consolidated memories for agent {agent_id}")
            
        except Exception as e:
            self.logger.error(f"Error consolidating memories: {e}")
    
    async def _coordinate_inter_agent_reasoning(self):
        """Coordinate reasoning between agents"""
        
        try:
            # Find active reasoning chains from all agents
            all_chains = []
            for cognitive_state in self.cognitive_agents.values():
                all_chains.extend(cognitive_state.active_reasoning_chains)
            
            if len(all_chains) < 2:
                return  # Need at least 2 chains for coordination
            
            # Create collaborative reasoning chain
            collaborative_topic = "Multi-Agent Collaborative Analysis"
            collaborative_goal = "Synthesize insights from multiple agent perspectives"
            
            collaborative_chain_id = await self.cognitive_system.reasoning_engine.start_reasoning_chain(
                topic=collaborative_topic,
                goal=collaborative_goal,
                reasoning_type=ReasoningType.META_COGNITIVE
            )
            
            # Add reasoning steps from individual chains
            for chain_id in all_chains[-5:]:  # Last 5 chains
                try:
                    chain = await self.cognitive_system.reasoning_engine.get_reasoning_chain(chain_id)
                    if chain and chain.reasoning_steps:
                        await self.cognitive_system.reasoning_engine.add_reasoning_step(
                            collaborative_chain_id,
                            premise=f"Agent reasoning from chain {chain_id}",
                            inference_rule="collaborative_synthesis",
                            evidence=[step.conclusion for step in chain.reasoning_steps[-3:]]
                        )
                except Exception as e:
                    self.logger.debug(f"Error processing chain {chain_id}: {e}")
                    continue
            
            # Complete collaborative reasoning
            final_chain = await self.cognitive_system.reasoning_engine.complete_reasoning_chain(
                collaborative_chain_id
            )
            
            if final_chain:
                self.inter_agent_reasoning_chains[collaborative_chain_id] = {
                    "created_at": datetime.now(),
                    "participating_agents": list(self.cognitive_agents.keys()),
                    "conclusion": final_chain.conclusion,
                    "confidence": final_chain.confidence
                }
                
                self.logger.info(f"Created collaborative reasoning chain: {collaborative_chain_id}")
            
        except Exception as e:
            self.logger.error(f"Error coordinating inter-agent reasoning: {e}")
    
    async def _synchronize_strategic_plans(self):
        """Synchronize strategic plans across agents"""
        
        try:
            # Collect all strategic plans
            all_plans = []
            for cognitive_state in self.cognitive_agents.values():
                all_plans.extend(cognitive_state.strategic_plans)
            
            if not all_plans:
                return
            
            # Create master strategic plan
            master_plan_id = await self.cognitive_system.strategic_planner.create_strategic_plan(
                title="Multi-Agent Coordinated Strategic Plan",
                primary_goal="Coordinate strategic objectives across all agents",
                template_type="cybersecurity_assessment"
            )
            
            # Add goals from individual plans
            for plan_id in all_plans[-10:]:  # Last 10 plans
                try:
                    plan = await self.cognitive_system.strategic_planner.get_strategic_plan(plan_id)
                    if plan:
                        # Add plan's goals as sub-goals
                        await self.cognitive_system.strategic_planner.add_goal_to_plan(
                            master_plan_id,
                            title=f"Sub-goal from plan {plan_id}",
                            description=plan.primary_goal,
                            priority=5
                        )
                except Exception as e:
                    self.logger.debug(f"Error processing plan {plan_id}: {e}")
                    continue
            
            self.collaborative_strategic_plans[master_plan_id] = {
                "created_at": datetime.now(),
                "participating_agents": list(self.cognitive_agents.keys()),
                "individual_plans": all_plans
            }
            
            self.logger.info(f"Created master strategic plan: {master_plan_id}")
            
        except Exception as e:
            self.logger.error(f"Error synchronizing strategic plans: {e}")
    
    async def _maintain_global_memory_graph(self):
        """Maintain global memory graph across agents"""
        
        try:
            # Build memory relationships across agents
            for agent_id, cognitive_state in self.cognitive_agents.items():
                
                # Get recent memories for this agent
                recent_memories = await self.cognitive_system.memory_manager.search_memories(
                    query=f"agent:{agent_id}",
                    limit=20
                )
                
                for memory in recent_memories:
                    # Look for related memories from other agents
                    for other_agent_id in self.cognitive_agents.keys():
                        if other_agent_id == agent_id:
                            continue
                        
                        # Search for related memories
                        related_memories = await self.cognitive_system.memory_manager.search_memories(
                            query=f"agent:{other_agent_id}",
                            memory_types=[memory.memory_type],
                            limit=5
                        )
                        
                        # Create relationships
                        for related_memory in related_memories:
                            relationship_key = f"{memory.memory_id}:{related_memory.memory_id}"
                            
                            if relationship_key not in self.global_memory_graph:
                                self.global_memory_graph[relationship_key] = {
                                    "source_agent": agent_id,
                                    "target_agent": other_agent_id,
                                    "relationship_type": "cross_agent_correlation",
                                    "strength": 0.3,  # Base correlation strength
                                    "created_at": datetime.now(),
                                    "access_count": 0
                                }
            
            self.logger.debug(f"Global memory graph has {len(self.global_memory_graph)} relationships")
            
        except Exception as e:
            self.logger.error(f"Error maintaining global memory graph: {e}")
    
    async def _initialize_cognitive_orchestrator(self):
        """Initialize orchestrator with cognitive enhancements"""
        
        try:
            # Create enhanced orchestrator
            self.orchestrator = Orchestrator()
            
            # Inject cognitive coordination methods
            async def coordinate_cognitive_analysis(scenario: Dict[str, Any]) -> Dict[str, Any]:
                """Coordinate cognitive analysis across all agents"""
                
                results = {}
                
                # Run scenario through each cognitive agent
                for agent_id, cognitive_state in self.cognitive_agents.items():
                    base_agent = cognitive_state.base_agent
                    
                    # Store scenario in agent memory
                    memory_id = await base_agent.remember(
                        content={
                            "scenario": scenario,
                            "analysis_requested": datetime.now().isoformat(),
                            "scenario_type": scenario.get("type", "unknown")
                        },
                        memory_type=MemoryType.EPISODIC,
                        importance=0.8,
                        tags={"scenario_analysis", "orchestrated"}
                    )
                    
                    # Start reasoning chain for analysis
                    reasoning_chain_id = await base_agent.reason(
                        topic=f"Scenario Analysis: {scenario.get('title', 'Untitled')}",
                        goal=f"Analyze scenario from {agent_id} perspective",
                        reasoning_type=cognitive_state.memory_profile.reasoning_preferences[0]
                    )
                    
                    # Create strategic plan if appropriate
                    if agent_id in ["recon", "c2", "post_exploit"]:
                        plan_id = await base_agent.plan_strategically(
                            title=f"Strategic Response to Scenario ({agent_id})",
                            primary_goal=scenario.get("primary_goal", "Address scenario requirements")
                        )
                        results[f"{agent_id}_strategic_plan"] = plan_id
                    
                    results[f"{agent_id}_memory"] = memory_id
                    results[f"{agent_id}_reasoning"] = reasoning_chain_id
                
                # Create collaborative analysis
                collaborative_analysis = await self.cognitive_system.process_complex_scenario({
                    **scenario,
                    "multi_agent_analysis": True,
                    "participating_agents": list(self.cognitive_agents.keys()),
                    "individual_results": results
                })
                
                results["collaborative_analysis"] = collaborative_analysis
                
                return results
            
            async def get_system_cognitive_status() -> Dict[str, Any]:
                """Get cognitive status of entire system"""
                
                system_status = {
                    "timestamp": datetime.now().isoformat(),
                    "agents": {},
                    "global_metrics": {
                        "total_active_reasoning_chains": sum(
                            len(cs.active_reasoning_chains) 
                            for cs in self.cognitive_agents.values()
                        ),
                        "total_strategic_plans": sum(
                            len(cs.strategic_plans) 
                            for cs in self.cognitive_agents.values()
                        ),
                        "inter_agent_reasoning_chains": len(self.inter_agent_reasoning_chains),
                        "collaborative_strategic_plans": len(self.collaborative_strategic_plans),
                        "global_memory_relationships": len(self.global_memory_graph)
                    },
                    "memory_stats": await self.cognitive_system.memory_manager.get_memory_stats()
                }
                
                # Get individual agent status
                for agent_id, cognitive_state in self.cognitive_agents.items():
                    system_status["agents"][agent_id] = await cognitive_state.base_agent.get_cognitive_status()
                
                return system_status
            
            # Inject methods into orchestrator
            setattr(self.orchestrator, 'coordinate_cognitive_analysis', coordinate_cognitive_analysis)
            setattr(self.orchestrator, 'get_system_cognitive_status', get_system_cognitive_status)
            setattr(self.orchestrator, 'cognitive_agents', self.cognitive_agents)
            
            self.logger.info("Initialized cognitive orchestrator")
            
        except Exception as e:
            self.logger.error(f"Error initializing cognitive orchestrator: {e}")
    
    async def run_cognitive_scenario(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Run a scenario through the cognitive multi-agent system"""
        
        try:
            self.logger.info(f"Starting cognitive scenario: {scenario.get('title', 'Untitled')}")
            
            # Use orchestrator's cognitive analysis
            if self.orchestrator and hasattr(self.orchestrator, 'coordinate_cognitive_analysis'):
                results = await self.orchestrator.coordinate_cognitive_analysis(scenario)
            else:
                # Fallback to direct cognitive processing
                results = await self.cognitive_system.process_complex_scenario(scenario)
            
            self.logger.info("Cognitive scenario completed")
            return results
            
        except Exception as e:
            self.logger.error(f"Error running cognitive scenario: {e}")
            return {"status": "error", "message": str(e)}
    
    async def get_agent(self, agent_id: str) -> Optional[Any]:
        """Get enhanced agent by ID"""
        
        cognitive_state = self.cognitive_agents.get(agent_id)
        return cognitive_state.base_agent if cognitive_state else None
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        
        if self.orchestrator and hasattr(self.orchestrator, 'get_system_cognitive_status'):
            return await self.orchestrator.get_system_cognitive_status()
        else:
            return {
                "status": "basic",
                "agents": list(self.cognitive_agents.keys()),
                "system_running": self.system_running
            }
    
    async def shutdown(self):
        """Graceful shutdown of the cognitive system"""
        
        try:
            self.logger.info("Shutting down persistent multi-agent system...")
            
            self.system_running = False
            
            # Cancel cognitive tasks
            for task in self.cognitive_tasks:
                task.cancel()
            
            if self.cognitive_tasks:
                await asyncio.gather(*self.cognitive_tasks, return_exceptions=True)
            
            # Shutdown server if running
            if self.server:
                await self.server.shutdown()
            
            self.logger.info("System shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")

# Factory function
def create_persistent_multi_agent_system(
    cognitive_db_path: str = "data/cognitive_system.db",
    enable_server: bool = True,
    server_port: int = 8080
) -> PersistentMultiAgentSystem:
    """Create persistent multi-agent system"""
    
    server_config = None
    if enable_server:
        server_config = create_server_config(port=server_port)
    
    return PersistentMultiAgentSystem(
        cognitive_db_path=cognitive_db_path,
        server_config=server_config
    )

# Example usage and testing
async def main():
    """Example usage of the persistent multi-agent system"""
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create system
    system = create_persistent_multi_agent_system()
    
    try:
        # Initialize
        await system.initialize_system()
        
        # Run example scenario
        scenario = {
            "title": "Advanced Persistent Threat Analysis",
            "type": "cybersecurity_assessment",
            "primary_goal": "Identify and analyze APT indicators",
            "target_environment": "corporate_network",
            "threat_indicators": [
                "suspicious_network_traffic",
                "unusual_authentication_patterns", 
                "lateral_movement_attempts"
            ],
            "time_constraints": "72_hours",
            "risk_tolerance": "low"
        }
        
        # Process scenario
        results = await system.run_cognitive_scenario(scenario)
        print(f"Scenario results: {json.dumps(results, indent=2, default=str)}")
        
        # Get system status
        status = await system.get_system_status()
        print(f"System status: {json.dumps(status, indent=2, default=str)}")
        
        # Keep system running
        while True:
            await asyncio.sleep(60)  # Check every minute
            
    except KeyboardInterrupt:
        print("Shutdown requested by user")
    finally:
        await system.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
