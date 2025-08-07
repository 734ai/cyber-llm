"""
Multi-Agent Collaboration Framework for Cyber-LLM
Advanced agent-to-agent communication and swarm intelligence

Author: Muzan Sano <sanosensei36@gmail.com>
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from collections import defaultdict, deque
import websockets
import aiohttp

from ..utils.logging_system import CyberLLMLogger, CyberLLMError, ErrorCategory
from ..memory.persistent_memory import PersistentMemoryManager
from ..cognitive.meta_cognitive import MetaCognitiveEngine

class MessageType(Enum):
    """Agent communication message types"""
    TASK_REQUEST = "task_request"
    TASK_RESPONSE = "task_response"
    INFORMATION_SHARE = "information_share"
    COORDINATION_REQUEST = "coordination_request"
    CONSENSUS_PROPOSAL = "consensus_proposal"
    CONSENSUS_VOTE = "consensus_vote"
    CAPABILITY_ANNOUNCEMENT = "capability_announcement"
    RESOURCE_REQUEST = "resource_request"
    RESOURCE_OFFER = "resource_offer"
    SWARM_DIRECTIVE = "swarm_directive"
    EMERGENCY_ALERT = "emergency_alert"

class AgentRole(Enum):
    """Agent roles in the collaboration framework"""
    LEADER = "leader"
    SPECIALIST = "specialist"
    COORDINATOR = "coordinator"
    SCOUT = "scout"
    ANALYZER = "analyzer"
    EXECUTOR = "executor"
    MONITOR = "monitor"

class ConsensusAlgorithm(Enum):
    """Consensus algorithms for decision making"""
    MAJORITY_VOTE = "majority_vote"
    WEIGHTED_VOTE = "weighted_vote"
    BYZANTINE_FAULT_TOLERANT = "byzantine_fault_tolerant"
    PROOF_OF_EXPERTISE = "proof_of_expertise"
    RAFT = "raft"

@dataclass
class AgentMessage:
    """Inter-agent communication message"""
    message_id: str
    sender_id: str
    recipient_id: Optional[str]  # None for broadcast
    message_type: MessageType
    timestamp: datetime
    
    # Content
    content: Dict[str, Any]
    priority: int = 5  # 1-10, 10 = highest
    
    # Routing and delivery
    ttl: int = 300  # Time to live in seconds
    requires_acknowledgment: bool = False
    correlation_id: Optional[str] = None
    
    # Security
    signature: Optional[str] = None
    encrypted: bool = False

@dataclass
class AgentCapability:
    """Agent capability description"""
    capability_id: str
    name: str
    description: str
    
    # Performance metrics
    accuracy: float
    speed: float  # Operations per second
    resource_cost: float
    
    # Availability
    available: bool = True
    current_load: float = 0.0
    max_concurrent: int = 10
    
    # Requirements
    required_resources: Dict[str, float] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)

@dataclass
class SwarmTask:
    """Task for swarm execution"""
    task_id: str
    description: str
    task_type: str
    
    # Requirements
    required_capabilities: List[str]
    estimated_complexity: float
    deadline: Optional[datetime] = None
    
    # Decomposition
    subtasks: List['SwarmTask'] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    
    # Assignment
    assigned_agents: List[str] = field(default_factory=list)
    status: str = "pending"
    
    # Results
    results: Dict[str, Any] = field(default_factory=dict)
    completion_time: Optional[datetime] = None

class AgentCommunicationProtocol:
    """Standardized protocol for agent communication"""
    
    def __init__(self, agent_id: str, logger: Optional[CyberLLMLogger] = None):
        self.agent_id = agent_id
        self.logger = logger or CyberLLMLogger(name="agent_protocol")
        
        # Communication infrastructure
        self.message_queue = asyncio.Queue()
        self.active_connections = {}
        self.message_handlers = {}
        self.acknowledgments = {}
        
        # Protocol state
        self.capabilities = {}
        self.peer_agents = {}
        self.conversation_contexts = {}
        
        # Security
        self.trusted_agents = set()
        self.encryption_keys = {}
        
        self.logger.info("Agent Communication Protocol initialized", agent_id=agent_id)
    
    async def send_message(self, message: AgentMessage) -> bool:
        """Send message to another agent or broadcast"""
        
        try:
            # Validate message
            if not self._validate_message(message):
                self.logger.error("Invalid message", message_id=message.message_id)
                return False
            
            # Add timestamp and sender
            message.timestamp = datetime.now()
            message.sender_id = self.agent_id
            
            # Sign message if required
            if message.encrypted or message.signature:
                message = await self._secure_message(message)
            
            # Route message
            if message.recipient_id:
                # Direct message
                success = await self._send_direct_message(message)
            else:
                # Broadcast message
                success = await self._broadcast_message(message)
            
            # Handle acknowledgment requirement
            if message.requires_acknowledgment and success:
                asyncio.create_task(self._wait_for_acknowledgment(message))
            
            self.logger.info("Message sent",
                           message_id=message.message_id,
                           recipient=message.recipient_id or "broadcast",
                           type=message.message_type.value)
            
            return success
            
        except Exception as e:
            self.logger.error("Failed to send message", error=str(e))
            return False
    
    async def receive_message(self) -> Optional[AgentMessage]:
        """Receive next message from queue"""
        
        try:
            # Get message from queue (with timeout)
            message = await asyncio.wait_for(self.message_queue.get(), timeout=1.0)
            
            # Validate and process message
            if self._validate_received_message(message):
                await self._process_received_message(message)
                return message
            
            return None
            
        except asyncio.TimeoutError:
            return None
        except Exception as e:
            self.logger.error("Failed to receive message", error=str(e))
            return None
    
    async def register_capability(self, capability: AgentCapability):
        """Register agent capability"""
        
        self.capabilities[capability.capability_id] = capability
        
        # Announce capability to other agents
        announcement = AgentMessage(
            message_id=str(uuid.uuid4()),
            sender_id=self.agent_id,
            recipient_id=None,  # Broadcast
            message_type=MessageType.CAPABILITY_ANNOUNCEMENT,
            timestamp=datetime.now(),
            content={
                "capability": {
                    "id": capability.capability_id,
                    "name": capability.name,
                    "description": capability.description,
                    "accuracy": capability.accuracy,
                    "speed": capability.speed,
                    "available": capability.available
                }
            }
        )
        
        await self.send_message(announcement)
        
        self.logger.info("Capability registered and announced",
                       capability_id=capability.capability_id,
                       name=capability.name)

class DistributedConsensus:
    """Distributed consensus mechanisms for multi-agent decisions"""
    
    def __init__(self, 
                 agent_id: str,
                 communication_protocol: AgentCommunicationProtocol,
                 logger: Optional[CyberLLMLogger] = None):
        
        self.agent_id = agent_id
        self.protocol = communication_protocol
        self.logger = logger or CyberLLMLogger(name="consensus")
        
        # Consensus state
        self.active_proposals = {}
        self.voting_history = deque(maxlen=1000)
        self.consensus_results = {}
        
        # Agent weights for weighted voting
        self.agent_weights = {}
        
        self.logger.info("Distributed Consensus initialized", agent_id=agent_id)
    
    async def propose_consensus(self, 
                              proposal_id: str,
                              proposal_content: Dict[str, Any],
                              algorithm: ConsensusAlgorithm = ConsensusAlgorithm.MAJORITY_VOTE,
                              timeout: int = 300) -> Dict[str, Any]:
        """Propose a decision for consensus"""
        
        try:
            proposal = {
                "proposal_id": proposal_id,
                "proposer": self.agent_id,
                "content": proposal_content,
                "algorithm": algorithm.value,
                "created_at": datetime.now().isoformat(),
                "timeout": timeout,
                "votes": {},
                "status": "active"
            }
            
            self.active_proposals[proposal_id] = proposal
            
            # Broadcast proposal
            message = AgentMessage(
                message_id=str(uuid.uuid4()),
                sender_id=self.agent_id,
                recipient_id=None,  # Broadcast
                message_type=MessageType.CONSENSUS_PROPOSAL,
                timestamp=datetime.now(),
                content=proposal,
                ttl=timeout
            )
            
            await self.protocol.send_message(message)
            
            # Wait for consensus or timeout
            result = await self._wait_for_consensus(proposal_id, timeout)
            
            self.logger.info("Consensus proposal completed",
                           proposal_id=proposal_id,
                           result=result.get("decision"),
                           votes_received=len(result.get("votes", {})))
            
            return result
            
        except Exception as e:
            self.logger.error("Consensus proposal failed", error=str(e))
            return {"decision": "failed", "error": str(e)}
    
    async def vote_on_proposal(self, 
                             proposal_id: str, 
                             vote: Union[bool, float, str],
                             justification: Optional[str] = None) -> bool:
        """Vote on an active proposal"""
        
        try:
            if proposal_id not in self.active_proposals:
                self.logger.warning("Unknown proposal", proposal_id=proposal_id)
                return False
            
            proposal = self.active_proposals[proposal_id]
            
            # Create vote message
            vote_content = {
                "proposal_id": proposal_id,
                "vote": vote,
                "voter": self.agent_id,
                "timestamp": datetime.now().isoformat(),
                "justification": justification
            }
            
            message = AgentMessage(
                message_id=str(uuid.uuid4()),
                sender_id=self.agent_id,
                recipient_id=proposal["proposer"],
                message_type=MessageType.CONSENSUS_VOTE,
                timestamp=datetime.now(),
                content=vote_content
            )
            
            await self.protocol.send_message(message)
            
            # Record vote locally
            self.voting_history.append((datetime.now(), proposal_id, vote))
            
            self.logger.info("Vote submitted",
                           proposal_id=proposal_id,
                           vote=vote)
            
            return True
            
        except Exception as e:
            self.logger.error("Failed to vote on proposal", error=str(e))
            return False

class SwarmIntelligence:
    """Swarm intelligence capabilities for emergent behavior"""
    
    def __init__(self, 
                 agent_id: str,
                 communication_protocol: AgentCommunicationProtocol,
                 memory_manager: PersistentMemoryManager,
                 logger: Optional[CyberLLMLogger] = None):
        
        self.agent_id = agent_id
        self.protocol = communication_protocol
        self.memory_manager = memory_manager
        self.logger = logger or CyberLLMLogger(name="swarm_intelligence")
        
        # Swarm state
        self.swarm_members = set()
        self.role = AgentRole.SPECIALIST
        self.current_tasks = {}
        
        # Intelligence mechanisms
        self.pheromone_trails = defaultdict(float)
        self.collective_knowledge = {}
        self.emergence_patterns = {}
        
        # Task distribution
        self.task_queue = asyncio.Queue()
        self.completed_tasks = deque(maxlen=1000)
        
        self.logger.info("Swarm Intelligence initialized", agent_id=agent_id)
    
    async def join_swarm(self, swarm_id: str, role: AgentRole = AgentRole.SPECIALIST):
        """Join a swarm with specified role"""
        
        try:
            self.role = role
            self.swarm_members.add(self.agent_id)
            
            # Announce joining
            message = AgentMessage(
                message_id=str(uuid.uuid4()),
                sender_id=self.agent_id,
                recipient_id=None,  # Broadcast
                message_type=MessageType.INFORMATION_SHARE,
                timestamp=datetime.now(),
                content={
                    "action": "join_swarm",
                    "swarm_id": swarm_id,
                    "role": role.value,
                    "agent_capabilities": list(self.protocol.capabilities.keys())
                }
            )
            
            await self.protocol.send_message(message)
            
            # Start swarm behaviors
            asyncio.create_task(self._run_swarm_behaviors())
            
            self.logger.info("Joined swarm",
                           swarm_id=swarm_id,
                           role=role.value)
            
        except Exception as e:
            self.logger.error("Failed to join swarm", error=str(e))
    
    async def distribute_task(self, task: SwarmTask) -> str:
        """Distribute task across swarm members"""
        
        try:
            # Analyze task requirements
            task_requirements = await self._analyze_task_requirements(task)
            
            # Find suitable agents
            suitable_agents = await self._find_suitable_agents(task_requirements)
            
            if not suitable_agents:
                self.logger.warning("No suitable agents found for task", task_id=task.task_id)
                return "failed"
            
            # Decompose task if needed
            if len(task.required_capabilities) > 1 or task.estimated_complexity > 0.7:
                subtasks = await self._decompose_task(task)
                if subtasks:
                    # Distribute subtasks
                    for subtask in subtasks:
                        await self.distribute_task(subtask)
                    return "distributed"
            
            # Assign task to best agent
            best_agent = await self._select_best_agent(suitable_agents, task_requirements)
            
            # Send task assignment
            task_message = AgentMessage(
                message_id=str(uuid.uuid4()),
                sender_id=self.agent_id,
                recipient_id=best_agent,
                message_type=MessageType.TASK_REQUEST,
                timestamp=datetime.now(),
                content={
                    "task": {
                        "id": task.task_id,
                        "description": task.description,
                        "type": task.task_type,
                        "complexity": task.estimated_complexity,
                        "deadline": task.deadline.isoformat() if task.deadline else None,
                        "requirements": task_requirements
                    }
                },
                requires_acknowledgment=True
            )
            
            await self.protocol.send_message(task_message)
            
            # Update task status
            task.assigned_agents = [best_agent]
            task.status = "assigned"
            self.current_tasks[task.task_id] = task
            
            self.logger.info("Task distributed",
                           task_id=task.task_id,
                           assigned_agent=best_agent)
            
            return "assigned"
            
        except Exception as e:
            self.logger.error("Task distribution failed", error=str(e))
            return "failed"
    
    async def execute_collective_problem_solving(self, 
                                               problem: Dict[str, Any]) -> Dict[str, Any]:
        """Execute collective problem solving using swarm intelligence"""
        
        try:
            problem_id = problem.get("id", str(uuid.uuid4()))
            
            self.logger.info("Starting collective problem solving", problem_id=problem_id)
            
            # Phase 1: Problem decomposition
            subproblems = await self._decompose_problem(problem)
            
            # Phase 2: Distribute subproblems
            partial_solutions = []
            for subproblem in subproblems:
                solution = await self._solve_subproblem_collectively(subproblem)
                partial_solutions.append(solution)
            
            # Phase 3: Solution synthesis
            final_solution = await self._synthesize_solutions(partial_solutions, problem)
            
            # Phase 4: Validation through consensus
            validation_result = await self._validate_solution_collectively(
                final_solution, problem)
            
            # Store in collective knowledge
            self.collective_knowledge[problem_id] = {
                "problem": problem,
                "solution": final_solution,
                "validation": validation_result,
                "timestamp": datetime.now().isoformat(),
                "participating_agents": list(self.swarm_members)
            }
            
            # Update pheromone trails for successful patterns
            if validation_result.get("valid", False):
                await self._update_pheromone_trails(problem, final_solution)
            
            self.logger.info("Collective problem solving completed",
                           problem_id=problem_id,
                           solution_quality=validation_result.get("quality", 0.0))
            
            return {
                "problem_id": problem_id,
                "solution": final_solution,
                "validation": validation_result,
                "collective_intelligence_applied": True
            }
            
        except Exception as e:
            self.logger.error("Collective problem solving failed", error=str(e))
            return {"problem_id": problem_id, "error": str(e)}

class TaskDistributionEngine:
    """Advanced task distribution and load balancing"""
    
    def __init__(self, logger: Optional[CyberLLMLogger] = None):
        self.logger = logger or CyberLLMLogger(name="task_distribution")
        self.agent_loads = defaultdict(float)
        self.task_history = deque(maxlen=10000)
        self.performance_metrics = defaultdict(dict)
    
    async def distribute_workload(self, 
                                tasks: List[SwarmTask],
                                available_agents: Dict[str, AgentCapability]) -> Dict[str, List[str]]:
        """Distribute workload optimally across agents"""
        
        try:
            # Calculate agent scores for each task
            task_assignments = {}
            
            for task in tasks:
                best_agent = await self._find_optimal_agent(task, available_agents)
                if best_agent:
                    if best_agent not in task_assignments:
                        task_assignments[best_agent] = []
                    task_assignments[best_agent].append(task.task_id)
                    
                    # Update agent load
                    self.agent_loads[best_agent] += task.estimated_complexity
            
            self.logger.info("Workload distributed",
                           tasks_count=len(tasks),
                           agents_used=len(task_assignments))
            
            return task_assignments
            
        except Exception as e:
            self.logger.error("Workload distribution failed", error=str(e))
            return {}

# Factory functions
def create_communication_protocol(agent_id: str, **kwargs) -> AgentCommunicationProtocol:
    """Create agent communication protocol"""
    return AgentCommunicationProtocol(agent_id, **kwargs)

def create_distributed_consensus(agent_id: str, 
                               protocol: AgentCommunicationProtocol, 
                               **kwargs) -> DistributedConsensus:
    """Create distributed consensus manager"""
    return DistributedConsensus(agent_id, protocol, **kwargs)

def create_swarm_intelligence(agent_id: str,
                            protocol: AgentCommunicationProtocol,
                            memory_manager: PersistentMemoryManager,
                            **kwargs) -> SwarmIntelligence:
    """Create swarm intelligence engine"""
    return SwarmIntelligence(agent_id, protocol, memory_manager, **kwargs)
