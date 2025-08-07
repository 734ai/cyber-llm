"""
Server-First Persistent Agent Architecture
Designed for continuous server operation with persistent memory and reasoning

Author: Cyber-LLM Development Team
Date: August 6, 2025
Version: 2.0.0
"""

import asyncio
import json
import logging
import sqlite3
import uuid
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from pathlib import Path
import pickle
import aiohttp
import websockets
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

# Server and networking imports
from aiohttp import web, WSMsgType
import ssl
import socket
from urllib.parse import urlparse

# Import our cognitive system
from cognitive.persistent_reasoning_system import (
    PersistentCognitiveSystem,
    create_persistent_cognitive_system,
    ReasoningType,
    MemoryType,
    MemoryEntry
)

@dataclass
class AgentSession:
    """Represents a persistent agent session"""
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str = ""
    start_time: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    status: str = "active"  # active, suspended, terminated
    memory_context: Dict[str, Any] = field(default_factory=dict)
    active_tasks: List[str] = field(default_factory=list)
    reasoning_chains: List[str] = field(default_factory=list)
    strategic_plans: List[str] = field(default_factory=list)

@dataclass
class ServerConfiguration:
    """Server configuration for persistent operation"""
    host: str = "0.0.0.0"
    port: int = 8080
    ssl_cert: Optional[str] = None
    ssl_key: Optional[str] = None
    max_connections: int = 1000
    session_timeout: int = 86400  # 24 hours
    memory_backup_interval: int = 3600  # 1 hour
    reasoning_persistence: bool = True
    distributed_mode: bool = False
    cluster_nodes: List[str] = field(default_factory=list)

class PersistentAgentServer:
    """Server-first agent architecture with persistent memory and reasoning"""
    
    def __init__(self, config: ServerConfiguration, db_path: str = "data/server_agent_system.db"):
        self.config = config
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Core components
        self.cognitive_system = PersistentCognitiveSystem(str(self.db_path))
        self.logger = logging.getLogger("persistent_agent_server")
        
        # Server state
        self.active_sessions = {}
        self.agent_registry = {}
        self.task_queue = asyncio.Queue()
        self.websocket_connections = set()
        
        # Background processes
        self.executor = ThreadPoolExecutor(max_workers=mp.cpu_count())
        self.background_tasks = []
        self.server_running = False
        
        # Initialize server database
        self._init_server_database()
        
        # Load existing sessions
        asyncio.create_task(self._load_persistent_sessions())
    
    def _init_server_database(self):
        """Initialize server-specific database tables"""
        
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        
        # Agent sessions table
        conn.execute("""
        CREATE TABLE IF NOT EXISTS agent_sessions (
            session_id TEXT PRIMARY KEY,
            agent_id TEXT NOT NULL,
            start_time REAL NOT NULL,
            last_activity REAL NOT NULL,
            status TEXT NOT NULL,
            memory_context BLOB NOT NULL,
            active_tasks BLOB NOT NULL,
            reasoning_chains BLOB NOT NULL,
            strategic_plans BLOB NOT NULL
        )
        """)
        
        # Server state table
        conn.execute("""
        CREATE TABLE IF NOT EXISTS server_state (
            key TEXT PRIMARY KEY,
            value BLOB NOT NULL,
            updated_at REAL NOT NULL
        )
        """)
        
        # Task queue table (for persistence across restarts)
        conn.execute("""
        CREATE TABLE IF NOT EXISTS persistent_tasks (
            task_id TEXT PRIMARY KEY,
            agent_id TEXT NOT NULL,
            task_type TEXT NOT NULL,
            task_data BLOB NOT NULL,
            priority INTEGER NOT NULL,
            created_at REAL NOT NULL,
            scheduled_at REAL,
            status TEXT NOT NULL
        )
        """)
        
        conn.commit()
        conn.close()
        
        self.logger.info("Server database initialized")
    
    async def start_server(self):
        """Start the persistent agent server"""
        
        self.server_running = True
        
        # Start background processes
        self._start_background_processes()
        
        # Create HTTP application
        app = web.Application()
        
        # Add routes
        self._setup_routes(app)
        
        # Configure SSL if certificates are provided
        ssl_context = None
        if self.config.ssl_cert and self.config.ssl_key:
            ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
            ssl_context.load_cert_chain(self.config.ssl_cert, self.config.ssl_key)
        
        # Start HTTP server
        runner = web.AppRunner(app)
        await runner.setup()
        
        site = web.TCPSite(runner, self.config.host, self.config.port, ssl_context=ssl_context)
        await site.start()
        
        self.logger.info(f"Persistent Agent Server started on {self.config.host}:{self.config.port}")
        
        # Keep server running
        try:
            while self.server_running:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            self.logger.info("Shutdown signal received")
        finally:
            await self.shutdown()
    
    def _setup_routes(self, app: web.Application):
        """Setup HTTP routes for the server"""
        
        # API routes
        app.router.add_get('/health', self.health_check)
        app.router.add_post('/agents/create', self.create_agent)
        app.router.add_get('/agents/{agent_id}', self.get_agent_status)
        app.router.add_post('/agents/{agent_id}/task', self.submit_task)
        app.router.add_get('/agents/{agent_id}/memory', self.get_agent_memory)
        app.router.add_post('/agents/{agent_id}/reasoning', self.start_reasoning_chain)
        app.router.add_post('/agents/{agent_id}/planning', self.create_strategic_plan)
        
        # WebSocket endpoint
        app.router.add_get('/ws', self.websocket_handler)
        
        # Administrative routes
        app.router.add_get('/admin/sessions', self.list_sessions)
        app.router.add_get('/admin/stats', self.get_server_stats)
        app.router.add_post('/admin/backup', self.backup_memory)
        app.router.add_post('/admin/restore', self.restore_memory)
    
    async def health_check(self, request):
        """Health check endpoint"""
        
        return web.json_response({
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "active_sessions": len(self.active_sessions),
            "server_uptime": (datetime.now() - self.server_start_time).total_seconds() if hasattr(self, 'server_start_time') else 0,
            "memory_stats": await self._get_memory_stats()
        })
    
    async def create_agent(self, request):
        """Create a new persistent agent"""
        
        try:
            data = await request.json()
            agent_id = data.get('agent_id', str(uuid.uuid4()))
            agent_type = data.get('type', 'general')
            
            # Create agent session
            session = AgentSession(
                agent_id=agent_id,
                memory_context={
                    'agent_type': agent_type,
                    'capabilities': data.get('capabilities', []),
                    'configuration': data.get('configuration', {})
                }
            )
            
            # Register agent
            self.active_sessions[session.session_id] = session
            self.agent_registry[agent_id] = session.session_id
            
            # Persist session
            await self._persist_session(session)
            
            # Initialize agent memory
            await self._initialize_agent_memory(session, data.get('initial_memory', {}))
            
            self.logger.info(f"Created persistent agent: {agent_id}")
            
            return web.json_response({
                "status": "success",
                "agent_id": agent_id,
                "session_id": session.session_id,
                "message": "Agent created successfully"
            })
            
        except Exception as e:
            self.logger.error(f"Error creating agent: {e}")
            return web.json_response({
                "status": "error",
                "message": str(e)
            }, status=500)
    
    async def submit_task(self, request):
        """Submit a task to an agent"""
        
        try:
            agent_id = request.match_info['agent_id']
            data = await request.json()
            
            if agent_id not in self.agent_registry:
                return web.json_response({
                    "status": "error",
                    "message": "Agent not found"
                }, status=404)
            
            # Create task
            task_id = str(uuid.uuid4())
            task = {
                "task_id": task_id,
                "agent_id": agent_id,
                "task_type": data.get('type', 'general'),
                "task_data": data.get('data', {}),
                "priority": data.get('priority', 5),
                "created_at": datetime.now().timestamp(),
                "status": "queued"
            }
            
            # Add to queue
            await self.task_queue.put(task)
            
            # Update agent session
            session_id = self.agent_registry[agent_id]
            session = self.active_sessions[session_id]
            session.active_tasks.append(task_id)
            session.last_activity = datetime.now()
            
            # Persist task
            await self._persist_task(task)
            
            self.logger.info(f"Task {task_id} submitted to agent {agent_id}")
            
            return web.json_response({
                "status": "success",
                "task_id": task_id,
                "message": "Task submitted successfully"
            })
            
        except Exception as e:
            self.logger.error(f"Error submitting task: {e}")
            return web.json_response({
                "status": "error",
                "message": str(e)
            }, status=500)
    
    async def get_agent_memory(self, request):
        """Get agent's memory state"""
        
        try:
            agent_id = request.match_info['agent_id']
            
            if agent_id not in self.agent_registry:
                return web.json_response({
                    "status": "error",
                    "message": "Agent not found"
                }, status=404)
            
            # Search agent's memories
            memories = await self.cognitive_system.memory_manager.search_memories(
                query=f"agent:{agent_id}",
                limit=50
            )
            
            memory_data = []
            for memory in memories:
                memory_data.append({
                    "memory_id": memory.memory_id,
                    "type": memory.memory_type.value,
                    "timestamp": memory.timestamp.isoformat(),
                    "importance": memory.importance,
                    "access_count": memory.access_count,
                    "tags": list(memory.tags)
                })
            
            return web.json_response({
                "status": "success",
                "agent_id": agent_id,
                "memory_count": len(memory_data),
                "memories": memory_data
            })
            
        except Exception as e:
            self.logger.error(f"Error getting agent memory: {e}")
            return web.json_response({
                "status": "error",
                "message": str(e)
            }, status=500)
    
    async def start_reasoning_chain(self, request):
        """Start a reasoning chain for an agent"""
        
        try:
            agent_id = request.match_info['agent_id']
            data = await request.json()
            
            if agent_id not in self.agent_registry:
                return web.json_response({
                    "status": "error",
                    "message": "Agent not found"
                }, status=404)
            
            # Start reasoning chain
            chain_id = await self.cognitive_system.reasoning_engine.start_reasoning_chain(
                topic=data.get('topic', 'Agent Reasoning'),
                goal=data.get('goal', 'Complete reasoning task'),
                reasoning_type=ReasoningType(data.get('reasoning_type', 'deductive'))
            )
            
            # Update agent session
            session_id = self.agent_registry[agent_id]
            session = self.active_sessions[session_id]
            session.reasoning_chains.append(chain_id)
            session.last_activity = datetime.now()
            
            await self._persist_session(session)
            
            self.logger.info(f"Started reasoning chain {chain_id} for agent {agent_id}")
            
            return web.json_response({
                "status": "success",
                "chain_id": chain_id,
                "agent_id": agent_id,
                "message": "Reasoning chain started"
            })
            
        except Exception as e:
            self.logger.error(f"Error starting reasoning chain: {e}")
            return web.json_response({
                "status": "error",
                "message": str(e)
            }, status=500)
    
    async def create_strategic_plan(self, request):
        """Create a strategic plan for an agent"""
        
        try:
            agent_id = request.match_info['agent_id']
            data = await request.json()
            
            if agent_id not in self.agent_registry:
                return web.json_response({
                    "status": "error",
                    "message": "Agent not found"
                }, status=404)
            
            # Create strategic plan
            plan_id = await self.cognitive_system.strategic_planner.create_strategic_plan(
                title=data.get('title', f'Strategic Plan for {agent_id}'),
                primary_goal=data.get('primary_goal', 'Complete strategic objectives'),
                template_type=data.get('template_type', 'cybersecurity_assessment')
            )
            
            # Update agent session
            session_id = self.agent_registry[agent_id]
            session = self.active_sessions[session_id]
            session.strategic_plans.append(plan_id)
            session.last_activity = datetime.now()
            
            await self._persist_session(session)
            
            self.logger.info(f"Created strategic plan {plan_id} for agent {agent_id}")
            
            return web.json_response({
                "status": "success",
                "plan_id": plan_id,
                "agent_id": agent_id,
                "message": "Strategic plan created"
            })
            
        except Exception as e:
            self.logger.error(f"Error creating strategic plan: {e}")
            return web.json_response({
                "status": "error",
                "message": str(e)
            }, status=500)
    
    async def websocket_handler(self, request):
        """WebSocket handler for real-time communication"""
        
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        
        self.websocket_connections.add(ws)
        
        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    await self._handle_websocket_message(ws, data)
                elif msg.type == WSMsgType.ERROR:
                    self.logger.error(f'WebSocket error: {ws.exception()}')
        
        finally:
            self.websocket_connections.discard(ws)
        
        return ws
    
    async def _handle_websocket_message(self, ws: web.WebSocketResponse, data: Dict[str, Any]):
        """Handle incoming WebSocket messages"""
        
        try:
            message_type = data.get('type')
            
            if message_type == 'subscribe_agent':
                agent_id = data.get('agent_id')
                # Subscribe to agent updates
                response = {
                    "type": "subscription_confirmed",
                    "agent_id": agent_id,
                    "timestamp": datetime.now().isoformat()
                }
                await ws.send_str(json.dumps(response))
            
            elif message_type == 'agent_command':
                agent_id = data.get('agent_id')
                command = data.get('command')
                
                # Process agent command
                result = await self._process_agent_command(agent_id, command)
                
                response = {
                    "type": "command_result",
                    "agent_id": agent_id,
                    "result": result,
                    "timestamp": datetime.now().isoformat()
                }
                await ws.send_str(json.dumps(response))
            
        except Exception as e:
            error_response = {
                "type": "error",
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            }
            await ws.send_str(json.dumps(error_response))
    
    async def _process_agent_command(self, agent_id: str, command: Dict[str, Any]) -> Dict[str, Any]:
        """Process agent command via WebSocket"""
        
        if agent_id not in self.agent_registry:
            return {"status": "error", "message": "Agent not found"}
        
        command_type = command.get('type')
        
        if command_type == 'get_status':
            session_id = self.agent_registry[agent_id]
            session = self.active_sessions[session_id]
            
            return {
                "status": "success",
                "agent_status": session.status,
                "active_tasks": len(session.active_tasks),
                "reasoning_chains": len(session.reasoning_chains),
                "strategic_plans": len(session.strategic_plans),
                "last_activity": session.last_activity.isoformat()
            }
        
        elif command_type == 'add_memory':
            memory_content = command.get('content', {})
            importance = command.get('importance', 0.5)
            tags = set(command.get('tags', []))
            tags.add(f"agent:{agent_id}")
            
            memory_entry = MemoryEntry(
                memory_type=MemoryType(command.get('memory_type', 'episodic')),
                content=memory_content,
                importance=importance,
                tags=tags
            )
            
            memory_id = await self.cognitive_system.memory_manager.store_memory(memory_entry)
            
            return {
                "status": "success",
                "memory_id": memory_id,
                "message": "Memory added successfully"
            }
        
        else:
            return {"status": "error", "message": f"Unknown command type: {command_type}"}
    
    def _start_background_processes(self):
        """Start background processes for server operation"""
        
        # Task processing worker
        async def task_processor():
            while self.server_running:
                try:
                    # Get task from queue
                    task = await asyncio.wait_for(self.task_queue.get(), timeout=1.0)
                    
                    # Process task
                    await self._process_task(task)
                    
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    self.logger.error(f"Task processing error: {e}")
        
        # Session cleanup worker
        async def session_cleanup():
            while self.server_running:
                try:
                    await asyncio.sleep(300)  # Every 5 minutes
                    await self._cleanup_inactive_sessions()
                except Exception as e:
                    self.logger.error(f"Session cleanup error: {e}")
        
        # Memory backup worker
        async def memory_backup():
            while self.server_running:
                try:
                    await asyncio.sleep(self.config.memory_backup_interval)
                    await self._backup_memory_state()
                except Exception as e:
                    self.logger.error(f"Memory backup error: {e}")
        
        # Start background tasks
        self.background_tasks = [
            asyncio.create_task(task_processor()),
            asyncio.create_task(session_cleanup()),
            asyncio.create_task(memory_backup())
        ]
        
        self.server_start_time = datetime.now()
        self.logger.info("Background processes started")
    
    async def _process_task(self, task: Dict[str, Any]):
        """Process a queued task"""
        
        try:
            task_id = task['task_id']
            agent_id = task['agent_id']
            task_type = task['task_type']
            task_data = task['task_data']
            
            self.logger.info(f"Processing task {task_id} for agent {agent_id}")
            
            # Update task status
            task['status'] = 'processing'
            await self._persist_task(task)
            
            # Process based on task type
            result = None
            
            if task_type == 'reasoning':
                result = await self._process_reasoning_task(agent_id, task_data)
            elif task_type == 'memory_search':
                result = await self._process_memory_search_task(agent_id, task_data)
            elif task_type == 'strategic_planning':
                result = await self._process_strategic_planning_task(agent_id, task_data)
            elif task_type == 'scenario_analysis':
                result = await self._process_scenario_analysis_task(agent_id, task_data)
            else:
                result = {"status": "error", "message": f"Unknown task type: {task_type}"}
            
            # Update task with result
            task['status'] = 'completed' if result.get('status') == 'success' else 'failed'
            task['result'] = result
            task['completed_at'] = datetime.now().timestamp()
            
            await self._persist_task(task)
            
            # Notify via WebSocket if connections exist
            await self._broadcast_task_update(task)
            
            self.logger.info(f"Completed task {task_id}")
            
        except Exception as e:
            self.logger.error(f"Error processing task {task.get('task_id', 'unknown')}: {e}")
            
            # Mark task as failed
            task['status'] = 'failed'
            task['error'] = str(e)
            await self._persist_task(task)
    
    async def _process_reasoning_task(self, agent_id: str, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a reasoning task"""
        
        try:
            # Start reasoning chain
            chain_id = await self.cognitive_system.reasoning_engine.start_reasoning_chain(
                topic=task_data.get('topic', 'Agent Reasoning Task'),
                goal=task_data.get('goal', 'Complete reasoning'),
                reasoning_type=ReasoningType(task_data.get('reasoning_type', 'deductive'))
            )
            
            # Add reasoning steps
            for step_data in task_data.get('steps', []):
                await self.cognitive_system.reasoning_engine.add_reasoning_step(
                    chain_id,
                    step_data.get('premise', ''),
                    step_data.get('inference_rule', ''),
                    step_data.get('evidence', [])
                )
            
            # Complete reasoning
            chain = await self.cognitive_system.reasoning_engine.complete_reasoning_chain(chain_id)
            
            return {
                "status": "success",
                "chain_id": chain_id,
                "conclusion": chain.conclusion if chain else "Failed to complete reasoning",
                "confidence": chain.confidence if chain else 0.0
            }
            
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    async def _process_memory_search_task(self, agent_id: str, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a memory search task"""
        
        try:
            query = task_data.get('query', '')
            memory_types = [MemoryType(mt) for mt in task_data.get('memory_types', ['episodic'])]
            limit = task_data.get('limit', 10)
            
            memories = await self.cognitive_system.memory_manager.search_memories(
                query, memory_types, limit
            )
            
            results = []
            for memory in memories:
                results.append({
                    "memory_id": memory.memory_id,
                    "type": memory.memory_type.value,
                    "importance": memory.importance,
                    "timestamp": memory.timestamp.isoformat(),
                    "tags": list(memory.tags)
                })
            
            return {
                "status": "success",
                "query": query,
                "results_count": len(results),
                "results": results
            }
            
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    async def _process_strategic_planning_task(self, agent_id: str, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a strategic planning task"""
        
        try:
            plan_id = await self.cognitive_system.strategic_planner.create_strategic_plan(
                title=task_data.get('title', f'Strategic Plan for {agent_id}'),
                primary_goal=task_data.get('primary_goal', 'Complete objectives'),
                template_type=task_data.get('template_type', 'cybersecurity_assessment')
            )
            
            return {
                "status": "success",
                "plan_id": plan_id,
                "message": "Strategic plan created successfully"
            }
            
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    async def _process_scenario_analysis_task(self, agent_id: str, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a scenario analysis task"""
        
        try:
            scenario = task_data.get('scenario', {})
            scenario['agent_id'] = agent_id  # Tag with agent ID
            
            result = await self.cognitive_system.process_complex_scenario(scenario)
            
            return {
                "status": "success",
                "scenario_id": result.get("scenario_id"),
                "analysis_results": result.get("results", {}),
                "message": "Scenario analysis completed"
            }
            
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    async def _persist_session(self, session: AgentSession):
        """Persist agent session to database"""
        
        try:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            
            conn.execute("""
            INSERT OR REPLACE INTO agent_sessions
            (session_id, agent_id, start_time, last_activity, status,
             memory_context, active_tasks, reasoning_chains, strategic_plans)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                session.session_id,
                session.agent_id,
                session.start_time.timestamp(),
                session.last_activity.timestamp(),
                session.status,
                pickle.dumps(session.memory_context),
                pickle.dumps(session.active_tasks),
                pickle.dumps(session.reasoning_chains),
                pickle.dumps(session.strategic_plans)
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error persisting session {session.session_id}: {e}")
    
    async def _persist_task(self, task: Dict[str, Any]):
        """Persist task to database"""
        
        try:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            
            conn.execute("""
            INSERT OR REPLACE INTO persistent_tasks
            (task_id, agent_id, task_type, task_data, priority, 
             created_at, scheduled_at, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                task['task_id'],
                task['agent_id'],
                task['task_type'],
                pickle.dumps(task.get('task_data', {})),
                task.get('priority', 5),
                task.get('created_at', datetime.now().timestamp()),
                task.get('scheduled_at'),
                task.get('status', 'queued')
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error persisting task {task.get('task_id', 'unknown')}: {e}")
    
    async def _load_persistent_sessions(self):
        """Load persistent sessions from database"""
        
        try:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            cursor = conn.execute("SELECT * FROM agent_sessions WHERE status = 'active'")
            
            for row in cursor.fetchall():
                session = AgentSession(
                    session_id=row[0],
                    agent_id=row[1],
                    start_time=datetime.fromtimestamp(row[2]),
                    last_activity=datetime.fromtimestamp(row[3]),
                    status=row[4],
                    memory_context=pickle.loads(row[5]),
                    active_tasks=pickle.loads(row[6]),
                    reasoning_chains=pickle.loads(row[7]),
                    strategic_plans=pickle.loads(row[8])
                )
                
                self.active_sessions[session.session_id] = session
                self.agent_registry[session.agent_id] = session.session_id
            
            conn.close()
            
            self.logger.info(f"Loaded {len(self.active_sessions)} persistent sessions")
            
            # Load persistent tasks
            await self._load_persistent_tasks()
            
        except Exception as e:
            self.logger.error(f"Error loading persistent sessions: {e}")
    
    async def _load_persistent_tasks(self):
        """Load persistent tasks from database"""
        
        try:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            cursor = conn.execute("SELECT * FROM persistent_tasks WHERE status IN ('queued', 'processing')")
            
            for row in cursor.fetchall():
                task = {
                    'task_id': row[0],
                    'agent_id': row[1],
                    'task_type': row[2],
                    'task_data': pickle.loads(row[3]),
                    'priority': row[4],
                    'created_at': row[5],
                    'scheduled_at': row[6],
                    'status': 'queued'  # Reset to queued for restart
                }
                
                await self.task_queue.put(task)
            
            conn.close()
            
            self.logger.info(f"Loaded {self.task_queue.qsize()} persistent tasks")
            
        except Exception as e:
            self.logger.error(f"Error loading persistent tasks: {e}")
    
    async def _cleanup_inactive_sessions(self):
        """Clean up inactive sessions"""
        
        try:
            current_time = datetime.now()
            sessions_to_remove = []
            
            for session_id, session in self.active_sessions.items():
                # Check if session has exceeded timeout
                inactive_duration = (current_time - session.last_activity).total_seconds()
                
                if inactive_duration > self.config.session_timeout:
                    sessions_to_remove.append(session_id)
            
            # Remove inactive sessions
            for session_id in sessions_to_remove:
                session = self.active_sessions[session_id]
                session.status = 'terminated'
                
                # Persist final state
                await self._persist_session(session)
                
                # Remove from active tracking
                del self.active_sessions[session_id]
                if session.agent_id in self.agent_registry:
                    del self.agent_registry[session.agent_id]
                
                self.logger.info(f"Cleaned up inactive session: {session_id}")
            
        except Exception as e:
            self.logger.error(f"Error cleaning up sessions: {e}")
    
    async def _backup_memory_state(self):
        """Backup memory state to persistent storage"""
        
        try:
            # Create backup of current system state
            backup_data = {
                "timestamp": datetime.now().isoformat(),
                "active_sessions": len(self.active_sessions),
                "agent_registry": dict(self.agent_registry),
                "task_queue_size": self.task_queue.qsize(),
                "memory_stats": await self._get_memory_stats()
            }
            
            # Store backup data
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            conn.execute("""
            INSERT OR REPLACE INTO server_state (key, value, updated_at)
            VALUES (?, ?, ?)
            """, (
                "backup_state",
                pickle.dumps(backup_data),
                datetime.now().timestamp()
            ))
            
            conn.commit()
            conn.close()
            
            self.logger.debug("Memory state backup completed")
            
        except Exception as e:
            self.logger.error(f"Error backing up memory state: {e}")
    
    async def _get_memory_stats(self) -> Dict[str, Any]:
        """Get memory system statistics"""
        
        try:
            conn = self.cognitive_system.memory_manager.conn
            
            # Get memory counts by type
            cursor = conn.execute("""
            SELECT memory_type, COUNT(*) as count
            FROM memory_entries
            GROUP BY memory_type
            """)
            
            memory_counts = dict(cursor.fetchall())
            
            # Get reasoning chain count
            cursor = conn.execute("SELECT COUNT(*) FROM reasoning_chains")
            reasoning_count = cursor.fetchone()[0]
            
            # Get strategic plan count
            cursor = conn.execute("SELECT COUNT(*) FROM strategic_plans")
            plan_count = cursor.fetchone()[0]
            
            return {
                "memory_counts": memory_counts,
                "reasoning_chains": reasoning_count,
                "strategic_plans": plan_count,
                "working_memory_size": len(self.cognitive_system.memory_manager.working_memory),
                "memory_cache_size": len(self.cognitive_system.memory_manager.memory_cache)
            }
            
        except Exception as e:
            self.logger.error(f"Error getting memory stats: {e}")
            return {}
    
    async def _broadcast_task_update(self, task: Dict[str, Any]):
        """Broadcast task updates to WebSocket connections"""
        
        if not self.websocket_connections:
            return
        
        message = {
            "type": "task_update",
            "task_id": task['task_id'],
            "agent_id": task['agent_id'],
            "status": task['status'],
            "timestamp": datetime.now().isoformat()
        }
        
        # Send to all connected clients
        disconnected = []
        for ws in self.websocket_connections:
            try:
                await ws.send_str(json.dumps(message))
            except Exception:
                disconnected.append(ws)
        
        # Remove disconnected clients
        for ws in disconnected:
            self.websocket_connections.discard(ws)
    
    async def shutdown(self):
        """Graceful server shutdown"""
        
        self.logger.info("Shutting down persistent agent server...")
        
        # Stop server
        self.server_running = False
        
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self.background_tasks:
            await asyncio.gather(*self.background_tasks, return_exceptions=True)
        
        # Save all session states
        for session in self.active_sessions.values():
            session.status = 'suspended'
            await self._persist_session(session)
        
        # Final backup
        await self._backup_memory_state()
        
        # Close WebSocket connections
        for ws in self.websocket_connections:
            await ws.close()
        
        self.logger.info("Server shutdown complete")

# Configuration builder
def create_server_config(
    host: str = "0.0.0.0",
    port: int = 8080,
    ssl_cert: Optional[str] = None,
    ssl_key: Optional[str] = None,
    max_connections: int = 1000,
    session_timeout: int = 86400,
    distributed_mode: bool = False
) -> ServerConfiguration:
    """Create server configuration"""
    
    return ServerConfiguration(
        host=host,
        port=port,
        ssl_cert=ssl_cert,
        ssl_key=ssl_key,
        max_connections=max_connections,
        session_timeout=session_timeout,
        distributed_mode=distributed_mode
    )

# Main server factory
def create_persistent_agent_server(
    config: Optional[ServerConfiguration] = None,
    db_path: str = "data/server_agent_system.db"
) -> PersistentAgentServer:
    """Create persistent agent server"""
    
    if config is None:
        config = create_server_config()
    
    return PersistentAgentServer(config, db_path)

# Main execution for standalone server
if __name__ == "__main__":
    import sys
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Parse command line arguments
    host = sys.argv[1] if len(sys.argv) > 1 else "0.0.0.0"
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8080
    
    # Create server configuration
    config = create_server_config(host=host, port=port)
    
    # Create and start server
    server = create_persistent_agent_server(config)
    
    try:
        asyncio.run(server.start_server())
    except KeyboardInterrupt:
        print("Server stopped by user")
    except Exception as e:
        logging.error(f"Server error: {e}")
        sys.exit(1)
