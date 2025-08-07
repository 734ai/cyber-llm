"""
Universal Tool Integration Framework for Cyber-LLM
Plugin architecture and standardized API for external security tools

Author: Muzan Sano <sanosensei36@gmail.com>
"""

import asyncio
import json
import logging
import importlib
import inspect
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import yaml
import aiohttp
import docker
import subprocess

from ..utils.logging_system import CyberLLMLogger, CyberLLMError, ErrorCategory
from ..memory.persistent_memory import PersistentMemoryManager

class ToolType(Enum):
    """Types of security tools"""
    SCANNER = "scanner"
    ANALYZER = "analyzer"
    MONITOR = "monitor"
    FORENSICS = "forensics"
    THREAT_INTEL = "threat_intel"
    VULNERABILITY_MGMT = "vulnerability_mgmt"
    INCIDENT_RESPONSE = "incident_response"
    COMPLIANCE = "compliance"
    REPORTING = "reporting"
    AUTOMATION = "automation"

class IntegrationMethod(Enum):
    """Tool integration methods"""
    REST_API = "rest_api"
    CLI_WRAPPER = "cli_wrapper"
    PYTHON_LIBRARY = "python_library"
    DOCKER_CONTAINER = "docker_container"
    WEBHOOK = "webhook"
    SOCKET = "socket"
    DATABASE = "database"
    FILE_SYSTEM = "file_system"

class ToolStatus(Enum):
    """Tool availability status"""
    AVAILABLE = "available"
    BUSY = "busy"
    ERROR = "error"
    OFFLINE = "offline"
    MAINTENANCE = "maintenance"

@dataclass
class ToolCapability:
    """Tool capability definition"""
    capability_id: str
    name: str
    description: str
    
    # Input/Output specification
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    
    # Performance characteristics
    typical_execution_time: float  # seconds
    resource_requirements: Dict[str, float]
    
    # Reliability metrics
    success_rate: float = 0.95
    error_rate: float = 0.05
    
    # Dependencies
    required_credentials: List[str] = field(default_factory=list)
    required_permissions: List[str] = field(default_factory=list)

@dataclass
class ToolMetadata:
    """Comprehensive tool metadata"""
    tool_id: str
    name: str
    version: str
    vendor: str
    
    # Classification
    tool_type: ToolType
    integration_method: IntegrationMethod
    
    # Capabilities
    capabilities: List[ToolCapability]
    supported_formats: List[str]
    
    # Integration details
    endpoint_url: Optional[str] = None
    api_key_required: bool = False
    authentication_method: Optional[str] = None
    
    # Docker configuration (if applicable)
    docker_image: Optional[str] = None
    docker_config: Dict[str, Any] = field(default_factory=dict)
    
    # CLI configuration (if applicable)
    executable_path: Optional[str] = None
    command_template: Optional[str] = None
    
    # Status and monitoring
    status: ToolStatus = ToolStatus.OFFLINE
    last_health_check: Optional[datetime] = None
    health_check_interval: int = 300  # seconds
    
    # Usage statistics
    total_invocations: int = 0
    successful_invocations: int = 0
    average_response_time: float = 0.0

@dataclass
class ToolExecutionRequest:
    """Tool execution request"""
    request_id: str
    tool_id: str
    capability_id: str
    
    # Input data
    input_data: Dict[str, Any]
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Execution settings
    timeout: int = 300  # seconds
    priority: int = 5   # 1-10
    retry_count: int = 3
    
    # Context
    correlation_id: Optional[str] = None
    requested_by: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class ToolExecutionResult:
    """Tool execution result"""
    request_id: str
    tool_id: str
    capability_id: str
    
    # Results
    success: bool
    output_data: Dict[str, Any]
    error_message: Optional[str] = None
    
    # Performance metrics
    execution_time: float = 0.0
    resource_usage: Dict[str, float] = field(default_factory=dict)
    
    # Metadata
    executed_at: datetime = field(default_factory=datetime.now)
    tool_version: Optional[str] = None

class UniversalToolRegistry:
    """Central registry for all integrated tools"""
    
    def __init__(self, logger: Optional[CyberLLMLogger] = None):
        self.logger = logger or CyberLLMLogger(name="tool_registry")
        
        # Tool storage
        self.registered_tools = {}
        self.tool_instances = {}
        self.capability_index = {}  # capability_id -> tool_id mapping
        
        # Discovery and validation
        self.discovery_paths = []
        self.validation_rules = {}
        
        # Monitoring
        self.health_monitors = {}
        self.usage_statistics = {}
        
        self.logger.info("Universal Tool Registry initialized")
    
    async def register_tool(self, metadata: ToolMetadata) -> bool:
        """Register a new tool"""
        
        try:
            # Validate tool metadata
            if not await self._validate_tool_metadata(metadata):
                self.logger.error("Invalid tool metadata", tool_id=metadata.tool_id)
                return False
            
            # Check for conflicts
            if metadata.tool_id in self.registered_tools:
                self.logger.warning("Tool already registered", tool_id=metadata.tool_id)
                return False
            
            # Create tool instance
            tool_instance = await self._create_tool_instance(metadata)
            if not tool_instance:
                self.logger.error("Failed to create tool instance", tool_id=metadata.tool_id)
                return False
            
            # Perform health check
            health_status = await self._perform_health_check(tool_instance)
            metadata.status = ToolStatus.AVAILABLE if health_status else ToolStatus.ERROR
            metadata.last_health_check = datetime.now()
            
            # Register tool
            self.registered_tools[metadata.tool_id] = metadata
            self.tool_instances[metadata.tool_id] = tool_instance
            
            # Index capabilities
            for capability in metadata.capabilities:
                self.capability_index[capability.capability_id] = metadata.tool_id
            
            # Start health monitoring
            asyncio.create_task(self._monitor_tool_health(metadata.tool_id))
            
            self.logger.info("Tool registered successfully",
                           tool_id=metadata.tool_id,
                           name=metadata.name,
                           capabilities_count=len(metadata.capabilities))
            
            return True
            
        except Exception as e:
            self.logger.error("Tool registration failed",
                            tool_id=metadata.tool_id,
                            error=str(e))
            return False
    
    async def discover_tools(self, discovery_paths: List[str]) -> List[ToolMetadata]:
        """Discover tools from specified paths"""
        
        discovered_tools = []
        
        for path in discovery_paths:
            try:
                if Path(path).is_file() and path.endswith('.yaml'):
                    # Tool definition file
                    metadata = await self._load_tool_from_yaml(path)
                    if metadata:
                        discovered_tools.append(metadata)
                
                elif Path(path).is_dir():
                    # Directory with tool definitions
                    for yaml_file in Path(path).glob('*.yaml'):
                        metadata = await self._load_tool_from_yaml(str(yaml_file))
                        if metadata:
                            discovered_tools.append(metadata)
            
            except Exception as e:
                self.logger.error("Tool discovery failed for path",
                                path=path,
                                error=str(e))
        
        self.logger.info("Tool discovery completed",
                       discovered_count=len(discovered_tools))
        
        return discovered_tools
    
    async def get_tool_by_capability(self, capability_id: str) -> Optional[ToolMetadata]:
        """Get tool that provides specific capability"""
        
        tool_id = self.capability_index.get(capability_id)
        if tool_id:
            return self.registered_tools.get(tool_id)
        return None
    
    async def list_tools_by_type(self, tool_type: ToolType) -> List[ToolMetadata]:
        """List all tools of specified type"""
        
        return [tool for tool in self.registered_tools.values() 
                if tool.tool_type == tool_type and tool.status == ToolStatus.AVAILABLE]

class ToolExecutionEngine:
    """Engine for executing tools with advanced features"""
    
    def __init__(self, 
                 registry: UniversalToolRegistry,
                 memory_manager: PersistentMemoryManager,
                 logger: Optional[CyberLLMLogger] = None):
        
        self.registry = registry
        self.memory_manager = memory_manager
        self.logger = logger or CyberLLMLogger(name="tool_execution")
        
        # Execution management
        self.active_executions = {}
        self.execution_queue = asyncio.Queue()
        self.execution_history = []
        
        # Resource management
        self.resource_limits = {
            "max_concurrent_executions": 10,
            "max_memory_per_execution": 2048,  # MB
            "max_cpu_per_execution": 2.0       # cores
        }
        
        # Performance optimization
        self.execution_cache = {}
        self.load_balancing = True
        
        # Start execution worker
        asyncio.create_task(self._execution_worker())
        
        self.logger.info("Tool Execution Engine initialized")
    
    async def execute_tool(self, request: ToolExecutionRequest) -> ToolExecutionResult:
        """Execute a tool with specified capability"""
        
        try:
            self.logger.info("Tool execution requested",
                           request_id=request.request_id,
                           tool_id=request.tool_id,
                           capability=request.capability_id)
            
            # Get tool metadata
            tool_metadata = self.registry.registered_tools.get(request.tool_id)
            if not tool_metadata:
                return ToolExecutionResult(
                    request_id=request.request_id,
                    tool_id=request.tool_id,
                    capability_id=request.capability_id,
                    success=False,
                    output_data={},
                    error_message="Tool not found"
                )
            
            # Check tool availability
            if tool_metadata.status != ToolStatus.AVAILABLE:
                return ToolExecutionResult(
                    request_id=request.request_id,
                    tool_id=request.tool_id,
                    capability_id=request.capability_id,
                    success=False,
                    output_data={},
                    error_message=f"Tool not available: {tool_metadata.status.value}"
                )
            
            # Check cache for identical requests
            cache_key = self._generate_cache_key(request)
            if cache_key in self.execution_cache:
                cached_result = self.execution_cache[cache_key]
                if self._is_cache_valid(cached_result):
                    self.logger.info("Returning cached result", request_id=request.request_id)
                    return cached_result
            
            # Execute tool
            tool_instance = self.registry.tool_instances[request.tool_id]
            start_time = datetime.now()
            
            try:
                # Execute based on integration method
                if tool_metadata.integration_method == IntegrationMethod.REST_API:
                    result = await self._execute_rest_api(tool_instance, request)
                elif tool_metadata.integration_method == IntegrationMethod.CLI_WRAPPER:
                    result = await self._execute_cli_wrapper(tool_instance, request)
                elif tool_metadata.integration_method == IntegrationMethod.PYTHON_LIBRARY:
                    result = await self._execute_python_library(tool_instance, request)
                elif tool_metadata.integration_method == IntegrationMethod.DOCKER_CONTAINER:
                    result = await self._execute_docker_container(tool_instance, request)
                else:
                    raise NotImplementedError(f"Integration method {tool_metadata.integration_method} not implemented")
                
                execution_time = (datetime.now() - start_time).total_seconds()
                
                execution_result = ToolExecutionResult(
                    request_id=request.request_id,
                    tool_id=request.tool_id,
                    capability_id=request.capability_id,
                    success=True,
                    output_data=result,
                    execution_time=execution_time,
                    executed_at=start_time,
                    tool_version=tool_metadata.version
                )
                
                # Cache successful result
                self.execution_cache[cache_key] = execution_result
                
                # Update statistics
                self._update_execution_statistics(tool_metadata, execution_result)
                
                self.logger.info("Tool execution completed",
                               request_id=request.request_id,
                               execution_time=execution_time,
                               success=True)
                
                return execution_result
                
            except asyncio.TimeoutError:
                return ToolExecutionResult(
                    request_id=request.request_id,
                    tool_id=request.tool_id,
                    capability_id=request.capability_id,
                    success=False,
                    output_data={},
                    error_message="Execution timeout",
                    execution_time=(datetime.now() - start_time).total_seconds()
                )
            
        except Exception as e:
            self.logger.error("Tool execution failed",
                            request_id=request.request_id,
                            error=str(e))
            
            return ToolExecutionResult(
                request_id=request.request_id,
                tool_id=request.tool_id,
                capability_id=request.capability_id,
                success=False,
                output_data={},
                error_message=str(e)
            )
    
    async def _execute_rest_api(self, tool_instance: Any, request: ToolExecutionRequest) -> Dict[str, Any]:
        """Execute REST API based tool"""
        
        async with aiohttp.ClientSession() as session:
            headers = await self._get_api_headers(tool_instance, request)
            
            async with session.post(
                tool_instance.endpoint,
                json=request.input_data,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=request.timeout)
            ) as response:
                
                if response.status == 200:
                    return await response.json()
                else:
                    raise Exception(f"API call failed with status {response.status}: {await response.text()}")
    
    async def _execute_cli_wrapper(self, tool_instance: Any, request: ToolExecutionRequest) -> Dict[str, Any]:
        """Execute CLI wrapper based tool"""
        
        # Build command from template
        command = tool_instance.command_template.format(**request.input_data)
        
        # Execute command
        process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), 
                timeout=request.timeout
            )
            
            if process.returncode == 0:
                return {"stdout": stdout.decode(), "stderr": stderr.decode()}
            else:
                raise Exception(f"Command failed with return code {process.returncode}: {stderr.decode()}")
        
        finally:
            if process.returncode is None:
                process.terminate()
                await process.wait()

class PluginManager:
    """Manager for dynamic plugin loading and lifecycle"""
    
    def __init__(self, 
                 registry: UniversalToolRegistry,
                 logger: Optional[CyberLLMLogger] = None):
        
        self.registry = registry
        self.logger = logger or CyberLLMLogger(name="plugin_manager")
        
        # Plugin management
        self.loaded_plugins = {}
        self.plugin_hooks = {}
        self.plugin_dependencies = {}
        
        self.logger.info("Plugin Manager initialized")
    
    async def load_plugin(self, plugin_path: str) -> bool:
        """Dynamically load a plugin"""
        
        try:
            # Import plugin module
            spec = importlib.util.spec_from_file_location("plugin", plugin_path)
            plugin_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(plugin_module)
            
            # Validate plugin interface
            if not hasattr(plugin_module, 'PLUGIN_METADATA'):
                raise Exception("Plugin missing PLUGIN_METADATA")
            
            if not hasattr(plugin_module, 'initialize_plugin'):
                raise Exception("Plugin missing initialize_plugin function")
            
            # Initialize plugin
            plugin_metadata = plugin_module.PLUGIN_METADATA
            plugin_instance = await plugin_module.initialize_plugin()
            
            # Register with tool registry
            if hasattr(plugin_instance, 'get_tool_metadata'):
                tool_metadata = await plugin_instance.get_tool_metadata()
                await self.registry.register_tool(tool_metadata)
            
            # Store plugin
            plugin_id = plugin_metadata['id']
            self.loaded_plugins[plugin_id] = {
                'module': plugin_module,
                'instance': plugin_instance,
                'metadata': plugin_metadata,
                'path': plugin_path
            }
            
            self.logger.info("Plugin loaded successfully",
                           plugin_id=plugin_id,
                           name=plugin_metadata.get('name'))
            
            return True
            
        except Exception as e:
            self.logger.error("Plugin loading failed",
                            plugin_path=plugin_path,
                            error=str(e))
            return False

# Factory functions
def create_tool_registry(**kwargs) -> UniversalToolRegistry:
    """Create universal tool registry"""
    return UniversalToolRegistry(**kwargs)

def create_tool_execution_engine(registry: UniversalToolRegistry,
                               memory_manager: PersistentMemoryManager,
                               **kwargs) -> ToolExecutionEngine:
    """Create tool execution engine"""
    return ToolExecutionEngine(registry, memory_manager, **kwargs)

def create_plugin_manager(registry: UniversalToolRegistry,
                        **kwargs) -> PluginManager:
    """Create plugin manager"""
    return PluginManager(registry, **kwargs)
