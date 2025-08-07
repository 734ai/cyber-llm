"""
Persistent Cognitive System Configuration and Startup
Central configuration and initialization for the complete persistent cognitive architecture

Author: Cyber-LLM Development Team
Date: August 6, 2025  
Version: 2.0.0
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict
import yaml

# Configuration imports
from cognitive.persistent_reasoning_system import PersistentCognitiveSystem
from server.persistent_agent_server import PersistentAgentServer, create_server_config
from integration.persistent_multi_agent_integration import (
    PersistentMultiAgentSystem, create_persistent_multi_agent_system
)

@dataclass
class DatabaseConfiguration:
    """Database configuration settings"""
    cognitive_db_path: str = "data/cognitive_system.db"
    server_db_path: str = "data/server_agent_system.db"
    backup_enabled: bool = True
    backup_interval_hours: int = 24
    backup_retention_days: int = 30
    auto_vacuum: bool = True
    wal_mode: bool = True
    sync_mode: str = "NORMAL"  # OFF, NORMAL, FULL

@dataclass
class ServerConfiguration:
    """Server configuration settings"""
    enabled: bool = True
    host: str = "0.0.0.0"
    port: int = 8080
    ssl_enabled: bool = False
    ssl_cert_path: Optional[str] = None
    ssl_key_path: Optional[str] = None
    max_connections: int = 1000
    session_timeout_hours: int = 24
    memory_backup_interval_hours: int = 1
    distributed_mode: bool = False
    cluster_nodes: List[str] = field(default_factory=list)

@dataclass
class CognitiveConfiguration:
    """Cognitive system configuration"""
    memory_consolidation_enabled: bool = True
    memory_consolidation_interval_hours: int = 6
    memory_decay_enabled: bool = True
    memory_decay_rate: float = 0.1
    working_memory_capacity: int = 20
    reasoning_chain_timeout_minutes: int = 60
    strategic_planning_enabled: bool = True
    meta_cognitive_enabled: bool = True
    background_processing_enabled: bool = True
    inter_agent_coordination: bool = True

@dataclass
class LoggingConfiguration:
    """Logging configuration settings"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_enabled: bool = True
    file_path: str = "logs/persistent_cognitive_system.log"
    file_max_size_mb: int = 100
    file_backup_count: int = 5
    console_enabled: bool = True
    structured_logging: bool = False

@dataclass
class SecurityConfiguration:
    """Security configuration settings"""
    authentication_enabled: bool = False
    api_key_required: bool = False
    rate_limiting_enabled: bool = True
    rate_limit_per_minute: int = 100
    encryption_enabled: bool = False
    audit_logging: bool = True
    secure_memory_erasure: bool = True
    safety_agent_required: bool = True

@dataclass
class PerformanceConfiguration:
    """Performance configuration settings"""
    max_worker_threads: int = 8
    memory_cache_size_mb: int = 512
    query_timeout_seconds: int = 30
    batch_processing_enabled: bool = True
    batch_size: int = 100
    connection_pool_size: int = 20
    async_processing: bool = True
    optimization_level: str = "balanced"  # conservative, balanced, aggressive

@dataclass 
class PersistentCognitiveConfiguration:
    """Complete configuration for persistent cognitive system"""
    database: DatabaseConfiguration = field(default_factory=DatabaseConfiguration)
    server: ServerConfiguration = field(default_factory=ServerConfiguration)
    cognitive: CognitiveConfiguration = field(default_factory=CognitiveConfiguration)
    logging: LoggingConfiguration = field(default_factory=LoggingConfiguration)
    security: SecurityConfiguration = field(default_factory=SecurityConfiguration)
    performance: PerformanceConfiguration = field(default_factory=PerformanceConfiguration)
    
    system_name: str = "Persistent Cognitive Multi-Agent System"
    version: str = "2.0.0"
    environment: str = "development"  # development, staging, production
    debug_mode: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return asdict(self)
    
    def save_to_file(self, filepath: str):
        """Save configuration to YAML file"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'PersistentCognitiveConfiguration':
        """Load configuration from YAML file"""
        with open(filepath, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'PersistentCognitiveConfiguration':
        """Create configuration from dictionary"""
        
        # Extract nested configurations
        database_config = DatabaseConfiguration(**config_dict.get('database', {}))
        server_config = ServerConfiguration(**config_dict.get('server', {}))
        cognitive_config = CognitiveConfiguration(**config_dict.get('cognitive', {}))
        logging_config = LoggingConfiguration(**config_dict.get('logging', {}))
        security_config = SecurityConfiguration(**config_dict.get('security', {}))
        performance_config = PerformanceConfiguration(**config_dict.get('performance', {}))
        
        # Create main configuration
        return cls(
            database=database_config,
            server=server_config,
            cognitive=cognitive_config,
            logging=logging_config,
            security=security_config,
            performance=performance_config,
            system_name=config_dict.get('system_name', 'Persistent Cognitive Multi-Agent System'),
            version=config_dict.get('version', '2.0.0'),
            environment=config_dict.get('environment', 'development'),
            debug_mode=config_dict.get('debug_mode', False)
        )

class PersistentCognitiveSystemManager:
    """
    Manager class for the complete persistent cognitive system
    Handles configuration, initialization, startup, and shutdown
    """
    
    def __init__(self, config: Optional[PersistentCognitiveConfiguration] = None):
        
        self.config = config or PersistentCognitiveConfiguration()
        self.logger = None
        
        # System components
        self.cognitive_system: Optional[PersistentCognitiveSystem] = None
        self.agent_server: Optional[PersistentAgentServer] = None
        self.multi_agent_system: Optional[PersistentMultiAgentSystem] = None
        
        # System state
        self.is_initialized = False
        self.is_running = False
        self.startup_time: Optional[datetime] = None
        self.background_tasks: List[asyncio.Task] = []
        
        # Initialize logging first
        self._setup_logging()
        
        self.logger.info(f"Created {self.config.system_name} v{self.config.version}")
    
    def _setup_logging(self):
        """Setup logging based on configuration"""
        
        # Create logs directory
        if self.config.logging.file_enabled:
            Path(self.config.logging.file_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, self.config.logging.level.upper()))
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # Create formatter
        if self.config.logging.structured_logging:
            formatter = logging.Formatter(
                '{"timestamp":"%(asctime)s","name":"%(name)s","level":"%(levelname)s","message":"%(message)s"}'
            )
        else:
            formatter = logging.Formatter(self.config.logging.format)
        
        # Console handler
        if self.config.logging.console_enabled:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            root_logger.addHandler(console_handler)
        
        # File handler
        if self.config.logging.file_enabled:
            from logging.handlers import RotatingFileHandler
            
            file_handler = RotatingFileHandler(
                self.config.logging.file_path,
                maxBytes=self.config.logging.file_max_size_mb * 1024 * 1024,
                backupCount=self.config.logging.file_backup_count
            )
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
        
        # Get logger for this class
        self.logger = logging.getLogger("persistent_cognitive_manager")
        
        self.logger.info("Logging configured")
    
    def _setup_directories(self):
        """Setup required directories"""
        
        directories = [
            Path(self.config.database.cognitive_db_path).parent,
            Path(self.config.database.server_db_path).parent,
            "data/backups",
            "data/exports",
            "data/imports",
            "logs",
            "temp"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
        
        self.logger.debug("Directories setup complete")
    
    def _validate_configuration(self):
        """Validate configuration settings"""
        
        issues = []
        
        # Database validation
        if not self.config.database.cognitive_db_path:
            issues.append("Cognitive database path is required")
        
        if not self.config.database.server_db_path:
            issues.append("Server database path is required")
        
        # Server validation
        if self.config.server.enabled:
            if self.config.server.port < 1 or self.config.server.port > 65535:
                issues.append("Server port must be between 1 and 65535")
            
            if self.config.server.ssl_enabled:
                if not self.config.server.ssl_cert_path:
                    issues.append("SSL certificate path required when SSL is enabled")
                if not self.config.server.ssl_key_path:
                    issues.append("SSL key path required when SSL is enabled")
        
        # Performance validation
        if self.config.performance.max_worker_threads < 1:
            issues.append("Max worker threads must be at least 1")
        
        if self.config.performance.memory_cache_size_mb < 64:
            issues.append("Memory cache size should be at least 64MB")
        
        if issues:
            raise ValueError(f"Configuration validation failed: {'; '.join(issues)}")
        
        self.logger.info("Configuration validation passed")
    
    async def initialize(self):
        """Initialize all system components"""
        
        if self.is_initialized:
            self.logger.warning("System already initialized")
            return
        
        try:
            self.logger.info("Initializing persistent cognitive system...")
            
            # Validate configuration
            self._validate_configuration()
            
            # Setup directories
            self._setup_directories()
            
            # Initialize cognitive system
            await self._initialize_cognitive_system()
            
            # Initialize agent server if enabled
            if self.config.server.enabled:
                await self._initialize_agent_server()
            
            # Initialize multi-agent system
            await self._initialize_multi_agent_system()
            
            self.is_initialized = True
            self.logger.info("System initialization complete")
            
        except Exception as e:
            self.logger.error(f"System initialization failed: {e}")
            raise
    
    async def _initialize_cognitive_system(self):
        """Initialize the persistent cognitive system"""
        
        try:
            self.cognitive_system = PersistentCognitiveSystem(
                self.config.database.cognitive_db_path
            )
            
            # Configure system parameters
            if hasattr(self.cognitive_system, 'memory_manager'):
                self.cognitive_system.memory_manager.working_memory_capacity = (
                    self.config.cognitive.working_memory_capacity
                )
            
            await self.cognitive_system.initialize()
            
            self.logger.info("Cognitive system initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize cognitive system: {e}")
            raise
    
    async def _initialize_agent_server(self):
        """Initialize the agent server"""
        
        try:
            server_config = create_server_config(
                host=self.config.server.host,
                port=self.config.server.port,
                ssl_cert=self.config.server.ssl_cert_path,
                ssl_key=self.config.server.ssl_key_path,
                max_connections=self.config.server.max_connections,
                session_timeout=self.config.server.session_timeout_hours * 3600,
                distributed_mode=self.config.server.distributed_mode
            )
            
            self.agent_server = PersistentAgentServer(
                server_config, 
                self.config.database.server_db_path
            )
            
            self.logger.info("Agent server initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize agent server: {e}")
            raise
    
    async def _initialize_multi_agent_system(self):
        """Initialize the multi-agent system"""
        
        try:
            server_config = None
            if self.config.server.enabled:
                server_config = create_server_config(
                    host=self.config.server.host,
                    port=self.config.server.port + 1,  # Use next port for multi-agent server
                    max_connections=self.config.server.max_connections
                )
            
            self.multi_agent_system = PersistentMultiAgentSystem(
                cognitive_db_path=self.config.database.cognitive_db_path,
                server_config=server_config
            )
            
            await self.multi_agent_system.initialize_system()
            
            self.logger.info("Multi-agent system initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize multi-agent system: {e}")
            raise
    
    async def start(self):
        """Start the system"""
        
        if not self.is_initialized:
            await self.initialize()
        
        if self.is_running:
            self.logger.warning("System already running")
            return
        
        try:
            self.logger.info("Starting persistent cognitive system...")
            self.startup_time = datetime.now()
            
            # Start background tasks
            await self._start_background_tasks()
            
            # Start agent server
            if self.agent_server:
                server_task = asyncio.create_task(self.agent_server.start_server())
                self.background_tasks.append(server_task)
            
            self.is_running = True
            
            self.logger.info(f"System started successfully on {self.startup_time}")
            
            # Print startup information
            await self._print_startup_info()
            
        except Exception as e:
            self.logger.error(f"System startup failed: {e}")
            await self.shutdown()
            raise
    
    async def _start_background_tasks(self):
        """Start background maintenance tasks"""
        
        # Database backup task
        if self.config.database.backup_enabled:
            backup_task = asyncio.create_task(self._backup_worker())
            self.background_tasks.append(backup_task)
        
        # System monitoring task
        monitor_task = asyncio.create_task(self._monitoring_worker())
        self.background_tasks.append(monitor_task)
        
        # Performance optimization task
        if self.config.performance.optimization_level != "conservative":
            optimization_task = asyncio.create_task(self._optimization_worker())
            self.background_tasks.append(optimization_task)
        
        self.logger.info(f"Started {len(self.background_tasks)} background tasks")
    
    async def _backup_worker(self):
        """Background database backup worker"""
        
        interval = self.config.database.backup_interval_hours * 3600
        
        while self.is_running:
            try:
                await asyncio.sleep(interval)
                await self._create_system_backup()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Backup worker error: {e}")
    
    async def _monitoring_worker(self):
        """Background system monitoring worker"""
        
        while self.is_running:
            try:
                await asyncio.sleep(300)  # Every 5 minutes
                await self._log_system_metrics()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Monitoring worker error: {e}")
    
    async def _optimization_worker(self):
        """Background performance optimization worker"""
        
        while self.is_running:
            try:
                await asyncio.sleep(1800)  # Every 30 minutes
                await self._optimize_system_performance()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Optimization worker error: {e}")
    
    async def _create_system_backup(self):
        """Create a complete system backup"""
        
        try:
            backup_dir = Path("data/backups") / datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            # Backup databases
            if self.cognitive_system:
                cognitive_backup = backup_dir / "cognitive_system.db"
                # Simple file copy for SQLite
                import shutil
                shutil.copy2(self.config.database.cognitive_db_path, cognitive_backup)
            
            if self.agent_server:
                server_backup = backup_dir / "server_system.db" 
                import shutil
                shutil.copy2(self.config.database.server_db_path, server_backup)
            
            # Backup configuration
            config_backup = backup_dir / "system_config.yaml"
            self.config.save_to_file(str(config_backup))
            
            self.logger.info(f"System backup created: {backup_dir}")
            
            # Cleanup old backups
            await self._cleanup_old_backups()
            
        except Exception as e:
            self.logger.error(f"Backup creation failed: {e}")
    
    async def _cleanup_old_backups(self):
        """Clean up old backup files"""
        
        try:
            backup_base = Path("data/backups")
            if not backup_base.exists():
                return
            
            retention_days = self.config.database.backup_retention_days
            cutoff_time = datetime.now() - timedelta(days=retention_days)
            
            for backup_dir in backup_base.iterdir():
                if backup_dir.is_dir():
                    dir_time = datetime.fromtimestamp(backup_dir.stat().st_mtime)
                    if dir_time < cutoff_time:
                        import shutil
                        shutil.rmtree(backup_dir)
                        self.logger.debug(f"Removed old backup: {backup_dir}")
            
        except Exception as e:
            self.logger.error(f"Backup cleanup failed: {e}")
    
    async def _log_system_metrics(self):
        """Log system performance metrics"""
        
        try:
            metrics = await self.get_system_metrics()
            
            self.logger.info(f"System Metrics - "
                           f"Uptime: {metrics.get('uptime_hours', 0):.1f}h, "
                           f"Memory: {metrics.get('memory_usage_mb', 0):.1f}MB, "
                           f"Active Sessions: {metrics.get('active_sessions', 0)}")
            
        except Exception as e:
            self.logger.error(f"Metrics logging failed: {e}")
    
    async def _optimize_system_performance(self):
        """Optimize system performance"""
        
        try:
            # Database optimization
            if self.cognitive_system:
                # Vacuum databases periodically
                if self.config.database.auto_vacuum:
                    # This would be implemented in the cognitive system
                    pass
            
            # Memory optimization
            if self.multi_agent_system:
                # Trigger memory consolidation
                await self.multi_agent_system.cognitive_system.memory_manager.consolidate_memories()
            
            self.logger.debug("System optimization completed")
            
        except Exception as e:
            self.logger.error(f"System optimization failed: {e}")
    
    async def _print_startup_info(self):
        """Print system startup information"""
        
        info_lines = [
            f"",
            f"{'='*60}",
            f"{self.config.system_name} v{self.config.version}",
            f"{'='*60}",
            f"Environment: {self.config.environment}",
            f"Started: {self.startup_time}",
            f"",
            f"Components:",
        ]
        
        if self.cognitive_system:
            info_lines.append(f"  ✓ Persistent Cognitive System")
        
        if self.agent_server:
            info_lines.append(f"  ✓ Agent Server (http://{self.config.server.host}:{self.config.server.port})")
        
        if self.multi_agent_system:
            info_lines.append(f"  ✓ Multi-Agent System")
        
        info_lines.extend([
            f"",
            f"Database:",
            f"  • Cognitive DB: {self.config.database.cognitive_db_path}",
            f"  • Server DB: {self.config.database.server_db_path}",
            f"",
            f"Background Tasks: {len(self.background_tasks)}",
            f"",
            f"System ready for cognitive operations!",
            f"{'='*60}",
            f""
        ])
        
        for line in info_lines:
            print(line)
    
    async def get_system_metrics(self) -> Dict[str, Any]:
        """Get comprehensive system metrics"""
        
        metrics = {
            "system_name": self.config.system_name,
            "version": self.config.version,
            "environment": self.config.environment,
            "is_running": self.is_running,
            "startup_time": self.startup_time.isoformat() if self.startup_time else None,
        }
        
        if self.startup_time:
            uptime = datetime.now() - self.startup_time
            metrics["uptime_hours"] = uptime.total_seconds() / 3600
        
        # Memory metrics
        import psutil
        process = psutil.Process()
        metrics["memory_usage_mb"] = process.memory_info().rss / 1024 / 1024
        metrics["cpu_percent"] = process.cpu_percent()
        
        # Component metrics
        if self.multi_agent_system:
            system_status = await self.multi_agent_system.get_system_status()
            metrics.update(system_status)
        
        # Background task metrics
        metrics["background_tasks"] = len(self.background_tasks)
        metrics["background_tasks_running"] = sum(1 for task in self.background_tasks if not task.done())
        
        return metrics
    
    async def run_scenario(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Run a scenario through the system"""
        
        if not self.is_running:
            raise RuntimeError("System is not running")
        
        if self.multi_agent_system:
            return await self.multi_agent_system.run_cognitive_scenario(scenario)
        else:
            raise RuntimeError("Multi-agent system not available")
    
    async def shutdown(self):
        """Graceful system shutdown"""
        
        if not self.is_running:
            return
        
        try:
            self.logger.info("Shutting down persistent cognitive system...")
            
            # Stop accepting new work
            self.is_running = False
            
            # Cancel background tasks
            for task in self.background_tasks:
                task.cancel()
            
            # Wait for tasks to complete
            if self.background_tasks:
                await asyncio.gather(*self.background_tasks, return_exceptions=True)
            
            # Shutdown components
            if self.multi_agent_system:
                await self.multi_agent_system.shutdown()
            
            if self.agent_server:
                await self.agent_server.shutdown()
            
            # Final backup
            if self.config.database.backup_enabled:
                await self._create_system_backup()
            
            uptime = datetime.now() - self.startup_time if self.startup_time else timedelta(0)
            self.logger.info(f"System shutdown complete. Uptime: {uptime}")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")

# Configuration templates
def create_development_config() -> PersistentCognitiveConfiguration:
    """Create development configuration"""
    
    config = PersistentCognitiveConfiguration()
    config.environment = "development"
    config.debug_mode = True
    config.logging.level = "DEBUG"
    config.database.backup_interval_hours = 1
    config.server.port = 8080
    
    return config

def create_production_config() -> PersistentCognitiveConfiguration:
    """Create production configuration"""
    
    config = PersistentCognitiveConfiguration()
    config.environment = "production"
    config.debug_mode = False
    config.logging.level = "INFO"
    config.logging.structured_logging = True
    config.database.backup_interval_hours = 24
    config.database.backup_retention_days = 90
    config.server.port = 443
    config.server.ssl_enabled = True
    config.security.authentication_enabled = True
    config.security.rate_limiting_enabled = True
    config.performance.optimization_level = "aggressive"
    
    return config

# CLI interface
async def main():
    """Main entry point for the system"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Persistent Cognitive Multi-Agent System")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--environment", choices=["development", "staging", "production"], 
                       default="development", help="Environment type")
    parser.add_argument("--port", type=int, default=8080, help="Server port")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        config = PersistentCognitiveConfiguration.load_from_file(args.config)
    elif args.environment == "production":
        config = create_production_config()
    else:
        config = create_development_config()
    
    # Override with command line arguments
    if args.port:
        config.server.port = args.port
    if args.host:
        config.server.host = args.host
    if args.debug:
        config.debug_mode = True
        config.logging.level = "DEBUG"
    
    config.environment = args.environment
    
    # Create and start system
    system_manager = PersistentCognitiveSystemManager(config)
    
    try:
        await system_manager.start()
        
        # Keep system running
        while system_manager.is_running:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
    except Exception as e:
        print(f"System error: {e}")
    finally:
        await system_manager.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
