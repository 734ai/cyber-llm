"""
Advanced Performance Optimization System for Cyber-LLM
GPU optimization, memory management, distributed inference, and scaling

Author: Muzan Sano <sanosensei36@gmail.com>
"""

import asyncio
import json
import logging
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import psutil
import gc
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict, deque
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

from ..utils.logging_system import CyberLLMLogger, CyberLLMError, ErrorCategory
from ..memory.persistent_memory import PersistentMemoryManager, MemoryType

class OptimizationTarget(Enum):
    """Performance optimization targets"""
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    MEMORY_EFFICIENCY = "memory_efficiency"
    GPU_UTILIZATION = "gpu_utilization"
    POWER_EFFICIENCY = "power_efficiency"
    COST_EFFICIENCY = "cost_efficiency"

class ResourceType(Enum):
    """Types of system resources"""
    CPU = "cpu"
    GPU = "gpu"
    MEMORY = "memory"
    STORAGE = "storage"
    NETWORK = "network"
    POWER = "power"

@dataclass
class PerformanceMetrics:
    """Performance metrics tracking"""
    timestamp: datetime
    
    # Latency metrics (milliseconds)
    inference_latency: float = 0.0
    preprocessing_latency: float = 0.0
    postprocessing_latency: float = 0.0
    total_latency: float = 0.0
    
    # Throughput metrics (requests/second)
    requests_per_second: float = 0.0
    tokens_per_second: float = 0.0
    
    # Resource utilization (0-1)
    cpu_utilization: float = 0.0
    gpu_utilization: float = 0.0
    memory_utilization: float = 0.0
    
    # Memory metrics (MB)
    cpu_memory_used: float = 0.0
    gpu_memory_used: float = 0.0
    gpu_memory_reserved: float = 0.0
    
    # Quality metrics
    accuracy: float = 0.0
    safety_score: float = 0.0
    
    # Cost metrics
    compute_cost: float = 0.0
    power_consumption: float = 0.0

@dataclass
class OptimizationConfiguration:
    """Performance optimization configuration"""
    optimization_targets: List[OptimizationTarget] = field(default_factory=lambda: [OptimizationTarget.LATENCY])
    
    # Model optimization
    use_mixed_precision: bool = True
    use_gradient_checkpointing: bool = False
    use_model_compilation: bool = True
    batch_size: int = 8
    max_sequence_length: int = 2048
    
    # Memory optimization
    enable_memory_pooling: bool = True
    memory_pool_size_mb: int = 4096
    enable_garbage_collection: bool = True
    gc_frequency: int = 100  # requests
    
    # Concurrency optimization
    max_concurrent_requests: int = 32
    request_queue_size: int = 128
    worker_processes: int = 4
    worker_threads_per_process: int = 8
    
    # Caching optimization
    enable_result_caching: bool = True
    cache_size_mb: int = 1024
    cache_ttl_seconds: int = 3600
    
    # GPU optimization
    enable_tensor_parallelism: bool = False
    enable_pipeline_parallelism: bool = False
    gpu_memory_fraction: float = 0.9
    
    # Monitoring
    metrics_collection_interval: float = 1.0  # seconds
    performance_logging_enabled: bool = True

class AdvancedPerformanceOptimizer:
    """Advanced performance optimization system with adaptive tuning"""
    
    def __init__(self, 
                 config: OptimizationConfiguration = None,
                 memory_manager: Optional[PersistentMemoryManager] = None,
                 logger: Optional[CyberLLMLogger] = None):
        
        self.config = config or OptimizationConfiguration()
        self.memory_manager = memory_manager
        self.logger = logger or CyberLLMLogger(name="performance_optimizer")
        
        # Performance tracking
        self.metrics_history = deque(maxlen=10000)
        self.current_metrics = PerformanceMetrics(timestamp=datetime.now())
        
        # Resource management
        self.resource_monitors = {}
        self.memory_pool = None
        self.result_cache = {}
        
        # Concurrency management
        self.request_queue = asyncio.Queue(maxsize=self.config.request_queue_size)
        self.worker_pool = None
        self.processing_semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)
        
        # Optimization state
        self.optimization_history = []
        self.current_optimization_strategy = {}
        self.adaptive_tuning_enabled = True
        
        # Monitoring and alerting
        self.performance_alerts = []
        self.monitoring_active = False
        self.monitoring_task = None
        
        # Initialize optimizer
        asyncio.create_task(self._initialize_optimizer())
        
        self.logger.info("Advanced Performance Optimizer initialized")
    
    async def _initialize_optimizer(self):
        """Initialize the performance optimization system"""
        
        try:
            # Initialize GPU optimization
            if torch.cuda.is_available():
                await self._initialize_gpu_optimization()
            
            # Initialize memory management
            await self._initialize_memory_management()
            
            # Initialize resource monitoring
            await self._initialize_resource_monitoring()
            
            # Initialize worker pools
            await self._initialize_worker_pools()
            
            # Start performance monitoring
            await self._start_performance_monitoring()
            
            self.logger.info("Performance optimizer initialization completed")
            
        except Exception as e:
            self.logger.error("Failed to initialize performance optimizer", error=str(e))
            raise CyberLLMError("Performance optimizer initialization failed", ErrorCategory.SYSTEM)
    
    async def _initialize_gpu_optimization(self):
        """Initialize GPU performance optimizations"""
        
        if not torch.cuda.is_available():
            self.logger.warning("CUDA not available, skipping GPU optimization")
            return
        
        try:
            # Set memory fraction
            torch.cuda.set_per_process_memory_fraction(self.config.gpu_memory_fraction)
            
            # Enable mixed precision if supported
            if self.config.use_mixed_precision and torch.cuda.is_bf16_supported():
                torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
                torch.backends.cudnn.allow_tf32 = True
            
            # Initialize memory pool
            if self.config.enable_memory_pooling:
                torch.cuda.empty_cache()
                
                # Custom memory pool (simplified)
                self.memory_pool = {
                    "allocated": 0,
                    "reserved": 0,
                    "max_reserved": self.config.memory_pool_size_mb * 1024 * 1024
                }
            
            # Initialize tensor parallelism if enabled
            if self.config.enable_tensor_parallelism:
                await self._setup_tensor_parallelism()
            
            self.logger.info("GPU optimization initialized",
                           devices=torch.cuda.device_count(),
                           memory_fraction=self.config.gpu_memory_fraction)
            
        except Exception as e:
            self.logger.error("GPU optimization initialization failed", error=str(e))
    
    async def _initialize_memory_management(self):
        """Initialize advanced memory management"""
        
        try:
            # Configure garbage collection
            if self.config.enable_garbage_collection:
                gc.set_threshold(700, 10, 10)  # Aggressive GC for memory efficiency
            
            # Initialize result cache
            if self.config.enable_result_caching:
                self.result_cache = {
                    "cache": {},
                    "access_times": {},
                    "max_size_mb": self.config.cache_size_mb,
                    "ttl_seconds": self.config.cache_ttl_seconds,
                    "current_size_mb": 0
                }
            
            # Set memory optimization flags
            if hasattr(torch.backends, 'opt_einsum'):
                torch.backends.opt_einsum.enabled = True
            
            self.logger.info("Memory management initialized")
            
        except Exception as e:
            self.logger.error("Memory management initialization failed", error=str(e))
    
    async def _initialize_resource_monitoring(self):
        """Initialize comprehensive resource monitoring"""
        
        try:
            # CPU monitoring
            self.resource_monitors['cpu'] = {
                'utilization_history': deque(maxlen=100),
                'temperature_history': deque(maxlen=100),
                'frequency_history': deque(maxlen=100)
            }
            
            # Memory monitoring  
            self.resource_monitors['memory'] = {
                'usage_history': deque(maxlen=100),
                'swap_history': deque(maxlen=100),
                'cache_history': deque(maxlen=100)
            }
            
            # GPU monitoring (if available)
            if torch.cuda.is_available():
                self.resource_monitors['gpu'] = {
                    'utilization_history': deque(maxlen=100),
                    'memory_history': deque(maxlen=100),
                    'temperature_history': deque(maxlen=100),
                    'power_history': deque(maxlen=100)
                }
            
            # Network monitoring
            self.resource_monitors['network'] = {
                'throughput_history': deque(maxlen=100),
                'latency_history': deque(maxlen=100),
                'error_history': deque(maxlen=100)
            }
            
            self.logger.info("Resource monitoring initialized")
            
        except Exception as e:
            self.logger.error("Resource monitoring initialization failed", error=str(e))
    
    async def _initialize_worker_pools(self):
        """Initialize worker pools for concurrent processing"""
        
        try:
            # Process pool for CPU-intensive tasks
            self.worker_pool = {
                'process_pool': ProcessPoolExecutor(max_workers=self.config.worker_processes),
                'thread_pool': ThreadPoolExecutor(max_workers=self.config.worker_threads_per_process * self.config.worker_processes)
            }
            
            self.logger.info("Worker pools initialized",
                           processes=self.config.worker_processes,
                           threads=self.config.worker_threads_per_process * self.config.worker_processes)
            
        except Exception as e:
            self.logger.error("Worker pool initialization failed", error=str(e))
    
    async def _start_performance_monitoring(self):
        """Start continuous performance monitoring"""
        
        self.monitoring_active = True
        self.monitoring_task = asyncio.create_task(self._performance_monitoring_loop())
        
        self.logger.info("Performance monitoring started")
    
    async def _performance_monitoring_loop(self):
        """Continuous performance monitoring loop"""
        
        while self.monitoring_active:
            try:
                # Collect current metrics
                current_metrics = await self._collect_performance_metrics()
                
                # Update metrics history
                self.metrics_history.append(current_metrics)
                self.current_metrics = current_metrics
                
                # Check for performance alerts
                await self._check_performance_alerts(current_metrics)
                
                # Adaptive optimization
                if self.adaptive_tuning_enabled:
                    await self._adaptive_performance_tuning(current_metrics)
                
                # Store metrics in memory system if available
                if self.memory_manager:
                    await self.memory_manager.store_memory(
                        memory_type=MemoryType.PROCEDURAL,
                        content=current_metrics.__dict__,
                        importance=0.3,
                        context_tags=["performance_metrics", "monitoring"],
                        agent_id="performance_optimizer"
                    )
                
                await asyncio.sleep(self.config.metrics_collection_interval)
                
            except Exception as e:
                self.logger.error("Performance monitoring error", error=str(e))
                await asyncio.sleep(5)  # Error recovery delay
    
    async def _collect_performance_metrics(self) -> PerformanceMetrics:
        """Collect comprehensive performance metrics"""
        
        metrics = PerformanceMetrics(timestamp=datetime.now())
        
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=None)
            metrics.cpu_utilization = cpu_percent / 100.0
            
            # Memory metrics
            memory_info = psutil.virtual_memory()
            metrics.memory_utilization = memory_info.percent / 100.0
            metrics.cpu_memory_used = memory_info.used / (1024 * 1024)  # MB
            
            # GPU metrics (if available)
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_stats()
                metrics.gpu_memory_used = gpu_memory.get('allocated_bytes.all.current', 0) / (1024 * 1024)
                metrics.gpu_memory_reserved = gpu_memory.get('reserved_bytes.all.current', 0) / (1024 * 1024)
                
                # GPU utilization (approximated)
                metrics.gpu_utilization = min(1.0, metrics.gpu_memory_used / (torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)))
            
            # Network metrics (simplified)
            network_io = psutil.net_io_counters()
            if hasattr(self, '_last_network_io'):
                bytes_sent_diff = network_io.bytes_sent - self._last_network_io.bytes_sent
                bytes_recv_diff = network_io.bytes_recv - self._last_network_io.bytes_recv
                time_diff = time.time() - self._last_network_time
                
                if time_diff > 0:
                    network_throughput = (bytes_sent_diff + bytes_recv_diff) / time_diff / (1024 * 1024)  # MB/s
            
            self._last_network_io = network_io
            self._last_network_time = time.time()
            
            # Update resource monitors
            self.resource_monitors['cpu']['utilization_history'].append(cpu_percent)
            self.resource_monitors['memory']['usage_history'].append(memory_info.percent)
            
            if torch.cuda.is_available():
                self.resource_monitors['gpu']['memory_history'].append(metrics.gpu_memory_used)
                self.resource_monitors['gpu']['utilization_history'].append(metrics.gpu_utilization * 100)
            
        except Exception as e:
            self.logger.error("Failed to collect performance metrics", error=str(e))
        
        return metrics
    
    async def optimize_inference_request(self, 
                                       request_data: Dict[str, Any],
                                       priority: int = 5) -> Dict[str, Any]:
        """Optimize and process an inference request"""
        
        request_id = request_data.get('request_id', f"req_{int(time.time())}")
        start_time = time.time()
        
        try:
            # Acquire processing semaphore
            async with self.processing_semaphore:
                
                # Check result cache first
                if self.config.enable_result_caching:
                    cached_result = await self._check_result_cache(request_data)
                    if cached_result:
                        self.logger.debug(f"Returning cached result for request: {request_id}")
                        return cached_result
                
                # Preprocess request
                preprocessing_start = time.time()
                preprocessed_data = await self._optimize_preprocessing(request_data)
                preprocessing_time = time.time() - preprocessing_start
                
                # Execute inference with optimizations
                inference_start = time.time()
                inference_result = await self._execute_optimized_inference(preprocessed_data)
                inference_time = time.time() - inference_start
                
                # Postprocess result
                postprocessing_start = time.time()
                final_result = await self._optimize_postprocessing(inference_result, request_data)
                postprocessing_time = time.time() - postprocessing_start
                
                total_time = time.time() - start_time
                
                # Update performance metrics
                await self._update_request_metrics(
                    total_time, preprocessing_time, inference_time, postprocessing_time
                )
                
                # Cache result if enabled
                if self.config.enable_result_caching:
                    await self._cache_result(request_data, final_result)
                
                # Add performance metadata
                final_result['performance'] = {
                    'total_latency_ms': total_time * 1000,
                    'preprocessing_latency_ms': preprocessing_time * 1000,
                    'inference_latency_ms': inference_time * 1000,
                    'postprocessing_latency_ms': postprocessing_time * 1000,
                    'cache_hit': False,
                    'optimization_applied': True
                }
                
                return final_result
                
        except Exception as e:
            self.logger.error(f"Request optimization failed: {request_id}", error=str(e))
            raise CyberLLMError("Request optimization failed", ErrorCategory.PROCESSING)
    
    async def _optimize_preprocessing(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize preprocessing with batching and parallelization"""
        
        try:
            # Batch similar requests for efficiency
            if 'batch_processing' in request_data:
                return await self._batch_preprocess(request_data)
            
            # Standard preprocessing with optimizations
            optimized_data = {
                'processed_input': request_data.get('input', ''),
                'max_length': min(request_data.get('max_length', 512), self.config.max_sequence_length),
                'optimization_flags': {
                    'use_mixed_precision': self.config.use_mixed_precision,
                    'gradient_checkpointing': self.config.use_gradient_checkpointing
                }
            }
            
            return optimized_data
            
        except Exception as e:
            self.logger.error("Preprocessing optimization failed", error=str(e))
            return request_data
    
    async def _execute_optimized_inference(self, preprocessed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute inference with performance optimizations"""
        
        try:
            # Apply memory optimizations
            if self.config.enable_garbage_collection and hasattr(self, '_request_count'):
                self._request_count += 1
                if self._request_count % self.config.gc_frequency == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            
            # Simulate optimized inference (in production, would call actual model)
            await asyncio.sleep(0.01)  # Simulate inference time
            
            result = {
                'output': f"Optimized inference result for: {preprocessed_data.get('processed_input', '')[:50]}...",
                'confidence': 0.95,
                'optimization_metrics': {
                    'memory_efficient': True,
                    'gpu_accelerated': torch.cuda.is_available(),
                    'mixed_precision': self.config.use_mixed_precision
                }
            }
            
            return result
            
        except Exception as e:
            self.logger.error("Optimized inference failed", error=str(e))
            raise
    
    async def _optimize_postprocessing(self, 
                                     inference_result: Dict[str, Any],
                                     original_request: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize postprocessing with parallel operations"""
        
        try:
            # Parallel postprocessing tasks
            tasks = []
            
            # Format output
            tasks.append(self._format_output(inference_result))
            
            # Apply safety filters
            tasks.append(self._apply_safety_filters(inference_result))
            
            # Generate metadata
            tasks.append(self._generate_response_metadata(inference_result, original_request))
            
            # Execute tasks in parallel
            formatted_output, safety_filtered, metadata = await asyncio.gather(*tasks)
            
            final_result = {
                'response': formatted_output,
                'safety_score': safety_filtered.get('safety_score', 1.0),
                'metadata': metadata,
                'request_id': original_request.get('request_id'),
                'timestamp': datetime.now().isoformat()
            }
            
            return final_result
            
        except Exception as e:
            self.logger.error("Postprocessing optimization failed", error=str(e))
            return inference_result
    
    async def _adaptive_performance_tuning(self, current_metrics: PerformanceMetrics):
        """Adaptive performance tuning based on current metrics"""
        
        try:
            # Analyze performance trends
            if len(self.metrics_history) < 10:
                return  # Need more data for adaptive tuning
            
            recent_metrics = list(self.metrics_history)[-10:]
            
            # Calculate performance trends
            latency_trend = self._calculate_trend([m.total_latency for m in recent_metrics])
            memory_trend = self._calculate_trend([m.memory_utilization for m in recent_metrics])
            gpu_trend = self._calculate_trend([m.gpu_utilization for m in recent_metrics])
            
            adaptations = []
            
            # Latency optimization
            if latency_trend > 0.1:  # Latency increasing
                if current_metrics.memory_utilization < 0.7:
                    # Increase batch size for better throughput
                    new_batch_size = min(self.config.batch_size * 2, 32)
                    if new_batch_size != self.config.batch_size:
                        self.config.batch_size = new_batch_size
                        adaptations.append(f"Increased batch size to {new_batch_size}")
                
                if not self.config.use_mixed_precision and torch.cuda.is_available():
                    self.config.use_mixed_precision = True
                    adaptations.append("Enabled mixed precision training")
            
            # Memory optimization
            if memory_trend > 0.1 and current_metrics.memory_utilization > 0.8:
                # Increase garbage collection frequency
                self.config.gc_frequency = max(self.config.gc_frequency // 2, 10)
                adaptations.append(f"Increased GC frequency to {self.config.gc_frequency}")
                
                # Reduce batch size if memory pressure
                if self.config.batch_size > 1:
                    self.config.batch_size = max(self.config.batch_size // 2, 1)
                    adaptations.append(f"Reduced batch size to {self.config.batch_size}")
            
            # GPU optimization
            if torch.cuda.is_available() and gpu_trend < -0.1:  # GPU underutilized
                if current_metrics.gpu_utilization < 0.5:
                    # Increase concurrent requests
                    new_max_concurrent = min(self.config.max_concurrent_requests + 8, 64)
                    if new_max_concurrent != self.config.max_concurrent_requests:
                        self.config.max_concurrent_requests = new_max_concurrent
                        self.processing_semaphore = asyncio.Semaphore(new_max_concurrent)
                        adaptations.append(f"Increased max concurrent requests to {new_max_concurrent}")
            
            # Log adaptations
            if adaptations:
                adaptation_record = {
                    'timestamp': datetime.now().isoformat(),
                    'adaptations': adaptations,
                    'trigger_metrics': {
                        'latency_trend': latency_trend,
                        'memory_trend': memory_trend,
                        'gpu_trend': gpu_trend
                    }
                }
                
                self.optimization_history.append(adaptation_record)
                
                if self.memory_manager:
                    await self.memory_manager.store_memory(
                        memory_type=MemoryType.PROCEDURAL,
                        content=adaptation_record,
                        importance=0.7,
                        context_tags=["adaptive_tuning", "performance_optimization"],
                        agent_id="performance_optimizer"
                    )
                
                self.logger.info("Applied adaptive optimizations", adaptations=adaptations)
        
        except Exception as e:
            self.logger.error("Adaptive tuning failed", error=str(e))
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend in performance metrics"""
        
        if len(values) < 2:
            return 0.0
        
        # Simple linear trend calculation
        x = np.arange(len(values))
        y = np.array(values)
        
        if np.std(y) == 0:
            return 0.0
        
        correlation = np.corrcoef(x, y)[0, 1]
        return correlation if not np.isnan(correlation) else 0.0
    
    def get_performance_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive performance dashboard data"""
        
        if not self.metrics_history:
            return {"status": "No metrics available"}
        
        recent_metrics = list(self.metrics_history)[-100:]  # Last 100 measurements
        
        return {
            "current_metrics": self.current_metrics.__dict__,
            "performance_trends": {
                "latency_trend": self._calculate_trend([m.total_latency for m in recent_metrics]),
                "throughput_trend": self._calculate_trend([m.requests_per_second for m in recent_metrics]),
                "memory_trend": self._calculate_trend([m.memory_utilization for m in recent_metrics]),
                "gpu_trend": self._calculate_trend([m.gpu_utilization for m in recent_metrics])
            },
            "optimization_config": {
                "batch_size": self.config.batch_size,
                "max_concurrent_requests": self.config.max_concurrent_requests,
                "gc_frequency": self.config.gc_frequency,
                "mixed_precision_enabled": self.config.use_mixed_precision,
                "caching_enabled": self.config.enable_result_caching
            },
            "system_resources": {
                "cpu_count": mp.cpu_count(),
                "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
                "total_memory_gb": psutil.virtual_memory().total / (1024**3),
                "available_memory_gb": psutil.virtual_memory().available / (1024**3)
            },
            "recent_adaptations": self.optimization_history[-10:] if self.optimization_history else [],
            "monitoring_status": {
                "active": self.monitoring_active,
                "metrics_collected": len(self.metrics_history),
                "adaptive_tuning_enabled": self.adaptive_tuning_enabled
            }
        }
    
    async def shutdown(self):
        """Graceful shutdown of performance optimizer"""
        
        self.logger.info("Shutting down performance optimizer")
        
        # Stop monitoring
        self.monitoring_active = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        # Shutdown worker pools
        if self.worker_pool:
            self.worker_pool['process_pool'].shutdown(wait=True)
            self.worker_pool['thread_pool'].shutdown(wait=True)
        
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Final garbage collection
        gc.collect()
        
        self.logger.info("Performance optimizer shutdown completed")

# Factory function
def create_performance_optimizer(config: OptimizationConfiguration = None, **kwargs) -> AdvancedPerformanceOptimizer:
    """Create advanced performance optimizer with configuration"""
    return AdvancedPerformanceOptimizer(config, **kwargs)
