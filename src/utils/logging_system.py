"""
Advanced Logging and Error Handling System for Cyber-LLM
Implements structured JSON logging, retry logic, and comprehensive error tracking
"""

import json
import logging
import time
import traceback
import functools
import asyncio
from typing import Dict, List, Any, Optional, Callable, Union
from datetime import datetime, timezone
from pathlib import Path
from enum import Enum
import uuid
import sys
import os
from contextlib import contextmanager

class LogLevel(Enum):
    """Logging levels for Cyber-LLM"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
    SECURITY = "SECURITY"
    AUDIT = "AUDIT"

class ErrorCategory(Enum):
    """Error categories for structured error handling"""
    SYSTEM = "system"
    NETWORK = "network"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    VALIDATION = "validation"
    TIMEOUT = "timeout"
    RATE_LIMIT = "rate_limit"
    RESOURCE = "resource"
    AGENT = "agent"
    MODEL = "model"
    DATA = "data"
    SECURITY = "security"

class CyberLLMLogger:
    """
    Advanced structured logger for Cyber-LLM with security-aware features
    """
    
    def __init__(self, 
                 name: str = "cyber-llm",
                 log_level: LogLevel = LogLevel.INFO,
                 log_dir: str = "src/monitoring/logs",
                 enable_console: bool = True,
                 enable_file: bool = True,
                 enable_security_log: bool = True):
        
        self.name = name
        self.log_level = log_level
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.session_id = str(uuid.uuid4())
        self.start_time = datetime.now(timezone.utc)
        
        # Setup loggers
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, log_level.value))
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Setup handlers
        if enable_console:
            self._setup_console_handler()
        
        if enable_file:
            self._setup_file_handler()
        
        if enable_security_log:
            self._setup_security_handler()
    
    def _setup_console_handler(self):
        """Setup console handler with JSON formatting"""
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(self._get_json_formatter())
        self.logger.addHandler(console_handler)
    
    def _setup_file_handler(self):
        """Setup file handler with rotation"""
        from logging.handlers import RotatingFileHandler
        
        log_file = self.log_dir / f"{self.name}.log"
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=50*1024*1024,  # 50MB
            backupCount=10
        )
        file_handler.setFormatter(self._get_json_formatter())
        self.logger.addHandler(file_handler)
    
    def _setup_security_handler(self):
        """Setup dedicated security event handler"""
        from logging.handlers import RotatingFileHandler
        
        security_log = self.log_dir / f"{self.name}_security.log"
        security_handler = RotatingFileHandler(
            security_log,
            maxBytes=100*1024*1024,  # 100MB
            backupCount=20
        )
        security_handler.setFormatter(self._get_json_formatter())
        security_handler.addFilter(self._security_filter)
        self.logger.addHandler(security_handler)
    
    def _get_json_formatter(self):
        """Get JSON formatter for structured logging"""
        class JSONFormatter(logging.Formatter):
            def format(self, record):
                log_entry = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "session_id": self.session_id,
                    "level": record.levelname,
                    "logger": record.name,
                    "message": record.getMessage(),
                    "module": record.module,
                    "function": record.funcName,
                    "line": record.lineno,
                }
                
                # Add extra fields if present
                if hasattr(record, 'extra'):
                    log_entry.update(record.extra)
                
                # Add exception info if present
                if record.exc_info:
                    log_entry["exception"] = {
                        "type": record.exc_info[0].__name__,
                        "message": str(record.exc_info[1]),
                        "traceback": traceback.format_exception(*record.exc_info)
                    }
                
                return json.dumps(log_entry, default=str)
        
        return JSONFormatter()
    
    def _security_filter(self, record):
        """Filter for security-related log entries"""
        security_keywords = [
            "security", "auth", "login", "logout", "permission", "access",
            "attack", "threat", "vulnerability", "exploit", "malware",
            "suspicious", "anomaly", "intrusion", "breach"
        ]
        
        message = record.getMessage().lower()
        return any(keyword in message for keyword in security_keywords) or \
               getattr(record, 'security_event', False)
    
    def log(self, level: LogLevel, message: str, **kwargs):
        """Log a message with structured data"""
        extra_data = {
            "extra": kwargs
        }
        
        if level == LogLevel.SECURITY:
            extra_data["extra"]["security_event"] = True
            self.logger.warning(message, extra=extra_data)
        elif level == LogLevel.AUDIT:
            extra_data["extra"]["audit_event"] = True
            self.logger.info(message, extra=extra_data)
        else:
            getattr(self.logger, level.value.lower())(message, extra=extra_data)
    
    def debug(self, message: str, **kwargs):
        self.log(LogLevel.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs):
        self.log(LogLevel.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        self.log(LogLevel.WARNING, message, **kwargs)
    
    def error(self, message: str, **kwargs):
        self.log(LogLevel.ERROR, message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        self.log(LogLevel.CRITICAL, message, **kwargs)
    
    def security(self, message: str, **kwargs):
        self.log(LogLevel.SECURITY, message, **kwargs)
    
    def audit(self, message: str, **kwargs):
        self.log(LogLevel.AUDIT, message, **kwargs)

class CyberLLMError(Exception):
    """Base exception class for Cyber-LLM"""
    
    def __init__(self, 
                 message: str,
                 category: ErrorCategory = ErrorCategory.SYSTEM,
                 details: Optional[Dict[str, Any]] = None,
                 retryable: bool = False):
        
        super().__init__(message)
        self.message = message
        self.category = category
        self.details = details or {}
        self.retryable = retryable
        self.timestamp = datetime.now(timezone.utc)
        self.error_id = str(uuid.uuid4())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for logging"""
        return {
            "error_id": self.error_id,
            "timestamp": self.timestamp.isoformat(),
            "message": self.message,
            "category": self.category.value,
            "details": self.details,
            "retryable": self.retryable,
            "traceback": traceback.format_exc()
        }

class RetryableError(CyberLLMError):
    """Error that can be retried"""
    def __init__(self, message: str, category: ErrorCategory = ErrorCategory.SYSTEM, **kwargs):
        super().__init__(message, category, retryable=True, **kwargs)

class NonRetryableError(CyberLLMError):
    """Error that should not be retried"""
    def __init__(self, message: str, category: ErrorCategory = ErrorCategory.SYSTEM, **kwargs):
        super().__init__(message, category, retryable=False, **kwargs)

def retry_with_backoff(max_attempts: int = 3,
                      initial_delay: float = 1.0,
                      max_delay: float = 60.0,
                      exponential_base: float = 2.0,
                      jitter: bool = True,
                      retryable_exceptions: tuple = (RetryableError,)):
    """
    Decorator for implementing retry logic with exponential backoff
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            logger = CyberLLMLogger(name=f"retry.{func.__name__}")
            
            for attempt in range(max_attempts):
                try:
                    if asyncio.iscoroutinefunction(func):
                        return await func(*args, **kwargs)
                    else:
                        return func(*args, **kwargs)
                
                except retryable_exceptions as e:
                    if attempt == max_attempts - 1:
                        logger.error(
                            f"Max retry attempts ({max_attempts}) exceeded for {func.__name__}",
                            error=e.to_dict() if hasattr(e, 'to_dict') else str(e),
                            function=func.__name__,
                            attempt_number=attempt + 1
                        )
                        raise
                    
                    # Calculate delay with exponential backoff
                    delay = min(initial_delay * (exponential_base ** attempt), max_delay)
                    if jitter:
                        import random
                        delay *= (0.5 + 0.5 * random.random())
                    
                    logger.warning(
                        f"Retry attempt {attempt + 1}/{max_attempts} for {func.__name__} after {delay:.2f}s",
                        error=e.to_dict() if hasattr(e, 'to_dict') else str(e),
                        function=func.__name__,
                        attempt_number=attempt + 1,
                        delay=delay
                    )
                    
                    await asyncio.sleep(delay)
                
                except Exception as e:
                    logger.error(
                        f"Non-retryable error in {func.__name__}",
                        error=str(e),
                        error_type=type(e).__name__,
                        function=func.__name__,
                        attempt_number=attempt + 1
                    )
                    raise
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            logger = CyberLLMLogger(name=f"retry.{func.__name__}")
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                
                except retryable_exceptions as e:
                    if attempt == max_attempts - 1:
                        logger.error(
                            f"Max retry attempts ({max_attempts}) exceeded for {func.__name__}",
                            error=e.to_dict() if hasattr(e, 'to_dict') else str(e),
                            function=func.__name__,
                            attempt_number=attempt + 1
                        )
                        raise
                    
                    # Calculate delay with exponential backoff
                    delay = min(initial_delay * (exponential_base ** attempt), max_delay)
                    if jitter:
                        import random
                        delay *= (0.5 + 0.5 * random.random())
                    
                    logger.warning(
                        f"Retry attempt {attempt + 1}/{max_attempts} for {func.__name__} after {delay:.2f}s",
                        error=e.to_dict() if hasattr(e, 'to_dict') else str(e),
                        function=func.__name__,
                        attempt_number=attempt + 1,
                        delay=delay
                    )
                    
                    time.sleep(delay)
                
                except Exception as e:
                    logger.error(
                        f"Non-retryable error in {func.__name__}",
                        error=str(e),
                        error_type=type(e).__name__,
                        function=func.__name__,
                        attempt_number=attempt + 1
                    )
                    raise
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

@contextmanager
def error_context(operation: str, logger: Optional[CyberLLMLogger] = None):
    """Context manager for structured error handling"""
    if logger is None:
        logger = CyberLLMLogger()
    
    operation_id = str(uuid.uuid4())
    start_time = time.time()
    
    logger.info(f"Starting operation: {operation}", 
                operation_id=operation_id,
                operation=operation)
    
    try:
        yield
        
        duration = time.time() - start_time
        logger.info(f"Operation completed successfully: {operation}",
                   operation_id=operation_id,
                   operation=operation,
                   duration=duration)
    
    except Exception as e:
        duration = time.time() - start_time
        
        if isinstance(e, CyberLLMError):
            logger.error(f"Operation failed: {operation}",
                        operation_id=operation_id,
                        operation=operation,
                        duration=duration,
                        error=e.to_dict())
        else:
            logger.error(f"Operation failed with unexpected error: {operation}",
                        operation_id=operation_id,
                        operation=operation,
                        duration=duration,
                        error=str(e),
                        error_type=type(e).__name__)
        
        raise

class HealthChecker:
    """Health monitoring and alerting system"""
    
    def __init__(self, logger: Optional[CyberLLMLogger] = None):
        self.logger = logger or CyberLLMLogger(name="health_checker")
        self.checks = {}
        self.alerts = []
    
    def register_check(self, name: str, check_func: Callable, interval: int = 60):
        """Register a health check"""
        self.checks[name] = {
            "function": check_func,
            "interval": interval,
            "last_run": 0,
            "last_result": None,
            "consecutive_failures": 0
        }
    
    async def run_checks(self):
        """Run all health checks"""
        current_time = time.time()
        
        for name, check in self.checks.items():
            if current_time - check["last_run"] >= check["interval"]:
                try:
                    result = await check["function"]() if asyncio.iscoroutinefunction(check["function"]) else check["function"]()
                    
                    check["last_result"] = result
                    check["last_run"] = current_time
                    check["consecutive_failures"] = 0
                    
                    self.logger.debug(f"Health check passed: {name}", 
                                    check_name=name, 
                                    result=result)
                
                except Exception as e:
                    check["consecutive_failures"] += 1
                    check["last_run"] = current_time
                    
                    self.logger.error(f"Health check failed: {name}",
                                    check_name=name,
                                    error=str(e),
                                    consecutive_failures=check["consecutive_failures"])
                    
                    if check["consecutive_failures"] >= 3:
                        self._trigger_alert(name, str(e))
    
    def _trigger_alert(self, check_name: str, error_message: str):
        """Trigger alert for failed health check"""
        alert = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "check_name": check_name,
            "error": error_message,
            "alert_id": str(uuid.uuid4())
        }
        
        self.alerts.append(alert)
        
        self.logger.critical(f"ALERT: Health check failure - {check_name}",
                           alert=alert)

# Example usage and testing
if __name__ == "__main__":
    # Initialize logger
    logger = CyberLLMLogger(name="test_logger")
    
    # Test basic logging
    logger.info("Starting Cyber-LLM system", component="main", version="0.4.0")
    logger.security("Potential security event detected", 
                    source_ip="192.168.1.100", 
                    event_type="suspicious_login")
    
    # Test error handling with context
    try:
        with error_context("test_operation", logger):
            raise RetryableError("Test retryable error", 
                                ErrorCategory.NETWORK,
                                details={"endpoint": "api.example.com"})
    except RetryableError as e:
        logger.error("Caught retryable error", error=e.to_dict())
    
    # Test retry decorator
    @retry_with_backoff(max_attempts=3)
    def flaky_function():
        import random
        if random.random() < 0.7:  # 70% chance of failure
            raise RetryableError("Random failure for testing")
        return "Success!"
    
    try:
        result = flaky_function()
        logger.info("Function succeeded", result=result)
    except Exception as e:
        logger.error("Function failed after all retries", error=str(e))
