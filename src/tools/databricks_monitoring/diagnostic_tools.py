"""
Diagnostic tools for analyzing Databricks job failures.
"""

import enum
from typing import Dict, Any, Tuple, Optional, List
import json
import re
import time
from opentelemetry import trace

# Import the logging configuration
from agent_core.logging_config import get_logger

# Get logger for this module
logger = get_logger(__name__)

# Get tracer for this module
tracer = trace.get_tracer(__name__)


class FailureType(enum.Enum):
    """Enumeration of failure types for Databricks jobs."""
    
    # Resource-related issues
    MEMORY_EXCEEDED = "memory_exceeded"
    DISK_SPACE_EXCEEDED = "disk_space_exceeded"
    DRIVER_FAILURE = "driver_failure"
    EXECUTOR_FAILURE = "executor_failure"
    CLUSTER_RESOURCE_EXHAUSTION = "cluster_resource_exhaustion"
    
    # Data-related issues
    DATA_VALIDATION_ERROR = "data_validation_error"
    SCHEMA_MISMATCH = "schema_mismatch"
    DATA_CORRUPTION = "data_corruption"
    DATA_NOT_FOUND = "data_not_found"
    
    # Code-related issues
    SYNTAX_ERROR = "syntax_error"
    DEPENDENCY_ERROR = "dependency_error"
    IMPORT_ERROR = "import_error"
    FUNCTION_ERROR = "function_error"
    TYPE_ERROR = "type_error"
    PERMISSION_ERROR = "permission_error"
    TIMEOUT = "timeout"
    
    # Configuration issues
    CONFIG_ERROR = "config_error"
    CONNECTION_ERROR = "connection_error"
    AUTH_ERROR = "auth_error"
    QUOTA_EXCEEDED = "quota_exceeded"
    
    # Unspecified or unknown
    UNKNOWN = "unknown"

@tracer.start_as_current_span("diagnose")
def diagnose(logs_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Diagnose the type of failure based on log content.
    
    Args:
        logs_data: Dictionary containing logs and metadata
        
    Returns:
        Dictionary with diagnosis information including issue_type and reasoning
    """
    logger.info("Diagnosing Databricks job failure from logs")
    
    # Extract the logs content
    stdout = logs_data.get("logs", {}).get("stdout", "")
    stderr = logs_data.get("logs", {}).get("stderr", "")
    
    # Combine stdout and stderr for analysis
    log_text = f"{stdout}\n{stderr}"
    
    # Check for memory-related issues
    memory_patterns = [
        r"OutOfMemoryError",
        r"GC overhead limit exceeded",
        r"java heap space",
        r"Memory limit exceeded",
        r"not enough memory"
    ]
    
    for pattern in memory_patterns:
        if re.search(pattern, log_text, re.IGNORECASE):
            issue_type = FailureType.MEMORY_EXCEEDED.value
            reason = f"Found memory error pattern: {pattern}"
            logger.info(f"Diagnosed: {issue_type} - {reason}")
            return {"issue_type": issue_type, "reasoning": reason}
    
    # Check for disk space issues
    disk_patterns = [
        r"No space left on device",
        r"Disk quota exceeded",
        r"cannot create temp file",
        r"disk full",
        r"disk space"
    ]
    
    for pattern in disk_patterns:
        if re.search(pattern, log_text, re.IGNORECASE):
            issue_type = FailureType.DISK_SPACE_EXCEEDED.value
            reason = f"Found disk space error pattern: {pattern}"
            logger.info(f"Diagnosed: {issue_type} - {reason}")
            return {"issue_type": issue_type, "reasoning": reason}
    
    # Check for dependency/import errors
    dependency_patterns = [
        r"ClassNotFoundException",
        r"NoClassDefFoundError",
        r"ModuleNotFoundError",
        r"ImportError",
        r"Could not resolve dependencies",
        r"Dependency .* not found",
        r"No module named"
    ]
    
    for pattern in dependency_patterns:
        if re.search(pattern, log_text, re.IGNORECASE):
            issue_type = FailureType.DEPENDENCY_ERROR.value
            reason = f"Found dependency error pattern: {pattern}"
            logger.info(f"Diagnosed: {issue_type} - {reason}")
            return {"issue_type": issue_type, "reasoning": reason}
    
    # Default to unknown issue type if no patterns matched
    issue_type = FailureType.UNKNOWN.value
    reason = "Unable to determine specific error type"
    logger.warning(f"Diagnosed: {issue_type} - {reason}")
    return {"issue_type": issue_type, "reasoning": reason}

def simulate_run(failure_type: str = "memory_exceeded") -> Dict[str, Any]:
    """
    Simulate a Databricks run with a specific failure type for testing.
    
    Args:
        failure_type: The type of failure to simulate
        
    Returns:
        A dictionary containing simulated run data
    """
    # Generate appropriate stderr logs based on failure type
    stderr_logs = ""
    
    if failure_type == "memory_exceeded":
        stderr_logs = """
        [DRIVER] ExecutorLostFailure: Job aborted due to stage failure:
        Task 12 in stage 4.0 failed 4 times, most recent failure: Lost task 12.3
        java.lang.OutOfMemoryError: Java heap space
        at java.util.Arrays.copyOf(Arrays.java:3181)
        at java.util.ArrayList.grow(ArrayList.java:265)
        """
    elif failure_type == "disk_space_exceeded":
        stderr_logs = """
        [EXECUTOR] ERROR: No space left on device: /tmp/blockmgr-12324-34232-353a
        java.io.IOException: No space left on device
        at java.io.FileOutputStream.writeBytes(Native Method)
        """
    elif failure_type == "dependency_error":
        stderr_logs = """
        [DRIVER] Py4JJavaError: An error occurred while calling z:org.apache.spark.sql.functions.col.

        : java.lang.ClassNotFoundException: org.apache.spark.sql.functions

        ModuleNotFoundError: No module named 'matplotlib'
        """
    else:
        stderr_logs = f"Unknown error type: {failure_type}"
    
    # Create a simulated run
    return {
        "run_id": f"simulated-run-{int(time.time())}",
        "job_id": "12345",
        "status": "TERMINATED", 
        "result": "FAILED",
        "start_time": int(time.time() * 1000) - 3600000,  # 1 hour ago
        "end_time": int(time.time() * 1000) - 3540000,  # 59 minutes ago
        "duration_seconds": 60,
        "run_name": "Test Job Run",
        "logs": {
            "stdout": "Starting job execution...\nProcessing data...\n",
            "stderr": stderr_logs
        },
        "metadata": {
            "cluster_id": "0123-456789-abcdef",
            "creator": "test-user",
            "cluster_size": "small",
            "driver_memory": "4g",
            "executor_memory": "4g"
        }
    } 