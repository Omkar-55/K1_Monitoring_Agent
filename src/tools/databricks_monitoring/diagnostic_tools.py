"""
Tools for diagnosing issues in Databricks logs.
"""

import random
import time
from enum import Enum
from typing import Dict, Any, List, Optional

# Import the logging configuration
from src.agent_core.logging_config import get_logger

# Get logger for this module
logger = get_logger(__name__)

class FailureType(str, Enum):
    """Types of failures that can be diagnosed."""
    MEMORY_EXCEEDED = "memory_exceeded"
    DISK_SPACE_EXCEEDED = "disk_space_exceeded"
    DEPENDENCY_ERROR = "dependency_error"
    UNKNOWN = "unknown"

def diagnose(logs_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Diagnose issues in Databricks logs.
    
    Args:
        logs_data: The logs data to analyze
        
    Returns:
        A diagnosis result with issue_type and reasoning
    """
    logger.info("Diagnosing Databricks logs")
    
    # Extract the logs
    logs = logs_data.get("logs", {})
    stdout = logs.get("stdout", "")
    stderr = logs.get("stderr", "")
    
    # Check for memory issues
    if "java.lang.OutOfMemoryError" in stderr or "MemoryError" in stderr:
        issue_type = FailureType.MEMORY_EXCEEDED
        reasoning = "Job failed due to insufficient memory. Found OutOfMemoryError in logs."
    
    # Check for disk space issues
    elif "No space left on device" in stderr or "Disk quota exceeded" in stderr:
        issue_type = FailureType.DISK_SPACE_EXCEEDED
        reasoning = "Job failed due to insufficient disk space. Found disk space error in logs."
    
    # Check for dependency issues
    elif "ModuleNotFoundError" in stderr or "ImportError" in stderr or "ClassNotFoundException" in stderr:
        issue_type = FailureType.DEPENDENCY_ERROR
        reasoning = "Job failed due to missing dependencies. Found import or module errors in logs."
    
    # Unknown issue
    else:
        issue_type = FailureType.UNKNOWN
        reasoning = "Could not identify a specific issue type from the logs."
    
    logger.info(f"Diagnosed issue: {issue_type}")
    
    return {
        "issue_type": issue_type,
        "reasoning": reasoning,
        "logs_analyzed": {
            "stdout_length": len(stdout),
            "stderr_length": len(stderr)
        }
    }

def simulate_run(failure_type: Optional[str] = None) -> Dict[str, Any]:
    """
    Generate simulated logs for testing.
    
    Args:
        failure_type: The type of failure to simulate
        
    Returns:
        Simulated logs data
    """
    logger.info(f"Simulating run with failure type: {failure_type}")
    
    # Generate a random run ID
    run_id = f"run_{random.randint(10000, 99999)}"
    job_id = f"job_{random.randint(1000, 9999)}"
    
    # Default to a random failure type if none specified
    if not failure_type:
        failure_types = list(FailureType)
        failure_type = random.choice(failure_types)
    
    # Ensure failure_type is a string
    if isinstance(failure_type, FailureType):
        failure_type = failure_type.value
    
    # Generate appropriate logs for the failure type
    stdout = "Starting Databricks job execution...\n"
    stdout += "Loading data...\n"
    stdout += "Processing data...\n"
    
    stderr = ""
    
    if failure_type == FailureType.MEMORY_EXCEEDED:
        stdout += "Processing large dataset...\n"
        stderr += "WARNING: Memory usage is high\n"
        stderr += "ERROR: java.lang.OutOfMemoryError: Java heap space\n"
        stderr += "  at org.apache.spark.sql.execution.aggregate.HashAggregateExec.doExecute(HashAggregateExec.scala:115)\n"
        stderr += "  at org.apache.spark.sql.execution.SparkPlan.execute(SparkPlan.scala:180)\n"
    
    elif failure_type == FailureType.DISK_SPACE_EXCEEDED:
        stdout += "Writing results to disk...\n"
        stderr += "WARNING: Disk usage is high\n"
        stderr += "ERROR: java.io.IOException: No space left on device\n"
        stderr += "  at java.io.FileOutputStream.writeBytes(Native Method)\n"
        stderr += "  at java.io.FileOutputStream.write(FileOutputStream.java:326)\n"
    
    elif failure_type == FailureType.DEPENDENCY_ERROR:
        stdout += "Importing libraries...\n"
        stderr += "ERROR: ModuleNotFoundError: No module named 'pandas'\n"
        stderr += "  at <frozen importlib._bootstrap>(219)._call_with_frames_removed\n"
        stderr += "  at <frozen importlib._bootstrap_external>(728).exec_module\n"
    
    else:
        stdout += "Executing job...\n"
        stderr += "ERROR: Unknown error occurred\n"
        stderr += "  at com.databricks.backend.common.rpc.InternalDriverConnectionProvider.lambda$getOrCreate$1(InternalDriverConnectionProvider.scala:102)\n"
    
    # Simulate job status
    status = "FAILED"
    
    # Simulate run duration
    duration_seconds = random.randint(60, 3600)
    
    # Return the simulated logs
    logs_data = {
        "run_id": run_id,
        "job_id": job_id,
        "status": status,
        "duration_seconds": duration_seconds,
        "logs": {
            "stdout": stdout,
            "stderr": stderr
        }
    }
    
    logger.info(f"Simulated run {run_id} with failure type {failure_type}")
    
    return logs_data 