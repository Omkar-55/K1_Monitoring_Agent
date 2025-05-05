"""
Tools for suggesting and applying fixes to Databricks job failures.
"""

import os
import time
from typing import Dict, Any, Optional
from opentelemetry import trace

# Import the logging configuration
from agent_core.logging_config import get_logger

# Get logger for this module
logger = get_logger(__name__)

# Get tracer for this module
tracer = trace.get_tracer(__name__)

@tracer.start_as_current_span("suggest_fix")
def suggest_fix(issue_type: str, logs_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Suggest a fix for a Databricks issue.
    
    Args:
        issue_type: The type of issue to suggest a fix for
        logs_data: Dictionary containing logs and metadata
        
    Returns:
        Dictionary with fix information including fix_type, confidence, and description
    """
    logger.info(f"Suggesting fix for failure type: {issue_type}")
    
    # Extract cluster metadata
    metadata = logs_data.get("metadata", {})
    cluster_size = metadata.get("cluster_size", "unknown")
    driver_memory = metadata.get("driver_memory", "unknown")
    executor_memory = metadata.get("executor_memory", "unknown")
    
    # Fixes for different issue types
    if issue_type == "memory_exceeded":
        if cluster_size == "small":
            fix_type = "increase_memory"
            confidence = 0.8
            description = f"Increase cluster resources from {cluster_size} to medium to address memory issues"
            logger.info(f"Suggested fix: {fix_type} with confidence {confidence}")
            
            return {
                "fix_type": fix_type,
                "confidence": confidence,
                "description": description,
                "parameters": {
                    "cluster_size": "medium",
                    "driver_memory": "Increase by 50%",
                    "executor_memory": "Increase by 50%"
                }
            }
        else:
            fix_type = "optimize_code"
            confidence = 0.6
            description = "Optimize data processing code to reduce memory usage"
            logger.info(f"Suggested fix: {fix_type} with confidence {confidence}")
            
            return {
                "fix_type": fix_type,
                "confidence": confidence,
                "description": description,
                "parameters": {
                    "suggestions": [
                        "Use caching more effectively",
                        "Partition data processing",
                        "Optimize SQL queries"
                    ]
                }
            }
    elif issue_type == "disk_space_exceeded":
        fix_type = "increase_disk_space"
        confidence = 0.75
        description = "Double the available disk space to prevent disk space errors"
        logger.info(f"Suggested fix: {fix_type} with confidence {confidence}")
        
        return {
            "fix_type": fix_type,
            "confidence": confidence,
            "description": description,
            "parameters": {
                "disk_size": "Increase by 100%"
            }
        }
    elif issue_type == "dependency_error":
        # Extract the missing dependency name from logs
        logs = logs_data.get("logs", {})
        stderr = logs.get("stderr", "")
        
        # Look for common patterns in dependency errors
        dependency_name = "unknown"
        if "ModuleNotFoundError: No module named" in stderr:
            import re
            match = re.search(r"No module named '([^']+)'", stderr)
            if match:
                dependency_name = match.group(1)
        
        fix_type = "install_dependencies"
        confidence = 0.7
        description = f"Install missing dependencies: {dependency_name}"
        logger.info(f"Suggested fix: {fix_type} with confidence {confidence}")
        
        return {
            "fix_type": fix_type,
            "confidence": confidence,
            "description": description,
            "parameters": {
                "dependencies": [dependency_name],
                "libraries": [{"pypi": {"package": dependency_name}}]
            }
        }
    else:
        # Generic fix for unknown issues
        fix_type = "restart_job"
        confidence = 0.3
        description = "Restart the job as a basic troubleshooting step"
        logger.info(f"Suggested fix: {fix_type} with confidence {confidence}")
        
        return {
            "fix_type": fix_type,
            "confidence": confidence,
            "description": description,
            "parameters": {}
        }

@tracer.start_as_current_span("apply_fix")
def apply_fix(job_id: str, run_id: str, fix_type: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply a fix to a Databricks job issue.
    
    Args:
        job_id: The ID of the Databricks job
        run_id: The ID of the failed run
        fix_type: The type of fix to apply
        parameters: Parameters for the fix
        
    Returns:
        Dictionary with the result of applying the fix, including success status and new run ID
    """
    logger.info(f"Applying fix '{fix_type}' to job {job_id}, run {run_id}")
    
    # In a real implementation, this would interact with the Databricks API
    # For now, we'll simulate the fix application
    
    # Simulate success with a new run
    new_run_id = f"fixed-run-{int(time.time())}"
    success = True
    message = f"Applied fix: {fix_type}"
    
    logger.info(f"Fix applied successfully. New run ID: {new_run_id}")
    
    return {
        "success": success,
        "message": message,
        "new_run_id": new_run_id,
        "job_id": job_id,
        "original_run_id": run_id,
        "fix_type": fix_type,
        "parameters": parameters
    } 