"""
Tools for suggesting and applying fixes to Databricks issues.
"""

import time
import re
from typing import Dict, Any, Optional

# Import the logging configuration
from src.agent_core.logging_config import get_logger

# Get logger for this module
logger = get_logger(__name__)

def suggest_fix(issue_type: str, logs_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Suggest a fix for a Databricks issue.
    
    Args:
        issue_type: The type of issue to fix
        logs_data: The logs data with context
        
    Returns:
        Fix suggestion details
    """
    logger.info(f"Suggesting fix for issue type: {issue_type}")
    
    # Initialize variables
    fix_type = "unknown"
    parameters = {}
    confidence = 0.5
    
    # Suggest fix based on issue type
    if issue_type == "memory_exceeded":
        fix_type = "increase_memory"
        parameters = {
            "memory_increment": "50%"
        }
        confidence = 0.9
        
    elif issue_type == "disk_space_exceeded":
        fix_type = "increase_disk_space"
        parameters = {
            "disk_increment": "100%"
        }
        confidence = 0.8
        
    elif issue_type == "dependency_error":
        fix_type = "install_dependencies"
        
        # Try to determine missing dependencies from logs
        logs = logs_data.get("logs", {})
        stderr = logs.get("stderr", "")
        
        # Extract missing package names from common error patterns
        missing_packages = []
        
        if "ModuleNotFoundError: No module named '" in stderr:
            # Extract from Python error
            matches = re.findall(r"No module named '([^']+)'", stderr)
            missing_packages.extend(matches)
        
        if "ClassNotFoundException" in stderr:
            # Extract from Java error
            matches = re.findall(r"ClassNotFoundException: ([^\s\n]+)", stderr)
            missing_packages.extend(matches)
        
        if missing_packages:
            parameters = {
                "packages": missing_packages
            }
            confidence = 0.9
        else:
            parameters = {
                "packages": ["unknown"]
            }
            confidence = 0.5
    
    else:
        fix_type = "unknown"
        parameters = {}
        confidence = 0.1
    
    logger.info(f"Suggested fix: {fix_type} with confidence {confidence}")
    
    return {
        "fix_type": fix_type,
        "parameters": parameters,
        "confidence": confidence,
        "timestamp": time.time()
    }

def apply_fix(
    fix_plan: Dict[str, Any],
    job_id: str, 
    run_id: str, 
    simulate: bool = True
) -> Dict[str, Any]:
    """
    Apply a fix to a Databricks job.
    
    Args:
        fix_plan: The fix plan to apply
        job_id: The Databricks job ID
        run_id: The run ID related to the issue
        simulate: Whether to simulate the fix
        
    Returns:
        Results of the fix application
    """
    fix_type = fix_plan.get("fix_type", "unknown")
    parameters = fix_plan.get("parameters", {})
        
    logger.info(f"Applying fix {fix_type} to job {job_id}, run {run_id}")
    
    # Simply log and return success for simulation
    if simulate:
        logger.info(f"Simulating fix application: {fix_type} with parameters {parameters}")
                return {
                    "success": True,
            "message": f"Simulated {fix_type} fix applied to job {job_id}",
                    "details": {
                "job_id": job_id,
                "run_id": run_id,
                "fix_type": fix_type,
                "parameters": parameters,
                "timestamp": time.time()
            }
                }
                
    # In a real implementation, this would connect to Databricks API
    # to modify job settings, cluster configurations, etc.
    
    # Apply fix based on type
    if fix_type == "increase_memory":
        memory_increment = parameters.get("memory_increment", "50%")
        # Implementation would update memory settings via API
        message = f"Increased memory by {memory_increment}"
        details = {
            "updated_memory": True,
            "increment": memory_increment,
            "job_id": job_id
                }
                
    elif fix_type == "increase_disk_space":
        disk_increment = parameters.get("disk_increment", "100%")
        # Implementation would update disk settings via API
        message = f"Increased disk space by {disk_increment}"
        details = {
            "updated_disk": True,
            "increment": disk_increment,
            "job_id": job_id
        }
        
    elif fix_type == "install_dependencies":
        packages = parameters.get("packages", [])
        # Implementation would add init script or update libraries via API
        message = f"Installed missing dependencies: {', '.join(packages)}"
        details = {
            "updated_dependencies": True,
            "packages": packages,
            "job_id": job_id
            }
            
        else:
        message = f"Unknown fix type: {fix_type}"
        details = {
                "success": False,
            "reason": "Unknown fix type",
            "job_id": job_id
            }
            
    logger.info(f"Applied fix: {message}")
    
        return {
        "success": True,
        "message": message,
        "details": details
        } 