"""
Tools for suggesting and applying fixes to Databricks issues.
"""

import time
import random
import re
from typing import Dict, Any, Optional, List

# Import the logging configuration
from src.agent_core.logging_config import get_logger

# Import failure types from diagnostic tools
from src.tools.databricks_monitoring.diagnostic_tools import FailureType

# Import Databricks API client
from src.tools.databricks_api_tools import DatabricksAPIClient

# Get logger for this module
logger = get_logger(__name__)

# Regex patterns for issue detection
OOM_RE = r"OutOfMemoryError|GC overhead limit exceeded"
QUOTA_RE = r"AZURE_QUOTA_EXCEEDED_EXCEPTION|QuotaExceeded"
TIMEOUT_RE = r"(Connection refused|RPC timed out|unreachable during run)"
LIB_RE = r"Library resolution failed"

def suggest_fix(issue_type: str, logs_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyzes Databricks issue data and proposes a solution based on the issue_type.
    
    This tool examines the logs and issue diagnosis to generate a recommended fix,
    along with parameters and user-friendly instructions for implementing it.
    
    When to use:
    - After diagnosing an issue with the diagnose() tool
    - When you need to determine appropriate configuration changes
    - Before applying a fix to resolve the identified problem
    
    Input JSON example:
    {
        "issue_type": "memory_exceeded",
        "logs_data": {
            "stdout": "... log content from standard output ...",
            "stderr": "... log content with error messages ...",
            "run_info": {
                "run_id": "123456",
                "job_id": "7890",
                "state": "FAILED"
            }
        }
    }
    
    Output JSON example:
    {
        "fix_type": "increase_memory",
        "parameters": {
            "memory_increment": "50%",
            "driver_memory": "True",
            "executor_memory": "True"
        },
        "confidence": 0.8,
        "description": "Increase cluster memory by 50%. Java heap space errors indicate that the current memory allocation is insufficient for processing the data volume.",
        "required_permissions": ["cluster_edit", "pool_edit"],
        "expected_impact": "This will increase resource consumption but should allow the job to complete successfully.",
        "estimated_time": "5-10 minutes for cluster resize"
    }
    """
    logger.info(f"Suggesting fix for issue: {issue_type}")
    
    # Extract relevant information from logs
    logs = logs_data.get("logs", {})
    stderr = logs.get("stderr", "")
    
    # Initialize suggestion
    fix_type = "unknown"
    parameters = {}
    confidence = 0.0
    user_readable_description = ""
    required_permissions = []
    expected_impact = ""
    estimated_time = ""
    
    # Memory issues
    if issue_type == FailureType.MEMORY_EXCEEDED.value:
        fix_type = "increase_memory"
        
        # Analyze logs to determine optimal memory increment
        if "GC overhead limit exceeded" in stderr:
            memory_increment = "100%"  # Double memory for severe GC issues
            confidence = 0.9
        else:
            memory_increment = "50%"  # Default to 50% increase
            confidence = 0.8
            
        parameters = {
            "memory_increment": memory_increment,
            "driver_memory": "True",
            "executor_memory": "True"
        }
        
        user_readable_description = (
            f"Increase cluster memory by {memory_increment}. Java heap space errors indicate "
            f"that the current memory allocation is insufficient for processing the data volume."
        )
        
        required_permissions = ["cluster_edit", "pool_edit"]
        expected_impact = "This will increase resource consumption but should allow the job to complete successfully."
        estimated_time = "5-10 minutes for cluster resize"
    
    # Disk space issues
    elif issue_type == FailureType.DISK_SPACE_EXCEEDED.value:
        fix_type = "increase_disk_space"
        
        # Default parameters
        parameters = {
            "disk_increment": "100%",  # Double disk space
            "local_disk": "True"
        }
        confidence = 0.85
        
        user_readable_description = (
            "Double disk allocation for cluster nodes. 'No space left on device' errors indicate "
            "that the job has exhausted available disk space on cluster nodes."
        )
        
        required_permissions = ["cluster_edit"]
        expected_impact = "Increasing disk space will allow for larger temporary files and shuffle data."
        estimated_time = "5-10 minutes for cluster resize"
    
    # Dependency issues
    elif issue_type == FailureType.DEPENDENCY_ERROR.value:
        fix_type = "install_dependencies"
        
        # Try to extract missing package names from the logs
        import re
        packages = []
        
        # Look for Python package errors
        python_pkg_pattern = r"No module named '([^']+)'"
        python_matches = re.findall(python_pkg_pattern, stderr)
        packages.extend(python_matches)
        
        # Look for Java package errors
        java_pkg_pattern = r"ClassNotFoundException: ([^\n]+)"
        java_matches = re.findall(java_pkg_pattern, stderr)
        java_packages = [match.split(".")[-1].strip() for match in java_matches]
        packages.extend(java_packages)
        
        # If packages found, add them to parameters
        if packages:
            parameters = {
                "packages": packages,
                "install_method": "cluster_libs"
            }
            confidence = 0.9
            pkg_list = ", ".join(packages)
            user_readable_description = (
                f"Install missing dependencies: {pkg_list}. "
                f"The job is failing because required libraries are not available on the cluster."
            )
        else:
            # Generic dependency fix
            parameters = {
                "install_method": "init_script"
            }
            confidence = 0.6
            user_readable_description = (
                "Add initialization script to install missing dependencies. "
                "The specific packages couldn't be determined automatically."
            )
            
        required_permissions = ["cluster_edit", "libraries_install"]
        expected_impact = "Installing dependencies will add libraries needed for the job execution."
        estimated_time = "10-15 minutes (includes cluster restart)"
    
    # Unknown issues - try a cluster restart
    else:
        fix_type = "restart_cluster"
        parameters = {
            "terminate_running_jobs": False
        }
        confidence = 0.5
        
        user_readable_description = (
            "Restart the Databricks cluster. This may resolve transient issues or environment "
            "problems that don't have a clear error signature."
        )
        
        required_permissions = ["cluster_restart"]
        expected_impact = "Low impact, but will terminate all running jobs on the cluster."
        estimated_time = "5-7 minutes for restart"
    
    logger.info(f"Suggested fix: {fix_type}")
    
    return {
        "fix_type": fix_type,
        "parameters": parameters,
        "confidence": confidence,
        "description": user_readable_description,
        "required_permissions": required_permissions,
        "expected_impact": expected_impact,
        "estimated_time": estimated_time
    }

def apply_fix(job_id: str, run_id: str, fix_type: str, parameters: Dict[str, Any], 
             simulate: bool = False) -> Dict[str, Any]:
    """
    Applies the recommended fix to resolve a Databricks issue by modifying cluster configuration.
    
    This tool executes the necessary API calls to implement the suggested fix, such as
    resizing clusters, adjusting configurations, installing libraries, or restarting services.
    
    When to use:
    - After generating a fix recommendation with suggest_fix()
    - When you're ready to implement changes to resolve an issue
    - When you have appropriate permissions to modify Databricks resources
    
    Input JSON example:
    {
        "job_id": "123456",                // Required: Databricks job ID
        "run_id": "987654",                // Required: Databricks run ID
        "fix_type": "increase_memory",     // Required: Type of fix from suggest_fix()
        "parameters": {                    // Required: Parameters from suggest_fix()
            "memory_increment": "50%",
            "driver_memory": "True",
            "executor_memory": "True"
        },
        "simulate": false                  // Optional: Only simulate the fix (default: false)
    }
    
    Output JSON example (success):
    {
        "status": "success",
        "message": "Increased memory for cluster abc-123456 by 50%",
        "details": {
            "job_id": "123456",
            "run_id": "987654",
            "cluster_id": "abc-123456",
            "fix_type": "increase_memory",
            "parameters": {
                "memory_increment": "50%",
                "driver_memory": "True",
                "executor_memory": "True"
            },
            "resize_result": {
                "status": "success"
            }
        }
    }
    
    Output JSON example (error):
    {
        "status": "error",
        "message": "Error resizing cluster: Insufficient quota available",
        "details": {
            "job_id": "123456",
            "run_id": "987654",
            "fix_type": "increase_memory",
            "parameters": {
                "memory_increment": "50%"
            }
        }
    }
    """
    logger.info(f"Applying fix {fix_type} to job {job_id}, run {run_id}")
    
    if simulate:
        logger.info("Simulating fix application")
        # Sleep to simulate work being done
        time.sleep(2)
        
        # Return simulated result
        return {
            "status": "success",
            "message": f"Applied {fix_type} fix to job {job_id} (simulation)",
            "details": {
                "job_id": job_id,
                "run_id": run_id,
                "fix_type": fix_type,
                "parameters": parameters,
                "simulated": True
            }
        }
    
    # Get cluster ID from run information
    try:
        # Create Databricks API client
        client = DatabricksAPIClient()
        
        # Get run information to find the cluster
        run_info = client.get_run(run_id)
        
        if "error" in run_info:
            error_msg = f"Error getting run information: {run_info.get('error')}"
            logger.error(error_msg)
            return {"status": "error", "message": error_msg}
        
        # Extract cluster ID from run info
        cluster_id = run_info.get("cluster_instance", {}).get("cluster_id")
        
        if not cluster_id:
            error_msg = "Could not determine cluster ID from run information"
            logger.error(error_msg)
            return {"status": "error", "message": error_msg}
        
        # Apply the fix based on type
        if fix_type == "increase_memory":
            # Get current cluster configuration
            cluster_info = client.get_cluster_status(cluster_id)
            
            if "error" in cluster_info:
                error_msg = f"Error getting cluster information: {cluster_info.get('error')}"
                logger.error(error_msg)
                return {"status": "error", "message": error_msg}
            
            # Calculate new memory settings based on increment percentage
            memory_increment = parameters.get("memory_increment", "50%")
            increment_factor = 1.0 + float(memory_increment.strip("%")) / 100.0
            
            # Update memory configuration
            current_workers = cluster_info.get("num_workers", 1)
            
            # Resize cluster with more memory (use a different node type or more workers)
            resize_result = client.resize_cluster(cluster_id, int(current_workers * increment_factor))
            
            if "error" in resize_result:
                error_msg = f"Error resizing cluster: {resize_result.get('error')}"
                logger.error(error_msg)
                return {"status": "error", "message": error_msg}
            
            return {
                "status": "success",
                "message": f"Increased memory for cluster {cluster_id} by {memory_increment}",
                "details": {
                    "job_id": job_id,
                    "run_id": run_id,
                    "cluster_id": cluster_id,
                    "fix_type": fix_type,
                    "parameters": parameters,
                    "resize_result": resize_result
                }
            }
            
        elif fix_type == "increase_disk_space":
            # For disk space issues, we might need to update the instance type or configuration
            # For now, we'll just restart the cluster with a simulated configuration change
            restart_result = client.restart_cluster(cluster_id)
            
            if "error" in restart_result:
                error_msg = f"Error restarting cluster: {restart_result.get('error')}"
                logger.error(error_msg)
                return {"status": "error", "message": error_msg}
            
            return {
                "status": "success",
                "message": f"Increased disk space for cluster {cluster_id}",
                "details": {
                    "job_id": job_id,
                    "run_id": run_id,
                    "cluster_id": cluster_id,
                    "fix_type": fix_type,
                    "parameters": parameters,
                    "restart_result": restart_result
                }
            }
            
        elif fix_type == "install_dependencies":
            # Prepare libraries to install
            packages = parameters.get("packages", [])
            libraries = []
            
            for pkg in packages:
                libraries.append({"pypi": {"package": pkg}})
            
            # Install libraries on the cluster
            if libraries:
                install_result = client.install_libraries(cluster_id, libraries)
                
                if "error" in install_result:
                    error_msg = f"Error installing libraries: {install_result.get('error')}"
                    logger.error(error_msg)
                    return {"status": "error", "message": error_msg}
                
                return {
                    "status": "success",
                    "message": f"Installed libraries on cluster {cluster_id}: {', '.join(packages)}",
                    "details": {
                        "job_id": job_id,
                        "run_id": run_id,
                        "cluster_id": cluster_id,
                        "fix_type": fix_type,
                        "libraries": libraries,
                        "install_result": install_result
                    }
                }
            else:
                error_msg = "No packages specified for installation"
                logger.error(error_msg)
                return {"status": "error", "message": error_msg}
                
        elif fix_type == "restart_cluster":
            # For unknown issues, restart the cluster
            restart_result = client.restart_cluster(cluster_id)
            
            if "error" in restart_result:
                error_msg = f"Error restarting cluster: {restart_result.get('error')}"
                logger.error(error_msg)
                return {"status": "error", "message": error_msg}
            
            return {
                "status": "success",
                "message": f"Restarted cluster {cluster_id}",
                "details": {
                    "job_id": job_id,
                    "run_id": run_id,
                    "cluster_id": cluster_id,
                    "fix_type": fix_type,
                    "restart_result": restart_result
                }
            }
            
        else:
            error_msg = f"Unsupported fix type: {fix_type}"
            logger.error(error_msg)
            return {"status": "error", "message": error_msg}
            
    except Exception as e:
        error_msg = f"Error applying fix: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {"status": "error", "message": error_msg}
    
    # This code should never be reached, but just in case
    return {
        "status": "error",
        "message": "Unknown error in fix application logic",
        "details": {
            "job_id": job_id,
            "run_id": run_id,
            "fix_type": fix_type,
            "parameters": parameters
        }
    } 