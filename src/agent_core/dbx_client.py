"""
Simplified Databricks Client for K1 Monitoring Agent.

This module provides a simplified client for interacting with Databricks in simulation mode.
"""

import os
import time
import json
import logging
from enum import Enum, auto
from typing import Dict, Any, List, Optional, Union, Tuple

# Configure logger
logger = logging.getLogger(__name__)

# Define if Databricks SDK is available (we'll assume it's not for simulation mode)
DATABRICKS_SDK_AVAILABLE = False

# Define dummy exception classes
class ApiError(Exception):
    """Exception for API errors."""
    pass

class ResourceDoesNotExist(Exception):
    """Exception for resources that don't exist."""
    pass

class RunStatus(Enum):
    """Enum for Databricks job run status."""
    PENDING = auto()
    RUNNING = auto()
    TERMINATED = auto()
    SKIPPED = auto()
    INTERNAL_ERROR = auto()
    UNKNOWN = auto()

# Simple API retry decorator that just passes through the function in simulation mode
def api_retry(func):
    """Decorator to retry API calls on failure (simplified for simulation)."""
    return func

class DbxClient:
    """Simplified client for interacting with Databricks workspaces."""
    
    def __init__(
        self,
        host: Optional[str] = None,
        token: Optional[str] = None,
        profile: Optional[str] = None,
        validate_env: bool = True
    ):
        """
        Initialize the Databricks client.
        
        Args:
            host: Databricks workspace URL
            token: Databricks API token
            profile: Databricks CLI profile to use
            validate_env: Whether to check environment variables if other params are not provided
        """
        # Get credentials from parameters or environment variables
        self.host = host
        self.token = token
        self.profile = profile
        
        if validate_env and not all([self.host, self.token]) and not self.profile:
            self.host = self.host or os.getenv("DATABRICKS_HOST")
            self.token = self.token or os.getenv("DATABRICKS_TOKEN")
        
        self._client = None
        
        # Log initialization status
        logger.info("Initialized simulated Databricks client")
    
    def _initialize_client(self) -> None:
        """Initialize the Databricks SDK client (no-op for simulation)."""
        pass
    
    def is_available(self) -> bool:
        """For simulation mode, we'll always return False to indicate SDK not available."""
        return False
    
    def _log_api_call(self, operation: str, params: Dict[str, Any], success: bool, 
                      result_size: Optional[int] = None, error: Optional[Exception] = None,
                      duration_seconds: Optional[float] = None) -> None:
        """
        Log details about an API call for telemetry and debugging.
        
        Args:
            operation: The API operation name
            params: Parameters passed to the operation
            success: Whether the call succeeded
            result_size: Size of the result (e.g., list length)
            error: Error that occurred, if any
            duration_seconds: Duration of the call in seconds
        """
        # Create a log record with all relevant information
        log_record = {
            "timestamp": time.time(),
            "operation": operation,
            "success": success,
            "parameters": {k: v for k, v in params.items() if k not in ["token", "password", "secret"]},
            "result_size": result_size,
            "duration_seconds": duration_seconds
        }
        
        # Add error information if applicable
        if error:
            log_record["error"] = {
                "type": type(error).__name__,
                "message": str(error)
            }
        
        # Log at appropriate level
        if success:
            logger.debug(f"API Call: {json.dumps(log_record)}")
        else:
            logger.error(f"Failed API Call: {json.dumps(log_record)}")
    
    # =========================================================================
    # Simulation methods - return dummy data
    # =========================================================================
    
    def list_jobs(self, limit: int = 20, name: Optional[str] = None, page_token: Optional[str] = None) -> List[Dict[str, Any]]:
        """Return simulated jobs list."""
        return [{"job_id": f"job_{i}", "name": f"Test Job {i}"} for i in range(1, 6)]
    
    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Return a simulated job."""
        return {"job_id": job_id, "name": f"Test Job {job_id}"}
    
    def list_runs(self, job_id: Optional[str] = None, active_only: bool = False, 
                 completed_only: bool = False, offset: int = 0, limit: int = 25) -> List[Dict[str, Any]]:
        """Return simulated run list."""
        return [{"run_id": f"run_{i}", "job_id": job_id or f"job_{i}"} for i in range(1, 6)]
    
    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Return a simulated run."""
        return {"run_id": run_id, "job_id": "job_123", "cluster_instance": {"cluster_id": "cluster_123"}}
    
    def get_run_status(self, run_id: str) -> Tuple[RunStatus, Optional[str]]:
        """Return simulated run status."""
        return RunStatus.TERMINATED, "Simulation mode - always terminated"
    
    def list_clusters(self) -> List[Dict[str, Any]]:
        """Return simulated cluster list."""
        return [{"cluster_id": f"cluster_{i}", "cluster_name": f"Test Cluster {i}"} for i in range(1, 4)]
    
    def get_cluster(self, cluster_id: str) -> Optional[Dict[str, Any]]:
        """Return a simulated cluster."""
        return {"cluster_id": cluster_id, "cluster_name": f"Test Cluster {cluster_id}"}
    
    def list_workspaces(self, path: str) -> List[Dict[str, Any]]:
        """Return simulated workspace list."""
        return [{"path": f"{path}/item_{i}", "object_type": "NOTEBOOK"} for i in range(1, 4)]
    
    def get_run_logs(self, run_id: str) -> Dict[str, str]:
        """Return simulated logs."""
        return {
            "stdout": "Simulated stdout content\nRunning job\nProcessing data",
            "stderr": "Simulated stderr content\nWARNING: Memory usage high\nERROR: Out of memory"
        }
    
    def get_run_info(self, run_id: str) -> Dict[str, Any]:
        """Return simulated run info."""
        return {
            "run_id": run_id,
            "status": "FAILED",
            "duration_seconds": 120,
            "start_time": time.time() - 120,
            "end_time": time.time()
        }
    
    def get_latest_run_id(self, job_id: str) -> str:
        """Return a simulated run ID."""
        return f"run_{int(time.time())}"
    
    def get_activity(self, days: int = 7, limit: int = 100) -> List[Dict[str, Any]]:
        """Return simulated activity data."""
        return [{"action_type": "JOB_RUN", "timestamp": time.time() - i * 3600} for i in range(10)]
    
    def get_logs(self, cluster_id: Optional[str] = None, run_id: Optional[str] = None,
                start_time: Optional[int] = None, limit: int = 100, log_type: str = "audit") -> List[Dict[str, Any]]:
        """Return simulated logs list."""
        return [{"timestamp": time.time() - i * 60, "message": f"Simulated log entry {i}"} for i in range(10)] 