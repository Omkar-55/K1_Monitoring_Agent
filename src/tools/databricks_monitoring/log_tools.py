"""
Tools for retrieving and analyzing Databricks logs.
"""

import time
from typing import Dict, Any, Optional

# Import the logging configuration
from src.agent_core.logging_config import get_logger
from src.agent_core.dbx_client import DbxClient, RunStatus

# Get logger for this module
logger = get_logger(__name__)

def get_logs(job_id: str, run_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Get logs for a specific Databricks job run.
    
    Args:
        job_id: The Databricks job ID
        run_id: Optional specific run ID to analyze. If None, uses the latest run.
        
    Returns:
        A dictionary containing run details and logs
    """
    logger.info(f"Getting logs for job {job_id}" + (f", run {run_id}" if run_id else ""))
    
    # Initialize Databricks client
    dbx_client = DbxClient()
    
    try:
        # Get the run ID if not provided (use latest run)
        if not run_id:
            run_id = dbx_client.get_latest_run_id(job_id)
            logger.info(f"Using latest run ID: {run_id}")
            
        # Get the run information
        run_info = dbx_client.get_run_info(run_id)
        
        # Get the run logs
        logs = dbx_client.get_run_logs(run_id)
        
        # Combine run info and logs
        result = {
            "job_id": job_id,
            "run_id": run_id,
            "status": run_info.get("status", "UNKNOWN"),
            "duration_seconds": run_info.get("duration_seconds", 0),
            "start_time": run_info.get("start_time", 0),
            "end_time": run_info.get("end_time", 0),
            "logs": logs,
            "timestamp": time.time()
        }
        
        logger.info(f"Successfully retrieved logs for job {job_id}, run {run_id}")
        return result
        
    except Exception as e:
        logger.error(f"Error getting logs for job {job_id}, run {run_id}: {e}", exc_info=True)
        # Return a minimal result with the error
        return {
            "job_id": job_id,
            "run_id": run_id,
            "status": "ERROR",
            "error": str(e),
            "logs": {"stdout": "", "stderr": f"Error retrieving logs: {str(e)}"},
            "timestamp": time.time()
        } 