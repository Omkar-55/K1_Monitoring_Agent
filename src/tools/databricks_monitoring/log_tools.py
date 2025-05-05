"""
Tools for fetching and processing Databricks job logs.
"""

import json
from typing import Dict, Any, Optional, List
import time
from opentelemetry import trace

# Import the Databricks client wrapper
from agent_core.dbx_client import DbxClient, RunStatus
from agent_core.logging_config import get_logger

# Get logger for this module
logger = get_logger(__name__)

# Get tracer for this module
tracer = trace.get_tracer(__name__)


@tracer.start_as_current_span("get_logs")
def get_logs(job_id: str, run_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Get logs and metadata for a Databricks job run.
    
    Args:
        job_id: The ID of the Databricks job
        run_id: The specific run ID to get logs for. If None, gets the most recent run.
        
    Returns:
        A dictionary containing:
            - run_id: The ID of the run
            - job_id: The ID of the job
            - status: The current status of the run
            - start_time: When the run started
            - end_time: When the run ended (if completed)
            - duration_seconds: How long the run took (if completed)
            - run_name: The name of the run
            - logs: Dictionary containing stdout and stderr logs
            - metadata: Additional metadata about the run
    """
    logger.info(f"Getting logs for job ID {job_id}, run ID {run_id if run_id else 'most recent'}")
    
    try:
        # Initialize Databricks client
        client = DbxClient()
        
        # If run_id is not provided, get the most recent run for this job
        if run_id is None:
            logger.info(f"No run ID provided, fetching most recent run for job {job_id}")
            runs = client.list_runs(job_id=job_id, limit=1)
            if not runs:
                logger.warning(f"No runs found for job {job_id}")
                return {
                    "status": "error",
                    "message": f"No runs found for job {job_id}"
                }
            run_id = runs[0].get("run_id")
            logger.info(f"Found most recent run ID: {run_id}")
        
        # Get run details
        run = client.get_run(run_id)
        if not run:
            logger.warning(f"Run {run_id} not found")
            return {
                "status": "error",
                "message": f"Run {run_id} not found"
            }
        
        # Extract relevant metadata
        state = run.get("state", {})
        life_cycle_state = state.get("life_cycle_state")
        result_state = state.get("result_state")
        
        start_time = run.get("start_time")
        end_time = run.get("end_time")
        
        # Calculate duration if both start and end times are available
        duration_seconds = None
        if start_time and end_time:
            duration_seconds = (end_time - start_time) / 1000
        
        # Get cluster ID if available
        cluster_id = None
        cluster_instance = run.get("cluster_instance", {})
        if cluster_instance:
            cluster_id = cluster_instance.get("cluster_id")
        
        # Get logs
        stdout_logs = ""
        stderr_logs = ""
        
        if cluster_id:
            logger.info(f"Getting logs from cluster {cluster_id} for run {run_id}")
            logs = client.get_logs(run_id=run_id, cluster_id=cluster_id)
            
            # Extract stdout and stderr logs
            for log_entry in logs:
                log_type = log_entry.get("log_type", "").lower()
                if log_type == "stdout":
                    stdout_logs += log_entry.get("message", "") + "\n"
                elif log_type == "stderr":
                    stderr_logs += log_entry.get("message", "") + "\n"
        else:
            logger.warning(f"No cluster ID found for run {run_id}, cannot get logs")
        
        # Compile results
        result = {
            "run_id": run_id,
            "job_id": job_id,
            "status": life_cycle_state,
            "result": result_state,
            "start_time": start_time,
            "end_time": end_time,
            "duration_seconds": duration_seconds,
            "run_name": run.get("run_name"),
            "logs": {
                "stdout": stdout_logs,
                "stderr": stderr_logs
            },
            "metadata": {
                "cluster_id": cluster_id,
                "creator": run.get("creator_user_name"),
                "tasks": run.get("tasks", []),
                "job_parameters": run.get("job_parameters", {})
            }
        }
        
        logger.info(f"Successfully retrieved logs for run {run_id}")
        return result
        
    except Exception as e:
        logger.error(f"Error getting logs for job {job_id}, run {run_id}: {str(e)}")
        return {
            "status": "error",
            "message": f"Error retrieving logs: {str(e)}",
            "job_id": job_id,
            "run_id": run_id
        } 