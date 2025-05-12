"""
Databricks Tools for K1 Monitoring Agent

This module contains tools for interacting with Databricks.
"""
from typing import Dict, List, Optional, Any, Tuple
import os
import logging
import asyncio
from datetime import datetime, timedelta

from src.agent_core.dbx_client import DbxClient, RunStatus

# Set up logger
logger = logging.getLogger(__name__)

class DatabricksTools:
    """Tools for interacting with Databricks."""
    
    def __init__(self, host: Optional[str] = None, token: Optional[str] = None, profile: Optional[str] = None):
        """Initialize the Databricks tools.
        
        Args:
            host: Databricks workspace URL
            token: Databricks API token
            profile: Databricks CLI profile to use
        """
        logger.info("Initializing Databricks tools")
        try:
            self.client = DbxClient(host=host, token=token, profile=profile)
            logger.info("Databricks client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Databricks client: {e}")
            self.client = None
            
    async def get_workspace_status(self) -> Dict[str, Any]:
        """Get the status of the Databricks workspace.
        
        Returns:
            Dictionary containing workspace status information
        """
        logger.info("Getting Databricks workspace status")
        if not self.client or not self.client.is_available():
            logger.error("Databricks client is not available")
            return {"status": "error", "message": "Databricks client is not available"}
            
        # Handle synchronous calls asynchronously
        result = await asyncio.to_thread(self._get_sync_workspace_status)
        return result
    
    def _get_sync_workspace_status(self) -> Dict[str, Any]:
        """Synchronous implementation of workspace status check."""
        try:
            clusters = self.client.list_clusters()
            workspaces = self.client.list_workspaces(path="/")
            
            active_clusters = sum(1 for c in clusters if c.get("state") == "RUNNING")
            
            return {
                "status": "available",
                "total_clusters": len(clusters),
                "active_clusters": active_clusters,
                "workspaces": len(workspaces),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting workspace status: {e}")
            return {"status": "error", "message": str(e)}
        
    async def list_clusters(self) -> List[Dict[str, Any]]:
        """
        Retrieves a list of all Databricks clusters with their current status and configuration.
        
        This tool provides simplified details about all clusters in the workspace including
        running state, worker count, and node types. It's useful for getting a quick overview
        of cluster resources or finding specific clusters for further analysis.
        
        When to use:
        - To get an inventory of available clusters
        - When looking for a specific cluster by name
        - Before performing operations that need a cluster ID
        - To check cluster states (running, terminated, etc.)
        
        No input parameters required.
        
        Output JSON example:
        [
            {
                "cluster_id": "0123-456789-abcdef",
                "name": "Production ETL Cluster",
                "state": "RUNNING",
                "creator": "john.smith@example.com",
                "node_type": "Standard_DS3_v2",
                "runtime": "10.4.x-scala2.12",
                "num_workers": 4
            },
            {
                "cluster_id": "0123-456789-ghijkl",
                "name": "ML Training Cluster",
                "state": "TERMINATED",
                "creator": "data.scientist@example.com",
                "node_type": "Standard_NC6s_v3",
                "runtime": "11.3.x-cpu-ml-scala2.12",
                "num_workers": 0
            }
        ]
        
        Error output example:
        [
            {
                "status": "error",
                "message": "Databricks client is not available"
            }
        ]
        """
        logger.info("Listing Databricks clusters")
        if not self.client or not self.client.is_available():
            logger.error("Databricks client is not available")
            return [{"status": "error", "message": "Databricks client is not available"}]
        
        # Handle synchronous calls asynchronously
        clusters = await asyncio.to_thread(self.client.list_clusters)
        
        # Simplify the response for better readability
        simplified_clusters = []
        for cluster in clusters:
            simplified_clusters.append({
                "cluster_id": cluster.get("cluster_id"),
                "name": cluster.get("cluster_name"),
                "state": cluster.get("state"),
                "creator": cluster.get("creator_user_name"),
                "node_type": cluster.get("node_type_id"),
                "runtime": cluster.get("spark_version"),
                "num_workers": cluster.get("num_workers", 0)
            })
            
        return simplified_clusters
    
    async def get_cluster_details(self, cluster_id: str) -> Dict[str, Any]:
        """Get detailed information about a specific cluster.
        
        Args:
            cluster_id: ID of the cluster to get details for
            
        Returns:
            Detailed cluster information
        """
        logger.info(f"Getting details for cluster {cluster_id}")
        if not self.client or not self.client.is_available():
            logger.error("Databricks client is not available")
            return {"status": "error", "message": "Databricks client is not available"}
        
        cluster = await asyncio.to_thread(self.client.get_cluster, cluster_id)
        if not cluster:
            return {"status": "error", "message": f"Cluster {cluster_id} not found"}
        
        return cluster
    
    async def list_jobs(self, limit: int = 50, name: Optional[str] = None) -> List[Dict[str, Any]]:
        """List jobs in the Databricks workspace.
        
        Args:
            limit: Maximum number of jobs to return
            name: Optional filter by job name
            
        Returns:
            List of job information dictionaries
        """
        logger.info(f"Listing Databricks jobs (limit={limit}, name={name})")
        if not self.client or not self.client.is_available():
            logger.error("Databricks client is not available")
            return [{"status": "error", "message": "Databricks client is not available"}]
        
        jobs = await asyncio.to_thread(self.client.list_jobs, limit=limit, name=name)
        
        # Simplify the response for better readability
        simplified_jobs = []
        for job in jobs:
            simplified_jobs.append({
                "job_id": job.get("job_id"),
                "name": job.get("settings", {}).get("name"),
                "creator": job.get("creator_user_name"),
                "created_time": job.get("created_time"),
                "schedule": job.get("settings", {}).get("schedule")
            })
            
        return simplified_jobs
    
    async def get_recent_job_runs(self, job_id: Optional[str] = None, limit: int = 25) -> List[Dict[str, Any]]:
        """Get recent job runs.
        
        Args:
            job_id: Optional job ID to filter by
            limit: Maximum number of runs to return
            
        Returns:
            List of job run information
        """
        logger.info(f"Getting recent job runs (job_id={job_id}, limit={limit})")
        if not self.client or not self.client.is_available():
            logger.error("Databricks client is not available")
            return [{"status": "error", "message": "Databricks client is not available"}]
        
        runs = await asyncio.to_thread(self.client.list_runs, job_id=job_id, limit=limit)
        
        # Simplify the response
        simplified_runs = []
        for run in runs:
            state = run.get("state", {})
            simplified_runs.append({
                "run_id": run.get("run_id"),
                "job_id": run.get("job_id"),
                "run_name": run.get("run_name"),
                "state": state.get("life_cycle_state"),
                "result_state": state.get("result_state"),
                "start_time": run.get("start_time"),
                "end_time": run.get("end_time", None)
            })
            
        return simplified_runs
    
    async def get_logs(self, 
                     cluster_id: Optional[str] = None,
                     run_id: Optional[str] = None,
                     log_type: str = "audit",
                     days: int = 7,
                     limit: int = 100) -> List[Dict[str, Any]]:
        """
        Retrieves various types of Databricks logs filtered by cluster, run, time period and type.
        
        This tool fetches logs from Databricks to help with troubleshooting and monitoring.
        It supports different log types including audit logs, cluster logs, and job run logs.
        Results are returned in chronological order with timestamps and metadata.
        
        When to use:
        - To investigate cluster or job issues
        - For security and compliance auditing
        - To monitor user activities in the workspace
        - To track historical performance patterns
        
        Input JSON example:
        {
            "cluster_id": "0123-456789-abcdef",   // Optional: Filter logs by specific cluster
            "run_id": "987654",                   // Optional: Filter logs by specific job run
            "log_type": "audit",                  // Optional: Type of logs to retrieve (default: "audit")
                                                 // Options: "audit", "cluster", "driver", "executor", "stderr", "stdout"
            "days": 7,                           // Optional: Number of days to look back (default: 7)
            "limit": 100                         // Optional: Maximum number of log entries to return (default: 100)
        }
        
        Output JSON example:
        [
            {
                "timestamp": "2023-04-15T08:25:31Z",
                "log_type": "stderr",
                "cluster_id": "0123-456789-abcdef",
                "user": "john.smith@example.com",
                "message": "ERROR: java.lang.OutOfMemoryError: Java heap space",
                "level": "ERROR",
                "source": "driver"
            },
            {
                "timestamp": "2023-04-15T08:24:15Z",
                "log_type": "stdout",
                "cluster_id": "0123-456789-abcdef", 
                "user": "john.smith@example.com",
                "message": "Starting job execution with parameters: {\"date\": \"2023-04-15\"}",
                "level": "INFO",
                "source": "driver"
            }
        ]
        
        Error output example:
        [
            {
                "status": "error",
                "message": "Databricks client is not available"
            }
        ]
        """
        logger.info(f"Getting Databricks logs (type={log_type}, days={days}, limit={limit})")
        if not self.client or not self.client.is_available():
            logger.error("Databricks client is not available")
            return [{"status": "error", "message": "Databricks client is not available"}]
        
        # Calculate start time in milliseconds
        start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
        
        logs = await asyncio.to_thread(
            self.client.get_logs,
            cluster_id=cluster_id,
            run_id=run_id,
            start_time=start_time,
            limit=limit,
            log_type=log_type
        )
        
        return logs
    
    async def get_activity(self, days: int = 7, limit: int = 100) -> Dict[str, Any]:
        """Get workspace activity data.
        
        Args:
            days: Number of days to look back
            limit: Maximum number of activity records to return
            
        Returns:
            Dictionary with workspace activity information
        """
        logger.info(f"Getting Databricks activity (days={days}, limit={limit})")
        if not self.client or not self.client.is_available():
            logger.error("Databricks client is not available")
            return {"status": "error", "message": "Databricks client is not available"}
        
        activity = await asyncio.to_thread(self.client.get_activity, days=days, limit=limit)
        
        # Organize activity by type
        activity_summary = {
            "clusters": [],
            "jobs": [],
            "notebooks": [],
            "warehouses": [],
            "count": len(activity),
            "days": days
        }
        
        for item in activity:
            action_type = item.get("action_type", "unknown")
            if "CLUSTER" in action_type:
                activity_summary["clusters"].append(item)
            elif "JOB" in action_type:
                activity_summary["jobs"].append(item)
            elif "NOTEBOOK" in action_type:
                activity_summary["notebooks"].append(item)
            elif "WAREHOUSE" in action_type:
                activity_summary["warehouses"].append(item)
        
        # Add count summaries
        activity_summary["cluster_count"] = len(activity_summary["clusters"])
        activity_summary["job_count"] = len(activity_summary["jobs"])
        activity_summary["notebook_count"] = len(activity_summary["notebooks"])
        activity_summary["warehouse_count"] = len(activity_summary["warehouses"])
        
        return activity_summary 