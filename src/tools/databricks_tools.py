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
        """List all clusters in the workspace.
        
        Returns:
            List of cluster information dictionaries
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
        """Get logs from Databricks.
        
        Args:
            cluster_id: Optional cluster ID to filter logs by
            run_id: Optional run ID to filter logs by
            log_type: Type of logs to retrieve (audit, cluster, etc.)
            days: Number of days to look back
            limit: Maximum number of log entries to return
            
        Returns:
            List of log entries
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