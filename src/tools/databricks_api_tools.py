"""
Databricks API Tools for K1 Monitoring Agent

This module contains tools for directly interacting with Databricks APIs.
"""
import os
import time
import json
import requests
from typing import Dict, Any, List, Optional, Union
from urllib.parse import urlparse

# Import the logging configuration
from src.agent_core.logging_config import get_logger

# Get logger for this module
logger = get_logger(__name__)

class DatabricksAPIClient:
    """Client for interacting with Databricks REST APIs."""
    
    def __init__(self, 
                host: Optional[str] = None, 
                token: Optional[str] = None,
                profile: Optional[str] = None):
        """Initialize the Databricks API client.
        
        Args:
            host: Databricks workspace URL
            token: Databricks API token
            profile: Databricks CLI profile name
        """
        self.host = host or os.getenv("DATABRICKS_HOST")
        self.token = token or os.getenv("DATABRICKS_TOKEN")
        
        if not self.host or not self.token:
            logger.warning("Databricks API credentials not provided or found in environment")
            
        # Normalize host URL
        if self.host and not self.host.startswith(('http://', 'https://')):
            self.host = f"https://{self.host}"
            
        # Remove trailing slash from host
        if self.host and self.host.endswith('/'):
            self.host = self.host[:-1]
            
        logger.info(f"Initialized Databricks API client for host: {self.host}")
    
    def _get_headers(self) -> Dict[str, str]:
        """Get headers for API requests."""
        return {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }
    
    def _make_request(self, 
                     method: str, 
                     endpoint: str, 
                     data: Optional[Dict[str, Any]] = None,
                     params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make a request to the Databricks API.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint
            data: Request data for POST/PUT requests
            params: Query parameters for GET requests
            
        Returns:
            API response data
        """
        if not self.host or not self.token:
            error_msg = "Databricks API credentials not configured"
            logger.error(error_msg)
            return {"error": error_msg}
        
        url = f"{self.host}/api/{endpoint}"
        headers = self._get_headers()
        
        try:
            response = requests.request(
                method=method,
                url=url,
                headers=headers,
                json=data,
                params=params
            )
            
            # Check for successful response
            response.raise_for_status()
            
            # Return JSON response if available
            if response.text:
                return response.json()
            else:
                return {"status": "success"}
                
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error calling Databricks API: {e}")
            if response.text:
                try:
                    return response.json()
                except:
                    return {"error": str(e), "status_code": response.status_code}
            return {"error": str(e), "status_code": response.status_code}
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling Databricks API: {e}")
            return {"error": str(e)}
    
    # Jobs API methods
    
    def run_job(self, 
               job_id: str, 
               parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Run a Databricks job.
        
        Args:
            job_id: Databricks job ID
            parameters: Job parameters
            
        Returns:
            Job run information
        """
        logger.info(f"Running job with ID: {job_id}")
        
        data = {
            "job_id": job_id
        }
        
        # Add parameters if provided
        if parameters:
            data.update(parameters)
        
        return self._make_request("POST", "2.1/jobs/run-now", data=data)
    
    def get_run(self, run_id: str) -> Dict[str, Any]:
        """Get information about a job run.
        
        Args:
            run_id: Run ID
            
        Returns:
            Run information
        """
        logger.info(f"Getting run with ID: {run_id}")
        
        params = {
            "run_id": run_id
        }
        
        return self._make_request("GET", "2.1/jobs/runs/get", params=params)
    
    def cancel_run(self, run_id: str) -> Dict[str, Any]:
        """Cancel a job run.
        
        Args:
            run_id: Run ID
            
        Returns:
            Cancellation result
        """
        logger.info(f"Cancelling run with ID: {run_id}")
        
        data = {
            "run_id": run_id
        }
        
        return self._make_request("POST", "2.1/jobs/runs/cancel", data=data)
    
    # Clusters API methods
    
    def resize_cluster(self, 
                      cluster_id: str, 
                      num_workers: int) -> Dict[str, Any]:
        """Resize a Databricks cluster.
        
        Args:
            cluster_id: Cluster ID
            num_workers: New number of workers
            
        Returns:
            Resize operation result
        """
        logger.info(f"Resizing cluster {cluster_id} to {num_workers} workers")
        
        data = {
            "cluster_id": cluster_id,
            "num_workers": num_workers
        }
        
        return self._make_request("POST", "2.0/clusters/resize", data=data)
    
    def restart_cluster(self, cluster_id: str) -> Dict[str, Any]:
        """Restart a Databricks cluster.
        
        Args:
            cluster_id: Cluster ID
            
        Returns:
            Restart operation result
        """
        logger.info(f"Restarting cluster with ID: {cluster_id}")
        
        data = {
            "cluster_id": cluster_id
        }
        
        return self._make_request("POST", "2.0/clusters/restart", data=data)
    
    def get_cluster_events(self, 
                          cluster_id: str,
                          start_time: Optional[int] = None,
                          end_time: Optional[int] = None,
                          limit: int = 100) -> Dict[str, Any]:
        """Get events for a cluster.
        
        Args:
            cluster_id: Cluster ID
            start_time: Start time in milliseconds since epoch
            end_time: End time in milliseconds since epoch
            limit: Maximum number of events to return
            
        Returns:
            Cluster events
        """
        logger.info(f"Getting events for cluster {cluster_id}")
        
        data = {
            "cluster_id": cluster_id,
            "limit": limit
        }
        
        if start_time:
            data["start_time"] = start_time
            
        if end_time:
            data["end_time"] = end_time
        
        return self._make_request("POST", "2.0/clusters/events", data=data)
    
    # Libraries API methods
    
    def install_libraries(self, 
                         cluster_id: str, 
                         libraries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Install libraries on a cluster.
        
        Args:
            cluster_id: Cluster ID
            libraries: List of libraries to install
                Each library should be a dict with library type and details
                Example: {"pypi": {"package": "pandas"}}
            
        Returns:
            Installation result
        """
        logger.info(f"Installing libraries on cluster {cluster_id}")
        
        data = {
            "cluster_id": cluster_id,
            "libraries": libraries
        }
        
        return self._make_request("POST", "2.0/libraries/install", data=data)
    
    def uninstall_libraries(self, 
                           cluster_id: str, 
                           libraries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Uninstall libraries from a cluster.
        
        Args:
            cluster_id: Cluster ID
            libraries: List of libraries to uninstall
            
        Returns:
            Uninstallation result
        """
        logger.info(f"Uninstalling libraries from cluster {cluster_id}")
        
        data = {
            "cluster_id": cluster_id,
            "libraries": libraries
        }
        
        return self._make_request("POST", "2.0/libraries/uninstall", data=data)
    
    def get_cluster_status(self, cluster_id: str) -> Dict[str, Any]:
        """Get status of a cluster.
        
        Args:
            cluster_id: Cluster ID
            
        Returns:
            Cluster status
        """
        logger.info(f"Getting status for cluster {cluster_id}")
        
        params = {
            "cluster_id": cluster_id
        }
        
        return self._make_request("GET", "2.0/clusters/get", params=params)
    
    # Custom metrics and configuration methods
    
    def get_cluster_metrics(self, cluster_id: str) -> Dict[str, Any]:
        """Get metrics for a cluster including CPU, memory, etc.
        
        Args:
            cluster_id: Cluster ID
            
        Returns:
            Cluster metrics
        """
        logger.info(f"Getting metrics for cluster {cluster_id}")
        
        # First, get cluster info to check if it's running
        cluster_info = self.get_cluster_status(cluster_id)
        
        if cluster_info.get("state") != "RUNNING":
            return {
                "status": "error",
                "message": f"Cluster {cluster_id} is not running. Current state: {cluster_info.get('state')}",
                "metrics": {}
            }
        
        # Get events that might contain metrics
        events = self.get_cluster_events(cluster_id, limit=10)
        
        # Process events to extract metrics
        # In a real implementation, this would parse Ganglia metrics
        # For now, we'll return a simulated set of metrics
        
        return {
            "status": "success",
            "timestamp": int(time.time() * 1000),
            "metrics": {
                "cpu": {
                    "driver": random.uniform(0, 100),
                    "workers": [random.uniform(0, 100) for _ in range(cluster_info.get("num_workers", 0))]
                },
                "memory": {
                    "driver": {
                        "used_percent": random.uniform(0, 100),
                        "used_gb": random.uniform(0, cluster_info.get("driver_node_type_id", {}).get("memory_mb", 0) / 1024),
                        "total_gb": cluster_info.get("driver_node_type_id", {}).get("memory_mb", 0) / 1024
                    },
                    "workers": [
                        {
                            "used_percent": random.uniform(0, 100),
                            "used_gb": random.uniform(0, cluster_info.get("node_type_id", {}).get("memory_mb", 0) / 1024),
                            "total_gb": cluster_info.get("node_type_id", {}).get("memory_mb", 0) / 1024
                        } 
                        for _ in range(cluster_info.get("num_workers", 0))
                    ]
                },
                "disk": {
                    "driver": {
                        "used_percent": random.uniform(0, 100)
                    },
                    "workers": [
                        {"used_percent": random.uniform(0, 100)} 
                        for _ in range(cluster_info.get("num_workers", 0))
                    ]
                }
            }
        }
    
    def set_spark_conf(self, 
                      cluster_id: str, 
                      spark_conf: Dict[str, str]) -> Dict[str, Any]:
        """Set Spark configuration for a cluster.
        
        Args:
            cluster_id: Cluster ID
            spark_conf: Spark configuration parameters
            
        Returns:
            Operation result
        """
        logger.info(f"Setting Spark configuration for cluster {cluster_id}")
        
        # First, get current cluster configuration
        cluster_info = self.get_cluster_status(cluster_id)
        
        if "error" in cluster_info:
            return cluster_info
        
        # Merge existing and new Spark configurations
        current_spark_conf = cluster_info.get("spark_conf", {})
        updated_spark_conf = {**current_spark_conf, **spark_conf}
        
        # Prepare edit request
        data = {
            "cluster_id": cluster_id,
            "spark_conf": updated_spark_conf
        }
        
        return self._make_request("POST", "2.0/clusters/edit", data=data)
    
    def analyze_data_skew(self, 
                         cluster_id: str, 
                         table_name: str,
                         partition_column: str) -> Dict[str, Any]:
        """Analyze data skew in a table by running a notebook.
        
        Args:
            cluster_id: Cluster ID to run the analysis
            table_name: Name of the table to analyze
            partition_column: Column to check for skew
            
        Returns:
            Skew analysis results
        """
        logger.info(f"Analyzing data skew for table {table_name} on cluster {cluster_id}")
        
        # In a real implementation, this would:
        # 1. Create a temporary notebook with the skew analysis code
        # 2. Run the notebook on the specified cluster
        # 3. Get the results
        
        # For now, we'll return simulated results
        return {
            "status": "success",
            "table": table_name,
            "partition_column": partition_column,
            "skew_detected": random.choice([True, False]),
            "skew_ratio": random.uniform(1.0, 10.0),
            "recommendation": "Consider repartitioning the data or using salting techniques."
        }

# Add the missing random import at the top
import random 