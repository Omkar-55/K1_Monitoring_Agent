"""
Monitoring Agent for K1

This module contains the implementation of the K1 Monitoring Agent.
"""
from typing import Dict, List, Optional, Any, Union
import os
import logging
import json
import re
from datetime import datetime, timedelta
from pydantic import BaseModel, Field, validator

# Import our tools
from src.tools.databricks_tools import DatabricksTools

# Set up logger
logger = logging.getLogger(__name__)

class MonitoringRequest(BaseModel):
    """Request model for the monitoring agent."""
    query: str = Field(..., description="User query to the agent")
    context: Optional[str] = Field(None, description="Additional context for the agent")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Additional parameters for the request")
    
    @validator("query")
    def query_not_empty(cls, v):
        """Validate that query is not empty."""
        if not v or not v.strip():
            raise ValueError("Query cannot be empty")
        return v.strip()
    
class MonitoringResponse(BaseModel):
    """Response model for the monitoring agent."""
    result: str = Field(..., description="Result of the agent processing")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional details and data")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat(), 
                          description="Timestamp of the response")

class MonitoringAgent:
    """K1 Monitoring Agent implementation."""
    
    def __init__(self, 
                databricks_host: Optional[str] = None, 
                databricks_token: Optional[str] = None,
                databricks_profile: Optional[str] = None):
        """Initialize the monitoring agent.
        
        Args:
            databricks_host: Databricks workspace URL
            databricks_token: Databricks API token
            databricks_profile: Databricks CLI profile name
        """
        logger.info("Initializing Monitoring Agent")
        
        # Initialize our tools
        self.databricks_tools = DatabricksTools(
            host=databricks_host or os.getenv("DATABRICKS_HOST"),
            token=databricks_token or os.getenv("DATABRICKS_TOKEN"),
            profile=databricks_profile or os.getenv("DATABRICKS_PROFILE")
        )
        
        # Initialize pattern matchers for query classification
        self._init_query_patterns()
        
    def _init_query_patterns(self):
        """Initialize regex patterns for query classification."""
        self.query_patterns = {
            "workspace_status": [
                r"(workspace|environment)\s+(status|health)",
                r"(status|health)\s+of\s+(workspace|environment)",
                r"(how|is)\s+(is|the)\s+(workspace|environment)\s+(doing|performing)",
            ],
            "list_clusters": [
                r"(list|show|get)\s+(all\s+)?(clusters)",
                r"what\s+clusters\s+(are|do\s+we\s+have)",
                r"cluster\s+list",
            ],
            "cluster_details": [
                r"(details|info|information)\s+(about|for|on)\s+cluster\s+(\w+)",
                r"cluster\s+(\w+)\s+(details|info)",
            ],
            "list_jobs": [
                r"(list|show|get)\s+(all\s+)?(jobs)",
                r"(what|which)\s+jobs\s+(are|do\s+we\s+have)",
                r"job\s+list",
            ],
            "recent_job_runs": [
                r"(recent|latest)\s+job\s+runs",
                r"(list|show|get)\s+(recent|latest)\s+job\s+(runs|executions)",
            ],
            "get_logs": [
                r"(get|fetch|show)\s+(cluster|job)?\s*logs",
                r"(logs|log)\s+(for|from)\s+(cluster|job)\s+(\w+)",
            ],
            "get_activity": [
                r"(get|fetch|show)\s+(recent\s+)?(activity|activities)",
                r"(what|which)\s+activity\s+(happened|occurred|took\s+place)",
                r"(recent|latest)\s+activity",
            ],
        }
        
        # Compile all patterns
        self.compiled_patterns = {
            category: [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
            for category, patterns in self.query_patterns.items()
        }
        
    async def process_request(self, request: MonitoringRequest) -> MonitoringResponse:
        """Process a monitoring request.
        
        Args:
            request: The monitoring request to process
            
        Returns:
            The response to the request
        """
        logger.info(f"Processing request: {request.query}")
        
        # Classify the query to determine what action to take
        action, params = self._classify_query(request.query, request.parameters)
        
        if action == "unknown":
            return MonitoringResponse(
                result="I'm not sure how to process that query. Please try a different query related to Databricks monitoring.",
                details={"status": "error", "action": "unknown"}
            )
        
        # Execute the appropriate action
        try:
            result = await self._execute_action(action, params)
            if isinstance(result, dict) and result.get("status") == "error":
                return MonitoringResponse(
                    result=f"Error processing request: {result.get('message', 'Unknown error')}",
                    details=result
                )
            
            # Format the response based on the action and result
            if action == "workspace_status":
                if result.get("status") == "available":
                    response_text = (
                        f"Workspace Status: AVAILABLE\n"
                        f"Total Clusters: {result.get('total_clusters', 0)}\n"
                        f"Active Clusters: {result.get('active_clusters', 0)}\n"
                        f"Workspaces: {result.get('workspaces', 0)}"
                    )
                else:
                    response_text = f"Workspace Status: {result.get('status', 'UNKNOWN')}"
                
            elif action == "list_clusters":
                if not result:
                    response_text = "No clusters found."
                else:
                    response_text = f"Found {len(result)} clusters:\n\n"
                    for i, cluster in enumerate(result[:10], 1):
                        response_text += (
                            f"{i}. {cluster.get('name', 'Unknown')} "
                            f"(ID: {cluster.get('cluster_id', 'Unknown')}, "
                            f"State: {cluster.get('state', 'Unknown')})\n"
                        )
                    if len(result) > 10:
                        response_text += f"\n...and {len(result) - 10} more clusters."
            
            elif action == "cluster_details":
                if not result or result.get("status") == "error":
                    response_text = f"Could not find details for cluster {params.get('cluster_id', 'Unknown')}."
                else:
                    response_text = (
                        f"Cluster: {result.get('cluster_name', 'Unknown')}\n"
                        f"ID: {result.get('cluster_id', 'Unknown')}\n"
                        f"State: {result.get('state', 'Unknown')}\n"
                        f"Creator: {result.get('creator_user_name', 'Unknown')}\n"
                        f"Spark Version: {result.get('spark_version', 'Unknown')}\n"
                        f"Node Type: {result.get('node_type_id', 'Unknown')}\n"
                        f"Workers: {result.get('num_workers', 0)}"
                    )
            
            elif action == "list_jobs":
                if not result:
                    response_text = "No jobs found."
                else:
                    response_text = f"Found {len(result)} jobs:\n\n"
                    for i, job in enumerate(result[:10], 1):
                        response_text += (
                            f"{i}. {job.get('name', 'Unknown')} "
                            f"(ID: {job.get('job_id', 'Unknown')})\n"
                        )
                    if len(result) > 10:
                        response_text += f"\n...and {len(result) - 10} more jobs."
            
            elif action == "recent_job_runs":
                if not result:
                    response_text = "No job runs found."
                else:
                    response_text = f"Found {len(result)} recent job runs:\n\n"
                    for i, run in enumerate(result[:10], 1):
                        response_text += (
                            f"{i}. {run.get('run_name', 'Unknown')} "
                            f"(Run ID: {run.get('run_id', 'Unknown')}, "
                            f"State: {run.get('state', 'Unknown')})\n"
                        )
                    if len(result) > 10:
                        response_text += f"\n...and {len(result) - 10} more job runs."
            
            elif action == "get_logs":
                if not result:
                    response_text = "No logs found."
                else:
                    response_text = f"Found {len(result)} log entries:\n\n"
                    for i, log in enumerate(result[:5], 1):
                        response_text += (
                            f"{i}. [{log.get('timestamp', 'Unknown')}] "
                            f"{log.get('message', 'Unknown')}\n"
                        )
                    if len(result) > 5:
                        response_text += f"\n...and {len(result) - 5} more log entries."
            
            elif action == "get_activity":
                response_text = (
                    f"Workspace activity summary (last {result.get('days', 7)} days):\n\n"
                    f"Total Activities: {result.get('count', 0)}\n"
                    f"Cluster Activities: {result.get('cluster_count', 0)}\n"
                    f"Job Activities: {result.get('job_count', 0)}\n"
                    f"Notebook Activities: {result.get('notebook_count', 0)}\n"
                    f"Warehouse Activities: {result.get('warehouse_count', 0)}"
                )
            
            else:
                response_text = "Processed request successfully."
            
            return MonitoringResponse(
                result=response_text,
                details={"action": action, "data": result}
            )
            
        except Exception as e:
            logger.error(f"Error processing request: {e}", exc_info=True)
            return MonitoringResponse(
                result=f"Error processing request: {str(e)}",
                details={"status": "error", "message": str(e), "action": action}
            )
    
    def _classify_query(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> tuple[str, Dict[str, Any]]:
        """Classify the query to determine what action to take.
        
        Args:
            query: The query to classify
            parameters: Optional parameters for the request
            
        Returns:
            Tuple of (action, parameters)
        """
        # Use explicit parameters if they indicate the action
        if parameters and "action" in parameters:
            return parameters["action"], parameters
        
        # Check against our patterns
        for category, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                match = pattern.search(query)
                if match:
                    # Extract parameters from the query if possible
                    params = self._extract_params(category, query, match, parameters)
                    return category, params
        
        # If no pattern matched, return unknown
        return "unknown", parameters or {}
    
    def _extract_params(self, category: str, query: str, match, parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Extract parameters from the query based on the category and pattern match.
        
        Args:
            category: The matched category
            query: The original query
            match: The regex match object
            parameters: Optional parameters for the request
            
        Returns:
            Dictionary of parameters
        """
        # Start with any existing parameters
        params = parameters.copy() if parameters else {}
        
        # Extract specific parameters based on category
        if category == "cluster_details":
            # Try to extract cluster_id from the query
            cluster_id_match = re.search(r"cluster\s+(\w+)", query)
            if cluster_id_match:
                params["cluster_id"] = cluster_id_match.group(1)
        
        elif category == "get_logs":
            # Try to extract cluster_id or run_id
            cluster_match = re.search(r"cluster\s+(\w+)", query)
            run_match = re.search(r"run\s+(\w+)", query)
            log_type_match = re.search(r"(audit|stdout|stderr)\s+logs", query)
            
            if cluster_match:
                params["cluster_id"] = cluster_match.group(1)
            if run_match:
                params["run_id"] = run_match.group(1)
            if log_type_match:
                params["log_type"] = log_type_match.group(1)
                
            # Try to extract days
            days_match = re.search(r"(last|past)\s+(\d+)\s+(day|days)", query)
            if days_match:
                params["days"] = int(days_match.group(2))
        
        elif category == "get_activity":
            # Try to extract days
            days_match = re.search(r"(last|past)\s+(\d+)\s+(day|days)", query)
            if days_match:
                params["days"] = int(days_match.group(2))
                
            # Try to extract limit
            limit_match = re.search(r"(top|limit)\s+(\d+)", query)
            if limit_match:
                params["limit"] = int(limit_match.group(2))
        
        return params
    
    async def _execute_action(self, action: str, params: Dict[str, Any]) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Execute the specified action with the given parameters.
        
        Args:
            action: The action to execute
            params: Parameters for the action
            
        Returns:
            The result of the action
        """
        # Map actions to tool methods
        if action == "workspace_status":
            result = await self.databricks_tools.get_workspace_status()
        
        elif action == "list_clusters":
            result = await self.databricks_tools.list_clusters()
        
        elif action == "cluster_details":
            if "cluster_id" not in params:
                return {"status": "error", "message": "Cluster ID is required"}
            result = await self.databricks_tools.get_cluster_details(params["cluster_id"])
        
        elif action == "list_jobs":
            limit = params.get("limit", 50)
            name = params.get("name")
            result = await self.databricks_tools.list_jobs(limit=limit, name=name)
        
        elif action == "recent_job_runs":
            job_id = params.get("job_id")
            limit = params.get("limit", 25)
            result = await self.databricks_tools.get_recent_job_runs(job_id=job_id, limit=limit)
        
        elif action == "get_logs":
            cluster_id = params.get("cluster_id")
            run_id = params.get("run_id")
            log_type = params.get("log_type", "audit")
            days = params.get("days", 7)
            limit = params.get("limit", 100)
            result = await self.databricks_tools.get_logs(
                cluster_id=cluster_id,
                run_id=run_id,
                log_type=log_type,
                days=days,
                limit=limit
            )
        
        elif action == "get_activity":
            days = params.get("days", 7)
            limit = params.get("limit", 100)
            result = await self.databricks_tools.get_activity(days=days, limit=limit)
        
        else:
            result = {"status": "error", "message": f"Unsupported action: {action}"}
        
        return result 