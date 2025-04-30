"""
Databricks client wrapper for the K1 Monitoring Agent.

This module provides a wrapper around the Databricks SDK, maintaining 
a one-to-one mapping with REST API methods while allowing for easy mocking
during testing.
"""

import os
import json
from typing import Dict, Any, List, Optional, Union, Tuple
from opentelemetry import trace
from enum import Enum, auto

# Import tenacity for retries
try:
    from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
    TENACITY_AVAILABLE = True
except ImportError:
    TENACITY_AVAILABLE = False

# Import the Databricks SDK
try:
    import databricks
    from databricks.sdk import WorkspaceClient
    # We don't import service modules directly as they might not be available
    # from databricks.sdk.service import jobs, clusters, workspace, compute, sql
    
    # Check if we can create a WorkspaceClient
    DATABRICKS_SDK_AVAILABLE = True
    
    # Define ApiError and ResourceDoesNotExist
    try:
        from databricks.sdk.errors import ApiError, ResourceDoesNotExist
    except ImportError:
        # Create fallback exceptions if not available
        class ApiError(Exception):
            pass
        class ResourceDoesNotExist(Exception):
            pass
            
except ImportError:
    DATABRICKS_SDK_AVAILABLE = False
    # Fallback for when SDK is not available
    class ApiError(Exception):
        pass
    class ResourceDoesNotExist(Exception):
        pass

# Import our logging configuration
from .logging_config import get_logger

# Initialize logger
logger = get_logger(__name__)

# Get tracer for this module
tracer = trace.get_tracer(__name__)

class RunStatus(Enum):
    """Enum for Databricks job run status."""
    PENDING = auto()
    RUNNING = auto()
    TERMINATED = auto()
    SKIPPED = auto()
    INTERNAL_ERROR = auto()
    UNKNOWN = auto()

class DbxClient:
    """
    Wrapper for the Databricks SDK that provides a consistent interface
    and allows for easy mocking during testing.
    """
    
    def __init__(
        self,
        host: Optional[str] = None,
        token: Optional[str] = None,
        profile: Optional[str] = None,
        validate_env: bool = True
    ):
        """
        Initialize the Databricks client wrapper.
        
        Args:
            host: Databricks workspace URL (e.g., https://your-workspace.cloud.databricks.com)
            token: Databricks Personal Access Token
            profile: Databricks CLI profile to use for authentication
            validate_env: Whether to validate required environment variables
            
        Raises:
            ValueError: If validate_env is True and required environment variables are missing
            ImportError: If required dependencies are not installed
        """
        self.host = host or os.getenv("DATABRICKS_HOST")
        self.token = token or os.getenv("DATABRICKS_TOKEN")
        self.profile = profile
        self._client = None
        
        # Check for required dependencies
        if not DATABRICKS_SDK_AVAILABLE:
            error_msg = "Databricks SDK not installed. Install with 'pip install databricks-sdk'"
            logger.error(error_msg)
            raise ImportError(error_msg)
            
        if not TENACITY_AVAILABLE:
            logger.warning("Tenacity not installed. Retries will not be available. Install with 'pip install tenacity'")
        
        # Validate environment variables if required
        if validate_env and not self.profile:
            missing_vars = []
            if not self.host:
                missing_vars.append("DATABRICKS_HOST")
            if not self.token:
                missing_vars.append("DATABRICKS_TOKEN")
            
            if missing_vars:
                error_msg = f"Missing required environment variables: {', '.join(missing_vars)}"
                logger.error(error_msg)
                raise ValueError(error_msg)
        
        self._initialize_client()
    
    def _initialize_client(self) -> None:
        """Initialize the Databricks SDK client."""
        try:
            # Create the client using the provided credentials
            if self.profile:
                # Use profile-based authentication
                self._client = WorkspaceClient(profile=self.profile)
                logger.info(f"Initialized Databricks client using profile '{self.profile}'")
            else:
                # Use host and token authentication
                self._client = WorkspaceClient(
                    host=self.host,
                    token=self.token
                )
                logger.info(f"Initialized Databricks client for host: {self.host}")
        except Exception as e:
            logger.error(f"Failed to initialize Databricks client: {e}", exc_info=True)
            self._client = None
    
    def is_available(self) -> bool:
        """Check if the Databricks client is available and initialized."""
        return self._client is not None
    
    def _log_api_call(self, operation: str, params: Dict[str, Any], success: bool, 
                      result_size: Optional[int] = None, error: Optional[Exception] = None) -> None:
        """
        Log structured information about an API call.
        
        Args:
            operation: The API operation being performed
            params: Parameters passed to the operation
            success: Whether the operation was successful
            result_size: Size of the result if applicable
            error: Exception if the operation failed
        """
        log_data = {
            "operation": operation,
            "params": {k: v for k, v in params.items() if v is not None},
            "success": success,
            "host": self.host
        }
        
        if result_size is not None:
            log_data["result_size"] = result_size
            
        if error:
            log_data["error"] = str(error)
            log_data["error_type"] = error.__class__.__name__
            
        # Log in a structured format
        if success:
            logger.info(f"Databricks API call: {json.dumps(log_data)}")
        else:
            logger.error(f"Databricks API call failed: {json.dumps(log_data)}")
    
    # Define retry decorator if tenacity is available
    if TENACITY_AVAILABLE:
        api_retry = retry(
            retry=retry_if_exception_type(ApiError),
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            before_sleep=lambda retry_state: logger.warning(
                f"Retrying API call (attempt {retry_state.attempt_number}/3) after error: {retry_state.outcome.exception()}"
            )
        )
    else:
        # No-op decorator if tenacity is not available
        def api_retry(func):
            return func
    
    # =========================================================================
    # Jobs API Methods
    # =========================================================================
    
    @api_retry
    def list_jobs(self, 
                  limit: Optional[int] = None,
                  name: Optional[str] = None,
                  page_token: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List jobs in the workspace.
        
        Args:
            limit: Maximum number of jobs to return
            name: Filter by job name
            page_token: Token for pagination
            
        Returns:
            List of job objects
        """
        operation = "list_jobs"
        params = {"limit": limit, "name": name, "page_token": page_token}
        
        with tracer.start_as_current_span("databricks.list_jobs") as span:
            if not self.is_available():
                error_msg = "Databricks client not available"
                logger.error(error_msg)
                self._log_api_call(operation, params, False, error=ValueError(error_msg))
                return []
            
            try:
                # In newer SDK versions, list() returns a generator directly
                jobs_generator = self._client.jobs.list(
                    limit=limit, 
                    name=name,
                    page_token=page_token
                )
                
                # Convert generator to list and extract dictionaries
                jobs_list = list(jobs_generator)
                result = [job.as_dict() if hasattr(job, 'as_dict') else job for job in jobs_list]
                
                # Apply limit if specified
                if limit and len(result) > limit:
                    result = result[:limit]
                
                # Set span attributes
                span.set_attribute("jobs.count", len(result))
                if name:
                    span.set_attribute("jobs.filter.name", name)
                
                # Log the successful call with structured data
                self._log_api_call(operation, params, True, result_size=len(result))
                
                return result
            except Exception as e:
                logger.error(f"Error listing jobs: {e}", exc_info=True)
                span.record_exception(e)
                span.set_status(trace.Status(trace.StatusCode.ERROR))
                
                # Log the failed call
                self._log_api_call(operation, params, False, error=e)
                return []
    
    @api_retry
    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a job by ID.
        
        Args:
            job_id: The job ID
            
        Returns:
            Job object or None if not found
        """
        operation = "get_job"
        params = {"job_id": job_id}
        
        with tracer.start_as_current_span("databricks.get_job") as span:
            span.set_attribute("job.id", job_id)
            
            if not self.is_available():
                error_msg = "Databricks client not available"
                logger.error(error_msg)
                self._log_api_call(operation, params, False, error=ValueError(error_msg))
                return None
            
            try:
                job = self._client.jobs.get(job_id=job_id)
                result = job.as_dict() if job else None
                
                # Log the successful call with structured data
                self._log_api_call(operation, params, True)
                
                return result
            except ResourceDoesNotExist:
                logger.warning(f"Job {job_id} not found")
                self._log_api_call(operation, params, False, error=ResourceDoesNotExist(f"Job {job_id} not found"))
                return None
            except Exception as e:
                logger.error(f"Error getting job {job_id}: {e}", exc_info=True)
                span.record_exception(e)
                span.set_status(trace.Status(trace.StatusCode.ERROR))
                
                # Log the failed call
                self._log_api_call(operation, params, False, error=e)
                return None
    
    @api_retry
    def list_runs(self, 
                  job_id: Optional[str] = None,
                  active_only: bool = False,
                  completed_only: bool = False,
                  offset: int = 0,
                  limit: int = 25) -> List[Dict[str, Any]]:
        """
        List job runs.
        
        Args:
            job_id: Filter by job ID
            active_only: List only active runs
            completed_only: List only completed runs
            offset: Pagination offset
            limit: Maximum number of runs to return
            
        Returns:
            List of job run objects
        """
        operation = "list_runs"
        params = {
            "job_id": job_id,
            "active_only": active_only,
            "completed_only": completed_only,
            "offset": offset,
            "limit": limit
        }
        
        with tracer.start_as_current_span("databricks.list_runs") as span:
            if job_id:
                span.set_attribute("job.id", job_id)
            span.set_attribute("active_only", active_only)
            span.set_attribute("completed_only", completed_only)
            
            if not self.is_available():
                error_msg = "Databricks client not available"
                logger.error(error_msg)
                self._log_api_call(operation, params, False, error=ValueError(error_msg))
                return []
            
            try:
                # In newer SDK versions, list_runs() returns a generator directly
                runs_generator = self._client.jobs.list_runs(
                    job_id=job_id,
                    active_only=active_only,
                    completed_only=completed_only,
                    offset=offset,
                    limit=limit
                )
                
                # Convert generator to list and extract dictionaries
                runs_list = list(runs_generator)
                result = [run.as_dict() if hasattr(run, 'as_dict') else run for run in runs_list]
                
                # Apply limit if specified
                if limit and len(result) > limit:
                    result = result[:limit]
                
                # Set span attributes and log data
                span.set_attribute("runs.count", len(result))
                self._log_api_call(operation, params, True, result_size=len(result))
                
                return result
            except Exception as e:
                logger.error(f"Error listing job runs: {e}", exc_info=True)
                span.record_exception(e)
                span.set_status(trace.Status(trace.StatusCode.ERROR))
                self._log_api_call(operation, params, False, error=e)
                return []
    
    @api_retry
    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a job run by ID.
        
        Args:
            run_id: The run ID
            
        Returns:
            Run object or None if not found
        """
        operation = "get_run"
        params = {"run_id": run_id}
        
        with tracer.start_as_current_span("databricks.get_run") as span:
            span.set_attribute("run.id", run_id)
            
            if not self.is_available():
                error_msg = "Databricks client not available"
                logger.error(error_msg)
                self._log_api_call(operation, params, False, error=ValueError(error_msg))
                return None
            
            try:
                run = self._client.jobs.get_run(run_id=run_id)
                result = run.as_dict() if run else None
                self._log_api_call(operation, params, True)
                return result
            except ResourceDoesNotExist:
                logger.warning(f"Run {run_id} not found")
                self._log_api_call(operation, params, False, error=ResourceDoesNotExist(f"Run {run_id} not found"))
                return None
            except Exception as e:
                logger.error(f"Error getting run {run_id}: {e}", exc_info=True)
                span.record_exception(e)
                span.set_status(trace.Status(trace.StatusCode.ERROR))
                self._log_api_call(operation, params, False, error=e)
                return None
    
    def get_run_status(self, run_id: str) -> Tuple[RunStatus, Optional[str]]:
        """
        Get the status of a job run.
        
        Args:
            run_id: The run ID
            
        Returns:
            Tuple containing (status enum, status message)
        """
        operation = "get_run_status"
        params = {"run_id": run_id}
        
        with tracer.start_as_current_span("databricks.get_run_status") as span:
            span.set_attribute("run.id", run_id)
            
            if not self.is_available():
                error_msg = "Databricks client not available"
                logger.error(error_msg)
                self._log_api_call(operation, params, False, error=ValueError(error_msg))
                return RunStatus.UNKNOWN, "Client not available"
            
            try:
                run = self._client.jobs.get_run(run_id=run_id)
                
                if not run or not run.state:
                    self._log_api_call(operation, params, False, 
                                      error=ValueError(f"Run {run_id} has no state information"))
                    return RunStatus.UNKNOWN, "No state information"
                
                # Extract status information
                life_cycle_state = run.state.life_cycle_state if run.state.life_cycle_state else "UNKNOWN"
                state_message = run.state.state_message if run.state.state_message else None
                
                # Map to our status enum
                status_map = {
                    "PENDING": RunStatus.PENDING,
                    "RUNNING": RunStatus.RUNNING,
                    "TERMINATING": RunStatus.RUNNING,
                    "TERMINATED": RunStatus.TERMINATED,
                    "SKIPPED": RunStatus.SKIPPED,
                    "INTERNAL_ERROR": RunStatus.INTERNAL_ERROR
                }
                
                status = status_map.get(life_cycle_state, RunStatus.UNKNOWN)
                
                # Set span attributes and log data
                span.set_attribute("run.status", life_cycle_state)
                self._log_api_call(operation, params, True)
                
                return status, state_message
            except Exception as e:
                logger.error(f"Error getting run status for {run_id}: {e}", exc_info=True)
                span.record_exception(e)
                span.set_status(trace.Status(trace.StatusCode.ERROR))
                self._log_api_call(operation, params, False, error=e)
                return RunStatus.UNKNOWN, str(e)
    
    # =========================================================================
    # Clusters API Methods
    # =========================================================================
    
    @api_retry
    def list_clusters(self) -> List[Dict[str, Any]]:
        """
        List clusters in the workspace.
        
        Returns:
            List of cluster objects
        """
        operation = "list_clusters"
        params = {}
        
        with tracer.start_as_current_span("databricks.list_clusters") as span:
            if not self.is_available():
                error_msg = "Databricks client not available"
                logger.error(error_msg)
                self._log_api_call(operation, params, False, error=ValueError(error_msg))
                return []
            
            try:
                # Convert generator to list directly
                clusters_list = list(self._client.clusters.list())
                result = [cluster.as_dict() if hasattr(cluster, 'as_dict') else cluster for cluster in clusters_list]
                
                # Set span attributes and log data
                span.set_attribute("clusters.count", len(result))
                self._log_api_call(operation, params, True, result_size=len(result))
                
                return result
            except Exception as e:
                logger.error(f"Error listing clusters: {e}", exc_info=True)
                span.record_exception(e)
                span.set_status(trace.Status(trace.StatusCode.ERROR))
                self._log_api_call(operation, params, False, error=e)
                return []
    
    @api_retry
    def get_cluster(self, cluster_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a cluster by ID.
        
        Args:
            cluster_id: The cluster ID
            
        Returns:
            Cluster object or None if not found
        """
        operation = "get_cluster"
        params = {"cluster_id": cluster_id}
        
        with tracer.start_as_current_span("databricks.get_cluster") as span:
            span.set_attribute("cluster.id", cluster_id)
            
            if not self.is_available():
                error_msg = "Databricks client not available"
                logger.error(error_msg)
                self._log_api_call(operation, params, False, error=ValueError(error_msg))
                return None
            
            try:
                cluster = self._client.clusters.get(cluster_id=cluster_id)
                result = cluster.as_dict() if cluster else None
                self._log_api_call(operation, params, True)
                return result
            except ResourceDoesNotExist:
                logger.warning(f"Cluster {cluster_id} not found")
                self._log_api_call(operation, params, False, 
                                  error=ResourceDoesNotExist(f"Cluster {cluster_id} not found"))
                return None
            except Exception as e:
                logger.error(f"Error getting cluster {cluster_id}: {e}", exc_info=True)
                span.record_exception(e)
                span.set_status(trace.Status(trace.StatusCode.ERROR))
                self._log_api_call(operation, params, False, error=e)
                return None
    
    # =========================================================================
    # Workspace API Methods
    # =========================================================================
    
    @api_retry
    def list_workspaces(self, path: str) -> List[Dict[str, Any]]:
        """
        List objects in a workspace directory.
        
        Args:
            path: The directory path
            
        Returns:
            List of workspace objects
        """
        operation = "list_workspaces"
        params = {"path": path}
        
        with tracer.start_as_current_span("databricks.list_workspaces") as span:
            span.set_attribute("workspace.path", path)
            
            if not self.is_available():
                error_msg = "Databricks client not available"
                logger.error(error_msg)
                self._log_api_call(operation, params, False, error=ValueError(error_msg))
                return []
            
            try:
                objects_list = list(self._client.workspace.list(path=path))
                result = [obj.as_dict() for obj in objects_list]
                
                # Set span attributes and log data
                span.set_attribute("workspace.objects.count", len(result))
                self._log_api_call(operation, params, True, result_size=len(result))
                
                return result
            except Exception as e:
                logger.error(f"Error listing workspace objects at {path}: {e}", exc_info=True)
                span.record_exception(e)
                span.set_status(trace.Status(trace.StatusCode.ERROR))
                self._log_api_call(operation, params, False, error=e)
                return []
    
    @api_retry
    def get_workspace_status(self, workspace_id: str) -> Optional[Dict[str, Any]]:
        """
        Get workspace status.
        
        Args:
            workspace_id: The workspace ID
            
        Returns:
            Workspace status or None if not found
        """
        operation = "get_workspace_status"
        params = {"workspace_id": workspace_id}
        
        with tracer.start_as_current_span("databricks.get_workspace_status") as span:
            span.set_attribute("workspace.id", workspace_id)
            
            if not self.is_available():
                error_msg = "Databricks client not available"
                logger.error(error_msg)
                self._log_api_call(operation, params, False, error=ValueError(error_msg))
                return None
            
            try:
                status = self._client.workspace.get_status(workspace_id=workspace_id)
                result = status.as_dict() if status else None
                self._log_api_call(operation, params, True)
                return result
            except ResourceDoesNotExist:
                logger.warning(f"Workspace {workspace_id} not found")
                self._log_api_call(operation, params, False, 
                                  error=ResourceDoesNotExist(f"Workspace {workspace_id} not found"))
                return None
            except Exception as e:
                logger.error(f"Error getting workspace status for {workspace_id}: {e}", exc_info=True)
                span.record_exception(e)
                span.set_status(trace.Status(trace.StatusCode.ERROR))
                self._log_api_call(operation, params, False, error=e)
                return None
    
    # =========================================================================
    # SQL API Methods
    # =========================================================================
    
    @api_retry
    def list_warehouses(self) -> List[Dict[str, Any]]:
        """
        List SQL warehouses in the workspace.
        
        Returns:
            List of SQL warehouse objects
        """
        operation = "list_warehouses"
        params = {}
        
        with tracer.start_as_current_span("databricks.list_warehouses") as span:
            if not self.is_available():
                error_msg = "Databricks client not available"
                logger.error(error_msg)
                self._log_api_call(operation, params, False, error=ValueError(error_msg))
                return []
            
            try:
                warehouses_list = list(self._client.warehouses.list())
                result = [warehouse.as_dict() for warehouse in warehouses_list]
                
                # Set span attributes and log data
                span.set_attribute("warehouses.count", len(result))
                self._log_api_call(operation, params, True, result_size=len(result))
                
                return result
            except Exception as e:
                logger.error(f"Error listing SQL warehouses: {e}", exc_info=True)
                span.record_exception(e)
                span.set_status(trace.Status(trace.StatusCode.ERROR))
                self._log_api_call(operation, params, False, error=e)
                return []
    
    @api_retry
    def get_warehouse(self, warehouse_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a SQL warehouse by ID.
        
        Args:
            warehouse_id: The warehouse ID
            
        Returns:
            Warehouse object or None if not found
        """
        operation = "get_warehouse"
        params = {"warehouse_id": warehouse_id}
        
        with tracer.start_as_current_span("databricks.get_warehouse") as span:
            span.set_attribute("warehouse.id", warehouse_id)
            
            if not self.is_available():
                error_msg = "Databricks client not available"
                logger.error(error_msg)
                self._log_api_call(operation, params, False, error=ValueError(error_msg))
                return None
            
            try:
                warehouse = self._client.warehouses.get(id=warehouse_id)
                result = warehouse.as_dict() if warehouse else None
                self._log_api_call(operation, params, True)
                return result
            except ResourceDoesNotExist:
                logger.warning(f"Warehouse {warehouse_id} not found")
                self._log_api_call(operation, params, False, 
                                  error=ResourceDoesNotExist(f"Warehouse {warehouse_id} not found"))
                return None
            except Exception as e:
                logger.error(f"Error getting SQL warehouse {warehouse_id}: {e}", exc_info=True)
                span.record_exception(e)
                span.set_status(trace.Status(trace.StatusCode.ERROR))
                self._log_api_call(operation, params, False, error=e)
                return None
    
    # =========================================================================
    # Logging API Methods
    # =========================================================================
    
    @api_retry
    def get_logs(self, 
                cluster_id: Optional[str] = None,
                run_id: Optional[str] = None,
                start_time: Optional[int] = None,
                end_time: Optional[int] = None,
                page_token: Optional[str] = None,
                offset: int = 0,
                limit: int = 100,
                log_type: str = "audit") -> List[Dict[str, Any]]:
        """
        Get logs from Databricks cluster, job run, or workspace.
        
        Args:
            cluster_id: Filter logs by cluster ID
            run_id: Filter logs by run ID
            start_time: Start time in milliseconds since epoch
            end_time: End time in milliseconds since epoch
            page_token: Token for pagination
            offset: Pagination offset
            limit: Maximum number of log entries to return
            log_type: Type of logs to retrieve (audit, cluster, run)
            
        Returns:
            List of log entries
        """
        operation = "get_logs"
        params = {
            "cluster_id": cluster_id,
            "run_id": run_id,
            "start_time": start_time,
            "end_time": end_time,
            "page_token": page_token,
            "offset": offset,
            "limit": limit,
            "log_type": log_type
        }
        
        with tracer.start_as_current_span("databricks.get_logs") as span:
            if cluster_id:
                span.set_attribute("cluster.id", cluster_id)
            if run_id:
                span.set_attribute("run.id", run_id)
            span.set_attribute("log_type", log_type)
            
            if not self.is_available():
                error_msg = "Databricks client not available"
                logger.error(error_msg)
                self._log_api_call(operation, params, False, error=ValueError(error_msg))
                return []
            
            try:
                results = []
                
                # Get cluster logs if cluster_id is provided
                if cluster_id:
                    # Try to get logs from cluster-level API
                    try:
                        logs = self._client.clusters.get_events(
                            cluster_id=cluster_id,
                            start_time=start_time,
                            end_time=end_time,
                            limit=limit,
                            offset=offset
                        )
                        
                        # Convert to list of dictionaries
                        events = list(logs)
                        for event in events:
                            if hasattr(event, 'as_dict'):
                                results.append(event.as_dict())
                            else:
                                results.append(event)
                    except Exception as cluster_error:
                        logger.warning(f"Could not get cluster events: {cluster_error}")
                
                # Get run logs if run_id is provided
                if run_id and not results:
                    try:
                        # Try to access logs using the runs API
                        logs = self._client.jobs.get_run_output(run_id=run_id)
                        if hasattr(logs, 'as_dict'):
                            log_output = logs.as_dict()
                            
                            # Extract logs from different sources
                            if 'logs' in log_output:
                                results.append({"logs": log_output['logs']})
                            if 'error' in log_output:
                                results.append({"error": log_output['error']})
                        else:
                            results.append(logs)
                    except Exception as run_error:
                        logger.warning(f"Could not get run logs: {run_error}")
                
                # Fall back to querying workspace audit logs
                if not results and log_type == "audit":
                    try:
                        # Try to get audit logs using a direct API call
                        import time
                        import requests
                        
                        # Default to last 24 hours if not specified
                        if not start_time:
                            start_time = int((time.time() - 86400) * 1000)  # 24 hours ago
                        if not end_time:
                            end_time = int(time.time() * 1000)  # now
                        
                        # Make direct API call instead of using the SDK
                        headers = {
                            "Authorization": f"Bearer {self.token}",
                            "Content-Type": "application/json"
                        }
                        
                        # Build the request URL - use the audit logs API which is more widely available
                        api_url = f"{self.host}/api/2.0/accounts/get-audit-logs"
                        
                        # Format the start and end times for the audit logs API
                        from datetime import datetime
                        start_time_str = datetime.fromtimestamp(start_time / 1000).strftime("%Y-%m-%dT%H:%M:%SZ")
                        end_time_str = datetime.fromtimestamp(end_time / 1000).strftime("%Y-%m-%dT%H:%M:%SZ")
                        
                        # Build the request body
                        filter_body = {
                            "start_time": start_time_str,
                            "end_time": end_time_str,
                            "max_results": limit
                        }
                        
                        # Make the API call
                        response = requests.post(api_url, headers=headers, json=filter_body)
                        
                        if response.status_code == 200:
                            data = response.json()
                            if "audit_logs" in data:
                                results = data["audit_logs"]
                                logger.info(f"Retrieved {len(results)} audit log entries")
                            else:
                                logger.info("No audit logs found in the response")
                        else:
                            logger.warning(f"Audit logs API call failed: {response.status_code} - {response.text}")
                            
                            # Try the Workspace Audit Logs API instead
                            try:
                                api_url = f"{self.host}/api/2.0/workspace-logs"
                                
                                # Build the request body
                                filter_body = {
                                    "start_time": start_time_str,
                                    "end_time": end_time_str,
                                    "page_size": limit
                                }
                                
                                # Make the API call
                                response = requests.get(api_url, headers=headers, params=filter_body)
                                
                                if response.status_code == 200:
                                    data = response.json()
                                    if "events" in data:
                                        results = data["events"]
                                        logger.info(f"Retrieved {len(results)} workspace log entries")
                                    else:
                                        logger.info("No workspace logs found in the response")
                                else:
                                    logger.warning(f"Workspace logs API call failed: {response.status_code} - {response.text}")
                            except Exception as workspace_error:
                                logger.warning(f"Could not get workspace logs: {workspace_error}")
                    except Exception as audit_error:
                        logger.warning(f"Could not get audit logs: {audit_error}")
                
                # Apply limit if needed
                if limit and len(results) > limit:
                    results = results[:limit]
                
                # Log the API call
                self._log_api_call(operation, params, True, result_size=len(results))
                span.set_attribute("logs.count", len(results))
                
                return results
            except Exception as e:
                logger.error(f"Error getting logs: {e}", exc_info=True)
                span.record_exception(e)
                span.set_status(trace.Status(trace.StatusCode.ERROR))
                self._log_api_call(operation, params, False, error=e)
                return []
    
    @api_retry
    def get_activity(self, days: int = 7, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get user activity information from the workspace.
        
        Args:
            days: Number of days of history to retrieve
            limit: Maximum number of activities to return
            
        Returns:
            List of activity entries
        """
        operation = "get_activity"
        params = {
            "days": days,
            "limit": limit
        }
        
        with tracer.start_as_current_span("databricks.get_activity") as span:
            span.set_attribute("days", days)
            
            if not self.is_available():
                error_msg = "Databricks client not available"
                logger.error(error_msg)
                self._log_api_call(operation, params, False, error=ValueError(error_msg))
                return []
            
            try:
                results = []
                
                # First try to get users in the workspace to reconstruct activity
                try:
                    import requests
                    import time
                    from datetime import datetime, timedelta
                    
                    # Make direct API call instead of using the SDK for SCIM API
                    headers = {
                        "Authorization": f"Bearer {self.token}",
                        "Content-Type": "application/scim+json"
                    }
                    
                    # Build the request URL
                    api_url = f"{self.host}/api/2.0/preview/scim/v2/Users"
                    
                    # Construct query to get users
                    response = requests.get(api_url, headers=headers)
                    
                    if response.status_code == 200:
                        data = response.json()
                        if "Resources" in data:
                            users = data["Resources"]
                            logger.info(f"Retrieved {len(users)} user records")
                            
                            for user in users:
                                # Extract last activity information if available
                                user_id = user.get("id")
                                name = user.get("userName")
                                display_name = user.get("displayName", name)
                                active = user.get("active", False)
                                
                                # Try to get last login time from meta
                                meta = user.get("meta", {})
                                last_modified = meta.get("lastModified")
                                
                                # For each user, build an activity record
                                activity = {
                                    "user_id": user_id,
                                    "user_name": name,
                                    "display_name": display_name,
                                    "active": active,
                                    "last_activity": last_modified or "Unknown",
                                    "timestamp": int(time.time() * 1000),  # Current time in ms
                                    "type": "user_information"
                                }
                                
                                results.append(activity)
                        else:
                            logger.info("No users found in the response")
                    else:
                        logger.warning(f"SCIM API call failed: {response.status_code} - {response.text}")
                        
                except Exception as user_error:
                    logger.warning(f"Could not get user activity information: {user_error}")
                
                # Next, try to get cluster usage information
                try:
                    clusters = self.list_clusters()
                    for cluster in clusters:
                        cluster_id = cluster.get("cluster_id")
                        cluster_name = cluster.get("cluster_name")
                        creator = cluster.get("creator_user_name", "Unknown")
                        
                        # Try to parse creation timestamp
                        created = cluster.get("cluster_source", {}).get("created_time")
                        
                        activity = {
                            "cluster_id": cluster_id,
                            "cluster_name": cluster_name,
                            "creator": creator,
                            "created_time": created,
                            "state": cluster.get("state"),
                            "timestamp": int(time.time() * 1000),
                            "type": "cluster_information"
                        }
                        
                        results.append(activity)
                except Exception as cluster_error:
                    logger.warning(f"Could not get cluster activity: {cluster_error}")
                
                # Also get workspace objects to provide some workspace activity
                try:
                    # Get objects from workspace root
                    workspace_objects = self.list_workspaces("/")
                    
                    for obj in workspace_objects:
                        activity = {
                            "path": obj.get("path"),
                            "object_type": obj.get("object_type"),
                            "language": obj.get("language") if "language" in obj else None,
                            "object_id": obj.get("object_id"),
                            "timestamp": int(time.time() * 1000),
                            "type": "workspace_object"
                        }
                        
                        results.append(activity)
                except Exception as workspace_error:
                    logger.warning(f"Could not get workspace objects: {workspace_error}")
                
                # Apply limit if needed
                if limit and len(results) > limit:
                    results = results[:limit]
                
                # Log the API call
                self._log_api_call(operation, params, True, result_size=len(results))
                span.set_attribute("activity.count", len(results))
                
                return results
            except Exception as e:
                logger.error(f"Error getting activity: {e}", exc_info=True)
                span.record_exception(e)
                span.set_status(trace.Status(trace.StatusCode.ERROR))
                self._log_api_call(operation, params, False, error=e)
                return [] 