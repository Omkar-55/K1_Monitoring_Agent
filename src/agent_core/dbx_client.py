"""
Databricks client wrapper for the K1 Monitoring Agent.

This module provides a wrapper around the Databricks SDK, maintaining 
a one-to-one mapping with REST API methods while allowing for easy mocking
during testing.
"""

import os
import json
import time
from typing import Dict, Any, List, Optional, Union, Tuple
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
                      result_size: Optional[int] = None, error: Optional[Exception] = None,
                      duration_seconds: Optional[float] = None) -> None:
        """
        Log structured information about an API call.
        
        Args:
            operation: The API operation being performed
            params: Parameters passed to the operation
            success: Whether the operation was successful
            result_size: Size of the result if applicable
            error: Exception if the operation failed
            duration_seconds: Time taken to complete the operation in seconds
        """
        log_data = {
            "operation": operation,
            "params": {k: v for k, v in params.items() if v is not None},
            "success": success,
            "host": self.host
        }
        
        if result_size is not None:
            log_data["result_size"] = result_size
            
        if duration_seconds is not None:
            log_data["duration_seconds"] = f"{duration_seconds:.4f}"
            
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
        start_time = time.time()
        
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
            
            # Log duration and other metadata
            duration = time.time() - start_time
            logger.info(f"Listed {len(result)} jobs in {duration:.4f} seconds")
            
            # Log the successful call with structured data
            self._log_api_call(
                operation, 
                params, 
                True, 
                result_size=len(result),
                duration_seconds=duration
            )
            
            return result
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Error listing jobs: {e}", exc_info=True)
            logger.error(f"Operation failed after {duration:.4f} seconds")
            
            # Log the failed call
            self._log_api_call(
                operation, 
                params, 
                False, 
                error=e,
                duration_seconds=duration
            )
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
        start_time = time.time()
        
        if not self.is_available():
            error_msg = "Databricks client not available"
            logger.error(error_msg)
            self._log_api_call(operation, params, False, error=ValueError(error_msg))
            return None
        
        try:
            job = self._client.jobs.get(job_id=job_id)
            result = job.as_dict() if job else None
        
            # Log duration
            duration = time.time() - start_time
            logger.info(f"Retrieved job {job_id} in {duration:.4f} seconds")
            
            # Log the successful call with structured data
            self._log_api_call(
                operation, 
                params, 
                True,
                duration_seconds=duration
            )
            
            return result
        except ResourceDoesNotExist:
            duration = time.time() - start_time
            logger.warning(f"Job {job_id} not found. Operation took {duration:.4f} seconds")
            self._log_api_call(
                operation, 
                params, 
                False, 
                error=ResourceDoesNotExist(f"Job {job_id} not found"),
                duration_seconds=duration
            )
            return None
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Error getting job {job_id}: {e}", exc_info=True)
            logger.error(f"Operation failed after {duration:.4f} seconds")
            
            # Log the failed call
            self._log_api_call(
                operation, 
                params, 
                False, 
                error=e,
                duration_seconds=duration
            )
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
        start_time = time.time()
        
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
            
            # Log duration and results
            duration = time.time() - start_time
            job_id_str = f" for job {job_id}" if job_id else ""
            logger.info(f"Listed {len(result)} runs{job_id_str} in {duration:.4f} seconds")
            
            # Log structured data
            self._log_api_call(
                operation, 
                params, 
                True, 
                result_size=len(result),
                duration_seconds=duration
            )
            
            return result
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Error listing job runs: {e}", exc_info=True)
            logger.error(f"Operation failed after {duration:.4f} seconds")
            
            self._log_api_call(
                operation, 
                params, 
                False, 
                error=e,
                duration_seconds=duration
            )
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
        start_time = time.time()
        
        if not self.is_available():
            error_msg = "Databricks client not available"
            logger.error(error_msg)
            self._log_api_call(operation, params, False, error=ValueError(error_msg))
            return None
        
        try:
            run = self._client.jobs.get_run(run_id=run_id)
            result = run.as_dict() if run else None
        
            # Log duration
            duration = time.time() - start_time
            logger.info(f"Retrieved run {run_id} in {duration:.4f} seconds")
            
            self._log_api_call(
                operation, 
                params, 
                True,
                duration_seconds=duration
            )
            return result
        except ResourceDoesNotExist:
            duration = time.time() - start_time
            logger.warning(f"Run {run_id} not found. Operation took {duration:.4f} seconds")
            
            self._log_api_call(
                operation, 
                params, 
                False, 
                error=ResourceDoesNotExist(f"Run {run_id} not found"),
                duration_seconds=duration
            )
            return None
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Error getting run {run_id}: {e}", exc_info=True)
            logger.error(f"Operation failed after {duration:.4f} seconds")
            
            self._log_api_call(
                operation, 
                params, 
                False, 
                error=e,
                duration_seconds=duration
            )
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
        start_time = time.time()
        
        if not self.is_available():
            error_msg = "Databricks client not available"
            logger.error(error_msg)
            self._log_api_call(operation, params, False, error=ValueError(error_msg))
            return RunStatus.UNKNOWN, "Client not available"
        
        try:
            run = self._client.jobs.get_run(run_id=run_id)
            
            if not run or not run.state:
                duration = time.time() - start_time
                self._log_api_call(
                    operation, 
                    params, 
                    False, 
                    error=ValueError(f"Run {run_id} has no state information"),
                    duration_seconds=duration
                )
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
            
            # Log duration and status
            duration = time.time() - start_time
            logger.info(f"Retrieved status '{life_cycle_state}' for run {run_id} in {duration:.4f} seconds")
            
            self._log_api_call(
                operation, 
                params, 
                True,
                duration_seconds=duration
            )
            
            return status, state_message
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Error getting run status for {run_id}: {e}", exc_info=True)
            logger.error(f"Operation failed after {duration:.4f} seconds")
            
            self._log_api_call(
                operation, 
                params, 
                False, 
                error=e,
                duration_seconds=duration
            )
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
        start_time = time.time()
        
        if not self.is_available():
            error_msg = "Databricks client not available"
            logger.error(error_msg)
            self._log_api_call(operation, params, False, error=ValueError(error_msg))
            return []
        
        try:
            # Convert generator to list directly
            clusters_list = list(self._client.clusters.list())
            result = [cluster.as_dict() if hasattr(cluster, 'as_dict') else cluster for cluster in clusters_list]
            
            # Log duration and results
            duration = time.time() - start_time
            logger.info(f"Listed {len(result)} clusters in {duration:.4f} seconds")
            
            self._log_api_call(
                operation, 
                params, 
                True, 
                result_size=len(result),
                duration_seconds=duration
            )
            
            return result
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Error listing clusters: {e}", exc_info=True)
            logger.error(f"Operation failed after {duration:.4f} seconds")
            
            self._log_api_call(
                operation, 
                params, 
                False, 
                error=e,
                duration_seconds=duration
            )
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
        start_time = time.time()
        
        if not self.is_available():
            error_msg = "Databricks client not available"
            logger.error(error_msg)
            self._log_api_call(operation, params, False, error=ValueError(error_msg))
            return None
        
        try:
            cluster = self._client.clusters.get(cluster_id=cluster_id)
            result = cluster.as_dict() if cluster else None
        
            # Log duration
            duration = time.time() - start_time
            logger.info(f"Retrieved cluster {cluster_id} in {duration:.4f} seconds")
            
            self._log_api_call(
                operation, 
                params, 
                True,
                duration_seconds=duration
            )
            
            return result
        except ResourceDoesNotExist:
            duration = time.time() - start_time
            logger.warning(f"Cluster {cluster_id} not found. Operation took {duration:.4f} seconds")
            
            self._log_api_call(
                operation, 
                params, 
                False, 
                error=ResourceDoesNotExist(f"Cluster {cluster_id} not found"),
                duration_seconds=duration
            )
            return None
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Error getting cluster {cluster_id}: {e}", exc_info=True)
            logger.error(f"Operation failed after {duration:.4f} seconds")
            
            self._log_api_call(
                operation, 
                params, 
                False, 
                error=e,
                duration_seconds=duration
            )
            return None
    
    # =========================================================================
    # Workspace API Methods
    # =========================================================================
    
    @api_retry
    def list_workspaces(self, path: str) -> List[Dict[str, Any]]:
        """
        List workspace items.
        
        Args:
            path: Workspace path to list
            
        Returns:
            List of workspace objects
        """
        operation = "list_workspaces"
        params = {"path": path}
        start_time = time.time()
        
        if not self.is_available():
            error_msg = "Databricks client not available"
            logger.error(error_msg)
            self._log_api_call(operation, params, False, error=ValueError(error_msg))
            return []
        
        try:
            workspace_list = list(self._client.workspace.list(path=path))
            result = [item.as_dict() if hasattr(item, 'as_dict') else item for item in workspace_list]
            
            # Log duration and results
            duration = time.time() - start_time
            logger.info(f"Listed {len(result)} workspace items at {path} in {duration:.4f} seconds")
            
            self._log_api_call(
                operation, 
                params, 
                True, 
                result_size=len(result),
                duration_seconds=duration
            )
            
            return result
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Error listing workspace items at {path}: {e}", exc_info=True)
            logger.error(f"Operation failed after {duration:.4f} seconds")
            
            self._log_api_call(
                operation, 
                params, 
                False, 
                error=e,
                duration_seconds=duration
            )
            return []
    
    @api_retry
    def get_workspace_status(self, workspace_id: str) -> Optional[Dict[str, Any]]:
        """
        Get workspace status by ID.
        
        Args:
            workspace_id: The workspace item ID
            
        Returns:
            Workspace status or None if not found
        """
        operation = "get_workspace_status"
        params = {"workspace_id": workspace_id}
        start_time = time.time()
        
        if not self.is_available():
            error_msg = "Databricks client not available"
            logger.error(error_msg)
            self._log_api_call(operation, params, False, error=ValueError(error_msg))
            return None
        
        try:
            # This is a placeholder as the actual implementation depends on the specific SDK version
            # In some versions, it might be workspace.get_status(), in others just workspace.get()
            # Adjust this based on your SDK version
            if hasattr(self._client.workspace, 'get_status'):
                workspace = self._client.workspace.get_status(id=workspace_id)
            else:
                workspace = self._client.workspace.get(path=workspace_id)  # Using path as fallback
                
            result = workspace.as_dict() if workspace else None
            
            # Log duration
            duration = time.time() - start_time
            logger.info(f"Retrieved workspace {workspace_id} in {duration:.4f} seconds")
            
            self._log_api_call(
                operation, 
                params, 
                True,
                duration_seconds=duration
            )
            
            return result
        except ResourceDoesNotExist:
            duration = time.time() - start_time
            logger.warning(f"Workspace {workspace_id} not found. Operation took {duration:.4f} seconds")
            
            self._log_api_call(
                operation, 
                params, 
                False, 
                error=ResourceDoesNotExist(f"Workspace {workspace_id} not found"),
                duration_seconds=duration
            )
            return None
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Error getting workspace {workspace_id}: {e}", exc_info=True)
            logger.error(f"Operation failed after {duration:.4f} seconds")
            
            self._log_api_call(
                operation, 
                params, 
                False, 
                error=e,
                duration_seconds=duration
            )
            return None
    
    # =========================================================================
    # SQL Warehouses API Methods
    # =========================================================================
    
    @api_retry
    def list_warehouses(self) -> List[Dict[str, Any]]:
        """
        List SQL warehouses.
        
        Returns:
            List of warehouse objects
        """
        operation = "list_warehouses"
        params = {}
        start_time = time.time()
        
        if not self.is_available():
            error_msg = "Databricks client not available"
            logger.error(error_msg)
            self._log_api_call(operation, params, False, error=ValueError(error_msg))
            return []
        
        try:
            # Check if SQL API is available - it might not be in all Databricks SDK versions
            if not hasattr(self._client, 'warehouses'):
                error_msg = "SQL Warehouses API not available in this version of Databricks SDK"
                logger.warning(error_msg)
                self._log_api_call(operation, params, False, error=ValueError(error_msg))
                return []
                
            warehouses_list = list(self._client.warehouses.list())
            result = [warehouse.as_dict() if hasattr(warehouse, 'as_dict') else warehouse for warehouse in warehouses_list]
            
            # Log duration and results
            duration = time.time() - start_time
            logger.info(f"Listed {len(result)} SQL warehouses in {duration:.4f} seconds")
            
            self._log_api_call(
                operation, 
                params, 
                True, 
                result_size=len(result),
                duration_seconds=duration
            )
            
            return result
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Error listing SQL warehouses: {e}", exc_info=True)
            logger.error(f"Operation failed after {duration:.4f} seconds")
            
            self._log_api_call(
                operation, 
                params, 
                False, 
                error=e,
                duration_seconds=duration
            )
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
        start_time = time.time()
        
        if not self.is_available():
            error_msg = "Databricks client not available"
            logger.error(error_msg)
            self._log_api_call(operation, params, False, error=ValueError(error_msg))
            return None
        
        try:
            # Check if SQL API is available - it might not be in all Databricks SDK versions
            if not hasattr(self._client, 'warehouses'):
                error_msg = "SQL Warehouses API not available in this version of Databricks SDK"
                logger.warning(error_msg)
                self._log_api_call(operation, params, False, error=ValueError(error_msg))
                return None
                
            warehouse = self._client.warehouses.get(id=warehouse_id)
            result = warehouse.as_dict() if warehouse else None
        
            # Log duration
            duration = time.time() - start_time
            logger.info(f"Retrieved warehouse {warehouse_id} in {duration:.4f} seconds")
            
            self._log_api_call(
                operation, 
                params, 
                True,
                duration_seconds=duration
            )
            
            return result
        except ResourceDoesNotExist:
            duration = time.time() - start_time
            logger.warning(f"Warehouse {warehouse_id} not found. Operation took {duration:.4f} seconds")
            
            self._log_api_call(
                operation, 
                params, 
                False, 
                error=ResourceDoesNotExist(f"Warehouse {warehouse_id} not found"),
                duration_seconds=duration
            )
            return None
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Error getting warehouse {warehouse_id}: {e}", exc_info=True)
            logger.error(f"Operation failed after {duration:.4f} seconds")
            
            self._log_api_call(
                operation, 
                params, 
                False, 
                error=e,
                duration_seconds=duration
            )
            return None
    
    # =========================================================================
    # Log Retrieval Methods
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
        Retrieve logs from Databricks.
        
        Args:
            cluster_id: Filter logs by cluster ID
            run_id: Filter logs by run ID
            start_time: Start time in milliseconds since epoch
            end_time: End time in milliseconds since epoch
            page_token: Token for pagination
            offset: Pagination offset
            limit: Maximum number of logs to return
            log_type: Type of logs to retrieve (audit, driver, stdout, stderr, etc.)
            
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
        start_time_exec = time.time()
        
        if not self.is_available():
            error_msg = "Databricks client not available"
            logger.error(error_msg)
            self._log_api_call(operation, params, False, error=ValueError(error_msg))
            return []
        
        # Based on what's provided, decide which API to use
        try:
            result = []
            
            if run_id:
                # Get run logs - typically stdout and stderr
                # First, get the run details to find out its cluster_id
                run_details = self.get_run(run_id)
                if not run_details:
                    logger.warning(f"Cannot get logs: Run {run_id} not found")
                    return []
                    
                # Check if the run has a cluster_id
                run_cluster_id = None
                if 'cluster_instance' in run_details:
                    run_cluster_id = run_details.get('cluster_instance', {}).get('cluster_id')
                
                # If we have a run and its cluster, we can get logs
                if run_cluster_id:
                    # Try to get the logs for this run's cluster
                    run_logs = {}
                    
                    # Try to get stdout logs
                    try:
                        stdout_logs = self._client.jobs.get_run_output(run_id=run_id)
                        if stdout_logs and hasattr(stdout_logs, 'as_dict'):
                            stdout_dict = stdout_logs.as_dict()
                            if 'logs' in stdout_dict:
                                run_logs['stdout'] = stdout_dict['logs']
                    except Exception as e:
                        logger.warning(f"Error getting stdout logs for run {run_id}: {e}")
                        run_logs['stdout'] = ""
                    
                    # Try to get stderr logs if available
                    # Note: In newer Databricks SDK versions, stderr might be included in stdout logs
                    try:
                        # For some SDK versions, there might be a specific stderr endpoint
                        if hasattr(self._client.jobs, 'get_run_stderr'):
                            stderr_logs = self._client.jobs.get_run_stderr(run_id=run_id)
                            if stderr_logs:
                                run_logs['stderr'] = stderr_logs
                    except Exception as e:
                        logger.debug(f"Error getting separate stderr logs for run {run_id}: {e}")
                        run_logs['stderr'] = ""
                    
                    # Add metadata to the result
                    result = [{
                        'run_id': run_id,
                        'job_id': run_details.get('job_id'),
                        'status': run_details.get('state', {}).get('state_message'),
                        'logs': run_logs,
                        'metadata': {
                            'cluster_id': run_cluster_id,
                            'start_time': run_details.get('start_time'),
                            'end_time': run_details.get('end_time'),
                            'run_name': run_details.get('run_name'),
                            'run_page_url': run_details.get('run_page_url')
                        }
                    }]
            
            elif cluster_id:
                # Get cluster logs - requires appropriate permissions
                # Not all SDK versions have direct log access, so have a fallback plan
                try:
                    if hasattr(self._client.clusters, 'get_logs'):
                        logs = self._client.clusters.get_logs(
                            cluster_id=cluster_id,
                            start=offset,
                            end=offset+limit,
                            log_type=log_type
                        )
                        if logs and hasattr(logs, 'as_dict'):
                            result = [logs.as_dict()]
                        else:
                            result = [{'logs': logs, 'cluster_id': cluster_id}]
                    else:
                        # Alternative log retrieval through events API if available
                        events_api_available = hasattr(self._client.clusters, 'events')
                        if events_api_available:
                            events = list(self._client.clusters.events(
                                cluster_id=cluster_id,
                                limit=limit,
                                order='DESC'
                            ))
                            result = [{'events': [e.as_dict() if hasattr(e, 'as_dict') else e for e in events], 
                                     'cluster_id': cluster_id}]
                        else:
                            logger.warning("Cluster logs API not available in this SDK version")
                except Exception as e:
                    logger.warning(f"Error accessing cluster logs: {e}")
                    # Fallback - at least return basic info about the cluster
                    cluster_info = self.get_cluster(cluster_id)
                    if cluster_info:
                        result = [{'cluster_info': cluster_info, 'logs': {log_type: "Log retrieval not supported"}}]
            
            else:
                # No specific target - try audit logs or workspace-level logs if available
                # This is very dependent on the SDK version and permissions
                logger.warning("General log retrieval without run_id or cluster_id is not well supported")
                result = [{'warning': 'Log retrieval requires run_id or cluster_id'}]
            
            # Log duration and results
            duration = time.time() - start_time_exec
            logger.info(f"Retrieved logs in {duration:.4f} seconds")
            
            self._log_api_call(
                operation, 
                params, 
                True, 
                result_size=len(result),
                duration_seconds=duration
            )
            
            return result
            
        except Exception as e:
            duration = time.time() - start_time_exec
            logger.error(f"Error retrieving logs: {e}", exc_info=True)
            logger.error(f"Operation failed after {duration:.4f} seconds")
            
            self._log_api_call(
                operation, 
                params, 
                False, 
                error=e,
                duration_seconds=duration
            )
            return []
    
    # =========================================================================
    # Activity Monitoring
    # =========================================================================
    
    @api_retry
    def get_activity(self, days: int = 7, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get recent activity (job runs, cluster events) from Databricks.
        
        Args:
            days: Number of days of history to retrieve
            limit: Maximum number of activities to return
            
        Returns:
            List of activity entries
        """
        operation = "get_activity"
        params = {"days": days, "limit": limit}
        start_time = time.time()
        
        if not self.is_available():
            error_msg = "Databricks client not available"
            logger.error(error_msg)
            self._log_api_call(operation, params, False, error=ValueError(error_msg))
            return []
        
        # We need to assemble activity from multiple API endpoints
        try:
            activity = []
            
            # 1. Recent job runs
            try:
                # Calculate the millisecond timestamp for X days ago
                start_time_ms = int((time.time() - (days * 24 * 60 * 60)) * 1000)
                
                # Get recent job runs 
                runs = self.list_runs(limit=limit)
                
                # Filter by date if needed
                if start_time_ms > 0:
                    runs = [r for r in runs if r.get('start_time', 0) >= start_time_ms]
                
                # Add runs to activity
                for run in runs:
                    activity.append({
                        'type': 'job_run',
                        'id': run.get('run_id'),
                        'time': run.get('start_time'),
                        'status': run.get('state', {}).get('life_cycle_state'),
                        'result': run.get('state', {}).get('result_state'),
                        'details': {
                            'job_id': run.get('job_id'),
                            'run_name': run.get('run_name'),
                            'creator': run.get('creator_user_name')
                        }
                    })
            except Exception as e:
                logger.warning(f"Error getting job runs for activity: {e}")
            
            # 2. Recent cluster events 
            try:
                clusters = self.list_clusters()
                
                for cluster in clusters:
                    cluster_id = cluster.get('cluster_id')
                    if not cluster_id:
                        continue
                        
                    # Check if events API is available
                    events_available = hasattr(self._client.clusters, 'events')
                    if events_available:
                        # Get events for this cluster
                        try:
                            events = list(self._client.clusters.events(
                                cluster_id=cluster_id,
                                limit=min(20, limit),  # Limit per cluster
                                order='DESC'
                            ))
                            
                            # Add events to activity
                            for event in events:
                                event_dict = event.as_dict() if hasattr(event, 'as_dict') else event
                                activity.append({
                                    'type': 'cluster_event',
                                    'id': f"{cluster_id}:{event_dict.get('timestamp', '')}",
                                    'time': event_dict.get('timestamp'),
                                    'status': event_dict.get('type'),
                                    'details': {
                                        'cluster_id': cluster_id,
                                        'details': event_dict.get('details', {}),
                                        'cluster_name': cluster.get('cluster_name')
                                    }
                                })
                        except Exception as e:
                            logger.warning(f"Error getting events for cluster {cluster_id}: {e}")
                    else:
                        # Just add basic cluster state as single "event"
                        activity.append({
                            'type': 'cluster_state',
                            'id': cluster_id,
                            'time': int(time.time() * 1000),  # Current time
                            'status': cluster.get('state'),
                            'details': {
                                'cluster_id': cluster_id,
                                'cluster_name': cluster.get('cluster_name'),
                                'creator': cluster.get('creator_user_name')
                            }
                        })
            except Exception as e:
                logger.warning(f"Error getting cluster activity: {e}")
            
            # 3. Sort all activity by time (descending)
            activity.sort(key=lambda x: x.get('time', 0), reverse=True)
            
            # 4. Limit to the requested number
            if len(activity) > limit:
                activity = activity[:limit]
            
            # Log duration and results
            duration = time.time() - start_time
            logger.info(f"Retrieved {len(activity)} activity records in {duration:.4f} seconds")
            
            self._log_api_call(
                operation, 
                params, 
                True, 
                result_size=len(activity),
                duration_seconds=duration
            )
            
            return activity
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Error getting activity: {e}", exc_info=True)
            logger.error(f"Operation failed after {duration:.4f} seconds")
            
            self._log_api_call(
                operation, 
                params, 
                False, 
                error=e,
                duration_seconds=duration
            )
            return [] 