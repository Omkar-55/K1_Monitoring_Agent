"""
Tests for the Databricks client wrapper of the K1 Monitoring Agent.
"""

import os
import json
import pytest
from unittest.mock import patch, MagicMock, AsyncMock, call
from typing import Dict, Any, List

# Import the client being tested
from agent_core.dbx_client import DbxClient, DATABRICKS_SDK_AVAILABLE, RunStatus
try:
    from databricks.sdk.errors import ApiError, ResourceDoesNotExist
    DATABRICKS_ERRORS_AVAILABLE = True
except ImportError:
    DATABRICKS_ERRORS_AVAILABLE = False
    # Create mock exception classes if not available
    class ApiError(Exception):
        pass
    class ResourceDoesNotExist(Exception):
        pass

# Skip all tests if Databricks SDK is not available
pytestmark = pytest.mark.skipif(not DATABRICKS_SDK_AVAILABLE, 
                              reason="Databricks SDK not installed")


@pytest.fixture
def mock_workspace_client():
    """Fixture to create a mock Databricks workspace client."""
    mock = MagicMock()
    
    # Mock jobs module
    mock.jobs = MagicMock()
    mock.jobs.list.return_value = MagicMock(
        jobs=[
            MagicMock(as_dict=lambda: {"job_id": "123", "name": "Test Job 1"}),
            MagicMock(as_dict=lambda: {"job_id": "456", "name": "Test Job 2"})
        ]
    )
    mock.jobs.get.return_value = MagicMock(
        as_dict=lambda: {"job_id": "123", "name": "Test Job 1", "settings": {"tasks": []}}
    )
    mock.jobs.list_runs.return_value = MagicMock(
        runs=[
            MagicMock(as_dict=lambda: {"run_id": "run-1", "job_id": "123", "state": {"life_cycle_state": "RUNNING"}}),
            MagicMock(as_dict=lambda: {"run_id": "run-2", "job_id": "123", "state": {"life_cycle_state": "COMPLETED"}})
        ]
    )
    
    # Mock run with state for get_run_status
    run_with_state = MagicMock()
    run_with_state.as_dict.return_value = {
        "run_id": "run-1", 
        "job_id": "123", 
        "state": {"life_cycle_state": "RUNNING", "state_message": "Job is running"}
    }
    run_with_state.state = MagicMock()
    run_with_state.state.life_cycle_state = "RUNNING"
    run_with_state.state.state_message = "Job is running"
    mock.jobs.get_run.return_value = run_with_state
    
    # Mock clusters module
    mock.clusters = MagicMock()
    mock.clusters.list.return_value = [
        MagicMock(as_dict=lambda: {"cluster_id": "cluster-1", "name": "Test Cluster"}),
        MagicMock(as_dict=lambda: {"cluster_id": "cluster-2", "name": "Another Cluster"})
    ]
    mock.clusters.get.return_value = MagicMock(
        as_dict=lambda: {"cluster_id": "cluster-1", "name": "Test Cluster", "state": "RUNNING"}
    )
    
    # Mock workspace module
    mock.workspace = MagicMock()
    mock.workspace.list.return_value = [
        MagicMock(as_dict=lambda: {"path": "/path/file1.py", "object_type": "NOTEBOOK"}),
        MagicMock(as_dict=lambda: {"path": "/path/file2.py", "object_type": "NOTEBOOK"})
    ]
    mock.workspace.get_status.return_value = MagicMock(
        as_dict=lambda: {"path": "/path/file1.py", "object_type": "NOTEBOOK", "language": "PYTHON"}
    )
    
    # Mock SQL warehouses
    mock.warehouses = MagicMock()
    mock.warehouses.list.return_value = [
        MagicMock(as_dict=lambda: {"id": "warehouse-1", "name": "Test Warehouse"}),
        MagicMock(as_dict=lambda: {"id": "warehouse-2", "name": "Another Warehouse"})
    ]
    mock.warehouses.get.return_value = MagicMock(
        as_dict=lambda: {"id": "warehouse-1", "name": "Test Warehouse", "state": "RUNNING"}
    )
    
    return mock


class TestDbxClient:
    """Tests for the Databricks client wrapper."""

    def test_client_initialization(self):
        """Test that the client initializes correctly."""
        # Skip test if Databricks SDK is not available
        if not DATABRICKS_SDK_AVAILABLE:
            pytest.skip("Databricks SDK not installed")
        
        # Patch os.getenv to return mock values
        with patch('os.getenv') as mock_getenv:
            mock_getenv.side_effect = lambda key, default=None: {
                "DATABRICKS_HOST": "https://test.cloud.databricks.com",
                "DATABRICKS_TOKEN": "test-token"
            }.get(key, default)
            
            client = DbxClient(validate_env=False)
            assert client.host == "https://test.cloud.databricks.com"
            assert client.token == "test-token"
            assert client.profile is None

    def test_client_initialization_with_explicit_values(self):
        """Test that the client initializes with explicit values."""
        # Skip test if Databricks SDK is not available
        if not DATABRICKS_SDK_AVAILABLE:
            pytest.skip("Databricks SDK not installed")
        
        client = DbxClient(
            host="https://explicit.cloud.databricks.com",
            token="explicit-token",
            profile="explicit-profile",
            validate_env=False
        )
        assert client.host == "https://explicit.cloud.databricks.com"
        assert client.token == "explicit-token"
        assert client.profile == "explicit-profile"

    def test_client_initialization_missing_env_vars(self):
        """Test that the client raises ValueError when env vars are missing."""
        # Skip test if Databricks SDK is not available
        if not DATABRICKS_SDK_AVAILABLE:
            pytest.skip("Databricks SDK not installed")
        
        # Patch os.getenv to return None
        with patch('os.getenv', return_value=None):
            # Should raise ValueError when validate_env is True
            with pytest.raises(ValueError) as excinfo:
                DbxClient(validate_env=True)
            
            assert "Missing required environment variables" in str(excinfo.value)
            
            # Should not raise when validate_env is False
            client = DbxClient(validate_env=False)
            assert client.host is None
            assert client.token is None
            assert not client.is_available()

    @patch('databricks.sdk.WorkspaceClient')
    def test_list_jobs(self, mock_client_class, mock_workspace_client):
        """Test listing jobs."""
        # Skip test if Databricks SDK is not available
        if not DATABRICKS_SDK_AVAILABLE:
            pytest.skip("Databricks SDK not installed")
        
        # Setup mock
        mock_client_class.return_value = mock_workspace_client
        
        # Create client and call method
        client = DbxClient(host="https://test.cloud.databricks.com", token="test-token", validate_env=False)
        jobs = client.list_jobs()
        
        # Verify results
        assert len(jobs) == 2
        assert jobs[0]["job_id"] == "123"
        assert jobs[1]["job_id"] == "456"
        mock_workspace_client.jobs.list.assert_called_once()

    @patch('databricks.sdk.WorkspaceClient')
    def test_get_job(self, mock_client_class, mock_workspace_client):
        """Test getting a job by ID."""
        # Skip test if Databricks SDK is not available
        if not DATABRICKS_SDK_AVAILABLE:
            pytest.skip("Databricks SDK not installed")
        
        # Setup mock
        mock_client_class.return_value = mock_workspace_client
        
        # Create client and call method
        client = DbxClient(host="https://test.cloud.databricks.com", token="test-token", validate_env=False)
        job = client.get_job("123")
        
        # Verify results
        assert job is not None
        assert job["job_id"] == "123"
        assert job["name"] == "Test Job 1"
        mock_workspace_client.jobs.get.assert_called_once_with(job_id="123")

    @patch('databricks.sdk.WorkspaceClient')
    def test_get_job_not_found(self, mock_client_class, mock_workspace_client):
        """Test getting a job that doesn't exist."""
        # Skip test if Databricks SDK is not available
        if not DATABRICKS_SDK_AVAILABLE or not DATABRICKS_ERRORS_AVAILABLE:
            pytest.skip("Databricks SDK or errors not installed")
        
        # Setup mock to raise ResourceDoesNotExist
        mock_client_class.return_value = mock_workspace_client
        mock_workspace_client.jobs.get.side_effect = ResourceDoesNotExist("Job not found")
        
        # Create client and call method
        client = DbxClient(host="https://test.cloud.databricks.com", token="test-token", validate_env=False)
        job = client.get_job("999")
        
        # Verify result is None
        assert job is None
        mock_workspace_client.jobs.get.assert_called_once_with(job_id="999")

    @patch('databricks.sdk.WorkspaceClient')
    def test_list_runs(self, mock_client_class, mock_workspace_client):
        """Test listing job runs."""
        # Skip test if Databricks SDK is not available
        if not DATABRICKS_SDK_AVAILABLE:
            pytest.skip("Databricks SDK not installed")
        
        # Setup mock
        mock_client_class.return_value = mock_workspace_client
        
        # Create client and call method
        client = DbxClient(host="https://test.cloud.databricks.com", token="test-token", validate_env=False)
        runs = client.list_runs(job_id="123")
        
        # Verify results
        assert len(runs) == 2
        assert runs[0]["run_id"] == "run-1"
        assert runs[1]["run_id"] == "run-2"
        mock_workspace_client.jobs.list_runs.assert_called_once()

    @patch('databricks.sdk.WorkspaceClient')
    def test_get_run_status(self, mock_client_class, mock_workspace_client):
        """Test getting run status."""
        # Skip test if Databricks SDK is not available
        if not DATABRICKS_SDK_AVAILABLE:
            pytest.skip("Databricks SDK not installed")
        
        # Setup mock
        mock_client_class.return_value = mock_workspace_client
        
        # Create client and call method
        client = DbxClient(host="https://test.cloud.databricks.com", token="test-token", validate_env=False)
        status, message = client.get_run_status("run-1")
        
        # Verify results
        assert status == RunStatus.RUNNING
        assert message == "Job is running"
        mock_workspace_client.jobs.get_run.assert_called_once_with(run_id="run-1")

    @patch('databricks.sdk.WorkspaceClient')
    def test_get_run_status_terminated(self, mock_client_class, mock_workspace_client):
        """Test getting terminated run status."""
        # Skip test if Databricks SDK is not available
        if not DATABRICKS_SDK_AVAILABLE:
            pytest.skip("Databricks SDK not installed")
        
        # Setup mock with a terminated run
        mock_client_class.return_value = mock_workspace_client
        run_with_state = MagicMock()
        run_with_state.state = MagicMock()
        run_with_state.state.life_cycle_state = "TERMINATED"
        run_with_state.state.state_message = "Job completed successfully"
        mock_workspace_client.jobs.get_run.return_value = run_with_state
        
        # Create client and call method
        client = DbxClient(host="https://test.cloud.databricks.com", token="test-token", validate_env=False)
        status, message = client.get_run_status("run-2")
        
        # Verify results
        assert status == RunStatus.TERMINATED
        assert message == "Job completed successfully"
        mock_workspace_client.jobs.get_run.assert_called_once_with(run_id="run-2")

    @patch('databricks.sdk.WorkspaceClient')
    def test_list_clusters(self, mock_client_class, mock_workspace_client):
        """Test listing clusters."""
        # Skip test if Databricks SDK is not available
        if not DATABRICKS_SDK_AVAILABLE:
            pytest.skip("Databricks SDK not installed")
        
        # Setup mock
        mock_client_class.return_value = mock_workspace_client
        
        # Create client and call method
        client = DbxClient(host="https://test.cloud.databricks.com", token="test-token", validate_env=False)
        clusters = client.list_clusters()
        
        # Verify results
        assert len(clusters) == 2
        assert clusters[0]["cluster_id"] == "cluster-1"
        assert clusters[1]["cluster_id"] == "cluster-2"
        mock_workspace_client.clusters.list.assert_called_once()

    @patch('databricks.sdk.WorkspaceClient')
    def test_list_workspaces(self, mock_client_class, mock_workspace_client):
        """Test listing workspace objects."""
        # Skip test if Databricks SDK is not available
        if not DATABRICKS_SDK_AVAILABLE:
            pytest.skip("Databricks SDK not installed")
        
        # Setup mock
        mock_client_class.return_value = mock_workspace_client
        
        # Create client and call method
        client = DbxClient(host="https://test.cloud.databricks.com", token="test-token", validate_env=False)
        objects = client.list_workspaces("/path")
        
        # Verify results
        assert len(objects) == 2
        assert objects[0]["path"] == "/path/file1.py"
        assert objects[1]["path"] == "/path/file2.py"
        mock_workspace_client.workspace.list.assert_called_once_with(path="/path")

    @patch('databricks.sdk.WorkspaceClient')
    def test_list_warehouses(self, mock_client_class, mock_workspace_client):
        """Test listing SQL warehouses."""
        # Skip test if Databricks SDK is not available
        if not DATABRICKS_SDK_AVAILABLE:
            pytest.skip("Databricks SDK not installed")
        
        # Setup mock
        mock_client_class.return_value = mock_workspace_client
        
        # Create client and call method
        client = DbxClient(host="https://test.cloud.databricks.com", token="test-token", validate_env=False)
        warehouses = client.list_warehouses()
        
        # Verify results
        assert len(warehouses) == 2
        assert warehouses[0]["id"] == "warehouse-1"
        assert warehouses[1]["id"] == "warehouse-2"
        mock_workspace_client.warehouses.list.assert_called_once()

    @patch('databricks.sdk.WorkspaceClient')
    def test_error_handling(self, mock_client_class, mock_workspace_client):
        """Test error handling when an API call fails."""
        # Skip test if Databricks SDK is not available
        if not DATABRICKS_SDK_AVAILABLE:
            pytest.skip("Databricks SDK not installed")
        
        # Setup mock to raise exception
        mock_client_class.return_value = mock_workspace_client
        mock_workspace_client.jobs.list.side_effect = Exception("API Error")
        
        # Create client and call method
        client = DbxClient(host="https://test.cloud.databricks.com", token="test-token", validate_env=False)
        jobs = client.list_jobs()
        
        # Verify empty result on error
        assert isinstance(jobs, list)
        assert len(jobs) == 0

    @patch('databricks.sdk.WorkspaceClient')
    @patch('agent_core.dbx_client.logger')
    def test_structured_logging(self, mock_logger, mock_client_class, mock_workspace_client):
        """Test structured logging."""
        # Skip test if Databricks SDK is not available
        if not DATABRICKS_SDK_AVAILABLE:
            pytest.skip("Databricks SDK not installed")
        
        # Setup mocks
        mock_client_class.return_value = mock_workspace_client
        
        # Create client and call method
        client = DbxClient(host="https://test.cloud.databricks.com", token="test-token", validate_env=False)
        client.list_jobs()
        
        # Verify that structured logging was called
        assert mock_logger.info.called
        log_message = mock_logger.info.call_args[0][0]
        assert "Databricks API call:" in log_message
        assert "list_jobs" in log_message

    @pytest.mark.skipif(not DATABRICKS_ERRORS_AVAILABLE, reason="Databricks errors not available")
    @patch('agent_core.dbx_client.TENACITY_AVAILABLE', True)  # Force retry to be available
    @patch('agent_core.dbx_client.api_retry')
    @patch('databricks.sdk.WorkspaceClient')
    def test_retry_mechanism(self, mock_client_class, mock_retry, mock_workspace_client):
        """Test retry mechanism for transient API errors."""
        # Skip test if Databricks SDK is not available
        if not DATABRICKS_SDK_AVAILABLE:
            pytest.skip("Databricks SDK not installed")
        
        # Setup mock
        mock_client_class.return_value = mock_workspace_client
        
        # Create client
        client = DbxClient(host="https://test.cloud.databricks.com", token="test-token", validate_env=False)
        
        # Call the method that should use the retry decorator
        client.list_jobs()
        
        # Verify the retry decorator was used
        assert mock_retry.called


@pytest.mark.integration
class TestDbxClientIntegration:
    """Integration tests for the Databricks client."""
    
    def test_real_client(self):
        """
        Test with the real Databricks client.
        
        This test requires Databricks credentials to be set in the environment:
        - DATABRICKS_HOST
        - DATABRICKS_TOKEN
        """
        # Skip if Databricks SDK not available
        if not DATABRICKS_SDK_AVAILABLE:
            pytest.skip("Databricks SDK not installed")
        
        # Skip if no credentials
        if not os.getenv("DATABRICKS_HOST") or not os.getenv("DATABRICKS_TOKEN"):
            pytest.skip("DATABRICKS_HOST or DATABRICKS_TOKEN not set")
        
        # Create a real client
        client = DbxClient(validate_env=True)
        
        # Skip if client is not available
        if not client.is_available():
            pytest.skip("Databricks client not available")
        
        # Basic test - list jobs
        jobs = client.list_jobs(limit=10)
        
        # Very basic verification - we don't know what's in the workspace
        assert isinstance(jobs, list) 