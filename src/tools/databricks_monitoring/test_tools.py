"""
Test script to demonstrate usage of the Databricks monitoring tools.
"""

import json
import time
from typing import Dict, Any, List
import logging

# Configure basic logging to see the output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s'
)

# Import our tools
from agent_core.logging_config import get_logger
from agent_core import enable_tracing
from .log_tools import get_logs
from .diagnostic_tools import diagnose, FailureType
from .fix_tools import suggest_fix, apply_fix
from .verification_tools import verify
from .reporting_tools import final_report

# Get logger for this module
logger = get_logger(__name__)


def simulate_job_run_with_failure() -> Dict[str, Any]:
    """
    Simulate a Databricks job run with a failure.
    Returns a mock log result similar to what get_logs would return.
    """
    # Mock log text with an out of memory error
    log_text = """
    22/04/30 10:15:22 INFO Jobs: Starting job execution.
    22/04/30 10:15:24 INFO Executor: Initializing execution.
    22/04/30 10:15:30 INFO Spark: Processing data batch #1
    22/04/30 10:15:42 INFO Spark: Processing large joined dataset
    22/04/30 10:15:55 ERROR Executor: java.lang.OutOfMemoryError: Java heap space
    22/04/30 10:15:55 ERROR Executor: at java.util.Arrays.copyOf(Arrays.java:3332)
    22/04/30 10:15:55 ERROR Executor: at java.io.ByteArrayOutputStream.grow(ByteArrayOutputStream.java:118)
    22/04/30 10:15:55 ERROR Executor: at org.apache.spark.util.collection.ExternalSorter.spillMemoryIteratorToDisk(ExternalSorter.scala:387)
    22/04/30 10:15:56 ERROR Spark: Job failed with error: Task failed due to JVM heap space
    22/04/30 10:15:56 INFO Jobs: Job execution completed with status: FAILED
    """
    
    # Mock run result similar to what the DbxClient would return
    return {
        "run_id": "mock-run-123",
        "job_id": "mock-job-456",
        "status": "TERMINATED",
        "status_message": "Job failed due to executor error",
        "start_time": int(time.time() - 1000),  # Started 1000 seconds ago
        "end_time": int(time.time() - 100),     # Ended 100 seconds ago
        "cluster_id": "mock-cluster-789",
        "logs": log_text,
        "metadata": {
            "job_id": "mock-job-456",
            "run_id": "mock-run-123",
            "creator_user_name": "mock-user@example.com",
            "state": {
                "life_cycle_state": "TERMINATED",
                "result_state": "FAILED",
                "state_message": "Job failed due to executor error"
            },
            "cluster_spec": {
                "existing_cluster_id": "mock-cluster-789"
            }
        }
    }


def simulate_job_run_success() -> Dict[str, Any]:
    """
    Simulate a successful Databricks job run.
    Returns a mock log result similar to what get_logs would return after a fix.
    """
    # Mock log text for a successful run
    log_text = """
    22/04/30 11:05:22 INFO Jobs: Starting job execution.
    22/04/30 11:05:24 INFO Executor: Initializing execution.
    22/04/30 11:05:30 INFO Spark: Processing data batch #1
    22/04/30 11:05:42 INFO Spark: Processing large joined dataset
    22/04/30 11:05:55 INFO Spark: Successfully processed all data
    22/04/30 11:05:56 INFO Jobs: Job execution completed with status: SUCCESS
    """
    
    # Mock run result similar to what the DbxClient would return
    return {
        "run_id": "mock-run-789",
        "job_id": "mock-job-456",
        "status": "TERMINATED",
        "status_message": "Job completed successfully",
        "start_time": int(time.time() - 500),  # Started 500 seconds ago
        "end_time": int(time.time() - 50),     # Ended 50 seconds ago
        "cluster_id": "mock-cluster-789",
        "logs": log_text,
        "metadata": {
            "job_id": "mock-job-456",
            "run_id": "mock-run-789",
            "creator_user_name": "mock-user@example.com",
            "state": {
                "life_cycle_state": "TERMINATED",
                "result_state": "SUCCESS",
                "state_message": "Job completed successfully"
            },
            "cluster_spec": {
                "existing_cluster_id": "mock-cluster-789"
            }
        }
    }


def run_monitoring_workflow() -> None:
    """
    Run the full monitoring workflow to demonstrate the tools.
    """
    # Enable tracing
    enable_tracing(service_name="monitoring-tools-test")
    
    job_id = "mock-job-456"
    
    # Keep track of the history for reporting
    history = []
    
    # Step 1: Get logs from a simulated failed run
    logger.info("Step 1: Getting logs from a failed job run")
    mock_logs = simulate_job_run_with_failure()
    history.append({
        "step": "get_logs",
        "timestamp": time.time(),
        "params": {"job_id": job_id, "run_id": mock_logs["run_id"]},
        "result": mock_logs
    })
    
    # Step 2: Diagnose the issue
    logger.info("Step 2: Diagnosing the issue")
    failure_type, reasoning = diagnose(mock_logs["logs"])
    history.append({
        "step": "diagnose",
        "timestamp": time.time(),
        "params": {"log_text": mock_logs["logs"]},
        "result": {"failure_type": failure_type.name, "reasoning": reasoning}
    })
    
    # Step 3: Suggest a fix
    logger.info("Step 3: Suggesting a fix")
    context = {
        "logs": mock_logs["logs"],
        "current_memory": "1g",
        "current_executors": 2,
        "job_id": job_id,
        "run_id": mock_logs["run_id"]
    }
    fix_plan = suggest_fix(failure_type, context)
    history.append({
        "step": "suggest_fix",
        "timestamp": time.time(),
        "params": {"failure_type": failure_type.name, "context": context},
        "result": fix_plan
    })
    
    # Step 4: Apply the fix (simulated)
    logger.info("Step 4: Applying the fix")
    fix_result = apply_fix(fix_plan, job_id, mock_logs["run_id"])
    history.append({
        "step": "apply_fix",
        "timestamp": time.time(),
        "params": {"plan": fix_plan, "job_id": job_id, "run_id": mock_logs["run_id"]},
        "result": fix_result
    })
    
    # Step 5: Get a new run (simulated)
    logger.info("Step 5: Getting logs from a new run after applying fix")
    new_run = simulate_job_run_success()
    history.append({
        "step": "get_logs",
        "timestamp": time.time(),
        "params": {"job_id": job_id, "run_id": new_run["run_id"]},
        "result": new_run
    })
    
    # Step 6: Verify the fix (simulated)
    logger.info("Step 6: Verifying the fix")
    # In a real scenario, we would call verify(new_run["run_id"])
    # For this test, we'll just use the mock result
    verification_result = "success"  # This would normally be returned by verify()
    history.append({
        "step": "verify",
        "timestamp": time.time(),
        "params": {"run_id": new_run["run_id"]},
        "result": verification_result
    })
    
    # Step 7: Generate the final report
    logger.info("Step 7: Generating the final report")
    report = final_report(history, job_id)
    
    # Print the report
    logger.info("Final Report:\n\n" + report)


if __name__ == "__main__":
    run_monitoring_workflow() 