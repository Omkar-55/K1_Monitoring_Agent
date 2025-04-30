"""
Test script for the Databricks monitoring agent workflow.

This script demonstrates a complete workflow using the tools we've implemented.
"""

import logging
import json
import time
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s'
)
logger = logging.getLogger("monitoring_agent_test")

# Import our monitoring tools
from src.tools.databricks_monitoring import (
    get_logs,
    diagnose, 
    FailureType,
    suggest_fix, 
    apply_fix,
    verify,
    final_report
)

# Import tracing if configured
try:
    from agent_core import enable_tracing
    # Initialize tracing with service name
    enable_tracing(service_name="monitoring-agent-test")
    logger.info("Tracing enabled")
except ImportError:
    logger.warning("Tracing not available")


def simulate_databricks_run(failure_type: str = "memory_exceeded") -> Dict[str, Any]:
    """
    Simulate a Databricks job run with a specific failure type.
    Used for testing when no real Databricks job is available.
    
    Args:
        failure_type: Type of failure to simulate
        
    Returns:
        Simulated run data
    """
    # Generate appropriate error logs based on failure type
    stderr_logs = ""
    if failure_type == "memory_exceeded":
        stderr_logs = """
        [DRIVER] ExecutorLostFailure: Job aborted due to stage failure: 
        Task 12 in stage 4.0 failed 4 times, most recent failure: Lost task 12.3
        java.lang.OutOfMemoryError: Java heap space
        at java.util.Arrays.copyOf(Arrays.java:3181)
        at java.util.ArrayList.grow(ArrayList.java:265)
        """
    elif failure_type == "dependency_error":
        stderr_logs = """
        [DRIVER] Py4JJavaError: An error occurred while calling z:org.apache.spark.sql.functions.col.
        : java.lang.ClassNotFoundException: org.apache.spark.sql.functions
        
        ModuleNotFoundError: No module named 'matplotlib'
        """
    elif failure_type == "disk_space_exceeded":
        stderr_logs = """
        [EXECUTOR] ERROR: No space left on device: /tmp/blockmgr-12324-34232-353a
        java.io.IOException: No space left on device
        at java.io.FileOutputStream.writeBytes(Native Method)
        """
    
    # Create a simulated run
    return {
        "run_id": f"simulated-run-{int(time.time())}",
        "job_id": "12345",
        "status": "TERMINATED", 
        "result": "FAILED",
        "start_time": int(time.time() * 1000) - 3600000,  # 1 hour ago
        "end_time": int(time.time() * 1000) - 3540000,  # 59 minutes ago
        "duration_seconds": 60,
        "run_name": "Test Job Run",
        "logs": {
            "stdout": "Starting job execution...\nProcessing data...\n",
            "stderr": stderr_logs
        },
        "metadata": {
            "cluster_id": "0123-456789-abcdef",
            "creator": "test-user",
            "cluster_size": "small",
            "driver_memory": "4g",
            "executor_memory": "4g"
        }
    }


def run_monitoring_workflow(job_id: str, run_id: str = None, simulate: bool = True, 
                           failure_type: str = "memory_exceeded") -> None:
    """
    Run the complete monitoring workflow.
    
    Args:
        job_id: The Databricks job ID to monitor
        run_id: Optional specific run ID to monitor
        simulate: Whether to use simulated data (for testing)
        failure_type: Type of failure to simulate if using simulation
    """
    # Initialize history to track steps
    history = []
    
    logger.info(f"Starting monitoring workflow for job {job_id}" + 
                (f", run {run_id}" if run_id else ""))
    
    # Step 1: Get logs
    logger.info("STEP 1: Fetching logs")
    
    if simulate:
        log_data = simulate_databricks_run(failure_type)
        logger.info(f"Using simulated run data for failure type: {failure_type}")
    else:
        log_data = get_logs(job_id, run_id)
    
    # Record in history
    history.append({
        "type": "logs",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "run_id": log_data.get("run_id"),
        "status": log_data.get("status")
    })
    
    run_id = log_data.get("run_id")
    logger.info(f"Retrieved logs for run ID: {run_id}")
    
    # Step 2: Diagnose the issue
    logger.info("STEP 2: Diagnosing issue")
    combined_logs = log_data.get("logs", {}).get("stdout", "") + "\n" + log_data.get("logs", {}).get("stderr", "")
    failure_type, reasoning = diagnose(combined_logs, log_data.get("metadata", {}))
    
    # Record in history
    history.append({
        "type": "diagnosis",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "failure_type": failure_type.value,
        "reasoning": reasoning,
        "log_excerpt": combined_logs[:500] + ("..." if len(combined_logs) > 500 else "")
    })
    
    logger.info(f"Diagnosed issue: {failure_type.value} - {reasoning}")
    
    # Step 3: Suggest a fix
    logger.info("STEP 3: Suggesting fix")
    fix_plan = suggest_fix(failure_type, {
        "logs": log_data.get("logs", {}),
        "metadata": log_data.get("metadata", {})
    })
    
    # Record in history
    history.append({
        "type": "fix",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "action": fix_plan.get("action"),
        "description": fix_plan.get("description"),
        "confidence": fix_plan.get("confidence")
    })
    
    logger.info(f"Suggested fix: {fix_plan.get('action')} - {fix_plan.get('description')}")
    
    # Step 4: Apply the fix
    logger.info("STEP 4: Applying fix")
    
    if simulate:
        # Simulate applying the fix
        new_run_id = f"simulated-fixed-run-{int(time.time())}"
        fix_result = {
            "success": True,
            "new_run_id": new_run_id,
            "message": f"Simulated fix applied: {fix_plan.get('action')}",
            "details": fix_plan.get("parameters", {})
        }
        logger.info(f"Simulated applying fix, new run ID: {new_run_id}")
    else:
        # Actually apply the fix
        fix_result = apply_fix(fix_plan, job_id, run_id)
    
    # Record in history
    history.append({
        "type": "fix",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "action": fix_plan.get("action"),
        "success": fix_result.get("success"),
        "new_run_id": fix_result.get("new_run_id"),
        "details": fix_result.get("details", {})
    })
    
    logger.info(f"Fix applied: {fix_result.get('success')} - {fix_result.get('message')}")
    
    # If the fix was applied and a new run was started, verify it
    if fix_result.get("success") and fix_result.get("new_run_id"):
        new_run_id = fix_result.get("new_run_id")
        
        # Step 5: Verify the fix
        logger.info(f"STEP 5: Verifying fix with run {new_run_id}")
        
        if simulate:
            # Simulate verification (successful for this test)
            verification_result = "success"
            logger.info(f"Simulated verification result: {verification_result}")
        else:
            # Actually verify the fix
            verification_result = verify(new_run_id, timeout_minutes=5, polling_interval_seconds=10)
        
        # Record in history
        history.append({
            "type": "verification",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "run_id": new_run_id,
            "result": verification_result,
            "duration_seconds": 300  # Just for illustration
        })
        
        logger.info(f"Verification result: {verification_result}")
    
    # Step 6: Generate final report
    logger.info("STEP 6: Generating final report")
    report = final_report(history, job_id)
    
    # Print the report
    print("\n" + "="*80 + "\n")
    print(report)
    print("\n" + "="*80 + "\n")
    
    logger.info("Monitoring workflow completed")


if __name__ == "__main__":
    # Run the workflow with simulated data for each failure type
    failure_types = ["memory_exceeded", "dependency_error", "disk_space_exceeded"]
    
    for failure_type in failure_types:
        print(f"\n\nTesting workflow with failure type: {failure_type}\n")
        run_monitoring_workflow(
            job_id="12345",
            simulate=True,
            failure_type=failure_type
        )
        time.sleep(1)  # Brief pause between runs 