"""
Test integration of monitoring tools from the databricks_monitoring module.
"""

import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s'
)
logger = logging.getLogger("tools_integration_test")

# Import tools directly from the module
from src.tools.databricks_monitoring import (
    get_logs,
    diagnose, 
    FailureType,
    suggest_fix, 
    apply_fix,
    verify,
    final_report
)

def test_integrated_workflow():
    """Test the integrated workflow using the tools."""
    logger.info("Starting integrated workflow test")
    
    # Step 1: Get logs (simulated)
    job_id = "test-job-123"
    log_data = get_logs(job_id)
    
    logger.info(f"Retrieved logs for job: {job_id}, run: {log_data.get('run_id')}")
    
    # Step 2: Diagnose
    combined_logs = "Test logs with OutOfMemoryError to trigger memory diagnosis"
    failure_type, reasoning = diagnose(combined_logs)
    
    logger.info(f"Diagnosed issue: {failure_type.value} - {reasoning}")
    
    # Step 3: Suggest fix
    fix_plan = suggest_fix(failure_type, {
        "logs": {"stdout": "", "stderr": combined_logs},
        "metadata": {"cluster_id": "test-cluster"}
    })
    
    logger.info(f"Suggested fix: {fix_plan.get('action')} - {fix_plan.get('description')}")
    
    # Step 4: Apply fix
    fix_result = apply_fix(fix_plan, job_id)
    
    logger.info(f"Fix applied: {fix_result.get('success')} - {fix_result.get('message')}")
    
    # Step 5: Verify fix
    verification_result = verify(fix_result.get("new_run_id"))
    
    logger.info(f"Verification result: {verification_result}")
    
    # Step 6: Generate final report
    history = [
        {"type": "logs", "timestamp": "2025-04-30 20:55:00", "run_id": log_data.get("run_id")},
        {"type": "diagnosis", "timestamp": "2025-04-30 20:55:01", "failure_type": failure_type.value},
        {"type": "fix", "timestamp": "2025-04-30 20:55:02", "action": fix_plan.get("action")},
        {"type": "verification", "timestamp": "2025-04-30 20:55:03", "result": verification_result}
    ]
    
    report = final_report(history, job_id)
    
    logger.info("Generated final report")
    print("\n" + "="*80 + "\n")
    print(report)
    print("\n" + "="*80 + "\n")
    
    logger.info("Integration test completed successfully!")

if __name__ == "__main__":
    test_integrated_workflow() 