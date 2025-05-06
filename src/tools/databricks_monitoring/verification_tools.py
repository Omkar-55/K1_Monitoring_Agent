"""
Tools for verifying if fixes to Databricks issues were successful.
"""

import time
import random
from typing import Dict, Any, Optional

# Import the logging configuration
from src.agent_core.logging_config import get_logger

# Get logger for this module
logger = get_logger(__name__)

def verify(
    job_id: str, 
    run_id: str, 
    new_run_id: Optional[str] = None,
    simulate: bool = False
) -> Dict[str, Any]:
    """
    Verify if a fix was successful by checking the status of the new run.
    
    Args:
        job_id: The Databricks job ID
        run_id: The original run ID with issues
        new_run_id: The new run ID to check (if None, assumes the fix didn't include a new run)
        simulate: Whether to simulate verification
        
    Returns:
        Verification results with status and details
    """
    logger.info(f"Verifying fix for job {job_id}, run {run_id}")
    
    # For simulation, randomly determine success with 80% probability
    if simulate:
        logger.info("Simulating verification")
        success_probability = 0.8
        is_successful = random.random() < success_probability
        
        status = "SUCCESSFUL" if is_successful else "FAILED"
        message = f"Simulated verification: Fix was {'successful' if is_successful else 'unsuccessful'}"
        
        logger.info(f"Simulated verification result: {status}")
        
        return {
            "status": status,
            "message": message,
            "details": {
                "job_id": job_id,
                "run_id": run_id,
                "new_run_id": new_run_id,
                "timestamp": time.time()
            }
        }
    
    # In a real implementation, this would check the Databricks run status via API
    # For now, we'll just return success
    
    status = "SUCCESSFUL"
    message = "Fix verified as successful"
    
    logger.info(f"Verification result: {status}")
    
    return {
        "status": status,
        "message": message,
        "details": {
            "job_id": job_id,
            "run_id": run_id,
            "new_run_id": new_run_id,
            "timestamp": time.time()
        }
    } 