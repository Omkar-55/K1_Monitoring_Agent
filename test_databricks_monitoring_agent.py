"""
Test script for the Databricks monitoring agent.

Tests the functionality of the agent including:
1. Logging functionality
2. Tracing functionality
3. Databricks integration
4. Hallucination detection
5. Safety checks
"""

import asyncio
import sys
import os
import logging
from typing import Dict, Any, Optional

# Add project root to path if needed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import our agent
from src.agents.databricks_monitoring_agent import (
    DatabricksMonitoringAgent,
    MonitoringRequest,
    MonitoringResponse
)

# Import logging configuration
from src.agent_core.logging_config import get_logger

# Set up logger
logger = get_logger(__name__)

async def test_monitoring_agent():
    """Test the Databricks monitoring agent."""
    logger.info("Starting test of Databricks monitoring agent")
    
    # Initialize the agent
    agent = DatabricksMonitoringAgent()
    logger.info("Initialized Databricks monitoring agent")
    
    # Test with different failure types
    failure_types = ["memory_exceeded", "dependency_error", "disk_space_exceeded"]
    
    for failure_type in failure_types:
        logger.info(f"Testing agent with failure type: {failure_type}")
        
        # Create a test request
        request = MonitoringRequest(
            job_id="12345",
            simulate=True,
            simulate_failure_type=failure_type
        )
        
        # Process the request
        response = await agent.process_request(request)
        
        # Log the summary of the response
        logger.info(f"Response summary for {failure_type}:")
        logger.info(f"  Issue detected: {response.issue_detected}")
        logger.info(f"  Issue type: {response.issue_type}")
        logger.info(f"  Fix attempts: {response.fix_attempts}")
        logger.info(f"  Fix successful: {response.fix_successful}")
        logger.info(f"  Hallucination detected: {response.hallucination_detected}")
        logger.info(f"  Safety issues: {response.safety_issues}")
        logger.info(f"  Confidence: {response.confidence}")
        
        # Print the report
        print(f"\n\n=== REPORT FOR {failure_type.upper()} ===\n")
        print(response.report)
        print("\n" + "=" * 80 + "\n")
        
        # Add a small delay between tests
        await asyncio.sleep(1)
    
    logger.info("Completed tests of Databricks monitoring agent")
    return True

async def test_monitoring_workflow(
    job_id: str, 
    run_id: Optional[str] = None,
    simulate: bool = False,
    failure_type: Optional[str] = None
):
    """
    Test the entire workflow.
    
    Args:
        job_id: The Databricks job ID to monitor
        run_id: Optional specific run ID to analyze
        simulate: Whether to simulate fix execution
        failure_type: Simulated failure type for testing
    """
    logger.info(f"Starting monitoring workflow for job {job_id}")
    
    # 1. Initialize agent
    agent = DatabricksMonitoringAgent()
    
    # 2. Create request
    request = MonitoringRequest(
        job_id=job_id,
        run_id=run_id,
        simulate=simulate,
        simulate_failure_type=failure_type
    )
    
    # 3. Process request
    response = await agent.process_request(request)
    
    # 4. Print response
    print("\n" + "=" * 80 + "\n")
    print(f"# Databricks Monitoring Results for Job {job_id}")
    print(f"- Issue Detected: {response.issue_detected}")
    if response.issue_detected:
        print(f"- Issue Type: {response.issue_type}")
        print(f"- Fix Attempts: {response.fix_attempts}")
        print(f"- Fix Successful: {response.fix_successful}")
        print(f"- Hallucination Check: {'Detected' if response.hallucination_detected else 'None'}")
        print(f"- Safety Check: {'Issues Detected' if response.safety_issues else 'Passed'}")
        print(f"- Confidence: {response.confidence:.2f}")
        print("\n## Reasoning Chain:\n")
        
        # Print the reasoning chain
        for i, step in enumerate(response.reasoning):
            print(f"{i+1}. **{step.get('step', 'Unknown').title()}** ({step.get('attempt', '')})")
            print(f"   {step.get('result', 'No result')}")
            print()
            
        # Print the full report
        print("\n## Final Report\n")
        print(response.report)
    
    print("\n" + "=" * 80 + "\n")
    
    logger.info("Monitoring workflow completed")
    return response

if __name__ == "__main__":
    try:
        # Run the test suite
        logger.info("Starting test script")
        
        if len(sys.argv) > 1 and sys.argv[1] == "workflow":
            # Run a single workflow test
            if len(sys.argv) > 2:
                failure_type = sys.argv[2]
            else:
                failure_type = "memory_exceeded"
                
            logger.info(f"Running workflow test with failure type: {failure_type}")
            asyncio.run(test_monitoring_workflow(
                job_id="12345",
                simulate=True,
                failure_type=failure_type
            ))
        else:
            # Run all tests
            asyncio.run(test_monitoring_agent())
            
        logger.info("Test script completed successfully")
        
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
    except Exception as e:
        logger.error(f"Error in test script: {e}", exc_info=True)
        print(f"Error: {e}")
        sys.exit(1) 