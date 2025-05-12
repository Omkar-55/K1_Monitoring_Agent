#!/usr/bin/env python
"""
Test script for the Databricks Monitoring Agent with AI-based analysis.
"""

import os
import asyncio
import json
import uuid
import time
from src.agents.databricks_monitoring_agent import monitor_databricks_job, MonitoringRequest
from src.agent_core.logging_config import setup_logging, get_logger

# Set up logging
setup_logging()
logger = get_logger(__name__)

async def run_test_with_approval(failure_type=None):
    """Run a complete test cycle with auto-approval of the fix."""
    logger.info(f"Testing with failure type: {failure_type}")
    
    # Create a unique conversation ID for this test
    conversation_id = str(uuid.uuid4())
    
    # First call to get the suggestion
    result = await monitor_databricks_job(
        job_id="test_job_123",
        simulate=True,
        simulate_failure_type=failure_type
    )
    
    # Print the initial diagnosis and suggested fix
    print("\n" + "=" * 80)
    print(f"\nTEST RESULTS FOR FAILURE TYPE: {failure_type}\n")
    print(f"Issue Detected: {result.get('issue_detected', False)}")
    print(f"Issue Type: {result.get('issue_type', 'Unknown')}")
    print(f"Confidence: {result.get('confidence', 0.0):.2f}")
    
    # Check if there's a suggested fix
    suggested_fix = result.get("suggested_fix")
    if suggested_fix:
        fix_id = suggested_fix.get("fix_id")
        print(f"\nSuggested Fix: {suggested_fix.get('fix_type')}")
        print(f"Fix Description: {suggested_fix.get('description')}")
        
        # Now approve the fix and run the second part
        print("\nAuto-approving fix...")
        
        # Create a direct request with the approved fix
        request = MonitoringRequest(
            job_id="test_job_123",
            simulate=True,
            simulate_failure_type=failure_type,
            approved_fix=fix_id,
            conversation_id=conversation_id
        )
        
        # Import the agent directly
        from src.agents.databricks_monitoring_agent import DatabricksMonitoringAgent
        agent = DatabricksMonitoringAgent()
        
        # Process the request with the approved fix
        result = await agent.process_request(request)
        result_dict = result.dict()
        
        # Print the final results
        print("\nFIX APPLIED AND VERIFIED")
        print(f"Fix Successful: {result.fix_successful}")
        if result.fix_successful:
            print("\nREPORT SUMMARY:")
            report_lines = result.report.split("\n") if result.report else []
            for line in report_lines[:10]:
                print(line)
            print("...")
    else:
        print("\nNo suggested fix found.")
    
    print("\n" + "=" * 80 + "\n")
    return result

async def main():
    """Run tests for all failure types."""
    logger.info("Starting AI-based diagnosis tests")
    
    try:
        # Set a timeout for the entire test run to prevent hanging
        await asyncio.wait_for(run_tests(), timeout=60)
    except asyncio.TimeoutError:
        logger.error("Tests timed out!")
        print("\nTests timed out! Check the logs for details.")
    except Exception as e:
        logger.error(f"Error running tests: {e}", exc_info=True)
        print(f"\nError: {e}")
    
    logger.info("All tests completed")

async def run_tests():
    """Run all the tests with different failure types."""
    # Test memory issues
    await run_test_with_approval("memory_exceeded")
    
    # Test disk space issues
    await run_test_with_approval("disk_space_exceeded")
    
    # Test dependency issues
    await run_test_with_approval("dependency_error")
    
    # Test unknown issues
    await run_test_with_approval("unknown")

if __name__ == "__main__":
    asyncio.run(main()) 