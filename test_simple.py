"""
Simple test script for the Databricks monitoring agent.

This test uses simulation to test the basic functionality
without requiring the Azure Agents SDK.
"""

import asyncio
import logging
import json
import os
import sys

# Add project root to path if needed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import our agent helper
from src.agents.databricks_monitoring_agent import monitor_databricks_job

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s'
)
logger = logging.getLogger(__name__)

async def main():
    """Run the simple test."""
    # Test with different failure types
    failure_types = ["memory_exceeded", "dependency_error", "disk_space_exceeded"]
    
    for failure_type in failure_types:
        logger.info(f"Testing with failure type: {failure_type}")
        
        # Call the utility function to monitor the job
        response = await monitor_databricks_job(
            job_id="12345",
            simulate=True,
            simulate_failure_type=failure_type
        )
        
        # Print a simple summary
        print("\n" + "=" * 80)
        print(f"\nTEST RESULTS FOR {failure_type.upper()}\n")
        
        print(f"Issue Type: {response.get('issue_type', 'Unknown')}")
        print(f"Fix Attempts: {response.get('fix_attempts', 0)}")
        print(f"Fix Successful: {response.get('fix_successful', False)}")
        print(f"Hallucination Check: {'Detected' if response.get('hallucination_detected', False) else 'None'}")
        print(f"Safety Check: {'Issues Detected' if response.get('safety_issues', False) else 'Passed'}")
        print(f"Confidence: {response.get('confidence', 0.0):.2f}")
        
        print("\n" + "-" * 80 + "\n")
        print("FINAL REPORT:\n")
        print(response.get("report", "No report generated"))
        
        print("\n" + "=" * 80 + "\n")
        
        # Small pause between tests
        await asyncio.sleep(1)
    
    logger.info("Tests completed successfully")

if __name__ == "__main__":
    try:
        # Run the test
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
    except Exception as e:
        logger.error(f"Error in test: {e}", exc_info=True)
        sys.exit(1) 