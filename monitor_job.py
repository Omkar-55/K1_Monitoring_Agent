#!/usr/bin/env python
"""
Streamlined CLI tool for monitoring Databricks jobs.

This tool runs the agent in a simpler, step-by-step fashion with
clear outputs and prompts for human approval.
"""

import os
import sys
import json
import asyncio
import uuid
from typing import Dict, Any

from src.agent_core.logging_config import setup_logging, get_logger
from src.agents.databricks_monitoring_agent import DatabricksMonitoringAgent, MonitoringRequest

# Set up logging
setup_logging()
logger = get_logger(__name__)

async def main():
    """Main entry point for the CLI tool"""
    job_id = "test_job_123"
    simulate = True
    failure_type = "memory_exceeded"
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        job_id = sys.argv[1]
    
    if len(sys.argv) > 2:
        failure_type = sys.argv[2]
    
    print("\n" + "=" * 80)
    print(f"K1 MONITORING AGENT - JOB: {job_id}")
    print("=" * 80)
    
    print(f"\nRunning in simulation mode with failure type: {failure_type}")
    print("This will show results step-by-step with pauses for user approval.")
    print("\nPress Enter to start...")
    input()
    
    # Initialize agent and request
    agent = DatabricksMonitoringAgent()
    request_id = str(uuid.uuid4())
    
    # Step 1: Initial diagnosis
    print("\n" + "=" * 80)
    print("STEP 1: DIAGNOSING ISSUES")
    print("=" * 80)
    print("Analyzing logs and diagnosing issues...")
    
    request = MonitoringRequest(
        job_id=job_id,
        simulate=simulate,
        simulate_failure_type=failure_type,
        conversation_id=request_id
    )
    
    response = await agent.process_request(request)
    
    # Check results
    if not response.issue_detected:
        print("\n✅ No issues detected in the job!")
        return
    
    # Display diagnosis
    issue_type = response.issue_type
    confidence = response.confidence
    
    print(f"\n🔍 Issue detected: {issue_type.replace('_', ' ')} (confidence: {confidence:.2f})")
    
    # Show reasoning steps
    if response.reasoning:
        print("\nReasoning Steps:")
        for step in response.reasoning:
            step_name = step.get("step", "unknown").replace("_", " ").title()
            print(f"- {step_name}")
            if "details" in step and step["details"]:
                print(f"  Details: {step['details']}")
    
    # Step 2: Get fix suggestion
    if not response.suggested_fix:
        print("\n❌ No fix suggestion available for this issue.")
        return
    
    print("\n" + "=" * 80)
    print("STEP 2: SUGGESTED FIX")
    print("=" * 80)
    
    fix = response.suggested_fix
    fix_id = fix.get("fix_id", "unknown")
    fix_type = fix.get("fix_type", "unknown")
    fix_description = fix.get("description", "No description available")
    
    print(f"Suggested fix: {fix_type.replace('_', ' ')}")
    print(f"Description: {fix_description}")
    
    if "parameters" in fix:
        print("\nParameters:")
        for key, value in fix["parameters"].items():
            print(f"- {key}: {value}")
    
    # Step 3: Get user approval
    print("\n" + "=" * 80)
    print("STEP 3: USER APPROVAL")
    print("=" * 80)
    
    while True:
        approval = input("\nApply this fix? (y/n): ").strip().lower()
        if approval in ("y", "yes"):
            break
        elif approval in ("n", "no"):
            print("\n❌ Fix rejected. Exiting.")
            return
        else:
            print("Please enter 'y' or 'n'")
    
    # Step 4: Apply the fix
    print("\n" + "=" * 80)
    print("STEP 4: APPLYING FIX")
    print("=" * 80)
    
    fix_request = MonitoringRequest(
        job_id=job_id,
        simulate=simulate,
        simulate_failure_type=failure_type,
        approved_fix=fix_id,
        conversation_id=request_id
    )
    
    print("\nApplying fix...")
    fix_response = await agent.process_request(fix_request)
    
    # Step 5: Show results
    print("\n" + "=" * 80)
    print("STEP 5: RESULTS")
    print("=" * 80)
    
    success = fix_response.fix_successful
    if success:
        print("\n✅ Fix applied successfully!")
    else:
        print("\n❌ Fix application failed!")
    
    # Show final report
    if fix_response.report:
        print("\n" + "=" * 80)
        print("FINAL REPORT")
        print("=" * 80)
        print(fix_response.report)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
    except Exception as e:
        print(f"\nError: {str(e)}")
        logger.exception("Unexpected error") 