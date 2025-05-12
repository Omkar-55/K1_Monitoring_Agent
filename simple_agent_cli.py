#!/usr/bin/env python
"""
Simple CLI interface for the Databricks Monitoring Agent.
This provides a clean interface with step-by-step progress,
user approval prompts, and clear summary reporting.
"""

import os
import asyncio
import json
import sys
import uuid
import time
from typing import Dict, Any, Optional, List

# Import the agent core package
from src.agent_core.logging_config import get_logger, setup_logging
from src.agent_core.dbx_client import DbxClient

# Import the DatabricksMonitoringAgent
from src.agents.databricks_monitoring_agent import (
    DatabricksMonitoringAgent, 
    MonitoringRequest,
    MonitoringResponse
)

# Set up logging
setup_logging()
logger = get_logger(__name__)

class AgentCLI:
    """Simple CLI interface for monitoring agent"""
    
    def __init__(self):
        """Initialize the CLI interface"""
        self.agent = DatabricksMonitoringAgent()
        self.conversation_id = str(uuid.uuid4())
        
    async def run(self, job_id: str, simulate: bool = True, failure_type: str = "memory_exceeded"):
        """Run the monitoring agent for a job"""
        print(f"\n{'='*80}")
        print(f"🔍 DATABRICKS MONITORING AGENT - JOB: {job_id}")
        print(f"{'='*80}")
        
        if simulate:
            print(f"\n📊 SIMULATION MODE: {failure_type}\n")
        
        # Step 1: Initial request - get diagnosis
        print("\n📋 STEP 1: DIAGNOSING ISSUES\n")
        
        request = MonitoringRequest(
            job_id=job_id,
            simulate=simulate,
            simulate_failure_type=failure_type,
            conversation_id=self.conversation_id
        )
        
        print("⏳ Analyzing logs and diagnosing issues...")
        response = await self.agent.process_request(request)
        
        # Check if an issue was detected
        if not response.issue_detected:
            print("✅ No issues detected in the job!")
            return
        
        # Step 2: Display diagnosis results
        issue_type = response.issue_type
        confidence = response.confidence
        
        print(f"\n🔎 Detected issue: {issue_type.replace('_', ' ')} (confidence: {confidence:.2f})")
        
        # Display reasoning steps
        if response.reasoning:
            print("\n📊 REASONING STEPS:")
            for idx, step in enumerate(response.reasoning, 1):
                step_name = step.get("step", "unknown").replace("_", " ").title()
                print(f"\n  {idx}. {step_name}")
                
                if "result" in step:
                    print(f"     Result: {step['result']}")
                
                if "details" in step and step["details"]:
                    # Truncate long details
                    details = step["details"]
                    if len(details) > 200:
                        details = details[:200] + "..."
                    print(f"     Details: {details}")
        
        # Step 3: Show suggested fix and ask for approval
        if response.suggested_fix:
            print("\n📋 STEP 2: SUGGESTED FIX\n")
            
            fix = response.suggested_fix
            fix_type = fix.get("fix_type", "unknown")
            fix_description = fix.get("description", "No description available")
            fix_parameters = fix.get("parameters", {})
            fix_id = fix.get("fix_id", "unknown")
            
            print(f"🔧 Suggested fix: {fix_type.replace('_', ' ')}")
            print(f"📝 Description: {fix_description}")
            print("\n📊 Parameters:")
            for param_name, param_value in fix_parameters.items():
                print(f"  - {param_name}: {param_value}")
            
            # Ask for user approval
            print("\n🔍 STEP 3: APPROVE FIX?\n")
            while True:
                approval = input("Would you like to apply this fix? (y/n): ").strip().lower()
                if approval in ('y', 'yes'):
                    break
                elif approval in ('n', 'no'):
                    print("\n❌ Fix rejected. Exiting...")
                    return
                else:
                    print("Please enter 'y' or 'n'")
            
            # Step 4: Apply the fix
            print("\n📋 STEP 4: APPLYING FIX\n")
            print("⏳ Applying fix...")
            
            # Create a new request with the approved fix ID
            fix_request = MonitoringRequest(
                job_id=job_id,
                simulate=simulate,
                simulate_failure_type=failure_type,
                approved_fix=fix_id,
                conversation_id=self.conversation_id
            )
            
            fix_response = await self.agent.process_request(fix_request)
            
            # Step 5: Show results
            print("\n📋 STEP 5: RESULTS\n")
            
            if fix_response.fix_successful:
                print("✅ Fix applied successfully!")
            else:
                print("❌ Fix application failed!")
            
            # Display final report
            if fix_response.report:
                print(f"\n{'='*80}")
                print("📊 FINAL REPORT")
                print(f"{'='*80}\n")
                print(fix_response.report)
                print(f"\n{'='*80}")
        else:
            print("\n❌ No fix was suggested for this issue.")

async def main():
    """Main entry point"""
    # Set up defaults
    job_id = "test_job_123"
    simulate = True
    failure_type = "memory_exceeded"
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        job_id = sys.argv[1]
    
    if len(sys.argv) > 2:
        simulate_arg = sys.argv[2].lower()
        simulate = simulate_arg in ("true", "yes", "y", "1")
    
    if len(sys.argv) > 3 and simulate:
        failure_type = sys.argv[3]
    
    # Run the CLI
    cli = AgentCLI()
    await cli.run(job_id, simulate, failure_type)

if __name__ == "__main__":
    asyncio.run(main()) 