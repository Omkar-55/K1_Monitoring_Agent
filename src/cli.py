#!/usr/bin/env python
"""
Command-line interface for the K1 Monitoring Agent.

Allows users to interact with the Databricks monitoring agent through the command line.
"""

import os
import sys
import argparse
import asyncio
from typing import Dict, Any, Optional

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our agent
from src.agents.databricks_monitoring_agent import (
    DatabricksMonitoringAgent,
    MonitoringRequest,
    MonitoringResponse,
    monitor_databricks_job
)

# Import logging configuration
from src.agent_core.logging_config import get_logger, setup_logging

# Initialize logging
setup_logging(log_level="info")
logger = get_logger(__name__)

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="K1 Monitoring Agent for Databricks workspaces",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Main command (subparser parent)
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Monitor command
    monitor_parser = subparsers.add_parser(
        "monitor", 
        help="Monitor a Databricks job and fix issues"
    )
    monitor_parser.add_argument(
        "--job-id", 
        type=str, 
        required=True,
        help="The Databricks job ID to monitor"
    )
    monitor_parser.add_argument(
        "--run-id", 
        type=str, 
        help="A specific run ID to analyze (if omitted, uses latest)"
    )
    monitor_parser.add_argument(
        "--workspace", 
        type=str, 
        help="The Databricks workspace URL"
    )
    monitor_parser.add_argument(
        "--max-attempts", 
        type=int, 
        default=5,
        help="Maximum number of fix attempts"
    )
    monitor_parser.add_argument(
        "--simulate", 
        action="store_true", 
        help="Simulate fix execution instead of actually applying fixes"
    )
    monitor_parser.add_argument(
        "--failure-type", 
        type=str, 
        choices=["memory_exceeded", "dependency_error", "disk_space_exceeded"],
        help="Simulated failure type for testing (only used with --simulate)"
    )
    
    # Version command
    version_parser = subparsers.add_parser(
        "version", 
        help="Show version information"
    )
    
    return parser.parse_args()

async def run_monitor_command(args):
    """Run the monitor command."""
    logger.info(f"Monitoring job {args.job_id}")
    
    try:
        # Use the utility function to monitor the job
        response = await monitor_databricks_job(
            job_id=args.job_id,
            run_id=args.run_id,
            simulate=args.simulate,
            simulate_failure_type=args.failure_type
        )
        
        # Print results
        print("\n" + "=" * 80)
        print(f"\nMONITORING RESULTS FOR JOB {args.job_id}\n")
        
        if response.get("issue_detected", False):
            print(f"Issue Type: {response.get('issue_type', 'Unknown')}")
            print(f"Fix Attempts: {response.get('fix_attempts', 0)}")
            print(f"Fix Successful: {response.get('fix_successful', False)}")
            print(f"Hallucination Check: {'Detected' if response.get('hallucination_detected', False) else 'None'}")
            print(f"Safety Check: {'Issues Detected' if response.get('safety_issues', False) else 'Passed'}")
            print(f"Confidence: {response.get('confidence', 0.0):.2f}")
            
            print("\n" + "-" * 80 + "\n")
            print("FINAL REPORT:\n")
            print(response.get("report", "No report generated"))
        else:
            print("No issues detected in the Databricks job.")
        
        print("\n" + "=" * 80 + "\n")
        
    except Exception as e:
        logger.error(f"Error monitoring job: {e}", exc_info=True)
        print(f"\nError: {e}\n")
        sys.exit(1)

def show_version():
    """Show version information."""
    print("K1 Monitoring Agent v0.1.0")
    print("An intelligent agent for monitoring and fixing Databricks workspaces")
    print("© 2025 K1 Technologies")

def main():
    """Main entry point for the CLI."""
    args = parse_args()
    
    if args.command == "monitor":
        asyncio.run(run_monitor_command(args))
    elif args.command == "version":
        show_version()
    else:
        # If no command specified, show help
        print("Please specify a command. Use --help for more information.")
        sys.exit(1)

if __name__ == "__main__":
    main() 