#!/usr/bin/env python
"""
Test script to get logs from Databricks using the DbxClient wrapper.
"""

import os
import json
import time
import logging
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Configure basic logging to see the output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s'
)

# Import our Databricks client
from agent_core.dbx_client import DbxClient, RunStatus

def format_timestamp(ts_ms):
    """Format a millisecond timestamp to a readable date string."""
    if not ts_ms:
        return "N/A"
    dt = datetime.fromtimestamp(ts_ms / 1000)  # Convert ms to seconds
    return dt.strftime("%Y-%m-%d %H:%M:%S")

def main():
    """Main function to test fetching Databricks logs."""
    # Load environment variables from .env file
    load_dotenv()
    
    print("\n=== DATABRICKS LOGS FETCH TEST ===\n")
    
    # Create the Databricks client
    print("Initializing Databricks client...")
    client = DbxClient(validate_env=True)
    
    if not client.is_available():
        print("ERROR: Databricks client not available. Check your credentials.")
        return
    
    print(f"Connected to Databricks workspace: {client.host}")
    
    # Calculate timestamps for last 7 days
    end_time = int(time.time() * 1000)  # Current time in ms
    start_time = end_time - (7 * 24 * 60 * 60 * 1000)  # 7 days ago in ms
    
    print(f"Fetching logs from {format_timestamp(start_time)} to {format_timestamp(end_time)}")
    
    # First, try to get SQL query logs
    try:
        print("\n=== SQL QUERY HISTORY ===")
        logs = client.get_logs(start_time=start_time, end_time=end_time, limit=10)
        print(f"Found {len(logs)} log entries")
        
        # Display the logs
        for i, log in enumerate(logs):
            print(f"\nLog Entry #{i+1}:")
            
            # Try to format nicer if possible
            if 'query_text' in log:
                print(f"Query: {log['query_text'][:150]}...")
            if 'user_name' in log:
                print(f"User: {log['user_name']}")
            if 'executed_at_ms' in log:
                print(f"Time: {format_timestamp(log['executed_at_ms'])}")
            if 'duration' in log:
                print(f"Duration: {log['duration']} ms")
            if 'status' in log:
                print(f"Status: {log['status']}")
                
            # Show any other fields that might be useful
            for key, value in log.items():
                if key not in ['query_text', 'user_name', 'executed_at_ms', 'duration', 'status']:
                    # Only show scalar values for readability
                    if isinstance(value, (str, int, float, bool)) and not isinstance(value, dict):
                        print(f"{key}: {value}")
    except Exception as e:
        print(f"Error fetching SQL query logs: {e}")
    
    # Try to get logs for a specific cluster if available
    try:
        # First get cluster list
        clusters = client.list_clusters()
        if clusters:
            cluster_id = clusters[0].get('cluster_id')
            print(f"\n=== CLUSTER LOGS (ID: {cluster_id}) ===")
            cluster_logs = client.get_logs(cluster_id=cluster_id, limit=10)
            
            if cluster_logs:
                print(f"Found {len(cluster_logs)} cluster log entries")
                
                for i, log in enumerate(cluster_logs[:5]):  # Show first 5 only
                    print(f"\nCluster Log #{i+1}:")
                    
                    # Try to format nicer if possible
                    if 'timestamp' in log:
                        print(f"Time: {format_timestamp(log['timestamp'])}")
                    if 'type' in log:
                        print(f"Type: {log['type']}")
                    if 'details' in log:
                        print(f"Details: {log['details']}")
                    
                    # Show any other useful fields
                    for key, value in log.items():
                        if key not in ['timestamp', 'type', 'details']:
                            # Only show scalar values for readability
                            if isinstance(value, (str, int, float, bool)) and not isinstance(value, dict):
                                print(f"{key}: {value}")
            else:
                print("No cluster logs found")
        else:
            print("No clusters available to fetch logs from")
    except Exception as e:
        print(f"Error fetching cluster logs: {e}")
    
    # Try to get logs for a specific job run if available
    try:
        # First get all jobs
        jobs = client.list_jobs(limit=5)
        if jobs:
            job_id = jobs[0].get('job_id')
            print(f"\n=== JOB RUNS (JOB ID: {job_id}) ===")
            
            # Get runs for this job
            runs = client.list_runs(job_id=job_id, limit=5)
            if runs:
                run_id = runs[0].get('run_id')
                print(f"Fetching logs for Run ID: {run_id}")
                
                # Get logs for this run
                run_logs = client.get_logs(run_id=run_id)
                
                if run_logs:
                    print(f"Found {len(run_logs)} run log entries")
                    
                    for i, log in enumerate(run_logs):
                        print(f"\nRun Log #{i+1}:")
                        
                        # Try to extract the actual log content
                        if 'logs' in log:
                            log_text = log['logs']
                            # Show only first 200 chars for readability
                            print(f"Log: {log_text[:200]}...")
                            print(f"(Log length: {len(log_text)} characters)")
                        elif 'error' in log:
                            print(f"Error: {log['error']}")
                        else:
                            # Just dump the whole log object
                            print(json.dumps(log, indent=2))
                else:
                    print("No run logs found")
            else:
                print("No runs found for this job")
        else:
            print("No jobs available to fetch logs from")
    except Exception as e:
        print(f"Error fetching job run logs: {e}")
    
    print("\n=== LOGS FETCH COMPLETE ===")
    print("Check the application logs to see structured logging of the API calls.")

if __name__ == "__main__":
    main() 