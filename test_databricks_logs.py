#!/usr/bin/env python
"""
Test script to pull logs from Databricks using the DbxClient wrapper.
This will demonstrate the client functionality and show how logs are structured.
"""

import os
import json
import logging
from dotenv import load_dotenv

# Configure basic logging to see the output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s'
)

# Import our Databricks client
from agent_core.dbx_client import DbxClient, RunStatus

def main():
    """Main function to test Databricks client and logging."""
    # Load environment variables from .env file
    load_dotenv()
    
    print("\n=== DATABRICKS CLIENT TEST ===\n")
    
    # Create the Databricks client
    print("Initializing Databricks client...")
    client = DbxClient(validate_env=True)
    
    if not client.is_available():
        print("ERROR: Databricks client not available. Check your credentials.")
        return
    
    print(f"Connected to Databricks workspace: {client.host}")
    
    # Test listing jobs
    try:
        print("\n=== LISTING JOBS ===")
        jobs = client.list_jobs(limit=5)
        print(f"Found {len(jobs)} jobs:")
        for job in jobs:
            print(f"  - Job ID: {job.get('job_id')}, Name: {job.get('name')}")
        
        # If we have jobs, get details for the first one
        if jobs:
            job_id = jobs[0].get('job_id')
            print(f"\n=== JOB DETAILS (ID: {job_id}) ===")
            job_details = client.get_job(job_id)
            if job_details:
                print(f"  - Name: {job_details.get('name')}")
                print(f"  - Creator: {job_details.get('creator_user_name', 'Unknown')}")
                settings = job_details.get('settings', {})
                schedule = settings.get('schedule', {})
                if schedule:
                    print(f"  - Schedule: {schedule.get('quartz_cron_expression', 'None')}")
            
            # List recent runs for this job
            print(f"\n=== RECENT RUNS FOR JOB {job_id} ===")
            runs = client.list_runs(job_id=job_id, limit=3)
            print(f"Found {len(runs)} recent runs:")
            for run in runs:
                run_id = run.get('run_id')
                state = run.get('state', {})
                life_cycle_state = state.get('life_cycle_state', 'UNKNOWN')
                result_state = state.get('result_state', 'UNKNOWN')
                print(f"  - Run ID: {run_id}, State: {life_cycle_state}, Result: {result_state}")
                
                # Get status for a run
                status, message = client.get_run_status(run_id)
                print(f"    Status: {status.name}, Message: {message or 'None'}")
    except Exception as e:
        print(f"Error during job operations: {e}")
    
    # Test listing clusters
    try:
        print("\n=== LISTING CLUSTERS ===")
        clusters = client.list_clusters()
        print(f"Found {len(clusters)} clusters:")
        for cluster in clusters:
            print(f"  - Cluster ID: {cluster.get('cluster_id')}, Name: {cluster.get('cluster_name')}")
            print(f"    State: {cluster.get('state')}")
    except Exception as e:
        print(f"Error during cluster operations: {e}")
    
    # Test listing SQL warehouses
    try:
        print("\n=== LISTING SQL WAREHOUSES ===")
        warehouses = client.list_warehouses()
        print(f"Found {len(warehouses)} SQL warehouses:")
        for warehouse in warehouses:
            print(f"  - Warehouse ID: {warehouse.get('id')}, Name: {warehouse.get('name')}")
            print(f"    State: {warehouse.get('state')}")
    except Exception as e:
        print(f"Error during SQL warehouse operations: {e}")
    
    print("\n=== OPERATION COMPLETE ===")
    print("Check the logs to see structured logging output.")

if __name__ == "__main__":
    main() 