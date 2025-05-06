"""
Simple script to get Databricks logs using DbxClient
"""
import os
import traceback
import json
from datetime import datetime
from dotenv import load_dotenv
from src.agent_core.dbx_client import DbxClient, DATABRICKS_SDK_AVAILABLE

# Load environment variables from .env file
print("Loading environment variables from .env file...")
load_dotenv()

def get_databricks_info():
    """Get logs and job info from Databricks workspace"""
    # Get Databricks credentials from environment variables
    host = os.getenv("DATABRICKS_HOST")
    token = os.getenv("DATABRICKS_TOKEN")
    
    print(f"Databricks SDK available: {DATABRICKS_SDK_AVAILABLE}")
    print(f"Host set: {'Yes' if host else 'No'}")
    print(f"Token set: {'Yes' if token else 'No'}")
    
    if not host or not token:
        print("Error: Databricks credentials not found in .env file")
        return
    
    # Store information for summary
    summary = {
        "jobs": [],
        "runs": [],
        "clusters": [],
        "logs_found": 0,
        "errors": []
    }
    
    try:
        # Initialize the client with credentials from .env
        print(f"Connecting to Databricks host: {host}")
        client = DbxClient(host=host, token=token)
        
        # Check if client is available
        if not client.is_available():
            print("Error: Databricks client not available. Check connection and credentials.")
            summary["errors"].append("Databricks client not available")
            return
        
        print("\n" + "="*80)
        print(" DATABRICKS WORKSPACE INFORMATION")
        print("="*80)
        
        # List recent jobs
        print("\n--- Recent Jobs ---")
        print("Fetching jobs...")
        jobs = client.list_jobs(limit=5)
        print(f"Found {len(jobs)} jobs")
        
        for job in jobs:
            job_id = job.get("job_id")
            job_name = job.get("settings", {}).get("name", "Unknown")
            print(f"Job ID: {job_id}, Name: {job_name}")
            
            summary["jobs"].append({
                "id": job_id,
                "name": job_name
            })
        
        # List recent runs
        print("\n--- Recent Runs ---")
        print("Fetching runs...")
        runs = client.list_runs(limit=5)
        print(f"Found {len(runs)} runs")
        
        run_ids = []
        for run in runs:
            run_id = run.get("run_id")
            job_id = run.get("job_id")
            state = run.get("state", {}).get("life_cycle_state", "Unknown")
            result_state = run.get("state", {}).get("result_state", "Unknown")
            print(f"Run ID: {run_id}, Job ID: {job_id}, State: {state}, Result: {result_state}")
            
            summary["runs"].append({
                "id": run_id,
                "job_id": job_id,
                "state": state,
                "result": result_state
            })
            
            if run_id:
                run_ids.append(run_id)
        
        print("\n" + "="*80)
        print(" JOB RUN LOGS")
        print("="*80)
        
        # Get logs for each run
        if run_ids:
            print("\n--- Job Run Logs ---")
            for i, run_id in enumerate(run_ids[:2]):  # Limit to first 2 runs to avoid too much output
                print(f"\nFetching logs for run {run_id}...")
                logs = client.get_logs(run_id=run_id)
                if logs:
                    for log_entry in logs:
                        summary["logs_found"] += 1
                        
                        print(f"Run: {log_entry.get('run_id')}")
                        print(f"Status: {log_entry.get('status')}")
                        
                        stdout = log_entry.get('logs', {}).get('stdout', '')
                        stderr = log_entry.get('logs', {}).get('stderr', '')
                        
                        if stdout:
                            print("--- Stdout Logs ---")
                            # Print first 10 lines and last 10 lines of logs
                            lines = stdout.splitlines()
                            if len(lines) > 20:
                                print("\n".join(lines[:10]))
                                print("...[truncated]...")
                                print("\n".join(lines[-10:]))
                            else:
                                print(stdout)
                        else:
                            print("[No stdout logs available]")
                        
                        if stderr:
                            print("--- Stderr Logs ---")
                            # Print first 10 lines and last 10 lines of logs
                            lines = stderr.splitlines()
                            if len(lines) > 20:
                                print("\n".join(lines[:10]))
                                print("...[truncated]...")
                                print("\n".join(lines[-10:]))
                            else:
                                print(stderr)
                else:
                    print(f"No logs found for run {run_id}")
                    summary["errors"].append(f"No logs found for run {run_id}")
        
        print("\n" + "="*80)
        print(" CLUSTER INFORMATION AND LOGS")
        print("="*80)
        
        # List clusters and get logs
        print("\n--- Clusters ---")
        print("Fetching clusters...")
        clusters = client.list_clusters()
        print(f"Found {len(clusters)} clusters")
        
        cluster_ids = []
        for cluster in clusters:
            cluster_id = cluster.get("cluster_id")
            cluster_name = cluster.get("cluster_name", "Unknown")
            state = cluster.get("state", "Unknown")
            
            print(f"Cluster ID: {cluster_id}, Name: {cluster_name}, State: {state}")
            
            summary["clusters"].append({
                "id": cluster_id,
                "name": cluster_name,
                "state": state
            })
            
            if cluster_id:
                cluster_ids.append(cluster_id)
        
        # Get logs for each cluster
        if cluster_ids:
            print("\n--- Cluster Logs ---")
            for i, cluster_id in enumerate(cluster_ids[:2]):  # Limit to first 2 clusters
                print(f"\nFetching logs for cluster {cluster_id}...")
                
                for log_type in ["driver", "stdout", "stderr"]:
                    print(f"Attempting to fetch {log_type} logs...")
                    logs = client.get_logs(cluster_id=cluster_id, log_type=log_type, limit=50)
                    
                    if logs:
                        print(f"Found {len(logs)} log entries for type {log_type}")
                        
                        for log_entry in logs:
                            if "events" in log_entry:
                                event_count = len(log_entry['events'])
                                print(f"Found {event_count} events")
                                summary["logs_found"] += event_count
                                
                                for event in log_entry["events"][:3]:  # First 3 events
                                    print(f"Event Type: {event.get('type')}")
                                    print(f"Timestamp: {event.get('timestamp')}")
                                    print(f"Details: {json.dumps(event.get('details', {}), indent=2)[:200]}...")
                                    if event_count > 3:
                                        print(f"... and {event_count - 3} more events")
                            
                            elif "logs" in log_entry:
                                log_content = log_entry.get("logs", {}).get(log_type)
                                if log_content:
                                    summary["logs_found"] += 1
                                    # Print first 10 lines of logs
                                    lines = log_content.splitlines()
                                    if len(lines) > 10:
                                        print("\n".join(lines[:10]))
                                        print("...[truncated]...")
                                    else:
                                        print(log_content)
                                else:
                                    print(f"[No {log_type} logs content available]")
                            else:
                                content_preview = json.dumps(log_entry)[:200]
                                print(f"Log entry format not recognized: {content_preview}...")
                                if "Log retrieval not supported" in content_preview:
                                    summary["errors"].append(f"Log retrieval not supported for {log_type} logs on cluster {cluster_id}")
                    else:
                        print(f"No {log_type} logs found for cluster {cluster_id}")
                        summary["errors"].append(f"No {log_type} logs found for cluster {cluster_id}")
        
        print("\n" + "="*80)
        print(" ACTIVITY INFORMATION")
        print("="*80)
        
        # Get activity information
        print("\n--- Activity Information ---")
        print("Fetching activity info...")
        activity = client.get_activity(days=7, limit=10)
        print(f"Found {len(activity)} activity records")
        
        if activity:
            print("\nMost recent activity:")
            for act in activity[:2]:
                print(json.dumps(act, indent=2))
                
            # Count activity types
            activity_types = {}
            for act in activity:
                act_type = act.get("type", "unknown")
                activity_types[act_type] = activity_types.get(act_type, 0) + 1
                
            print("\nActivity type breakdown:")
            for act_type, count in activity_types.items():
                print(f"- {act_type}: {count} records")
        
        # Print summary
        print("\n" + "="*80)
        print(" SUMMARY")
        print("="*80)
        
        print(f"Databricks Host: {host}")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Jobs found: {len(summary['jobs'])}")
        print(f"Runs found: {len(summary['runs'])}")
        print(f"Clusters found: {len(summary['clusters'])}")
        print(f"Log entries found: {summary['logs_found']}")
        
        if summary["errors"]:
            print(f"\nErrors encountered: {len(summary['errors'])}")
            for i, error in enumerate(summary["errors"][:5], 1):  # Show first 5 errors
                print(f"{i}. {error}")
            if len(summary["errors"]) > 5:
                print(f"... and {len(summary['errors']) - 5} more errors")
        
    except Exception as e:
        print(f"Error accessing Databricks: {str(e)}")
        print("Stack trace:")
        traceback.print_exc()

if __name__ == "__main__":
    print("Starting Databricks logs retrieval script...")
    get_databricks_info()
    print("\nScript completed.") 