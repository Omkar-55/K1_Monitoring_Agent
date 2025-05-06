"""
Simple script to get Databricks logs using DbxClient and direct API access
"""
import os
import traceback
import json
import requests
from datetime import datetime
from dotenv import load_dotenv
from src.agent_core.dbx_client import DbxClient, DATABRICKS_SDK_AVAILABLE

# Load environment variables from .env file
print("Loading environment variables from .env file...")
load_dotenv()

def make_databricks_api_call(host, token, endpoint, method="GET", data=None):
    """
    Make a direct API call to Databricks REST API.
    
    Args:
        host: Databricks host URL
        token: Databricks access token
        endpoint: API endpoint path (starting with /api/...)
        method: HTTP method (GET, POST, etc.)
        data: Request payload for POST/PUT requests
    
    Returns:
        Response JSON or None if failed
    """
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    
    # Ensure endpoint starts with a slash
    if not endpoint.startswith('/'):
        endpoint = f'/{endpoint}'
    
    # Ensure host doesn't end with a slash
    if host.endswith('/'):
        host = host[:-1]
    
    url = f"{host}{endpoint}"
    
    try:
        if method.upper() == "GET":
            if data:
                # For GET with data, convert to query params
                response = requests.get(url, headers=headers, params=data)
            else:
                response = requests.get(url, headers=headers)
        elif method.upper() == "POST":
            response = requests.post(url, headers=headers, json=data)
        else:
            print(f"Unsupported method: {method}")
            return None
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"API call failed: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"Exception in API call to {endpoint}: {str(e)}")
        return None

def get_run_output(host, token, run_id):
    """Get job run output logs using admin API access"""
    return make_databricks_api_call(
        host, token, 
        "/api/2.0/jobs/runs/get-output", 
        "GET", 
        {"run_id": int(run_id)}
    )

def get_run_details(host, token, run_id):
    """Get detailed information about a job run using admin API access"""
    return make_databricks_api_call(
        host, token, 
        "/api/2.0/jobs/runs/get", 
        "GET", 
        {"run_id": int(run_id)}
    )

def get_cluster_events(host, token, cluster_id, limit=20):
    """Get cluster events using admin API access"""
    return make_databricks_api_call(
        host, token, 
        "/api/2.0/clusters/events", 
        "GET", 
        {"cluster_id": cluster_id, "limit": limit}
    )

def get_cluster_logs(host, token, cluster_id, log_type="driver"):
    """Get cluster logs using admin API access"""
    if log_type == "driver":
        return make_databricks_api_call(
            host, token, 
            f"/api/2.0/clusters/{cluster_id}/driver-logs"
        )
    else:
        return make_databricks_api_call(
            host, token, 
            f"/api/2.0/clusters/log", 
            "GET", 
            {"cluster_id": cluster_id, "log_type": log_type}
        )

def safe_get(obj, key, default=None):
    """Safely get a value from a dict or return default"""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return default

def get_databricks_info():
    """Get logs and job info from Databricks workspace using admin access"""
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
        print(" DATABRICKS WORKSPACE INFORMATION (ADMIN ACCESS)")
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
        print(" JOB RUN DETAILS AND LOGS (ADMIN API)")
        print("="*80)
        
        # Get detailed run information and logs for each run
        if run_ids:
            print("\n--- Job Run Details and Logs ---")
            for i, run_id in enumerate(run_ids[:3]):  # Limit to first 3 runs
                print(f"\n==== Run {run_id} Details ====")
                
                # 1. Get detailed run information using the get API
                run_details = get_run_details(host, token, run_id)
                if run_details:
                    print("Successfully retrieved run details!")
                    summary["logs_found"] += 1
                    
                    # Extract key information
                    state = safe_get(run_details, "state", {})
                    life_cycle_state = safe_get(state, "life_cycle_state", "Unknown")
                    result_state = safe_get(state, "result_state", "Unknown")
                    state_message = safe_get(state, "state_message", "No message")
                    
                    # Display run information
                    print(f"State: {life_cycle_state}")
                    print(f"Result: {result_state}")
                    print(f"Message: {state_message}")
                    
                    # Show trigger information if available
                    if "trigger" in run_details:
                        trigger = run_details.get("trigger")
                        if isinstance(trigger, dict):
                            print(f"Trigger Type: {trigger.get('type', 'Unknown')}")
                        else:
                            print(f"Trigger: {trigger}")
                    
                    # Show cluster information
                    if "cluster_instance" in run_details:
                        cluster_info = run_details.get("cluster_instance", {})
                        cluster_id = cluster_info.get("cluster_id", "Unknown")
                        print(f"Cluster ID: {cluster_id}")
                        
                        # For errors, check for termination reason
                        if "termination_reason" in cluster_info:
                            term_reason = cluster_info.get("termination_reason", {})
                            code = safe_get(term_reason, "code", "Unknown")
                            type_str = safe_get(term_reason, "type", "Unknown")
                            params = safe_get(term_reason, "parameters", {})
                            
                            print(f"Termination Code: {code}")
                            print(f"Termination Type: {type_str}")
                            if params:
                                print("Termination Parameters:")
                                for k, v in params.items():
                                    print(f"  {k}: {v}")
                    
                    # Extract any errors or task information
                    tasks = run_details.get("tasks", [])
                    if tasks:
                        print(f"\nRun Tasks: {len(tasks)}")
                        for idx, task in enumerate(tasks):
                            task_key = safe_get(task, "task_key", "Unknown")
                            task_state = safe_get(safe_get(task, "state", {}), "life_cycle_state", "Unknown")
                            task_result = safe_get(safe_get(task, "state", {}), "result_state", "Unknown")
                            
                            print(f"Task {idx+1}: {task_key} - State: {task_state}, Result: {task_result}")
                            
                            # Check for exceptions in task
                            task_message = safe_get(safe_get(task, "state", {}), "state_message")
                            if task_message:
                                print(f"Task Message: {task_message}")
                
                # 2. Get run output logs
                print(f"\n==== Run {run_id} Logs ====")
                run_output = get_run_output(host, token, run_id)
                
                if run_output:
                    print("Successfully retrieved run output logs!")
                    summary["logs_found"] += 1
                    
                    # Extract logs
                    logs = run_output.get("logs", "")
                    
                    if logs:
                        print("\n--- Log Output ---")
                        # Print first 15 lines and last 15 lines
                        lines = logs.splitlines()
                        if len(lines) > 30:
                            print("\n".join(lines[:15]))
                            print("\n... [truncated] ...")
                            print("\n".join(lines[-15:]))
                        else:
                            print(logs)
                    else:
                        print("No logs available in run output")
                        
                    # Check for error stack trace
                    error_trace = run_output.get("error_trace", "")
                    if error_trace:
                        print("\n--- Error Stack Trace ---")
                        print(error_trace)
                        
                    # Check for notebook output
                    notebook_output = run_output.get("notebook_output", {})
                    if notebook_output:
                        print("\n--- Notebook Output Summary ---")
                        if "result" in notebook_output:
                            print(f"Result: {notebook_output.get('result')}")
                        if "truncated" in notebook_output:
                            print(f"Output Truncated: {notebook_output.get('truncated')}")
                else:
                    print(f"Failed to retrieve logs for run {run_id}")
                    summary["errors"].append(f"No logs found for run {run_id}")
        
        print("\n" + "="*80)
        print(" CLUSTER INFORMATION AND EVENTS")
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
        
        # Get cluster events which are more reliable than logs
        if cluster_ids:
            print("\n--- Cluster Events (Admin API) ---")
            for i, cluster_id in enumerate(cluster_ids[:2]):  # Limit to first 2 clusters
                print(f"\n==== Cluster {cluster_id} Events ====")
                
                events = get_cluster_events(host, token, cluster_id)
                
                if events and "events" in events:
                    event_list = events.get("events", [])
                    print(f"Found {len(event_list)} events")
                    summary["logs_found"] += len(event_list)
                    
                    # Print the most recent events
                    for idx, event in enumerate(event_list[:5]):  # Show first 5 events
                        event_type = event.get("type", "Unknown")
                        timestamp = event.get("timestamp", 0)
                        
                        # Convert timestamp to readable format
                        if timestamp > 0:
                            try:
                                event_time = datetime.fromtimestamp(timestamp/1000).strftime('%Y-%m-%d %H:%M:%S')
                            except:
                                event_time = f"{timestamp} (raw timestamp)"
                        else:
                            event_time = "Unknown time"
                            
                        print(f"\nEvent {idx+1}: {event_type} at {event_time}")
                        
                        # Check for details
                        details = event.get("details", {})
                        if details:
                            print("Event details:")
                            # Format the details to show key information
                            for k, v in details.items():
                                print(f"  {k}: {v}")
                            
                        # For cluster events, check for specific error information
                        if event_type in ["CREATING", "TERMINATING", "UNEXPECTED_ERROR"]:
                            reason = safe_get(details, "reason", "No reason provided")
                            user = safe_get(details, "user", "Unknown")
                            print(f"  Reason: {reason}")
                            print(f"  User: {user}")
                else:
                    print(f"No events found for cluster {cluster_id}")
                
                # Also try to get logs but they may not be available
                print(f"\n==== Cluster {cluster_id} Logs ====")
                for log_type in ["driver", "stdout", "stderr"]:
                    logs = get_cluster_logs(host, token, cluster_id, log_type)
                    
                    if logs:
                        print(f"\nRetrieved {log_type} logs!")
                        summary["logs_found"] += 1
                        
                        # Extract log content based on possible formats
                        log_content = None
                        if isinstance(logs, dict):
                            if "text" in logs:
                                log_content = logs.get("text", "")
                            elif "contents" in logs:
                                log_content = logs.get("contents", "")
                            elif "logs" in logs:
                                log_content = logs.get("logs", "")
                            elif "log" in logs:
                                log_content = logs.get("log", "")
                        
                        if log_content:
                            # Print a sample
                            print(f"\n--- {log_type} log sample ---")
                            # Show first few lines
                            lines = log_content.splitlines() if isinstance(log_content, str) else []
                            sample = "\n".join(lines[:10]) if len(lines) > 10 else log_content
                            print(sample)
                            print("... [more log content available] ...")
                        else:
                            print(f"No {log_type} log content available")
                    else:
                        print(f"Failed to retrieve {log_type} logs for cluster {cluster_id}")
        
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
    print("Starting Databricks logs retrieval script with admin access...")
    get_databricks_info()
    print("\nScript completed.") 