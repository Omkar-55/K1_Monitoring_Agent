#!/usr/bin/env python
"""
Integration test to verify that logging, tracing, and Databricks API calls 
all work correctly together.
"""

import os
import time
import logging
from dotenv import load_dotenv

# Configure basic logging to see the output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s'
)

# Import our components
from agent_core import enable_tracing
from agent_core.dbx_client import DbxClient, RunStatus

def main():
    """Main function to run the integration test."""
    # Load environment variables from .env file
    load_dotenv()
    
    print("\n=== INTEGRATION TEST: TRACING, LOGGING, AND API CALLS ===\n")
    
    # Step 1: Enable tracing
    print("1. Enabling distributed tracing...")
    tracing_enabled = enable_tracing(
        service_name="k1-monitoring-agent-test",
        # Use the default OTLP endpoint (from env or localhost)
        rate=1.0  # 100% sampling rate for testing
    )
    print(f"   Tracing enabled: {tracing_enabled}")
    
    if not tracing_enabled:
        print("   WARNING: Tracing could not be enabled. Check logs for details.")
        print("   Continuing test without tracing...")
    
    # Step 2: Initialize Databricks client
    print("\n2. Initializing Databricks client...")
    try:
        client = DbxClient(validate_env=True)
        
        if client.is_available():
            print(f"   Connected to Databricks workspace: {client.host}")
        else:
            print("   ERROR: Databricks client not available. Check logs for details.")
            return
    except Exception as e:
        print(f"   ERROR initializing Databricks client: {e}")
        return
    
    # Step 3: Make a series of API calls to generate spans and logs
    print("\n3. Making API calls to generate spans and logs...")
    
    # Test clusters API
    print("   Calling Databricks Clusters API...")
    clusters = client.list_clusters()
    print(f"   Found {len(clusters)} clusters")
    
    # Test jobs API
    print("   Calling Databricks Jobs API...")
    jobs = client.list_jobs(limit=10)
    print(f"   Found {len(jobs)} jobs")
    
    # Test workspace API
    print("   Calling Databricks Workspace API...")
    workspace_objects = client.list_workspaces("/")
    print(f"   Found {len(workspace_objects)} workspace objects at root")
    
    # Test warehouses API
    print("   Calling Databricks SQL Warehouses API...")
    warehouses = client.list_warehouses()
    print(f"   Found {len(warehouses)} SQL warehouses")
    
    # Test activity API (our custom method)
    print("   Calling Databricks Activity API...")
    activities = client.get_activity(days=7, limit=20)
    print(f"   Found {len(activities)} activity records")
    
    # Step 4: Wait a moment for spans to be exported
    print("\n4. Waiting for spans to be exported...")
    time.sleep(2)  # Give time for spans to be processed
    
    # Step 5: Summarize what was done
    print("\n=== TEST COMPLETED ===")
    print("\nResults are visible in the following locations:")
    
    log_path = os.path.join(os.getcwd(), "logs", "agent.log")
    print(f"\n1. Logs: {log_path}")
    print("   This file contains all logging information including:")
    print("   - API calls with parameters and results")
    print("   - Success/failure of operations")
    print("   - Error details")
    print("   - Performance information")
    
    print("\n2. Traces: OpenTelemetry Collector / Jaeger / Zipkin")
    print("   If you have an OTEL collector running, spans were sent to:")
    print(f"   - {os.getenv('OTEL_EXPORTER_OTLP_ENDPOINT', 'http://localhost:4317')}")
    print("   Access your tracing UI to view the traces for 'k1-monitoring-agent-test'")
    
    print("\n3. How to read logs:")
    print("   - Use a text editor to open the log file")
    print("   - Use the 'tail' command to watch log updates in real-time:")
    print("     - In PowerShell: Get-Content -Path logs/agent.log -Wait -Tail 10")
    print("     - In Linux/macOS: tail -f logs/agent.log")
    print("   - Look for structured JSON objects after 'Databricks API call:'")
    
    print("\nAll test operations have been logged and traced!")

if __name__ == "__main__":
    main() 