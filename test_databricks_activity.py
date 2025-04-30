#!/usr/bin/env python
"""
Test script to get activity information from Databricks using the DbxClient wrapper.
"""

import os
import json
import time
import logging
import traceback
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
    try:
        if isinstance(ts_ms, str):
            # Try to parse ISO format
            dt = datetime.fromisoformat(ts_ms.replace('Z', '+00:00'))
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        else:
            # Assume it's a millisecond timestamp
            dt = datetime.fromtimestamp(ts_ms / 1000)  # Convert ms to seconds
            return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return str(ts_ms)  # Return as is if parsing fails

def main():
    """Main function to test fetching Databricks activity."""
    # Load environment variables from .env file
    load_dotenv()
    
    print("\n=== DATABRICKS ACTIVITY FETCH TEST ===\n")
    
    # Create the Databricks client
    print("Initializing Databricks client...")
    try:
        client = DbxClient(validate_env=True)
        
        if not client.is_available():
            print("ERROR: Databricks client not available. Check your credentials.")
            return
        
        print(f"Connected to Databricks workspace: {client.host}")
        
        # Try to get activity information
        try:
            # Get the activity records
            print("\n=== WORKSPACE ACTIVITY ===")
            print("Fetching activity records...")
            activities = client.get_activity(days=30, limit=50)
            print(f"Found {len(activities)} activity records")
            
            if not activities:
                print("No activity records found. This could be due to:")
                print("1. Limited permissions on your Databricks account")
                print("2. No recent activity in the workspace")
                print("3. API access restrictions")
                return
            
            # Group by type
            activity_types = {}
            for activity in activities:
                activity_type = activity.get("type", "unknown")
                if activity_type not in activity_types:
                    activity_types[activity_type] = []
                activity_types[activity_type].append(activity)
            
            # Display by type
            for activity_type, records in activity_types.items():
                print(f"\n=== {activity_type.upper()} ({len(records)} records) ===")
                
                if activity_type == "user_information":
                    for user in records:
                        print(f"\nUser: {user.get('display_name')} ({user.get('user_name')})")
                        print(f"  ID: {user.get('user_id')}")
                        print(f"  Active: {user.get('active')}")
                        print(f"  Last Activity: {user.get('last_activity')}")
                
                elif activity_type == "cluster_information":
                    for cluster in records:
                        print(f"\nCluster: {cluster.get('cluster_name')} ({cluster.get('cluster_id')})")
                        print(f"  Creator: {cluster.get('creator')}")
                        print(f"  State: {cluster.get('state')}")
                        print(f"  Created: {cluster.get('created_time')}")
                
                elif activity_type == "workspace_object":
                    for obj in records:
                        print(f"\nWorkspace Object: {obj.get('path')}")
                        print(f"  Type: {obj.get('object_type')}")
                        if obj.get("language"):
                            print(f"  Language: {obj.get('language')}")
                
                else:
                    # Generic display for other types
                    for i, record in enumerate(records[:5]):  # Show only first 5
                        print(f"\nRecord #{i+1}:")
                        for key, value in record.items():
                            if isinstance(value, (str, int, float, bool)) and not isinstance(value, dict):
                                print(f"  {key}: {value}")
                    
                    if len(records) > 5:
                        print(f"... and {len(records) - 5} more records")
        
        except Exception as e:
            print(f"Error fetching activity information: {e}")
            print(traceback.format_exc())
        
        print("\n=== ACTIVITY FETCH COMPLETE ===")
        print("Check the application logs to see structured logging of the API calls.")
    except Exception as init_error:
        print(f"Error initializing client: {init_error}")
        print(traceback.format_exc())

if __name__ == "__main__":
    main() 