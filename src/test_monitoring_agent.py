#!/usr/bin/env python
"""
Test script for the K1 Monitoring Agent.

This script demonstrates how to use the monitoring agent to process queries.
"""

import asyncio
import os
import json
import sys
from dotenv import load_dotenv

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from src.agents.monitoring_agent import MonitoringAgent, MonitoringRequest
from src.agent_core.logging_config import get_logger

# Load environment variables
load_dotenv()

# Set up logger
logger = get_logger(__name__)

async def main():
    """Test the monitoring agent with sample queries."""
    # Initialize the agent (credentials will be pulled from environment variables)
    logger.info("Initializing Monitoring Agent")
    agent = MonitoringAgent()
    
    # Sample queries to test with
    test_queries = [
        "What is the status of the Databricks workspace?",
        "List all clusters",
        "Show me the recent job runs",
        "Get workspace activity for the last 3 days",
        "Show logs for cluster abc123",
        "What is the weather like today?" # This should return an unknown action
    ]
    
    # Process each query
    for query in test_queries:
        logger.info(f"Processing query: {query}")
        
        request = MonitoringRequest(query=query)
        response = await agent.process_request(request)
        
        # Print the response
        print("\n" + "=" * 80)
        print(f"Query: {query}")
        print("-" * 80)
        print(f"Result:\n{response.result}")
        print("=" * 80)
        
        # Wait a moment between requests
        await asyncio.sleep(1)

if __name__ == "__main__":
    # Set up asyncio event loop
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nTest interrupted by user.")
    except Exception as e:
        logger.error(f"Error running test: {e}", exc_info=True)
        print(f"\nError: {e}") 