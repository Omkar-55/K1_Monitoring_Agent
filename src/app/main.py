"""
K1 Monitoring Agent - Streamlit Application

This application demonstrates logging and tracing capabilities.
"""

import os
import streamlit as st
import asyncio
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import the agent core package
import src.agent_core
from src.agent_core.logging_config import get_logger, setup_logging
from src.agent_core.core_logic import Agent, AgentInput, AgentResponse
from src.agent_core.dbx_client import DbxClient

# Get logger for this module
logger = get_logger(__name__)

# Set page configuration
st.set_page_config(
    page_title="K1 Monitoring Agent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize tracing in a function so we can call it from the UI
def initialize_tracing():
    """Initialize distributed tracing"""
    success = src.agent_core.enable_tracing(
        service_name="k1-monitoring-agent-ui", 
        rate=1.0
    )
    return success

# Configure logging level from the UI
def configure_logging(level="info"):
    """Configure logging with the specified level"""
    setup_logging(log_level=level)
    logger.info(f"Logging level set to {level}")
    return True

# Initialize agent
@st.cache_resource
def get_agent():
    """Get or create the agent instance"""
    return Agent()

# Function to get Databricks logs
def get_databricks_logs(run_id=None):
    """Get logs from Databricks workspace"""
    # Get Databricks credentials from environment variables
    host = os.getenv("DATABRICKS_HOST")
    token = os.getenv("DATABRICKS_TOKEN")
    
    if not host or not token:
        logger.error("Databricks credentials not found in .env file")
        return {"error": "Databricks credentials not found in .env file"}
    
    try:
        # Initialize the client with credentials from .env
        client = DbxClient(host=host, token=token)
        
        # Retrieve the logs for the specified run
        if run_id:
            logs = client.get_logs(run_id=run_id)
            logger.info(f"Retrieved logs for run {run_id}")
            return logs
        else:
            # If no run_id is provided, list recent runs
            runs = client.list_runs(limit=5)
            logger.info(f"Listed {len(runs)} recent runs")
            return runs
    except Exception as e:
        logger.error(f"Error getting Databricks logs: {e}", exc_info=True)
        return {"error": str(e)}

# Functions to demonstrate logging at different levels
def log_demo():
    """Generate log messages at different levels"""
    logger.debug("This is a DEBUG message from the UI")
    logger.info("This is an INFO message from the UI")
    logger.warning("This is a WARNING message from the UI")
    logger.error("This is an ERROR message from the UI")
    return True

# Async function to process agent query
async def process_agent_query(query, context=None):
    """Process a query with the agent"""
    agent = get_agent()
    input_data = AgentInput(
        query=query,
        context=context or {},
        conversation_id=f"demo-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    )
    return await agent.process_query(input_data)

# Streamlit UI layout
def main():
    """Main Streamlit application"""
    st.title("K1 Monitoring Agent - Logging & Tracing Demo")
    
    # Sidebar for configuration
    st.sidebar.title("Configuration")
    
    # Logging configuration
    st.sidebar.header("Logging")
    log_level = st.sidebar.selectbox(
        "Log Level", 
        options=["debug", "info", "warning", "error", "critical"],
        index=1
    )
    if st.sidebar.button("Apply Logging Configuration"):
        success = configure_logging(log_level)
        if success:
            st.sidebar.success(f"Logging configured at {log_level.upper()} level")
    
    # Tracing configuration
    st.sidebar.header("Tracing")
    if st.sidebar.button("Enable Tracing"):
        success = initialize_tracing()
        if success:
            st.sidebar.success("Tracing enabled successfully!")
        else:
            st.sidebar.error("Failed to enable tracing. Check logs for details.")
    
    # Main content area with tabs
    tab1, tab2, tab3 = st.tabs(["Logging Demo", "Agent Query", "Databricks Logs"])
    
    # Tab 1: Logging Demo
    with tab1:
        st.header("Logging Demonstration")
        st.write("Click the button below to generate log messages at different levels")
        if st.button("Generate Log Messages"):
            log_demo()
            st.success("Log messages generated! Check the console output and log file.")
        
        # Display path to log file
        from pathlib import Path
        log_path = Path(__file__).parent.parent.parent / "logs" / "agent.log"
        st.info(f"Log file location: {log_path}")
    
    # Tab 2: Agent Query
    with tab2:
        st.header("Agent Query with Tracing")
        query = st.text_area("Enter your query", "What is the current status of the system?")
        
        if st.button("Process Query"):
            try:
                st.info("Processing query with distributed tracing...")
                
                # Use asyncio to run the async function
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                response = loop.run_until_complete(process_agent_query(query))
                loop.close()
                
                # Display response
                st.write("### Response")
                st.write(response.response)
                st.write(f"Confidence: {response.confidence:.2f}")
                
                # Log the interaction
                logger.info(f"Processed query: '{query}' - Confidence: {response.confidence:.2f}")
                st.success("Query processed successfully with tracing!")
                
            except Exception as e:
                logger.error(f"Error processing query: {e}", exc_info=True)
                st.error(f"Error: {str(e)}")
    
    # Tab 3: Databricks Logs
    with tab3:
        st.header("Databricks Logs")
        
        # Input for specific run ID
        run_id = st.text_input("Run ID (leave empty to list recent runs)")
        
        if st.button("Get Databricks Logs"):
            with st.spinner("Retrieving logs from Databricks..."):
                logs = get_databricks_logs(run_id)
                
                if "error" in logs:
                    st.error(logs["error"])
                else:
                    st.success("Successfully retrieved data from Databricks")
                    
                    # Display the logs or runs
                    if run_id:
                        st.subheader(f"Logs for Run: {run_id}")
                    else:
                        st.subheader("Recent Runs")
                        
                    # Show the data as JSON
                    st.json(logs)
    
if __name__ == "__main__":
    # Initial logging setup - this will be updated from the UI
    setup_logging(log_level="info")
    logger.info("Application started")
    
    # Run the Streamlit app
    main()
