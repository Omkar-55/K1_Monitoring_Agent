"""
K1 Monitoring Agent - Streamlit Application

This application demonstrates logging and tracing capabilities.
"""

import os
import streamlit as st
import asyncio
from datetime import datetime

# Import the agent core package
import agent_core
from agent_core.logging_config import get_logger, setup_logging
from agent_core.core_logic import Agent, AgentInput, AgentResponse

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
    success = agent_core.enable_tracing(
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
    
    # Main content area
    st.header("Logging Demonstration")
    st.write("Click the button below to generate log messages at different levels")
    if st.button("Generate Log Messages"):
        log_demo()
        st.success("Log messages generated! Check the console output and log file.")
    
    # Display path to log file
    from pathlib import Path
    log_path = Path(__file__).parent.parent / "logs" / "agent.log"
    st.info(f"Log file location: {log_path}")
    
    # Agent demonstration
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

if __name__ == "__main__":
    # Initial logging setup - this will be updated from the UI
    setup_logging(log_level="info")
    logger.info("Application started")
    
    # Run the Streamlit app
    main()
