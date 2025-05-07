"""
Pipeline Doctor - Streamlit Application

An intelligent assistant for diagnosing and fixing Databricks pipeline issues.
"""

import os
import streamlit as st
import asyncio
import json
import uuid
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import the agent core package
import src.agent_core
from src.agent_core.logging_config import get_logger, setup_logging
from src.agent_core.dbx_client import DbxClient

# Import the DatabricksMonitoringAgent
from src.agents.databricks_monitoring_agent import (
    DatabricksMonitoringAgent, 
    MonitoringRequest,
    MonitoringResponse,
    monitor_databricks_job
)

# Get logger for this module
logger = get_logger(__name__)

# Set page configuration
st.set_page_config(
    page_title="Pipeline Doctor",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = str(uuid.uuid4())

if "current_monitoring_response" not in st.session_state:
    st.session_state.current_monitoring_response = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "agent_thinking" not in st.session_state:
    st.session_state.agent_thinking = False

# Initialize tracing
def initialize_tracing():
    """Initialize distributed tracing"""
    success = src.agent_core.enable_tracing(
        service_name="pipeline-doctor", 
        rate=1.0
    )
    return success

# Configure logging
def configure_logging(level="info"):
    """Configure logging with the specified level"""
    setup_logging(log_level=level)
    logger.info(f"Logging level set to {level}")
    return True

# Initialize Databricks Monitoring Agent
@st.cache_resource
def get_agent():
    """Get or create the DatabricksMonitoringAgent instance"""
    return DatabricksMonitoringAgent()

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

# Async function to process Databricks monitoring job
async def process_monitoring_job(job_id, run_id=None, approved_fix=None, simulate=False, failure_type=None):
    """Process a monitoring job with the DatabricksMonitoringAgent"""
    try:
        # Create the request
        request = MonitoringRequest(
            job_id=job_id,
            run_id=run_id,
            simulate=simulate,
            simulate_failure_type=failure_type,
            approved_fix=approved_fix,
            conversation_id=st.session_state.conversation_id
        )
        
        # Create or get the agent
    agent = get_agent()
        
        # Process the request
        response = await agent.process_request(request)
        
        return response
    except Exception as e:
        logger.error(f"Error processing monitoring job: {e}", exc_info=True)
        return {"error": str(e)}

# Function to handle user messages
def handle_user_message(user_message):
    """Handle user message and update conversation history"""
    # Add user message to the conversation
    st.session_state.messages.append({"role": "user", "content": user_message})
    
    # Check if the message contains job information
    if "job" in user_message.lower() and any(word in user_message.lower() for word in ["diagnose", "check", "fix", "monitor"]):
        # Try to extract job ID
        import re
        job_id_match = re.search(r"job[_\s-]?id[:\s]+([a-zA-Z0-9_-]+)", user_message, re.IGNORECASE)
        run_id_match = re.search(r"run[_\s-]?id[:\s]+([a-zA-Z0-9_-]+)", user_message, re.IGNORECASE)
        
        job_id = job_id_match.group(1) if job_id_match else None
        run_id = run_id_match.group(1) if run_id_match else None
        
        if job_id:
            # Set simulate to True for development testing
            simulate = True
            simulate_failure = "memory_exceeded"
            
            # Add thinking message
            st.session_state.agent_thinking = True
            st.session_state.messages.append({
                "role": "assistant", 
                "content": "I'm analyzing this Databricks job...",
                "thinking": True
            })
            
            # Run the monitoring job asynchronously
            st.session_state.current_job_id = job_id
            st.session_state.current_run_id = run_id
            
            # Start monitoring task
            st.experimental_rerun()
        else:
            # Add assistant message asking for job ID
            st.session_state.messages.append({
                "role": "assistant", 
                "content": "I need the Job ID to diagnose issues. Can you provide it? For example: 'Check job_id ABC123'"
            })
    else:
        # Add basic assistant response
        st.session_state.messages.append({
            "role": "assistant", 
            "content": "I'm here to help diagnose Databricks pipeline issues. Please provide a Job ID for me to analyze."
        })

# Function to approve a fix
def approve_fix(fix_id):
    """Approve a suggested fix and apply it"""
    if st.session_state.current_monitoring_response:
        # Set flag to apply the fix in the next run
        st.session_state.approved_fix = fix_id
        st.session_state.agent_thinking = True
        
        # Add a message indicating approval
        st.session_state.messages.append({
            "role": "user", 
            "content": f"I approve the suggested fix."
        })
        
        st.session_state.messages.append({
            "role": "assistant", 
            "content": "Applying the approved fix...",
            "thinking": True
        })
        
        # Trigger a rerun to apply the fix
        st.experimental_rerun()

# Function to display the fix suggestion with approval button
def display_fix_suggestion(response):
    """Display the fix suggestion with an approval button"""
    if not response.pending_approval or not response.suggested_fix:
        return
    
    with st.chat_message("assistant"):
        # Create a container for the fix suggestion
        fix = response.suggested_fix
        
        st.markdown("### Suggested Fix")
        st.markdown(fix.get("description", "No description available"))
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Confidence", f"{fix.get('confidence', 0.0):.0%}")
        with col2:
            st.metric("Estimated Time", fix.get("estimated_time", "Unknown"))
        with col3:
            st.metric("Fix Attempt", fix.get("attempt", 1))
        
        if "expected_impact" in fix:
            st.markdown(f"**Expected Impact**: {fix['expected_impact']}")
        
        if "required_permissions" in fix:
            st.markdown(f"**Required Permissions**: {', '.join(fix['required_permissions'])}")
        
        # Display technical details in an expander
        with st.expander("Technical Details"):
            st.json({
                "fix_type": fix.get("fix_type"),
                "parameters": fix.get("parameters", {})
            })
        
        # Show approve button
        fix_id = fix.get("fix_id")
        st.button("Apply Fix", key=f"approve_{fix_id}", on_click=approve_fix, args=(fix_id,))
        st.button("Reject and Ask for Alternative", key=f"reject_{fix_id}")

# Function to display the agent's reasoning
def display_reasoning(response):
    """Display the agent's reasoning in an expander"""
    with st.expander("View Agent Reasoning", expanded=False):
        for step in response.reasoning:
            step_name = step.get("step", "unknown").replace("_", " ").title()
            st.markdown(f"### {step_name}")
            
            # Display result
            if "result" in step:
                st.markdown(f"**Result**: {step['result']}")
            
            # Display details if available
            if "details" in step and step["details"]:
                st.markdown("**Details:**")
                st.markdown(step["details"])
            
            # Show hallucination check if available
            if "hallucination_check" in step:
                hall_check = step["hallucination_check"]
                st.markdown("**Hallucination Check:**")
                if hall_check.get("detected", False):
                    st.error(f"Hallucination detected (score: {hall_check.get('score', 0.0):.2f})")
                    if "reason" in hall_check:
                        st.markdown(hall_check["reason"])
                else:
                    st.success("No hallucination detected")
            
            # Show safety check if available
            if "safety_check" in step:
                safety = step["safety_check"]
                st.markdown("**Safety Check:**")
                if safety.get("issues_detected", False):
                    st.error("Safety issues detected")
                    scores = {k: v for k, v in safety.items() if k.endswith("_score")}
                    for k, v in scores.items():
                        st.markdown(f"- {k}: {v:.2f}")
                else:
                    st.success("No safety issues detected")
            
            st.markdown("---")

# Function to display agent's recommendations
def display_report(response):
    """Display the agent's final report"""
    if response.report:
        with st.chat_message("assistant"):
            st.markdown("## Final Analysis Report")
            st.markdown(response.report)

# Main UI function
def main():
    """Main Streamlit application"""
    st.title("📊 Pipeline Doctor")
    st.markdown("Your intelligent assistant for diagnosing and fixing Databricks pipeline issues.")
    
    # Sidebar for configuration
    st.sidebar.title("Configuration")
    
    # Job configuration section
    st.sidebar.header("Job Settings")
    job_id_input = st.sidebar.text_input("Job ID", key="job_id_input")
    run_id_input = st.sidebar.text_input("Run ID (optional)", key="run_id_input")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        simulate = st.checkbox("Simulate", value=True, key="simulate_checkbox")
    with col2:
        failure_type = st.selectbox(
            "Simulated Failure",
            options=["memory_exceeded", "dependency_error", "disk_space_exceeded"],
            index=0,
            key="failure_type_select",
            disabled=not simulate
        )
    
    if st.sidebar.button("Monitor Job", disabled=not job_id_input):
        st.session_state.current_job_id = job_id_input
        st.session_state.current_run_id = run_id_input if run_id_input else None
        st.session_state.agent_thinking = True
        st.session_state.messages = []
        st.session_state.messages.append({
            "role": "user", 
            "content": f"Please diagnose job_id {job_id_input}{' run_id ' + run_id_input if run_id_input else ''}"
        })
        st.session_state.messages.append({
            "role": "assistant", 
            "content": "I'm analyzing this Databricks job...",
            "thinking": True
        })
        st.experimental_rerun()
    
    # Logging configuration
    st.sidebar.header("Advanced Settings")
    log_level = st.sidebar.selectbox(
        "Log Level", 
        options=["debug", "info", "warning", "error", "critical"],
        index=1,
        key="log_level_select"
    )
    if st.sidebar.button("Apply Logging Configuration"):
        success = configure_logging(log_level)
        if success:
            st.sidebar.success(f"Logging configured at {log_level.upper()} level")
    
    # Tracing configuration
    if st.sidebar.button("Enable Tracing"):
        success = initialize_tracing()
        if success:
            st.sidebar.success("Tracing enabled successfully!")
        else:
            st.sidebar.error("Failed to enable tracing. Check logs for details.")
    
    # Display the chat interface
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message.get("thinking", False):
                with st.status("Analyzing...", state="running", expanded=True) as status:
                    st.markdown(message["content"])
                    st.write("This may take a moment...")
            else:
                st.markdown(message["content"])
    
    # Handle async monitoring job if agent is thinking
    if st.session_state.agent_thinking and hasattr(st.session_state, 'current_job_id'):
        with st.spinner("Processing..."):
            try:
                # Run the monitoring job asynchronously
                job_id = st.session_state.current_job_id
                run_id = getattr(st.session_state, 'current_run_id', None)
                approved_fix = getattr(st.session_state, 'approved_fix', None)
                
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
                
                response = loop.run_until_complete(process_monitoring_job(
                    job_id=job_id,
                    run_id=run_id,
                    approved_fix=approved_fix,
                    simulate=st.session_state.simulate_checkbox,
                    failure_type=st.session_state.failure_type_select
                ))
                
            loop.close()
            
                # Update session state
                st.session_state.current_monitoring_response = response
                st.session_state.agent_thinking = False
                
                # Remove the thinking message
                if st.session_state.messages and st.session_state.messages[-1].get("thinking", False):
                    st.session_state.messages.pop()
                
                # Add diagnostic results
                if getattr(st.session_state, 'approved_fix', None):
                    # This was a fix application
                    if response.fix_successful:
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": f"✅ Fix applied successfully! The issue with job {job_id} has been resolved."
                        })
                    elif response.pending_approval:
                        # Another fix suggestion
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": "The previous fix didn't completely resolve the issue. I have another suggestion."
                        })
                    else:
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": "The fix was applied but didn't fully resolve the issue. Let me analyze what went wrong."
                        })
                    
                    # Clear the approved fix
                    st.session_state.approved_fix = None
                else:
                    # Initial diagnosis
                    if response.issue_detected:
                        issue_message = (
                            f"I've detected an issue with job {job_id}: **{response.issue_type}**\n\n"
                            "I've analyzed the logs and have a suggested fix."
                        )
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": issue_message
                        })
                    else:
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": f"I've analyzed job {job_id} and everything looks good! No issues detected."
                        })
                
                # Force a rerun to display the results
                st.experimental_rerun()
            
        except Exception as e:
                logger.error(f"Error in monitoring job: {e}", exc_info=True)
            st.error(f"Error: {str(e)}")
                
                # Update session state
                st.session_state.agent_thinking = False
                if st.session_state.messages and st.session_state.messages[-1].get("thinking", False):
                    st.session_state.messages.pop()
                
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": f"I encountered an error while analyzing the job: {str(e)}"
                })
                
                st.experimental_rerun()
    
    # Display the fix suggestion if available
    if not st.session_state.agent_thinking and st.session_state.current_monitoring_response:
        response = st.session_state.current_monitoring_response
        
        # Show fix suggestion if pending approval
        if response.pending_approval and response.suggested_fix:
            display_fix_suggestion(response)
        
        # Show reasoning
        display_reasoning(response)
        
        # Show final report if fix was successful
        if response.fix_successful and response.report:
            display_report(response)
    
    # Chat input
    user_message = st.chat_input("Ask me about your Databricks pipelines...")
    if user_message:
        handle_user_message(user_message)
        st.experimental_rerun()

if __name__ == "__main__":
    # Initial logging setup
    setup_logging(log_level="info")
    logger.info("Application started")
    
    # Run the Streamlit app
    main()
