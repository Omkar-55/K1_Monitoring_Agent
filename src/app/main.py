"""
Pipeline Doctor - Streamlit Application

An intelligent assistant for diagnosing and fixing Databricks pipeline issues.
"""

import os
import sys
import streamlit as st
import asyncio
import json
import uuid
from datetime import datetime
from dotenv import load_dotenv
import re
import time

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

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

if "waiting_for_approval" not in st.session_state:
    st.session_state.waiting_for_approval = False

if "current_fix" not in st.session_state:
    st.session_state.current_fix = None

if "fix_approved" not in st.session_state:
    st.session_state.fix_approved = False
    
if "process_monitor" not in st.session_state:
    st.session_state.process_monitor = False

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
    """
    Process a monitoring job and return the result
    """
    try:
        # Create a unique conversation ID if we don't have one
        if "conversation_id" not in st.session_state:
            st.session_state.conversation_id = str(uuid.uuid4())
        
        logger.info(f"Processing monitoring job: {job_id}")
        
        # Handle simulation settings
        if simulate:
            if not failure_type or failure_type == "random":
                logger.info("Using random failure type for simulation")
                failure_type = None  # Let the system choose randomly
            else:
                logger.info(f"Using explicit failure type for simulation: {failure_type}")
                # Ensure failure type is in the correct format
                if failure_type in ["memory", "memory_error", "out_of_memory", "oom"]:
                    failure_type = "memory_exceeded"
                elif failure_type in ["disk", "disk_error", "space", "storage"]:
                    failure_type = "disk_space_exceeded"
                elif failure_type in ["depend", "dependency", "package", "import", "library"]:
                    failure_type = "dependency_error"
        
        # Generate a unique conversation ID if needed
        if not st.session_state.get("conversation_id"):
            st.session_state.conversation_id = str(uuid.uuid4())
            
        # Create the request
        request = MonitoringRequest(
            job_id=job_id,
            run_id=run_id,
            approved_fix=approved_fix,
            simulate=simulate,
            simulate_failure_type=failure_type,
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

# Function to display the agent's reasoning
def display_reasoning(response):
    """Display the agent's reasoning with animation and save to session state for later viewing"""
    # Store reasoning in session state for later retrieval
    if "reasoning_history" not in st.session_state:
        st.session_state.reasoning_history = {}
    
    # Generate a timestamp key
    import time
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    
    # Store this reasoning for later viewing
    if hasattr(response, 'reasoning') and response.reasoning:
        # Add timestamps to reasoning steps if not present
        timestamped_reasoning = []
        for step in response.reasoning:
            step_with_time = step.copy()
            if 'timestamp' not in step_with_time:
                step_with_time['timestamp'] = time.strftime("%Y-%m-%d %H:%M:%S")
            timestamped_reasoning.append(step_with_time)
        
        reasoning_data = {
            "timestamp": timestamp,
            "job_id": getattr(response, 'job_id', 'unknown'),
            "issue_type": getattr(response, 'issue_type', 'unknown'),
            "reasoning": timestamped_reasoning,
            "hallucination": getattr(response, 'hallucination_detected', False),
            "hallucination_details": getattr(response, 'hallucination_details', None),
            "safety": getattr(response, 'safety_issues', False),
            "safety_issue_details": getattr(response, 'safety_issue_details', None),
            "confidence": getattr(response, 'confidence', None),
            "metadata": {
                "fix_successful": getattr(response, 'fix_successful', None),
                "fix_attempts": getattr(response, 'fix_attempts', None),
                "pending_approval": getattr(response, 'pending_approval', None)
            }
        }
        st.session_state.reasoning_history[timestamp] = reasoning_data
    
    # Display the analysis header
    st.markdown("## 🧠 Agent Analysis")
    
    # Display hallucination and safety check results
    with st.container():
        col1, col2 = st.columns(2)
        
        with col1:
            hallucination_detected = getattr(response, 'hallucination_detected', False)
            hallucination_status = "🔍 No Hallucination Detected" if not hallucination_detected else "⚠️ Potential Hallucination Detected"
            status_color = "success" if not hallucination_detected else "warning"
            st.markdown(f"**{hallucination_status}**")
            if hasattr(response, 'hallucination_details') and response.hallucination_details:
                st.warning(response.hallucination_details)
            else:
                st.success("Response validated for factual accuracy")
        
        with col2:
            safety_issues = getattr(response, 'safety_issues', False)
            safety_status = "✅ No Safety Issues" if not safety_issues else "⚠️ Safety Issues Detected"
            st.markdown(f"**{safety_status}**")
            if hasattr(response, 'safety_issue_details') and response.safety_issue_details:
                st.error(response.safety_issue_details)
            else:
                st.success("Response passed safety checks")
    
    # Display reasoning steps with animation for better visibility
    if hasattr(response, 'reasoning') and response.reasoning:
        st.markdown("### 🔍 Analysis Steps")
        
        # Create a container for reasoning steps
        steps_container = st.container()
        
        # Show a progress bar to slow things down and give user time to see steps
        progress_bar = st.progress(0)
        
        # Display steps with animation
        for i, step in enumerate(response.reasoning):
            # Update progress
            progress = (i + 1) / len(response.reasoning)
            progress_bar.progress(progress)
            
            # Slight delay for better visibility (not too much to block the UI)
            time.sleep(0.5)
            
            with steps_container:
                step_name = step.get("step", "unknown").replace("_", " ").title()
                
                # Create a container for each step
                with st.container():
                    st.markdown(f"**{step_name}**")
                    
                    # Display details if available
                    if "details" in step and step["details"]:
                        st.markdown(step["details"])
                    
                    # Display result in a different color/style
                    if "result" in step:
                        st.success(f"Result: {step['result']}")
                    
                    # Add a separator between steps
                    st.markdown("---")
                    
        # Complete the progress bar
        progress_bar.progress(1.0)
        st.success("Analysis complete! You can review these steps later in the 'Reasoning History' section.")
    else:
        st.warning("No detailed reasoning steps available for this analysis.")

# Function to display agent's recommendations
def display_report(response):
    """Display the agent's final report"""
    if response.report:
        with st.container():
            st.markdown("## Final Analysis Report")
            
            # Display any hallucination or safety warnings if present
            if response.hallucination_detected:
                with st.warning("⚠️ **Potential Hallucination Warning**"):
                    st.markdown("""
                        The response may contain potential hallucinations. 
                        Please verify critical information before proceeding.
                    """)
                    if response.hallucination_details:
                        st.markdown(f"**Details:** {response.hallucination_details}")
            
            if response.safety_issues:
                with st.error("⚠️ **Safety Issues Detected**"):
                    st.markdown("""
                        Safety concerns were identified in the response. 
                        Please review carefully before taking any action.
                    """)
                    if response.safety_issue_details:
                        st.markdown(f"**Details:**\n{response.safety_issue_details}")
            
            # Let the report display the confidence information
            # The confidence is now included directly in the report
            st.markdown(response.report)
    else:
        st.warning("No final report available")

# Main UI function
def main():
    """Main Streamlit application"""
    st.title("📊 Pipeline Doctor")
    st.markdown("Your intelligent assistant for diagnosing and fixing Databricks pipeline issues.")
    
    # Initialize active_tab if not set
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = "Main Interface"
    
    # Add tabs for main interface and reasoning history
    tab_labels = ["Main Interface", "Reasoning History"]
    selected_index = tab_labels.index(st.session_state.active_tab) if st.session_state.active_tab in tab_labels else 0
    main_tab, history_tab = st.tabs(tab_labels)
    
    # Store the current active tab to avoid infinite rerun loops
    current_tab = tab_labels[selected_index]
    st.session_state.active_tab = current_tab
    
    with main_tab:
        # Debug section (conditional)
        if st.session_state.get("debug_mode", False):
            with st.expander("Debug Info", expanded=False):
                if st.session_state.current_monitoring_response:
                    response = st.session_state.current_monitoring_response
                    st.write("Current Monitoring Response:")
                    st.write(f"- fix_successful: {response.fix_successful}")
                    st.write(f"- pending_approval: {response.pending_approval}")
                    st.write(f"- has report: {response.report is not None}")
                    st.write(f"- issue_type: {response.issue_type}")
                    st.write(f"- fix_attempts: {response.fix_attempts}")
                    st.write(f"- waiting_for_approval: {st.session_state.waiting_for_approval}")
                    st.write(f"- fix_approved: {st.session_state.fix_approved}")
                
                # Add live log viewer
                st.markdown("### Recent Logs")
                try:
                    with open("logs/agent.log", "r") as f:
                        log_lines = f.readlines()
                        # Show the most recent 20 log lines
                        log_text = "".join(log_lines[-20:])
                        st.text_area("Log Entries", log_text, height=300)
                except Exception as e:
                    st.error(f"Error reading logs: {e}")
        
        # Initialize sidebar
        initialize_sidebar()
        
        # Initialize chat state
        initialize_chat()
        
        # Display the chat interface
        display_chat_history()
        
        # Show approval buttons if waiting for approval
        if st.session_state.waiting_for_approval and st.session_state.current_fix:
            show_approval_buttons()
            
        # Show final report if available and fix was successful
        if st.session_state.get("show_report", False) and st.session_state.current_monitoring_response and st.session_state.current_monitoring_response.report:
            display_report(st.session_state.current_monitoring_response)
            st.session_state.show_report = False
    
    # Display reasoning history in a separate tab
    with history_tab:
        st.header("📝 Reasoning History")
        
        if "reasoning_history" in st.session_state and st.session_state.reasoning_history:
            # Get available sessions in reverse chronological order
            history_items = list(st.session_state.reasoning_history.items())
            history_items.sort(reverse=True)  # Sort by timestamp (most recent first)
            
            # Create a selectbox for users to choose which analysis to view
            timestamps = [f"{ts} - Job: {data['job_id']} - Issue: {data['issue_type']}" 
                         for ts, data in history_items]
            
            selected_entry = st.selectbox(
                "Select analysis to review:",
                options=timestamps,
                index=0
            )
            
            if selected_entry:
                # Extract timestamp from the selected entry
                timestamp = selected_entry.split(" - ")[0]
                
                # Get the data for this timestamp
                if timestamp in st.session_state.reasoning_history:
                    data = st.session_state.reasoning_history[timestamp]
                    
                    # Display the reasoning data
                    st.subheader(f"Analysis for Job: {data['job_id']}")
                    st.markdown(f"**Issue Type:** {data['issue_type']}")
                    st.markdown(f"**Analysis Time:** {data['timestamp']}")
                    
                    # Hallucination and safety results in expandable sections
                    with st.expander("🔍 Hallucination Analysis", expanded=True):
                        st.markdown("### Hallucination Check Results")
                        if data['hallucination']:
                            st.warning("⚠️ **Potential Hallucination Detected**")
                            if 'hallucination_details' in data:
                                st.markdown(f"**Details:** {data['hallucination_details']}")
                            st.markdown("""
                            **Steps Taken:**
                            1. Analyzed response content for factual consistency
                            2. Checked against known patterns and data
                            3. Validated technical recommendations
                            """)
                        else:
                            st.success("✅ **No Hallucination Detected**")
                            st.markdown("""
                            **Validation Steps:**
                            1. Verified response against Databricks documentation
                            2. Confirmed error patterns match known issues
                            3. Validated technical accuracy of recommendations
                            """)
                    
                    with st.expander("🛡️ Safety Analysis", expanded=True):
                        st.markdown("### Safety Check Results")
                        if data['safety']:
                            st.error("⚠️ **Safety Issues Found**")
                            if 'safety_issue_details' in data:
                                st.markdown(f"**Details:** {data['safety_issue_details']}")
                            st.markdown("""
                            **Safety Concerns:**
                            1. Reviewed potential impact on production systems
                            2. Analyzed data security implications
                            3. Checked for destructive operations
                            """)
                        else:
                            st.success("✅ **No Safety Issues**")
                            st.markdown("""
                            **Safety Verification Steps:**
                            1. Confirmed changes are non-destructive
                            2. Verified data security compliance
                            3. Checked resource access boundaries
                            """)
                    
                    # Display reasoning steps
                    st.markdown("### 🧠 Detailed Analysis Steps")
                    
                    for i, step in enumerate(data['reasoning']):
                        step_name = step.get('step', 'unknown').replace('_', ' ').title()
                        with st.expander(f"Step {i+1}: {step_name}", expanded=False):
                            # Add timing information if available
                            if 'timestamp' in step:
                                st.markdown(f"**Time:** {step['timestamp']}")
                                
                            # Display step details with improved formatting
                            if 'details' in step:
                                st.markdown("**Analysis:**")
                                st.markdown(step['details'])
                            
                            # Display result with status indicator
                            if 'result' in step:
                                result = step['result']
                                if any(success_term in result.lower() for success_term in ['success', 'passed', 'resolved']):
                                    st.success(f"**Result:** {result}")
                                elif any(warning_term in result.lower() for warning_term in ['warning', 'caution', 'attention']):
                                    st.warning(f"**Result:** {result}")
                                elif any(error_term in result.lower() for error_term in ['error', 'failed', 'issue']):
                                    st.error(f"**Result:** {result}")
                                else:
                                    st.info(f"**Result:** {result}")
                            
                            # Display any additional metadata
                            if 'metadata' in step:
                                with st.expander("Additional Details"):
                                    st.json(step['metadata'])
            else:
                st.info("Select an analysis from the dropdown to view its details")
        else:
            st.info("No reasoning history available yet. Run some analyses first.")
    
    # Process messages (wait for user input or handle previous requests)
    process_messages()

# Function to display agent's chat responses
def display_chat_history():
    """Display the chat history"""
    # Display all messages
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Add a button to view reasoning after AI responses that mention analysis steps
            if (message["role"] == "assistant" and 
                "Click the button below to see detailed analysis steps" in message["content"]):
                if st.button("🧠 View Analysis Steps", key=f"view_steps_{i}", type="primary", 
                            on_click=lambda: setattr(st.session_state, 'active_tab', 'Reasoning History')):
                    st.rerun()

# Function to process new messages
def process_messages():
    """Process new messages and agent responses"""
    # Check if we need to process a monitoring request from sidebar
    if st.session_state.get("process_monitor", False):
        job_id = st.session_state.get("job_id_input", "")
        if job_id:
            logger.info(f"Processing monitoring request from sidebar for job ID: {job_id}")
            st.session_state.process_monitor = False
            asyncio.run(handle_message(f"monitor {job_id}"))
            return
    
    # If we just approved a fix, process it
    if st.session_state.get("fix_approved", False) and st.session_state.get("approved_fix", None):
        logger.info("Processing approved fix")
        st.session_state.fix_approved = False
        asyncio.run(process_approved_fix())
        return
    
    # Handle user input if not waiting for approval
    if not st.session_state.waiting_for_approval:
        message = st.chat_input("How can I help diagnose your Databricks job issues?")
        
        if message:
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": message})
            
            # Process user message
            asyncio.run(handle_message(message))

# Function to process an approved fix
async def process_approved_fix():
    """Process a fix that was approved by the user"""
    try:
        # Get the current response and approved fix
        response = st.session_state.current_monitoring_response
        fix = st.session_state.approved_fix
        
        # Add a processing message
        with st.chat_message("assistant"):
            st.markdown("Applying the approved fix...")
        
        # Process the fix
        new_response = await process_monitoring_job(
            job_id=response.job_id,
            approved_fix=fix.get("fix_id"),
            simulate=st.session_state.get("simulate", True),
            failure_type=st.session_state.get("failure_type")
        )
        
        # Update the response
        st.session_state.current_monitoring_response = new_response
        
        # Check for success in multiple ways (handle both response formats)
        is_successful = False
        
        # Check various potential success indicators
        if hasattr(new_response, 'fix_successful') and new_response.fix_successful:
            is_successful = True
            logger.info("Fix marked successful via fix_successful property")
        
        # Also check if there's a status field with 'success' value (from fix_tools.py)
        if hasattr(new_response, 'status') and new_response.status == 'success':
            is_successful = True
            logger.info("Fix marked successful via status='success' property")
        
        # Check dictionary-style response (tools might return plain dict)
        if isinstance(new_response, dict) and new_response.get('status') == 'success':
            is_successful = True
            logger.info("Fix marked successful via dict status='success'")
        
        # For BaseModel responses, check internal __dict__
        if hasattr(new_response, '__dict__') and getattr(new_response, '__dict__', {}).get('status') == 'success':
            is_successful = True
            logger.info("Fix marked successful via __dict__.status='success'")
        
        # Log the full response structure in debug mode
        logger.debug(f"Fix response structure: {new_response}")
        
        # Show the appropriate message based on success status
        if is_successful:
            st.session_state.messages.append({
                "role": "assistant", 
                "content": f"✅ Fix applied successfully! The issue with job {response.job_id} has been resolved."
            })
            
            # Set flag to show report
            st.session_state.show_report = True
            
        else:
            st.session_state.messages.append({
                "role": "assistant", 
                "content": f"❌ The fix was not fully successful. Let me analyze and suggest another approach."
            })
        
        # Reset fix-related state
        st.session_state.approved_fix = None
        
    except Exception as e:
        logger.error(f"Error processing approved fix: {e}", exc_info=True)
        st.session_state.messages.append({
            "role": "assistant", 
            "content": f"Error processing the fix: {str(e)}"
        })
    
    # Force a refresh
    st.rerun()

def show_approval_buttons():
    """Show approval buttons for the current fix suggestion"""
    if not st.session_state.get("current_fix"):
        return
        
    fix = st.session_state.current_fix
    
    fix_id = fix.get("fix_id", "unknown")
    fix_description = fix.get("description", "No description available")
    fix_params = fix.get("parameters", {})
    
    with st.container():
        # Create a well-structured fix suggestion display
        st.markdown("### 🔧 Suggested Fix")
        st.markdown(fix_description)
        
        # Display additional info in a clean format
        st.info(f"This fix will address the {st.session_state.current_monitoring_response.issue_type.replace('_', ' ')} issue detected earlier.")
        
        # Display parameters in an expandable container
        with st.expander("Technical Details"):
            st.json(fix_params)
        
        # Add a clear separator
        st.markdown("---")
        st.markdown("**Would you like to apply this fix?**")
        
        # Create a container for the approve/reject buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Apply Fix", key="approve", type="primary"):
                logger.info(f"User approved fix {fix_id}")
                st.session_state.messages.append({
                    "role": "user", 
                    "content": "I approve this fix."
                })
                st.session_state.waiting_for_approval = False
                st.session_state.approved_fix = fix
                st.session_state.fix_approved = True
                st.session_state.current_fix = None
                st.rerun()
        
        with col2:
            if st.button("❌ Reject Fix", key="reject"):
                logger.info(f"User rejected fix {fix_id}")
                st.session_state.messages.append({
                    "role": "user", 
                    "content": "I reject this fix. Please suggest another approach."
                })
                st.session_state.waiting_for_approval = False
                st.session_state.current_fix = None
                st.rerun()

# Handle user messages
async def handle_message(message):
    """Handle user message and process with the monitoring agent"""
    try:
        # Check if message is a monitoring request
        monitor_match = re.search(r'(?i)monitor\s+(\S+)', message)
        job_id = None
        
        if monitor_match:
            job_id = monitor_match.group(1)
            logger.info(f"Extracted job ID from message: {job_id}")
        else:
            # Try other patterns
            job_id_match = re.search(r'job[_-]?(\d+)', message, re.IGNORECASE)
            if job_id_match:
                job_id = f"job_{job_id_match.group(1)}"
            
            # Look for phrases like "check job 12345"
            check_job_match = re.search(r'(?:check|diagnose|analyze)\s+(?:job|run)?\s*(\d+)', message, re.IGNORECASE)
            if check_job_match:
                job_id = f"job_{check_job_match.group(1)}"
        
        # Check for approval message
        is_approval = re.search(r'(?i)approve|yes|accept|apply', message) and not re.search(r'(?i)don\'t|not|no', message)
        is_rejection = re.search(r'(?i)reject|no|decline|don\'t', message) 
        
        # Handle approval or rejection via chat
        if is_approval and st.session_state.waiting_for_approval and st.session_state.current_fix:
            fix = st.session_state.current_fix
            
            logger.info(f"User approved fix via chat: {fix.get('fix_id')}")
            st.session_state.waiting_for_approval = False
            st.session_state.approved_fix = fix
            st.session_state.fix_approved = True
            st.session_state.current_fix = None
            
            # Will process in the next cycle
            st.rerun()
            return
            
        elif is_rejection and st.session_state.waiting_for_approval and st.session_state.current_fix:
            fix = st.session_state.current_fix
            
            logger.info(f"User rejected fix via chat: {fix.get('fix_id')}")
            st.session_state.waiting_for_approval = False
            st.session_state.current_fix = None
            
            # Add rejection message
            st.session_state.messages.append({
                "role": "assistant", 
                "content": "I'll look for another solution approach. Please let me know if you have specific concerns about the suggested fix."
            })
            
            st.rerun()
            return
        
        if job_id or re.search(r'(?i)simulate|test|try', message):
            # Initialize simulate and failure_type from sidebar settings first
            sidebar_simulate = st.session_state.get("simulate", False)
            sidebar_failure_type = st.session_state.get("failure_type", "random") if sidebar_simulate else None

            # Default to sidebar settings
            simulate = sidebar_simulate
            failure_type = sidebar_failure_type
            
            # Check for simulation keywords and specific failure types in the chat message
            # These will override sidebar settings if present
            chat_requests_simulation = bool(re.search(r'(?i)simulate|test|try', message))
            
            if chat_requests_simulation:
                simulate = True # Ensure simulation is on if chat requests it
                # Try to parse a specific failure type from the chat message
                if re.search(r'(?i)memory|out\\s+of\\s+memory|oom', message):
                    failure_type = "memory_exceeded"
                elif re.search(r'(?i)disk|space|storage', message):
                    failure_type = "disk_space_exceeded"
                elif re.search(r'(?i)depend|package|import|library', message):
                    failure_type = "dependency_error"
                elif failure_type is None: # If chat asks to simulate but gives no type, and sidebar had no type
                    failure_type = "random"
                # If chat says "simulate" but gives no type, and sidebar *did* have a type,
                # we keep the sidebar_failure_type (already assigned to failure_type)
            
            # If job ID found, start monitoring process
            if job_id:
                # Display thinking message
                with st.chat_message("assistant"):
                    st.markdown(f"Starting monitoring for job ID: `{job_id}`...")
                    if simulate: # Now 'simulate' reflects combined logic
                        st.markdown(f"*Simulation mode enabled with failure type: {failure_type or 'random'}*")
                
                # Reset for new monitoring session
                st.session_state.current_monitoring_response = None
                st.session_state.current_fix = None
                st.session_state.approved_fix = None
                st.session_state.waiting_for_approval = False
                st.session_state.fix_approved = False
                
                # Process the monitoring request
                response = await process_monitoring_job(
                    job_id=job_id,
                    simulate=simulate,
                    failure_type=failure_type
                )
                
                # Store the response in session state
                st.session_state.current_monitoring_response = response
                
                # Create a container for reasoning button that will stay visible
                step_button_container = st.container()
                
                # Always display agent reasoning steps with animation
                st.write("") # Add space
                display_reasoning(response)
                
                # Add a button in the container to view reasoning after it disappears
                with step_button_container:
                    st.button("🧠 View Analysis Steps Again", 
                             key=f"view_again_{int(time.time())}", 
                             on_click=lambda: setattr(st.session_state, 'active_tab', 'Reasoning History'),
                             type="primary")
                
                # If issue detected, show diagnosis and fix suggestions
                if response.issue_detected:
                    issue_message = f"I've analyzed job `{job_id}` and found a {response.issue_type.replace('_', ' ')} issue."
                    
                    # Add evidence and details to the message
                    evidence_text = ""
                    if hasattr(response, "evidence") and response.evidence:
                        evidence_text = "\n\n**Evidence:**\n" + "\n".join([f"- {item}" for item in response.evidence[:3]])
                    
                    details_text = ""
                    for step in response.reasoning:
                        if step.get("step") == "diagnosis" and step.get("details"):
                            details_text = f"\n\n**Details:**\n{step.get('details')}"
                            break
                    
                    # Add full message with context
                    diagnosis_message = f"{issue_message}{evidence_text}{details_text}"
                    
                    # Save timestamp for reasoning
                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                    
                    # Add a button key to session state
                    button_key = f"reasoning_btn_{timestamp}"
                    st.session_state[button_key] = False
                    
                    # Check if there's a suggested fix waiting for approval
                    if response.suggested_fix:
                        # Store the fix for approval
                        st.session_state.waiting_for_approval = True
                        st.session_state.current_fix = response.suggested_fix
                        
                        # Add explanation message about the issue first
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": f"{diagnosis_message}\n\n**Important:** Click the button below to see detailed analysis steps."
                        })
                        
                        logger.info("Waiting for user approval of fix...")
                    else:
                        # Just show the diagnosis
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": f"{diagnosis_message}\n\n**Important:** Click the button below to see detailed analysis steps."
                        })
                else:
                    # No issues detected
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": f"I've analyzed job `{job_id}` and didn't find any issues. The job appears to be running normally."
                    })
                
                # Force a rerun to display the results
                st.rerun()
                
            else:
                # No job ID provided, but simulate requested
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": "To begin monitoring, please provide a Databricks job ID. For example: 'Monitor job_123456' or 'Check job 789012'"
                })
                st.rerun()
        
        else:
            # General conversation
            if st.session_state.current_monitoring_response and st.session_state.current_monitoring_response.issue_type:
                response = st.session_state.current_monitoring_response
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": f"I'm currently analyzing a {response.issue_type.replace('_', ' ')} issue with your Databricks job. What else would you like to know?"
                })
            else:
                # Handle general questions or prompts
                if "help" in message.lower() or "what can you do" in message.lower():
                    help_message = "I can help diagnose and fix issues with your Databricks jobs. Please provide a job ID to start. For example: 'Check job_123456' or 'Monitor job 789012'"
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": help_message
                    })
                else:
                    generic_message = "To begin monitoring a Databricks job, please provide a job ID. For example: 'Monitor job_123456' or 'Check job 789012'. You can also use the 'Monitor Job' button in the sidebar."
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": generic_message
                    })
            # Force a rerun to update UI
            st.rerun()
        
    except Exception as e:
        logger.error(f"Error handling message: {e}", exc_info=True)
        st.session_state.messages.append({
            "role": "assistant", 
            "content": f"I encountered an error: {str(e)}. Please try again or contact support if the issue persists."
        })
            
        # Reset waiting status if error occurs
        st.session_state.waiting_for_approval = False
        st.rerun()

# Initialize sidebar configuration
def initialize_sidebar():
    """Initialize the sidebar configuration"""
    st.sidebar.title("Configuration")
    
    # Job configuration section
    st.sidebar.header("Job Settings")
    job_id_input = st.sidebar.text_input("Job ID", value="test_job_123", key="job_id_input")
    
    # Store simulate setting in session state
    st.session_state.simulate = st.sidebar.checkbox("Simulation Mode", value=True)
    
    # Only show failure type if simulate is enabled
    if st.session_state.simulate:
        # Ensure the option names exactly match the enum values in diagnostic_tools.py
        st.session_state.failure_type = st.sidebar.selectbox(
            "Simulated Failure Type",
            options=["memory_exceeded", "disk_space_exceeded", "dependency_error", "random"],
            index=0,
            help="Select the type of failure to simulate, or 'random' to get different failures each time"
        )
        
        # Display current simulation settings with clear labels
        simulation_display = {
            "memory_exceeded": "Memory Exceeded (Java heap space, GC overhead)",
            "disk_space_exceeded": "Disk Space Exceeded (No space left on device)",
            "dependency_error": "Dependency Error (Missing libraries, ClassNotFoundException)",
            "random": "Random Failure (choose randomly each time)"
        }
        
        st.sidebar.info(f"Current simulation: {simulation_display.get(st.session_state.failure_type, st.session_state.failure_type)}")
    
    # Add a monitoring button
    if st.sidebar.button("Monitor Job", key="monitor_button"):
        if job_id_input:
            # Add the monitoring message to chat
            st.session_state.messages.append({
                "role": "user", 
                "content": f"monitor {job_id_input}"
            })
            # Set flag to process the monitoring request
            st.session_state.process_monitor = True
            st.rerun()
    
    # Reasoning History section
    st.sidebar.header("Analysis History")
    
    # Count number of analyses in history
    history_count = 0
    if "reasoning_history" in st.session_state:
        history_count = len(st.session_state.reasoning_history)
    
    # Add a button to view reasoning history
    if st.sidebar.button(f"View Reasoning History ({history_count} analyses)", key="view_history_button"):
        # Set a session state flag to switch to the history tab
        st.session_state.active_tab = "Reasoning History"
        st.rerun()
    
    # Debug mode toggle
    st.sidebar.header("Developer Options")
    st.session_state.debug_mode = st.sidebar.checkbox("Debug Mode", value=False)
    
    # Logging configuration
    log_level = st.sidebar.selectbox(
        "Log Level", 
        options=["debug", "info", "warning", "error", "critical"],
        index=1
    )
    if st.sidebar.button("Apply Logging Configuration"):
        success = configure_logging(log_level)
        if success:
            st.sidebar.success(f"Logging configured at {log_level.upper()} level")

# Initialize chat state
def initialize_chat():
    """Initialize the chat state if not already done"""
    # Initialize the session state for messages
    if "messages" not in st.session_state:
        st.session_state.messages = []
        # Add welcome message
        st.session_state.messages.append({
            "role": "assistant",
            "content": "👋 I'm Pipeline Doctor, your AI assistant for diagnosing and fixing Databricks pipeline issues. How can I help you today?"
        })

# Function to show reasoning in a popup dialog
def show_reasoning_popup():
    """Display reasoning steps in a popup dialog"""
    if "current_monitoring_response" in st.session_state and st.session_state.current_monitoring_response:
        response = st.session_state.current_monitoring_response
        
        if hasattr(response, 'reasoning') and response.reasoning:
            # Create an expander for the reasoning steps
            with st.expander("🧠 Reasoning Steps", expanded=True):
                # Display the reasoning steps
                for i, step in enumerate(response.reasoning):
                    step_name = step.get("step", "unknown").replace("_", " ").title()
                    st.markdown(f"### Step {i+1}: {step_name}")
                    
                    # Display details if available
                    if "details" in step and step["details"]:
                        st.markdown(step["details"])
                    
                    # Display result in a different color/style
                    if "result" in step:
                        st.success(f"Result: {step['result']}")
                    
                    # Add a separator between steps (except for the last one)
                    if i < len(response.reasoning) - 1:
                        st.markdown("---")
        else:
            st.warning("No reasoning steps available for the current analysis.")
    else:
        st.warning("No current analysis available. Please run a diagnosis first.")

if __name__ == "__main__":
    # Initial logging setup
    setup_logging(log_level="info")
    logger.info("Application started")
    
    # Run the Streamlit app
    main()
