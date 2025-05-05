"""
Tools for generating reports about Databricks job monitoring.
"""

import time
from typing import Dict, Any, List, Optional
from opentelemetry import trace

# Import the logging configuration
from agent_core.logging_config import get_logger

# Get logger for this module
logger = get_logger(__name__)

# Get tracer for this module
tracer = trace.get_tracer(__name__)

@tracer.start_as_current_span("final_report")
def final_report(report_data: Dict[str, Any]) -> str:
    """
    Generate a final report for a Databricks job monitoring session.
    
    Args:
        report_data: Dictionary containing report information
        
    Returns:
        Formatted Markdown report
    """
    logger.info(f"Generating final report for job {report_data.get('job_id')} with {len(report_data.get('steps', []))} steps")
    
    job_id = report_data.get("job_id", "unknown")
    steps = report_data.get("steps", [])
    issue_type = report_data.get("issue_type")
    fix_attempts = report_data.get("fix_attempts", 0)
    fix_successful = report_data.get("fix_successful", False)
    
    # Extract various steps for the report
    log_step = _find_step(steps, "log_collection")
    diagnosis_steps = _find_all_steps(steps, "diagnosis")
    fix_steps = _find_all_steps(steps, "fix_suggestion")
    application_steps = _find_all_steps(steps, "fix_application")
    verification_steps = _find_all_steps(steps, "verification")
    
    # Format the timestamp
    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    
    # Generate the report in Markdown format
    report = f"# Databricks Job Monitoring Report\n"
    report += f"## Summary\n"
    report += f"- **Job ID**: {job_id}\n"
    report += f"- **Report Generated**: {current_time}\n"
    report += f"- **Total Steps**: {len(steps)}\n"
    report += f"- **Errors Detected**: {1 if issue_type else 0}\n"
    report += f"- **Fixes Attempted**: {fix_attempts}\n"
    report += f"- **Successful Fixes**: {1 if fix_successful else 0}\n"
    report += f"\n"
    
    # Timeline
    report += f"## Timeline\n"
    for i, step in enumerate(steps, 1):
        step_type = step.get("step", "unknown")
        timestamp = step.get("timestamp", 0)
        time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp)) if timestamp else "unknown"
        
        if step_type == "log_collection":
            run_id = step.get("logs_summary", {}).get("run_id", "unknown")
            status = step.get("logs_summary", {}).get("status", "unknown")
            report += f"{i}. **Logs** ({time_str}): Retrieved logs for run {run_id} (status: {status})\n"
        elif step_type == "diagnosis":
            issue = step.get("result", "No issue detected")
            report += f"\n{i}. **Diagnosis** ({time_str}): {issue}\n"
        elif step_type == "fix_suggestion":
            attempt = step.get("attempt", "")
            result = step.get("result", "No suggestion")
            # Format the attempt number if it exists
            attempt_str = f" (Attempt {attempt})" if attempt else ""
            report += f"{i}. **Fix{attempt_str}** ({time_str}): {result}\n"
        elif step_type == "fix_application":
            attempt = step.get("attempt", "")
            result = step.get("result", "No fix applied")
            # Skip unsuccessful attempts to reduce noise
            if "Fix applied: True" not in result:
                continue
            report += f"{i}. **Fix** ({time_str}): {result}\n"
        elif step_type == "verification":
            attempt = step.get("attempt", "")
            result = step.get("result", "Unknown verification result")
            run_id = step.get("new_run_id", "unknown")
            report += f"{i}. **Verification** ({time_str}): Verified run {run_id} with result: {result.split(':')[-1].strip()}\n"
    report += f"\n"
    
    # Diagnoses section
    if diagnosis_steps:
        report += f"## Diagnoses\n"
        for i, step in enumerate(diagnosis_steps, 1):
            result = step.get("result", "No diagnosis")
            issue_parts = result.split(" - ", 1)
            issue_type = issue_parts[0].replace("Diagnosed issue: ", "") if len(issue_parts) > 0 else "Unknown"
            reasoning = issue_parts[1] if len(issue_parts) > 1 else "No reasoning provided"
            
            report += f"### Diagnosis {i}: {issue_type}\n"
            report += f"**Reasoning**: {reasoning}\n\n"
            
            # Extract relevant logs if available
            if log_step:
                logs = log_step.get("logs_summary", {})
                stderr = logs.get("stderr", "")
                if stderr:
                    # Show a snippet of the logs
                    report += f"**Relevant Logs**:\n```\n{stderr[:500]}\n```\n\n"
    
    # Applied fixes section
    if application_steps:
        report += f"## Applied Fixes\n"
        for i, step in enumerate(application_steps, 1):
            result = step.get("result", "")
            attempt = step.get("attempt", "")
            
            # Only include successful applications
            if "Fix applied: True" not in result:
                continue
                
            # Extract fix details
            fix_step = _find_step(fix_steps, "fix_suggestion", attempt)
            fix_result = fix_step.get("result", "") if fix_step else ""
            fix_parts = fix_result.split(" - ", 1)
            fix_type = fix_parts[0].replace("Suggested fix: ", "") if len(fix_parts) > 0 else "Unknown"
            fix_desc = fix_parts[1] if len(fix_parts) > 1 else ""
            
            # Determine success based on verification
            verification_step = _find_step(verification_steps, "verification", attempt)
            success = "success" in verification_step.get("result", "").lower() if verification_step else False
            
            report += f"### Fix {i}: {fix_type} ({'✅ Successful' if success else '❌ Failed'})\n"
            if fix_desc:
                report += f"**Description**: {fix_desc}\n"
                
            # Check for parameters
            if "parameters" in step:
                params = step.get("parameters", {})
                if params:
                    report += "**Details**:\n"
                    for key, value in params.items():
                        report += f"- {key}: {value}\n"
            report += "\n"
    
    # Verifications section
    if verification_steps:
        report += f"## Verifications\n"
        for i, step in enumerate(verification_steps, 1):
            run_id = step.get("new_run_id", "unknown")
            result = step.get("result", "")
            success = "success" in result.lower()
            duration = step.get("duration", 0)
            
            report += f"### Verification {i}: Run {run_id} ({'✅ Successful' if success else '❌ Failed'})\n"
            if duration:
                report += f"**Duration**: {duration} seconds\n\n"
    
    # Recommendations section
    report += f"## Recommendations\n"
    
    if fix_successful:
        report += f"✅ **Job is now running successfully**\n"
        report += f"- Continue monitoring for any future issues\n"
        if issue_type == "dependency_error":
            report += f"- Consider setting up alerts for similar issues\n"
        elif issue_type == "memory_exceeded":
            report += f"- Monitor resource usage to ensure the new memory limits are sufficient\n"
        elif issue_type == "disk_space_exceeded":
            report += f"- Consider data cleanup to prevent future disk space issues\n"
            report += f"- Set up alerts for disk utilization\n"
    else:
        report += f"❌ **Job is still failing**\n"
        report += f"- Consider manual intervention and review\n"
        if fix_attempts >= 3:
            report += f"- The issue may require advanced troubleshooting beyond automated fixes\n"
        if issue_type:
            report += f"- Specifically review the {issue_type} issue\n"
    
    return report

def _find_step(steps: List[Dict[str, Any]], step_type: str, attempt: Any = None) -> Optional[Dict[str, Any]]:
    """
    Find a specific step in the list of steps.
    
    Args:
        steps: List of step dictionaries
        step_type: Type of step to find
        attempt: Optional attempt number to match
        
    Returns:
        The step dictionary or None if not found
    """
    for step in steps:
        if step.get("step") == step_type:
            if attempt is not None:
                if step.get("attempt") == attempt:
                    return step
            else:
                return step
    return None

def _find_all_steps(steps: List[Dict[str, Any]], step_type: str) -> List[Dict[str, Any]]:
    """
    Find all steps of a given type.
    
    Args:
        steps: List of step dictionaries
        step_type: Type of steps to find
        
    Returns:
        List of matching step dictionaries
    """
    return [step for step in steps if step.get("step") == step_type] 