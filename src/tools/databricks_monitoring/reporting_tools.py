"""
Tools for generating reports about Databricks job monitoring.
"""

import time
from typing import Dict, Any, List, Optional

# Import the logging configuration
from src.agent_core.logging_config import get_logger

# Get logger for this module
logger = get_logger(__name__)

def final_report(
    issue_type: Optional[str], 
    steps: List[Dict[str, Any]], 
    fix_successful: Optional[bool] = None
) -> str:
    """
    Generate a final report for the monitoring run.
    
    Args:
        issue_type: The type of issue that was detected, if any
        steps: The reasoning steps taken during monitoring
        fix_successful: Whether the fix was successful, if applicable
        
    Returns:
        A formatted report string
    """
    logger.info("Generating final report")
    
    # Start building the report
    report = "# Databricks Monitoring Report\n\n"
    
    # Add timestamp
    report += f"**Generated at:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    # Add issue summary
    report += "## Issue Summary\n\n"
    
    if issue_type:
        report += f"**Issue Type:** {issue_type}\n\n"
    else:
        report += "No issues detected.\n\n"
    
    # Add resolution status
    report += "## Resolution Status\n\n"
    
    if fix_successful is None:
        report += "No fix was attempted.\n\n"
    elif fix_successful:
        report += "✅ **Fix was successful!**\n\n"
    else:
        report += "❌ **Fix was unsuccessful.**\n\n"
    
    # Add detailed steps
    report += "## Monitoring Steps\n\n"
    
    for i, step in enumerate(steps):
        step_type = step.get("step", "unknown")
        timestamp = step.get("timestamp", 0)
        formatted_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp))
        result = step.get("result", "No result recorded")
        
        report += f"### Step {i+1}: {step_type.replace('_', ' ').title()}\n\n"
        report += f"**Time:** {formatted_time}\n\n"
        
        # Add attempt number if present
        if "attempt" in step:
            report += f"**Attempt:** {step['attempt']}\n\n"
        
        report += f"**Result:** {result}\n\n"
        
        # Add details if present
        if "details" in step and step["details"]:
            if isinstance(step["details"], str):
                report += f"**Details:**\n\n```\n{step['details']}\n```\n\n"
            else:
                report += f"**Details:**\n\n```\n{step['details']}\n```\n\n"
        
        # Add hallucination check if present
        if "hallucination_check" in step:
            hallucination = step["hallucination_check"]
            if hallucination.get("detected", False):
                report += f"⚠️ **Hallucination Check:** Detected with score {hallucination.get('score', 'N/A')}\n\n"
                report += f"**Reason:** {hallucination.get('reason', 'Unknown')}\n\n"
            else:
                report += "✅ **Hallucination Check:** No issues detected\n\n"
        
        # Add safety check if present
        if "safety_check" in step:
            safety = step["safety_check"]
            if safety.get("issues_detected", False):
                report += f"⚠️ **Safety Check:** Issues detected\n\n"
                if "harmful_score" in safety:
                    report += f"- Harmful content score: {safety['harmful_score']}\n"
                if "harassment_score" in safety:
                    report += f"- Harassment content score: {safety['harassment_score']}\n"
                if "hate_score" in safety:
                    report += f"- Hate content score: {safety['hate_score']}\n"
                if "sexual_score" in safety:
                    report += f"- Sexual content score: {safety['sexual_score']}\n"
                if "self_harm_score" in safety:
                    report += f"- Self-harm content score: {safety['self_harm_score']}\n"
                report += "\n"
    else:
                report += "✅ **Safety Check:** No issues detected\n\n"
        
        # Add guardrail trigger information if present
        if "guardrail_triggered" in step:
            guardrail = step["guardrail_triggered"]
            report += f"⚠️ **Guardrail Triggered:** {guardrail.get('message', 'Unknown reason')}\n\n"
            report += f"**Step:** {guardrail.get('step', 'Unknown')}\n\n"
        
        # For diagnosis step, include logs summary if present
        if step_type == "log_collection" and "logs_summary" in step:
            logs_summary = step["logs_summary"]
            report += "**Logs Summary:**\n\n"
            for key, value in logs_summary.items():
                report += f"- {key}: {value}\n"
            report += "\n"
        
        # For fix suggestion, include parameters if present
        if step_type == "fix_suggestion" and "parameters" in step:
            params = step["parameters"]
            report += "**Fix Parameters:**\n\n"
            for key, value in params.items():
                report += f"- {key}: {value}\n"
            report += "\n"
    
    # Add recommendations
    report += "## Recommendations\n\n"
    
    if issue_type == "memory_exceeded":
        report += "- Consider optimizing the job's memory usage by refining data processing logic\n"
        report += "- Monitor memory usage trends to proactively adjust cluster settings\n"
        report += "- Implement checkpointing for large operations to minimize memory requirements\n"
    elif issue_type == "disk_space_exceeded":
        report += "- Implement regular data cleanup in your workflows\n"
        report += "- Consider using more efficient data formats (Parquet, Delta)\n"
        report += "- Monitor disk usage patterns and set up alerts\n"
    elif issue_type == "dependency_error":
        report += "- Document all required dependencies in cluster configuration\n"
        report += "- Consider using an init script for consistent dependency installation\n"
        report += "- Implement dependency version pinning to prevent compatibility issues\n"
    else:
        report += "- Regular monitoring of job performance metrics\n"
        report += "- Review logs periodically for potential issues\n"
        report += "- Consider implementing automated alerts for failures\n"
    
    # Add footer
    report += "\n---\n"
    report += "*This report was automatically generated by the Databricks Monitoring Agent.*\n"
    
    logger.info("Final report generated successfully")
    return report 