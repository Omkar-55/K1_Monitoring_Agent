"""
Tools for generating reports about Databricks job issues and resolutions.
"""

from typing import Dict, Any, List
import time

def final_report(history: List[Dict[str, Any]], job_id: str) -> str:
    """
    Generate a final report in Markdown format summarizing the monitoring, diagnosis, and resolution process.
    """
    # Simple report generation for testing
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    
    report = [
        f"# Databricks Job Monitoring Report",
        f"## Summary",
        f"- **Job ID**: {job_id}",
        f"- **Report Generated**: {timestamp}",
        f"- **Total Steps**: {len(history)}",
        f"",
        f"## Timeline",
    ]
    
    for i, step in enumerate(history):
        step_type = step.get("type", "Unknown")
        report.append(f"{i+1}. **{step_type.title()}** - {step.get('timestamp', 'Unknown time')}")
    
    return "\n".join(report)