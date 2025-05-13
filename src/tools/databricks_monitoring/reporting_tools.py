"""
Tools for generating reports about Databricks job monitoring.
"""

import time
from datetime import datetime
from typing import Dict, Any, List, Optional, Union

# Import the logging configuration
from src.agent_core.logging_config import get_logger

# Get logger for this module
logger = get_logger(__name__)

def final_report(issue_type: Union[str, Dict[str, Any]], reasoning: List[Dict[str, Any]] = None, fix_successful: bool = False, job_id: str = "unknown", confidence: float = 0.0) -> str:
    """
    Generate a final report for the user after monitoring and fix application.
    
    Args:
        issue_type: The type of issue that was identified, can be string or dict with fix details
        reasoning: List of reasoning steps taken
        fix_successful: Whether the fix was successful
        job_id: The ID of the job that was analyzed
        confidence: Confidence level of the diagnosis (0.0 to 1.0)
    
    Returns:
        A markdown-formatted report string
    """
    logger.info(f"Generating final report with confidence: {confidence}")
    
    # Get timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Extract issue_type and parameters if it's a dictionary
    parameters = {}
    resolution_details = ""
    
    if isinstance(issue_type, dict):
        # It's actually fix_details
        fix_details = issue_type
        parameters = fix_details.get("parameters", {})
        issue_type = fix_details.get("fix_type", "unknown")
        
        # Get resolution details
        if issue_type == "increase_memory":
            memory_increment = parameters.get("memory_increment", "unknown")
            resolution_details = f"Increased cluster memory by {memory_increment}."
        elif issue_type == "increase_disk_space":
            disk_increment = parameters.get("disk_increment", "unknown")
            resolution_details = f"Increased disk space by {disk_increment}."
        elif issue_type == "install_dependencies":
            packages = parameters.get("packages", [])
            if packages:
                packages_str = ", ".join(packages)
                resolution_details = f"Installed missing dependencies: {packages_str}."
            else:
                resolution_details = "Installed required dependencies."
        elif issue_type == "restart_cluster":
            resolution_details = "Restarted the Databricks cluster to resolve transient issues."
    
    # Format the issue type for display
    display_issue_type = issue_type.replace('_', ' ')
    
    # Generate report header with summary status
    report = f"# Databricks Monitoring Report\n\n"
    
    # Status section
    if fix_successful:
        report += f"## ✅ Fix Successful\n\n"
    else:
        report += f"## ❌ Unable to resolve issue. Please contact a data engineer.\n\n"
    
    # Confidence section
    confidence_percentage = confidence * 100
    confidence_text = ""
    
    if confidence_percentage >= 80:
        confidence_text = f"**High confidence** ({confidence_percentage:.1f}%)"
    elif confidence_percentage >= 50:
        confidence_text = f"**Moderate confidence** ({confidence_percentage:.1f}%)"
    elif confidence_percentage > 0:
        confidence_text = f"**Low confidence** ({confidence_percentage:.1f}%)"
    else:
        confidence_text = "**Confidence level unavailable**"
    
    report += f"Diagnosis confidence: {confidence_text}\n\n"
    
    # Situation section
    report += f"## Situation\n\n"
    report += f"Job ID **{job_id}** encountered a **{display_issue_type}** issue at {timestamp}.\n\n"
    
    # Root Cause section
    report += f"## Root Cause\n\n"
    
    if issue_type == "memory_exceeded":
        report += "The job failed due to insufficient memory allocation. The cluster ran out of memory while processing large datasets, "
        report += "likely during a memory-intensive operation such as a join or aggregation on high-cardinality columns.\n\n"
    elif issue_type == "disk_space_exceeded":
        report += "The job failed due to insufficient disk space. The cluster exhausted available storage while writing shuffle data or "
        report += "temporary files, preventing the job from completing successfully.\n\n"
    elif issue_type == "dependency_error":
        report += "The job failed due to missing or incompatible dependencies. Required libraries or packages were not available "
        report += "on the cluster, causing execution to fail when attempting to use unavailable functions.\n\n"
    else:
        report += "The job encountered an unexpected error that prevented successful completion. "
        report += "Analysis suggests possible resource constraints or configuration issues.\n\n"
    
    # Resolution section
    report += f"## Resolution\n\n"
    
    if fix_successful:
        if resolution_details:
            report += f"{resolution_details} "
        
        if issue_type == "memory_exceeded":
            report += "This provides sufficient memory for the job's data processing requirements, "
            report += "allowing memory-intensive operations to complete without exhausting available resources.\n\n"
        elif issue_type == "disk_space_exceeded":
            report += "This provides adequate storage for intermediate shuffle data and temporary files, "
            report += "ensuring the job can complete all data processing stages successfully.\n\n"
        elif issue_type == "dependency_error":
            report += "All required libraries are now available on the cluster, "
            report += "allowing the job to access necessary functions and complete successfully.\n\n"
        else:
            report += "The applied fix addressed the underlying issue, allowing the job to run successfully.\n\n"
    else:
        report += "The attempted fix was not successful. Further investigation is required to resolve this issue. "
        report += "Consider engaging your data engineering team for additional troubleshooting.\n\n"
    
    # Recommendations section
    report += f"## Recommendations\n\n"
    
    if issue_type == "memory_exceeded":
        report += "1. **Optimize Data Processing**: Review joins and aggregations for optimization opportunities\n"
        report += "2. **Implement Monitoring**: Set up memory usage alerts with Ganglia or Datadog\n"
        report += "3. **Tune Spark Settings**: Adjust spark.memory.fraction and spark.memory.storageFraction\n"
        report += "4. **Consider Data Partitioning**: Restructure data processing to work with smaller chunks\n"
    elif issue_type == "disk_space_exceeded":
        report += "1. **Implement Storage Monitoring**: Set up disk usage alerts\n"
        report += "2. **Optimize Storage Usage**: Use more efficient file formats like Parquet or Delta\n"
        report += "3. **Adjust Shuffle Partitions**: Tune spark.sql.shuffle.partitions for optimal performance\n"
        report += "4. **Data Lifecycle Management**: Implement policies to clean up temporary files\n"
    elif issue_type == "dependency_error":
        report += "1. **Standardize Environment**: Use initialization scripts for consistent dependency management\n"
        report += "2. **Document Requirements**: Maintain a comprehensive list of required packages\n"
        report += "3. **Version Control**: Pin specific versions of libraries to prevent compatibility issues\n"
        report += "4. **Environment Testing**: Validate dependencies before production deployment\n"
    else:
        report += "1. **Regular Monitoring**: Schedule periodic reviews of job performance\n"
        report += "2. **Proactive Alerts**: Set up notifications for potential issues\n"
        report += "3. **Resource Planning**: Allocate appropriate resources based on workload patterns\n"
        report += "4. **Logging Improvements**: Enhance logging to capture more diagnostic information\n"
    
    # Footer
    report += "\n---\n*This report was automatically generated by the Databricks Monitoring Agent.*\n"
    
    logger.info("Final report generated successfully")
    return report 