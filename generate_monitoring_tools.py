"""
Generate monitoring tools files.

This script creates simplified versions of our monitoring tools files for testing.
"""

import os

# Create the directory structure if it doesn't exist
os.makedirs("src/tools/databricks_monitoring", exist_ok=True)

# Create log_tools.py
log_tools = """
\"\"\"
Tools for fetching and processing Databricks job logs.
\"\"\"

import json
from typing import Dict, Any, Optional, List
import time

# Import the Databricks client wrapper
from agent_core.dbx_client import DbxClient

# For testing purposes, this is a simplified version
def get_logs(job_id: str, run_id: Optional[str] = None) -> Dict[str, Any]:
    \"\"\"
    Get logs and metadata for a Databricks job run.
    \"\"\"
    print(f"Getting logs for job ID {job_id}, run ID {run_id if run_id else 'most recent'}")
    
    # Return simulated data for testing
    return {
        "run_id": run_id or f"simulated-run-{int(time.time())}",
        "job_id": job_id,
        "status": "TERMINATED",
        "result": "FAILED",
        "logs": {
            "stdout": "Simulated stdout logs",
            "stderr": "Simulated stderr logs with errors"
        },
        "metadata": {
            "cluster_id": "simulated-cluster",
            "creator": "test-user"
        }
    }
"""

# Create diagnostic_tools.py
diagnostic_tools = """
\"\"\"
Diagnostic tools for analyzing Databricks job failures.
\"\"\"

import enum
from typing import Dict, Any, Tuple, Optional

class FailureType(enum.Enum):
    \"\"\"Enumeration of failure types for Databricks jobs.\"\"\"
    MEMORY_EXCEEDED = "memory_exceeded"
    DISK_SPACE_EXCEEDED = "disk_space_exceeded"
    DEPENDENCY_ERROR = "dependency_error"
    UNKNOWN = "unknown"

def diagnose(log_text: str, metadata: Optional[Dict[str, Any]] = None) -> Tuple[FailureType, str]:
    \"\"\"
    Analyze job logs to diagnose the root cause of failures.
    \"\"\"
    # Simple pattern matching for testing
    if "OutOfMemoryError" in log_text or "MemoryError" in log_text:
        return FailureType.MEMORY_EXCEEDED, "Found memory error in logs"
    elif "No space left on device" in log_text:
        return FailureType.DISK_SPACE_EXCEEDED, "Found disk space error in logs"
    elif "ModuleNotFoundError" in log_text or "ImportError" in log_text:
        return FailureType.DEPENDENCY_ERROR, "Found dependency error in logs"
    else:
        return FailureType.UNKNOWN, "Could not determine the cause of failure"
"""

# Create fix_tools.py
fix_tools = """
\"\"\"
Tools for suggesting and applying fixes to Databricks job issues.
\"\"\"

from typing import Dict, Any, Optional
import time
from .diagnostic_tools import FailureType

def suggest_fix(failure_type: FailureType, context: Dict[str, Any]) -> Dict[str, Any]:
    \"\"\"
    Suggest a fix for a specific failure type with context.
    \"\"\"
    # Simple fixes for testing
    if failure_type == FailureType.MEMORY_EXCEEDED:
        return {
            "action": "increase_memory",
            "parameters": {"memory": "8g"},
            "description": "Increase memory allocation",
            "confidence": 0.8
        }
    elif failure_type == FailureType.DISK_SPACE_EXCEEDED:
        return {
            "action": "increase_disk_space",
            "parameters": {"disk": "100g"},
            "description": "Increase disk space allocation",
            "confidence": 0.7
        }
    elif failure_type == FailureType.DEPENDENCY_ERROR:
        return {
            "action": "install_dependencies",
            "parameters": {"libraries": ["missing-lib"]},
            "description": "Install missing dependencies",
            "confidence": 0.9
        }
    else:
        return {
            "action": "manual_review",
            "parameters": {},
            "description": "Manual review required",
            "confidence": 0.3
        }

def apply_fix(plan: Dict[str, Any], job_id: str, run_id: Optional[str] = None) -> Dict[str, Any]:
    \"\"\"
    Apply a suggested fix to a Databricks job.
    \"\"\"
    # Simulate fix application
    new_run_id = f"simulated-fixed-run-{int(time.time())}"
    
    return {
        "success": True,
        "new_run_id": new_run_id,
        "message": f"Applied fix: {plan.get('action')}",
        "details": plan.get("parameters", {})
    }
"""

# Create verification_tools.py
verification_tools = """
\"\"\"
Tools for verifying that applied fixes have resolved Databricks job issues.
\"\"\"

from typing import Literal

def verify(run_id: str, timeout_minutes: int = 60, polling_interval_seconds: int = 30) -> Literal["success", "needs_retry", "failed"]:
    \"\"\"
    Poll a Databricks job run to verify if it completes successfully.
    \"\"\"
    # For testing, always return success
    print(f"Verifying run {run_id}")
    return "success"
"""

# Create reporting_tools.py
reporting_tools = """
\"\"\"
Tools for generating reports about Databricks job issues and resolutions.
\"\"\"

from typing import Dict, Any, List
import time

def final_report(history: List[Dict[str, Any]], job_id: str) -> str:
    \"\"\"
    Generate a final report in Markdown format summarizing the monitoring, diagnosis, and resolution process.
    \"\"\"
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
    
    return "\\n".join(report)
"""

# Create __init__.py
init_py = """
\"\"\"
Databricks monitoring tools module.

This module contains tools for monitoring, diagnosing, and fixing Databricks job issues.
\"\"\"

from .log_tools import get_logs
from .diagnostic_tools import diagnose, FailureType
from .fix_tools import suggest_fix, apply_fix
from .verification_tools import verify
from .reporting_tools import final_report

__all__ = [
    'get_logs',
    'diagnose',
    'FailureType',
    'suggest_fix',
    'apply_fix',
    'verify',
    'final_report'
]
"""

# Write all files
with open("src/tools/databricks_monitoring/__init__.py", "w") as f:
    f.write(init_py.strip())

with open("src/tools/databricks_monitoring/log_tools.py", "w") as f:
    f.write(log_tools.strip())

with open("src/tools/databricks_monitoring/diagnostic_tools.py", "w") as f:
    f.write(diagnostic_tools.strip())

with open("src/tools/databricks_monitoring/fix_tools.py", "w") as f:
    f.write(fix_tools.strip())

with open("src/tools/databricks_monitoring/verification_tools.py", "w") as f:
    f.write(verification_tools.strip())

with open("src/tools/databricks_monitoring/reporting_tools.py", "w") as f:
    f.write(reporting_tools.strip())

print("All monitoring tool files have been generated successfully!")
print("Files created:")
print("- src/tools/databricks_monitoring/__init__.py")
print("- src/tools/databricks_monitoring/log_tools.py")
print("- src/tools/databricks_monitoring/diagnostic_tools.py")
print("- src/tools/databricks_monitoring/fix_tools.py")
print("- src/tools/databricks_monitoring/verification_tools.py")
print("- src/tools/databricks_monitoring/reporting_tools.py") 