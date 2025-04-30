"""
Diagnostic tools for analyzing Databricks job failures.
"""

import enum
from typing import Dict, Any, Tuple, Optional

class FailureType(enum.Enum):
    """Enumeration of failure types for Databricks jobs."""
    MEMORY_EXCEEDED = "memory_exceeded"
    DISK_SPACE_EXCEEDED = "disk_space_exceeded"
    DEPENDENCY_ERROR = "dependency_error"
    UNKNOWN = "unknown"

def diagnose(log_text: str, metadata: Optional[Dict[str, Any]] = None) -> Tuple[FailureType, str]:
    """
    Analyze job logs to diagnose the root cause of failures.
    """
    # Simple pattern matching for testing
    if "OutOfMemoryError" in log_text or "MemoryError" in log_text:
        return FailureType.MEMORY_EXCEEDED, "Found memory error in logs"
    elif "No space left on device" in log_text:
        return FailureType.DISK_SPACE_EXCEEDED, "Found disk space error in logs"
    elif "ModuleNotFoundError" in log_text or "ImportError" in log_text:
        return FailureType.DEPENDENCY_ERROR, "Found dependency error in logs"
    else:
        return FailureType.UNKNOWN, "Could not determine the cause of failure"