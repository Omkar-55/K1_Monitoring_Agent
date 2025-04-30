# Databricks Monitoring Tools

This module provides tools for monitoring, diagnosing, and fixing Databricks job issues.

## Overview

The `databricks_monitoring` module provides a suite of tools for identifying and resolving issues with Databricks jobs. The tools follow a logical workflow:

1. Fetch logs from a job run
2. Diagnose any failures
3. Suggest fixes based on the diagnosis
4. Apply the fixes
5. Verify the fixes resolved the issue
6. Generate a report summarizing the process

## Tools

### `get_logs(job_id, run_id=None)`

Retrieves logs and metadata for a Databricks job run.

- **Input**: `job_id` and optional `run_id` (if not provided, gets most recent run)
- **Output**: JSON with run metadata and logs (stdout/stderr)
- **Use case**: First step in monitoring process to gather information

### `diagnose(log_text, metadata=None)`

Analyzes logs to determine the type of failure.

- **Input**: Raw log text and optional metadata
- **Output**: `FailureType` enum and reasoning text
- **Use case**: Identifying the root cause of job failures

### `suggest_fix(failure_type, context)`

Suggests a fix for the diagnosed issue.

- **Input**: `FailureType` and context dictionary
- **Output**: Structured plan with action, parameters, description, and confidence
- **Use case**: Determining how to resolve the detected issue

### `apply_fix(plan, job_id, run_id=None)`

Applies the suggested fix to a Databricks job.

- **Input**: Fix plan, job ID, and optional run ID
- **Output**: Success status, new run ID, and details
- **Use case**: Modifying job configuration or settings to resolve issues

### `verify(run_id, timeout_minutes=60, polling_interval_seconds=30)`

Polls a job run to verify if it completes successfully after fixing.

- **Input**: Run ID to monitor
- **Output**: "success", "needs_retry", or "failed"
- **Use case**: Confirming that applied fixes resolved the issue

### `final_report(history, job_id)`

Generates a comprehensive report summarizing the monitoring process.

- **Input**: History of steps taken and job ID
- **Output**: Markdown formatted report
- **Use case**: Creating documentation of the monitoring and fixes applied

## Failure Types

The module defines a `FailureType` enum with these categories:

- `MEMORY_EXCEEDED`: Job ran out of memory
- `DISK_SPACE_EXCEEDED`: Job ran out of disk space
- `DEPENDENCY_ERROR`: Missing libraries or dependencies
- `UNKNOWN`: Unidentified failure type

## Usage Example

```python
from src.tools.databricks_monitoring import (
    get_logs, diagnose, suggest_fix, apply_fix, verify, final_report
)

# Initialize history to track steps
history = []

# Step 1: Get logs
log_data = get_logs("12345")
history.append({"type": "logs", "timestamp": "2025-04-30 12:00:00", "run_id": log_data.get("run_id")})

# Step 2: Diagnose the issue
combined_logs = log_data.get("logs", {}).get("stdout", "") + "\n" + log_data.get("logs", {}).get("stderr", "")
failure_type, reasoning = diagnose(combined_logs)
history.append({"type": "diagnosis", "timestamp": "2025-04-30 12:01:00", "failure_type": failure_type.value})

# Step 3: Suggest a fix
fix_plan = suggest_fix(failure_type, {"logs": log_data.get("logs", {}), "metadata": log_data.get("metadata", {})})
history.append({"type": "fix", "timestamp": "2025-04-30 12:02:00", "action": fix_plan.get("action")})

# Step 4: Apply the fix
fix_result = apply_fix(fix_plan, log_data.get("job_id"))
history.append({"type": "fix", "timestamp": "2025-04-30 12:03:00", "success": fix_result.get("success")})

# Step 5: Verify the fix worked
if fix_result.get("new_run_id"):
    verification_result = verify(fix_result.get("new_run_id"))
    history.append({"type": "verification", "timestamp": "2025-04-30 12:04:00", "result": verification_result})

# Step 6: Generate report
report = final_report(history, log_data.get("job_id"))
print(report)
```

## Extension

The monitoring tools are designed to be extensible. To add support for new failure types:

1. Add a new value to the `FailureType` enum
2. Update the `diagnose` function to detect the new failure type
3. Update the `suggest_fix` function to provide appropriate fixes
4. Update the apply_fix function to implement the fix 