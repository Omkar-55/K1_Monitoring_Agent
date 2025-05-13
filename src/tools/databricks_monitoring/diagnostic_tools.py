"""
Tools for diagnostics and issue detection in Databricks jobs.
"""

import os
import re
import json
import time
import random
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple, Union

# Import logging configuration
from src.agent_core.logging_config import get_logger

# Get logger for this module
logger = get_logger(__name__)

class FailureType(Enum):
    """Types of failures that can occur in Databricks jobs."""
    MEMORY_EXCEEDED = "memory_exceeded"
    DISK_SPACE_EXCEEDED = "disk_space_exceeded"
    DEPENDENCY_ERROR = "dependency_error"
    UNKNOWN = "unknown"

def diagnose(logs_data: Dict[str, Any], cluster_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Analyzes Databricks logs to identify and diagnose job failures or performance issues.
    
    This tool uses AI-powered analysis for error identification.
    
    When to use:
    - After collecting logs from a failed or problematic Databricks job
    - To identify the root cause of a job failure
    - To get contextual information about performance issues
    
    Input JSON example:
    {
        "stdout": "... log content from standard output ...",
        "stderr": "... log content from standard error ...",
        "run_id": "12345",
        "job_id": "67890",
        "status": "FAILED",
        "duration_seconds": 3600,
        "cluster_id": "cluster-98765"  // Optional
    }
    
    Output JSON example:
    {
        "issue_type": "memory_exceeded",
        "confidence": 0.85,
        "evidence": [
            "java.lang.OutOfMemoryError: Java heap space",
            "Container killed by YARN for exceeding memory limits"
        ],
        "details": "The job failed due to insufficient memory allocation. Detected heap space errors and container termination.",
        "recommendations": [
            "Increase executor memory by at least 2GB",
            "Optimize join operations to reduce memory pressure",
            "Consider using disk spill for large operations"
        ],
        "cluster_context": {  // Optional, if available
            "executor_memory": "4g",
            "driver_memory": "8g",
            "max_executors": 10
        }
    }
    """
    logger.info(f"Diagnosing logs for run {logs_data.get('run_id', 'unknown')}")
    
    # Use simulation if this is a simulated run
    if logs_data.get("simulated", False):
        logger.info("Simulated run detected, generating simulated diagnosis")
        return _simulate_diagnosis(logs_data.get("simulate_failure_type"))
    
    # Always use AI-based diagnosis
    try:
        logger.info("Performing AI-based diagnosis")
        ai_diagnosis = diagnose_with_ai(logs_data)
        
        # Add reasoning steps to the diagnosis
        reasoning_steps = [
            {
                "step": "log_analysis",
                "details": "Analyzed Databricks job logs for error patterns and performance issues",
                "result": "Found relevant log entries for diagnosis"
            },
            {
                "step": "issue_identification",
                "details": f"Identified issue type: {ai_diagnosis.get('issue_type', 'unknown')}",
                "result": f"Confidence: {ai_diagnosis.get('confidence', 0.0)}"
            },
            {
                "step": "evidence_collection",
                "details": "Collected supporting evidence from logs",
                "result": "\n".join(ai_diagnosis.get('evidence', []))
            }
        ]
        
        # Add reasoning steps to the diagnosis
        ai_diagnosis["reasoning"] = reasoning_steps
        
        # Check confidence threshold
        if ai_diagnosis.get("confidence", 0) >= 0.3:
            logger.info(f"AI diagnosis complete with confidence {ai_diagnosis.get('confidence')}")
            return ai_diagnosis
        else:
            logger.warning(f"AI diagnosis confidence too low ({ai_diagnosis.get('confidence')})")
            # Even with low confidence, still return the AI diagnosis
            return ai_diagnosis
    except Exception as e:
        logger.error(f"AI diagnosis failed: {e}")
        # Return a basic unknown error diagnosis
        return {
            "issue_type": FailureType.UNKNOWN.value,
            "confidence": 0.1,
            "evidence": [],
            "details": f"Failed to diagnose the issue: {str(e)}",
            "recommendations": ["Check logs manually", "Contact support"],
            "reasoning": [
                {
                    "step": "error",
                    "details": f"Error during diagnosis: {str(e)}",
                    "result": "Failed to complete diagnosis"
                }
            ]
        }

def _extract_issue_type_from_text(text: str) -> FailureType:
    """
    Extract issue type from text analysis.
    
    Args:
        text: The text to analyze
        
    Returns:
        The failure type
    """
    text_lower = text.lower()
    
    if "memory" in text_lower and any(term in text_lower for term in ["oom", "out of memory", "heap space"]):
        return FailureType.MEMORY_EXCEEDED
    elif "disk" in text_lower and ("space" in text_lower or "quota" in text_lower):
        return FailureType.DISK_SPACE_EXCEEDED
    elif any(term in text_lower for term in ["dependency", "module", "package", "import", "library"]):
        return FailureType.DEPENDENCY_ERROR
    else:
        return FailureType.UNKNOWN

def diagnose_with_ai(logs_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Diagnose issues in Databricks logs using Azure OpenAI.
    
    Args:
        logs_data: Dictionary containing log data
        
    Returns:
        Dictionary with diagnosis results
    """
    logger.info("Using AI to diagnose logs")
    
    # Extract logs
    stdout = logs_data.get("stdout", "")
    stderr = logs_data.get("stderr", "")
    
    # Limit log size to avoid token limits
    stdout = stdout[-5000:] if len(stdout) > 5000 else stdout
    stderr = stderr[-10000:] if len(stderr) > 10000 else stderr
    
    # Get Azure OpenAI credentials
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2023-05-15")
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1")
    
    if not api_key or not endpoint:
        logger.warning("Azure OpenAI credentials not available")
        return {
            "issue_type": FailureType.UNKNOWN.value,
            "confidence": 0.1,
            "evidence": [],
            "details": "Could not connect to AI service for analysis",
            "recommendations": ["Check OpenAI configuration", "Check logs manually"]
        }
    
    try:
        from openai import AzureOpenAI
        
        # Create OpenAI client
        client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_version
        )
        
        # Prepare prompt
        prompt = f"""
        Analyze the following Databricks job logs and identify the most likely type of issue.
        
        STDERR LOGS:
        {stderr}
        
        STDOUT LOGS:
        {stdout}
        
        I need to categorize the issue into one of these types:
        1. memory_exceeded: Out of memory errors, heap space issues
        2. disk_space_exceeded: No space left on device, disk quota issues
        3. dependency_error: Missing dependencies, import errors, library issues
        4. unknown: Can't determine from the logs
        
        Also tell me:
        - The confidence in your diagnosis (0.0 to 1.0)
        - Specific evidence from the logs that supports your diagnosis
        - Likely root cause of the issue
        
        Return your analysis as a JSON object.
        """
        
        # Call the API without async (synchronous call in a synchronous function)
        response = client.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "system", "content": "You are a Databricks diagnostics expert. Be concise and accurate."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=800
        )
        
        try:
            # Try to parse JSON from the response
            content = response.choices[0].message.content
            result = json.loads(content)
            
            # Extract the diagnosis
            issue_type = result.get("issue_type", "unknown")
            confidence = result.get("confidence", 0.5)
            evidence = result.get("evidence", [])
            root_cause = result.get("root_cause", "Unknown")
            
            logger.info(f"AI diagnosis complete: {issue_type} (confidence: {confidence})")
            
            # Convert to standard format
            return {
                "issue_type": issue_type,
                "confidence": confidence,
                "evidence": evidence,
                "details": root_cause
            }
        except json.JSONDecodeError:
            # If can't parse JSON, try to extract insights directly
            content = response.choices[0].message.content
            issue_type = _extract_issue_type_from_text(content)
            
            logger.warning(f"Couldn't parse JSON response, extracted issue type: {issue_type}")
        
        return {
                "issue_type": issue_type.value,
                "confidence": 0.6,
                "evidence": [],
                "details": "Extracted from AI analysis text"
            }
            
    except ImportError:
        logger.warning("OpenAI package not available")
        return {
            "issue_type": FailureType.UNKNOWN.value,
            "confidence": 0.1,
            "evidence": [],
            "details": "OpenAI package not available for AI analysis",
            "recommendations": ["Install OpenAI package", "Check logs manually"]
        }
    except Exception as e:
        logger.error(f"Error in AI diagnosis: {e}")
        return {
            "issue_type": FailureType.UNKNOWN.value,
            "confidence": 0.1,
            "evidence": [],
            "details": f"Error in AI diagnosis: {str(e)}",
            "recommendations": ["Check logs manually", "Review API access"]
        }

def _simulate_diagnosis(failure_type_str: Optional[str] = None) -> Dict[str, Any]:
    """Simulate a diagnosis result for testing."""
    logger.info(f"Simulating run with failure type: {failure_type_str}")
    
    # Determine failure type - either use requested type or random
    if failure_type_str:
        try:
            # Direct mapping - ensure UI selection maps to correct failure type
            failure_type = FailureType(failure_type_str)
            logger.info(f"Using specified failure type: {failure_type.value}")
        except ValueError:
            # If it's not a valid enum, check if it matches any enum value case-insensitively
            found = False
            for enum_val in FailureType:
                if enum_val.value.lower() == failure_type_str.lower():
                    failure_type = enum_val
                    found = True
                    logger.info(f"Mapped input '{failure_type_str}' to failure type: {failure_type.value}")
                    break
            
            if not found:
                logger.warning(f"Invalid failure type: {failure_type_str}. Using random failure type.")
            failure_type = random.choice(list(FailureType))
    else:
        # If no failure type specified, use random
        failure_type = random.choice(list(FailureType))
    
    run_id = f"run_{int(time.time())}"
    logger.info(f"Simulated run {run_id} with failure type: {failure_type.value}")
    
    # Common reasoning steps for all diagnoses
    common_reasoning_steps = [
        {
            "step": "log_collection",
            "details": "Retrieved and parsed Databricks job logs from the specified job run",
            "result": "Successfully collected log data for analysis"
        },
        {
            "step": "pattern_analysis",
            "details": "Analyzed logs for common error patterns and signatures",
            "result": "Identified key error patterns in the logs"
        }
    ]
    
    # Detailed error logs for each failure type
    if failure_type == FailureType.MEMORY_EXCEEDED:
        # Choose one of several memory error patterns
        memory_error_patterns = [
            {
                "error": "Java Heap Space Error",
                "evidence": [
                    "java.lang.OutOfMemoryError: Java heap space",
                    "at org.apache.spark.sql.execution.joins.BroadcastHashJoinExec.executeBroadcast(BroadcastHashJoinExec.scala:163)",
                    "at org.apache.spark.sql.execution.joins.BroadcastHashJoinExec.doExecuteBroadcast(BroadcastHashJoinExec.scala:98)"
                ],
                "details": "The JVM could not allocate more memory for objects due to heap space exhaustion during a join operation on large datasets."
            },
            {
                "error": "Driver Out of Memory",
                "evidence": [
                    "ERROR JobProgressReporter: Exception while running job: Driver: Out of Memory",
                    "org.apache.spark.SparkException: Job aborted due to stage failure: Total size of serialized results of 67 tasks (1024.0 MB) is bigger than spark.driver.maxResultSize (1024.0 MB)"
                ],
                "details": "The size of data being collected to the driver exceeded the configured limit during a collect() operation."
            },
            {
                "error": "GC Overhead Limit Exceeded",
                "evidence": [
                    "java.lang.OutOfMemoryError: GC overhead limit exceeded",
                    "at java.util.Arrays.copyOfRange(Arrays.java:3664)",
                    "at java.lang.String.<init>(String.java:207)",
                    "at org.apache.spark.sql.catalyst.expressions.GeneratedClass$GeneratedIteratorForCodegenStage1.processNext(Unknown Source)"
                ],
                "details": "The garbage collector is running almost continuously and recovering very little memory, indicating the system is spending too much time performing garbage collection with minimal progress."
            },
            {
                "error": "ExecutorLostFailure due to Out of Memory",
                "evidence": [
                    "ERROR TaskSetManager: Lost executor 5 on 10.139.64.11: ExecutorLostFailure (executor 5 exited caused by one of the running tasks)",
                    "Reason: Container killed by YARN for exceeding memory limits. 16.0 GB of 16 GB physical memory used.",
                    "Consider boosting spark.yarn.executor.memoryOverhead."
                ],
                "details": "An executor was terminated because it exceeded its allocated memory, possibly due to data skew or insufficient memory allocation."
            }
        ]
        
        # Select one pattern randomly
        selected_error = random.choice(memory_error_patterns)
        
        # Add memory-specific reasoning steps
        memory_reasoning = common_reasoning_steps + [
            {
                "step": "memory_analysis",
                "details": "Examined memory usage patterns and allocation in executor logs",
                "result": "Detected memory pressure during job execution"
            },
            {
                "step": "error_classification",
                "details": f"Analyzed error type: {selected_error['error']}",
                "result": "Classified as memory exceeded issue with high confidence"
            },
            {
                "step": "root_cause_identification",
                "details": "Traced error to specific operation in job execution",
                "result": selected_error["details"]
            }
        ]
        
        return {
            "issue_type": failure_type.value,
            "confidence": 0.85,
            "evidence": selected_error["evidence"],
            "details": selected_error["details"],
            "recommendations": [
                "Increase executor memory",
                "Optimize join operations",
                "Implement data partitioning to reduce memory pressure",
                "Consider using disk spill for large operations"
            ],
            "cluster_context": {
                "executor_memory": "4g",
                "driver_memory": "8g",
                "spark.memory.fraction": "0.6",
                "spark.memory.storageFraction": "0.5"
            },
            "issue_detected": True,
            "reasoning": memory_reasoning
        }
    elif failure_type == FailureType.DISK_SPACE_EXCEEDED:
        # Different disk space error patterns
        disk_error_patterns = [
            {
                "error": "No Space Left on Device",
                "evidence": [
                    "ERROR StorageManager: Error writing data to local disk",
                    "java.io.IOException: No space left on device",
                    "at java.io.FileOutputStream.writeBytes(Native Method)",
                    "at org.apache.spark.storage.DiskBlockObjectWriter.write(DiskBlockObjectWriter.scala:233)"
                ],
                "details": "The job ran out of disk space while writing shuffle data to local storage."
            },
            {
                "error": "Disk Quota Exceeded",
                "evidence": [
                    "java.io.IOException: Disk quota exceeded",
                    "at java.io.UnixFileSystem.createFileExclusively(Native Method)",
                    "at org.apache.hadoop.fs.RawLocalFileSystem.create(RawLocalFileSystem.java:297)"
                ],
                "details": "The job exceeded the available disk quota when writing output files."
            },
            {
                "error": "Storage Directory Full",
                "evidence": [
                    "ERROR DiskBlockManager: Disk write failed at /local_disk0/spark/storage",
                    "java.io.FileSystemException: /local_disk0/spark/storage: No space left on device",
                    "Cannot write to output stream: No space left on device"
                ],
                "details": "The local storage directory for Spark shuffle data is full."
            }
        ]
        
        selected_error = random.choice(disk_error_patterns)
        
        # Add disk-specific reasoning steps
        disk_reasoning = common_reasoning_steps + [
            {
                "step": "disk_space_analysis",
                "details": "Examined disk usage and storage allocation patterns",
                "result": "Detected filesystem capacity issues during job execution"
            },
            {
                "step": "error_classification",
                "details": f"Analyzed error type: {selected_error['error']}",
                "result": "Classified as disk space exceeded issue with high confidence"
            },
            {
                "step": "file_operation_tracing",
                "details": "Identified file operations that triggered the failure",
                "result": "Found critical failure during shuffle write operations"
            }
        ]
        
        return {
            "issue_type": failure_type.value,
            "confidence": 0.78,
            "evidence": selected_error["evidence"],
            "details": selected_error["details"],
            "recommendations": [
                "Clean up unused data files",
                "Use more efficient storage formats (Parquet, Delta)",
                "Implement data lifecycle policies",
                "Adjust shuffle partition count to create smaller files"
            ],
            "cluster_context": {
                "available_disk_space": "2.1GB",
                "used_disk_space": "98.5%",
                "filesystem_type": "ext4"
            },
            "issue_detected": True,
            "reasoning": disk_reasoning
        }
    elif failure_type == FailureType.DEPENDENCY_ERROR:
        # Various dependency error patterns
        dependency_error_patterns = [
            {
                "error": "Class Not Found Exception",
                "evidence": [
                    "java.lang.ClassNotFoundException: org.apache.spark.sql.delta.DeltaTable",
                    "at java.net.URLClassLoader.findClass(URLClassLoader.java:382)",
                    "at java.lang.ClassLoader.loadClass(ClassLoader.java:418)"
                ],
                "details": "The required Java class for Delta Lake operations was not found in the classpath."
            },
            {
                "error": "Module Not Found Error",
                "evidence": [
                    "ModuleNotFoundError: No module named 'pyarrow.fs'",
                    "at line 14 in command",
                    "from pyarrow import fs"
                ],
                "details": "The Python module 'pyarrow.fs' is missing from the cluster environment."
            },
            {
                "error": "Library Installation Failure",
                "evidence": [
                    "pip install --upgrade /local_disk0/tmp/addedFilec/mypackage --disable-pip-version-check",
                    "ERROR: COMMAND FAILED WITH STATUS 1",
                    "Error: An error occurred while installing package 'mypackage'"
                ],
                "details": "Failed to install a required library package on the cluster."
            },
            {
                "error": "Version Conflict Error",
                "evidence": [
                    "ERROR Executor: Exception in task 0.0 in stage 2.0 (TID 2)",
                    "java.lang.NoSuchMethodError: org.apache.hadoop.fs.azurebfs.AzureBlobFileSystem.initialize(Ljava/net/URI;Lorg/apache/hadoop/conf/Configuration;)V",
                    "at org.apache.hadoop.fs.FileSystem.createFileSystem(FileSystem.java:3303)"
                ],
                "details": "Incompatible versions of libraries are causing method conflicts."
            }
        ]
        
        selected_error = random.choice(dependency_error_patterns)
        
        # Add dependency-specific reasoning steps
        dependency_reasoning = common_reasoning_steps + [
            {
                "step": "dependency_analysis",
                "details": "Examined library and package dependencies in the cluster configuration",
                "result": "Detected missing or incompatible dependencies"
            },
            {
                "step": "error_classification",
                "details": f"Analyzed error type: {selected_error['error']}",
                "result": "Classified as dependency error with high confidence"
            },
            {
                "step": "stack_trace_analysis",
                "details": "Analyzed stack trace to pinpoint missing or incompatible components",
                "result": f"Identified specific dependency issue: {selected_error['error']}"
            }
        ]
        
        return {
            "issue_type": failure_type.value,
            "confidence": 0.82,
            "evidence": selected_error["evidence"],
            "details": selected_error["details"],
            "recommendations": [
                "Install missing dependencies",
                "Check library compatibility",
                "Add initialization scripts to install required packages",
                "Properly configure cluster with required libraries"
            ],
            "cluster_context": {
                "spark_version": "10.4.x-scala2.12",
                "python_version": "3.8.10",
                "installed_libraries": [
                    {"name": "pandas", "version": "1.3.4"},
                    {"name": "numpy", "version": "1.21.3"}
                ]
            },
            "issue_detected": True,
            "reasoning": dependency_reasoning
        }
    else:
        # Generic unknown issue
        unknown_reasoning = common_reasoning_steps + [
            {
                "step": "error_pattern_search",
                "details": "Searched for known error patterns in log output",
                "result": "No clear error pattern detected in the logs"
            },
            {
                "step": "resource_analysis",
                "details": "Checked cluster resources and configuration",
                "result": "No obvious resource constraints or misconfigurations found"
            },
            {
                "step": "process_monitoring",
                "details": "Analyzed process timing and execution flow",
                "result": "Process terminated abnormally without specific error"
            }
        ]
        
        return {
            "issue_type": failure_type.value,
            "confidence": 0.4,
            "evidence": [
                "Task failed without specific error messages",
                "Process exited with non-zero status",
                "Cluster event logs show unexpected termination"
            ],
            "details": "The job failed for unknown reasons. No specific error pattern was detected.",
            "recommendations": [
                "Check Databricks job configuration",
                "Review data quality and pipeline inputs",
                "Examine cluster usage metrics",
                "Enable debug logging for more detailed diagnostics"
            ],
            "cluster_context": {},
            "issue_detected": True,
            "reasoning": unknown_reasoning
        } 