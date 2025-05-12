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

# Regular expressions for common errors
OOM_RE = r"java\.lang\.OutOfMemoryError|OutOfMemoryError|Memory limit exceeded|GC overhead limit exceeded|Container killed.*exceeding memory limits"
DISK_RE = r"No space left on device|Disk quota exceeded|IOException.*space|DiskBlockManager.*disk write failed"
DEP_RE = r"ClassNotFoundException|ModuleNotFoundError|ImportError|NoClassDefFoundError|Can\'t find resource file|library not loaded"
QUOTA_RE = r"Quota exceeded|Resource limit exceeded|exceeded your quota"
TIMEOUT_RE = r"Connection timed out|Read timed out|Socket timeout|Connection reset|unreachable"
DATA_SKEW_RE = r"data skew|skewed data|imbalanced partitions"

class FailureType(Enum):
    """Types of failures that can occur in Databricks jobs."""
    MEMORY_EXCEEDED = "memory_exceeded"
    DISK_SPACE_EXCEEDED = "disk_space_exceeded"
    DEPENDENCY_ERROR = "dependency_error"
    UNKNOWN = "unknown"

def diagnose(logs_data: Dict[str, Any], cluster_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Analyzes Databricks logs to identify and diagnose job failures or performance issues.
    
    This tool uses a multi-layered approach to diagnose issues:
    1. Pattern-based detection using regex for common errors
    2. AI-powered analysis for complex or unclear errors
    3. Correlation with cluster metadata and job configuration
    
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
    
    # Determine which diagnostic method to use
    use_ai = os.getenv("USE_AI_DIAGNOSIS", "true").lower() in ["true", "1", "yes"]
    
    # Try AI diagnosis first if enabled
    if use_ai:
        try:
            logger.info("Attempting AI-based diagnosis")
            ai_diagnosis = diagnose_with_ai(logs_data)
            
            # Check confidence threshold
            if ai_diagnosis.get("confidence", 0) >= 0.6:
                logger.info(f"AI diagnosis successful with confidence {ai_diagnosis.get('confidence')}")
                return ai_diagnosis
            else:
                logger.info(f"AI diagnosis confidence too low ({ai_diagnosis.get('confidence')}), falling back to pattern matching")
        except Exception as e:
            logger.warning(f"AI diagnosis failed: {e}, falling back to pattern matching")
    
    # Fall back to pattern matching
    logger.info("Using pattern matching for diagnosis")
    return diagnose_pattern_matching(logs_data)

def _extract_issue_type_from_text(text: str) -> FailureType:
    """
    Extract issue type from text analysis.
    
    Args:
        text: The text to analyze
        
    Returns:
        The failure type
    """
    text_lower = text.lower()
    
    if re.search(OOM_RE, text_lower, re.IGNORECASE) or "memory" in text_lower:
        return FailureType.MEMORY_EXCEEDED
    elif re.search(DISK_RE, text_lower, re.IGNORECASE) or "disk" in text_lower:
        return FailureType.DISK_SPACE_EXCEEDED
    elif re.search(DEP_RE, text_lower, re.IGNORECASE) or "dependency" in text_lower:
        return FailureType.DEPENDENCY_ERROR
    else:
        return FailureType.UNKNOWN

def _basic_diagnosis(stdout: str, stderr: str) -> Dict[str, Any]:
    """
    Perform a basic diagnosis using pattern matching.
    
    Args:
        stdout: Standard output logs
        stderr: Standard error logs
        
    Returns:
        Dictionary with diagnosis results
    """
    # Combine logs
    combined_logs = f"{stdout}\n{stderr}"
    
    # Check for memory issues
    if re.search(OOM_RE, combined_logs, re.IGNORECASE):
        return {
            "issue_type": FailureType.MEMORY_EXCEEDED.value,
            "confidence": 0.7,
            "evidence": ["Out of memory error detected in logs"],
            "details": "Memory limit exceeded"
        }
    
    # Check for disk issues
    if re.search(DISK_RE, combined_logs, re.IGNORECASE):
        return {
            "issue_type": FailureType.DISK_SPACE_EXCEEDED.value,
            "confidence": 0.7,
            "evidence": ["Disk space error detected in logs"],
            "details": "Disk space limit exceeded"
        }
    
    # Check for dependency issues
    if re.search(DEP_RE, combined_logs, re.IGNORECASE):
        return {
            "issue_type": FailureType.DEPENDENCY_ERROR.value,
            "confidence": 0.7,
            "evidence": ["Dependency error detected in logs"],
            "details": "Missing dependencies"
        }
    
    # Unknown issue
    return {
        "issue_type": FailureType.UNKNOWN.value,
        "confidence": 0.3,
        "evidence": [],
        "details": "Could not determine issue type from logs"
    }

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
        return _basic_diagnosis(stdout, stderr)
    
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
        return _basic_diagnosis(stdout, stderr)
    except Exception as e:
        logger.error(f"Error in AI diagnosis: {e}")
        return _basic_diagnosis(stdout, stderr)

def extract_issue_type(text: str) -> str:
    """
    Extract the issue type from AI analysis.
    
    Args:
        text: The AI analysis text
        
    Returns:
        The issue type as a string
    """
    # Convert to lowercase for easier matching
    text_lower = text.lower()
    
    # Check for memory issues
    if "memory" in text_lower and any(term in text_lower for term in ["oom", "out of memory", "heap space"]) or re.search(OOM_RE, text_lower, re.IGNORECASE):
        return FailureType.MEMORY_EXCEEDED.value
    elif "disk" in text_lower and ("space" in text_lower or "quota" in text_lower):
        return FailureType.DISK_SPACE_EXCEEDED.value
    elif any(term in text_lower for term in ["dependency", "module", "package", "import", "library"]) or re.search(DEP_RE, text_lower, re.IGNORECASE):
        return FailureType.DEPENDENCY_ERROR.value
    elif re.search(QUOTA_RE, text_lower, re.IGNORECASE):
        return FailureType.UNKNOWN.value
    elif re.search(TIMEOUT_RE, text_lower, re.IGNORECASE):
        return FailureType.UNKNOWN.value
    elif "skew" in text_lower or "imbalance" in text_lower:
        return FailureType.UNKNOWN.value
    else:
        return FailureType.UNKNOWN.value

def diagnose_pattern_matching(logs_data: Dict[str, Any]) -> Dict[str, Any]:
    """Diagnose using pattern matching."""
    logger.info("Diagnosing Databricks logs using pattern matching")
    
    # Extract logs
    stdout = logs_data.get("stdout", "")
    stderr = logs_data.get("stderr", "")
    
    # Default diagnosis
    issue_type = FailureType.UNKNOWN
    reasoning = "Could not determine the issue from logs"
    confidence = 0.3
    
    # Check for memory issues (highest priority)
    if "OutOfMemoryError" in stderr or "GC overhead limit exceeded" in stderr or re.search(OOM_RE, stderr, re.IGNORECASE):
        issue_type = FailureType.MEMORY_EXCEEDED
        reasoning = "Job failed due to memory issues. Found OutOfMemoryError or GC overhead limit exceeded in logs."
        confidence = 0.8
    
    # Check for disk space issues
    elif "No space left on device" in stderr or "Disk quota exceeded" in stderr or re.search(DISK_RE, stderr, re.IGNORECASE):
        issue_type = FailureType.DISK_SPACE_EXCEEDED
        reasoning = "Job failed due to disk space issues. Found 'No space left on device' or similar in logs."
        confidence = 0.8
    
    # Check for dependency issues
    elif "ModuleNotFoundError" in stderr or "ImportError" in stderr or "ClassNotFoundException" in stderr or re.search(DEP_RE, stderr, re.IGNORECASE):
        issue_type = FailureType.DEPENDENCY_ERROR
        reasoning = "Job failed due to missing dependencies. Found import or module errors in logs."
        confidence = 0.8
    
    # Check for quota issues
    elif re.search(QUOTA_RE, stderr, re.IGNORECASE):
        issue_type = FailureType.UNKNOWN
        reasoning = "Job failed due to unknown quota issues."
        confidence = 0.7
    
    # Check for timeout issues
    elif re.search(TIMEOUT_RE, stderr, re.IGNORECASE):
        issue_type = FailureType.UNKNOWN
        reasoning = "Job failed due to connection timeouts or unreachable components."
        confidence = 0.7
    
    # If nothing specific found, try to gather some metrics for further analysis
    elif "Exception" in stderr or "Error" in stderr:
        issue_type = FailureType.UNKNOWN
        reasoning = "Job failed with an exception, but could not determine the specific cause."
        confidence = 0.4
        
        # Extract some metrics if available
        try:
            # Extract memory usage metrics if available
            metrics = {}
            worker_mems = []
            
            # Memory pattern: "Worker X: Y% memory used"
            mem_pattern = r"Worker \d+: (\d+)% memory used"
            for match in re.finditer(mem_pattern, stdout):
                worker_mems.append(float(match.group(1)))
                
            if worker_mems:
                metrics["worker_memory_usage"] = {
                    "min": min(worker_mems),
                    "max": max(worker_mems),
                    "avg": sum(worker_mems) / len(worker_mems)
                }
                    
                # Check for potential data skew if timeout or unknown issue
                if issue_type in [FailureType.UNKNOWN]:
                    # Check for imbalanced worker memory usage
                    if worker_mems and max(worker_mems) > 2 * min(worker_mems) and max(worker_mems) > 75:
                        issue_type = FailureType.UNKNOWN
                        reasoning = f"Potential data skew detected. Worker memory usage varies widely from {min(worker_mems):.1f}% to {max(worker_mems):.1f}%, suggesting imbalanced data processing."
                        confidence = 0.6
        except Exception as e:
            logger.warning(f"Error extracting metrics: {e}")
    
    # Create the diagnosis result
    basic_diagnosis = {
        "issue_type": issue_type.value,
        "confidence": confidence,
        "reasoning": reasoning,
        "method": "pattern_matching",
        "evidence": []
    }
    
    # Extract evidence for the diagnosis
    if "OutOfMemoryError" in stderr:
        for line in stderr.split("\n"):
            if "OutOfMemoryError" in line:
                basic_diagnosis["evidence"].append(line.strip())
                break
    
    # Add recommendations based on issue type
    if issue_type == FailureType.MEMORY_EXCEEDED:
        basic_diagnosis["recommendations"] = [
            "Increase executor memory",
            "Optimize join operations",
            "Reduce shuffle partitions"
        ]
    elif issue_type == FailureType.DISK_SPACE_EXCEEDED:
        basic_diagnosis["recommendations"] = [
            "Clean up unused data files",
            "Use more efficient storage formats"
        ]
    elif issue_type == FailureType.DEPENDENCY_ERROR:
        basic_diagnosis["recommendations"] = [
            "Install missing dependencies",
            "Check library compatibility"
        ]
    else:
        basic_diagnosis["recommendations"] = [
            "Check logs for more details",
            "Inspect job configuration"
        ]
    
    logger.info(f"Pattern matching diagnosis complete: {issue_type.value} with confidence {confidence:.2f}")
    return basic_diagnosis

def _pattern_detection(stdout: str, stderr: str) -> Tuple[FailureType, float, List[str]]:
    """
    Detect issues based on pattern matching in logs.
    
    Args:
        stdout: Standard output logs
        stderr: Standard error logs
        
    Returns:
        Tuple of (issue_type, confidence, evidence)
    """
    # Combine logs for analysis
    combined_logs = f"{stdout}\n{stderr}"
    evidence = []
    
    # Check for memory issues
    oom_matches = re.findall(OOM_RE, combined_logs, re.IGNORECASE)
    if oom_matches:
        # Extract surrounding context for evidence
        for match in oom_matches[:3]:  # Limit to first 3 matches
            # Find the line containing the match
            for line in combined_logs.splitlines():
                if match in line:
                    evidence.append(line.strip())
                    break
        
        # Calculate confidence based on number and type of matches
        confidence = min(0.6 + (len(oom_matches) * 0.1), 0.9)
        return FailureType.MEMORY_EXCEEDED, confidence, evidence
    
    # Check for disk space issues
    disk_matches = re.findall(DISK_RE, combined_logs, re.IGNORECASE)
    if disk_matches:
        for match in disk_matches[:3]:
            for line in combined_logs.splitlines():
                if match in line:
                    evidence.append(line.strip())
                    break
        
        confidence = min(0.6 + (len(disk_matches) * 0.1), 0.9)
        return FailureType.DISK_SPACE_EXCEEDED, confidence, evidence
    
    # Check for dependency issues
    dep_matches = re.findall(DEP_RE, combined_logs, re.IGNORECASE)
    if dep_matches:
        for match in dep_matches[:3]:
            for line in combined_logs.splitlines():
                if match in line:
                    evidence.append(line.strip())
                    break
        
        confidence = min(0.6 + (len(dep_matches) * 0.1), 0.9)
        return FailureType.DEPENDENCY_ERROR, confidence, evidence
    
    # If no clear pattern, check for other common error indicators
    if "error" in combined_logs.lower() or "exception" in combined_logs.lower():
        # Extract some error lines as evidence
        for line in combined_logs.splitlines():
            if "error" in line.lower() or "exception" in line.lower():
                evidence.append(line.strip())
                if len(evidence) >= 3:
                    break
        
        return FailureType.UNKNOWN, 0.4, evidence
    
    # No issues detected
    return FailureType.UNKNOWN, 0.1, evidence

def _simulate_diagnosis(failure_type_str: Optional[str] = None) -> Dict[str, Any]:
    """Simulate a diagnosis result for testing."""
    logger.info(f"Simulating run with failure type: {failure_type_str}")
    
    if failure_type_str:
        try:
            failure_type = FailureType(failure_type_str)
        except ValueError:
            logger.warning(f"Invalid failure type: {failure_type_str}. Using random type.")
            failure_type = random.choice(list(FailureType))
    else:
        failure_type = random.choice(list(FailureType))
    
    run_id = f"run_{int(time.time())}"
    logger.info(f"Simulated run {run_id} with failure type: {failure_type.value}")
    
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
        }
    else:
        # Generic unknown issue
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
        } 