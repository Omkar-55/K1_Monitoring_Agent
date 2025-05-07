"""
Tools for diagnosing issues in Databricks logs.
"""

import random
import time
import json
import os
import re
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple

# Import the logging configuration
from src.agent_core.logging_config import get_logger

# Import the Databricks API client
from src.tools.databricks_api_tools import DatabricksAPIClient

# Get logger for this module
logger = get_logger(__name__)

# Regex patterns for issue detection
OOM_RE = r"OutOfMemoryError|GC overhead limit exceeded"
QUOTA_RE = r"AZURE_QUOTA_EXCEEDED_EXCEPTION|QuotaExceeded"
TIMEOUT_RE = r"(Connection refused|RPC timed out|unreachable during run)"
LIB_RE = r"Library resolution failed"

class FailureType(str, Enum):
    """Types of failures that can be diagnosed."""
    MEMORY_EXCEEDED = "memory_exceeded"
    DISK_SPACE_EXCEEDED = "disk_space_exceeded"
    DEPENDENCY_ERROR = "dependency_error"
    QUOTA_EXCEEDED = "quota_exceeded"
    TIMEOUT = "timeout"
    DATA_SKEW = "data_skew"
    UNKNOWN = "unknown"

async def diagnose_with_ai(logs_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Diagnose issues in Databricks logs using Azure OpenAI.
    
    Args:
        logs_data: The logs data to analyze
        
    Returns:
        A diagnosis result with issue_type and reasoning
    """
    logger.info("Diagnosing Databricks logs with AI")
    
    try:
        # Import Azure OpenAI package
        from azure.openai import AsyncAzureOpenAI
        
        # Setup OpenAI client
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2023-05-15")
        deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4")
        
        if not api_key or not endpoint:
            logger.warning("Azure OpenAI credentials not available, falling back to pattern matching")
            return diagnose_pattern_matching(logs_data)
            
        # Initialize the client
        client = AsyncAzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=endpoint
        )
        
        # Extract the logs
        logs = logs_data.get("logs", {})
        stdout = logs.get("stdout", "")
        stderr = logs.get("stderr", "")
        
        # Create the system and user messages
        system_message = """You are a Databricks diagnostic expert. Your task is to analyze logs from Databricks 
        jobs and identify the root cause of issues. Categorize the issue into one of these specific types:
        1. memory_exceeded - When there are memory-related errors (OutOfMemoryError, GC overhead limit exceeded)
        2. disk_space_exceeded - When there are disk space issues (No space left on device)
        3. dependency_error - When there are missing packages or libraries (ModuleNotFoundError, ClassNotFoundException)
        4. quota_exceeded - When Azure quota limits are hit (AZURE_QUOTA_EXCEEDED_EXCEPTION, QuotaExceeded)
        5. timeout - When connections time out (Connection refused, RPC timed out, unreachable during run)
        6. data_skew - When data skew causes performance issues
        7. unknown - When the issue doesn't fit into the above categories
        
        Provide your analysis with a specific issue_type and detailed reasoning."""
        
        user_message = f"""Analyze these Databricks job logs:
        
        STDOUT:
        {stdout}
        
        STDERR:
        {stderr}
        
        Status: {logs_data.get('status', 'UNKNOWN')}
        
        Identify the issue type and explain your reasoning in detail.
        """
        
        # Get response from Azure OpenAI
        response = await client.chat.completions.create(
            deployment_name=deployment,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            temperature=0.3,
            max_tokens=1000
        )
        
        if not response.choices:
            logger.error("No response from Azure OpenAI")
            return diagnose_pattern_matching(logs_data)
            
        analysis_text = response.choices[0].message.content
        
        # Extract the issue type from the response
        issue_type = extract_issue_type(analysis_text)
        
        logger.info(f"AI diagnosed issue: {issue_type}")
        
        return {
            "issue_type": issue_type,
            "reasoning": analysis_text,
            "logs_analyzed": {
                "stdout_length": len(stdout),
                "stderr_length": len(stderr)
            },
            "method": "ai"
        }
        
    except Exception as e:
        logger.error(f"Error using AI for diagnosis: {e}", exc_info=True)
        logger.warning("Falling back to pattern matching diagnosis")
        return diagnose_pattern_matching(logs_data)

def extract_issue_type(text: str) -> str:
    """Extract issue type from AI analysis text."""
    text_lower = text.lower()
    
    if "issue_type" in text_lower and ":" in text_lower:
        # Try to extract explicit issue_type field
        for line in text_lower.split('\n'):
            if "issue_type" in line and ":" in line:
                value = line.split(":", 1)[1].strip()
                for issue_type in FailureType:
                    if issue_type.value in value:
                        return issue_type.value
    
    # If no explicit issue_type field, infer from content
    if re.search(OOM_RE, text_lower, re.IGNORECASE) or ("memory" in text_lower and ("error" in text_lower or "exceed" in text_lower)):
        return FailureType.MEMORY_EXCEEDED
    elif "disk" in text_lower and ("space" in text_lower or "quota" in text_lower):
        return FailureType.DISK_SPACE_EXCEEDED
    elif any(term in text_lower for term in ["dependency", "module", "package", "import", "library"]) or re.search(LIB_RE, text_lower, re.IGNORECASE):
        return FailureType.DEPENDENCY_ERROR
    elif re.search(QUOTA_RE, text_lower, re.IGNORECASE):
        return FailureType.QUOTA_EXCEEDED
    elif re.search(TIMEOUT_RE, text_lower, re.IGNORECASE):
        return FailureType.TIMEOUT
    elif "skew" in text_lower or "imbalance" in text_lower:
        return FailureType.DATA_SKEW
    else:
        return FailureType.UNKNOWN

def diagnose_pattern_matching(logs_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Diagnose issues in Databricks logs using pattern matching.
    This is the fallback method when AI is not available.
    
    Args:
        logs_data: The logs data to analyze
        
    Returns:
        A diagnosis result with issue_type and reasoning
    """
    logger.info("Diagnosing Databricks logs using pattern matching")
    
    # Extract the logs
    logs = logs_data.get("logs", {})
    stdout = logs.get("stdout", "")
    stderr = logs.get("stderr", "")
    
    # Check for memory issues using regex
    if re.search(OOM_RE, stderr, re.IGNORECASE):
        issue_type = FailureType.MEMORY_EXCEEDED
        reasoning = "Job failed due to insufficient memory. Found OutOfMemoryError in logs."
    
    # Check for disk space issues
    elif "No space left on device" in stderr or "Disk quota exceeded" in stderr:
        issue_type = FailureType.DISK_SPACE_EXCEEDED
        reasoning = "Job failed due to insufficient disk space. Found disk space error in logs."
    
    # Check for dependency issues
    elif "ModuleNotFoundError" in stderr or "ImportError" in stderr or "ClassNotFoundException" in stderr or re.search(LIB_RE, stderr, re.IGNORECASE):
        issue_type = FailureType.DEPENDENCY_ERROR
        reasoning = "Job failed due to missing dependencies. Found import or module errors in logs."
    
    # Check for quota issues
    elif re.search(QUOTA_RE, stderr, re.IGNORECASE):
        issue_type = FailureType.QUOTA_EXCEEDED
        reasoning = "Job failed due to Azure quota limits being exceeded."
    
    # Check for timeout issues
    elif re.search(TIMEOUT_RE, stderr, re.IGNORECASE):
        issue_type = FailureType.TIMEOUT
        reasoning = "Job failed due to connection timeouts or unreachable components."
    
    # Unknown issue
    else:
        issue_type = FailureType.UNKNOWN
        reasoning = "Could not identify a specific issue type from the logs."
    
    logger.info(f"Pattern matching diagnosed issue: {issue_type}")
    
    return {
        "issue_type": issue_type,
        "reasoning": reasoning,
        "logs_analyzed": {
            "stdout_length": len(stdout),
            "stderr_length": len(stderr)
        },
        "method": "pattern_matching"
    }

def diagnose_with_metrics(logs_data: Dict[str, Any], cluster_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Enhance diagnosis with cluster metrics if available.
    
    Args:
        logs_data: The logs data to analyze
        cluster_id: Optional cluster ID to fetch metrics for
        
    Returns:
        Enhanced diagnosis with metrics
    """
    # First do basic diagnosis
    basic_diagnosis = diagnose_pattern_matching(logs_data)
    issue_type = basic_diagnosis["issue_type"]
    
    # If we have a cluster ID, try to fetch metrics
    if cluster_id:
        try:
            # Create Databricks API client
            client = DatabricksAPIClient()
            
            # Get cluster metrics
            metrics = client.get_cluster_metrics(cluster_id)
            
            if metrics.get("status") == "success":
                # Look for specific metrics that might indicate issues
                memory_metrics = metrics.get("metrics", {}).get("memory", {})
                
                # Check for memory pressure
                driver_mem = memory_metrics.get("driver", {}).get("used_percent", 0)
                worker_mems = [w.get("used_percent", 0) for w in memory_metrics.get("workers", [])]
                
                if driver_mem > 90 or any(mem > 90 for mem in worker_mems):
                    # Confirm or update memory diagnosis
                    issue_type = FailureType.MEMORY_EXCEEDED
                    enhanced_reasoning = (
                        f"High memory usage detected: Driver at {driver_mem:.1f}%, "
                        f"Workers max at {max(worker_mems) if worker_mems else 0:.1f}%. "
                        f"This confirms memory pressure as the likely cause."
                    )
                    
                    # Add metrics to the diagnosis
                    return {
                        **basic_diagnosis,
                        "issue_type": issue_type,
                        "reasoning": enhanced_reasoning,
                        "metrics": metrics.get("metrics"),
                        "method": "metrics_enhanced"
                    }
                    
                # Check for potential data skew if timeout or unknown issue
                if issue_type in [FailureType.TIMEOUT, FailureType.UNKNOWN]:
                    # Check for imbalanced worker memory usage
                    if worker_mems and max(worker_mems) > 2 * min(worker_mems) and max(worker_mems) > 75:
                        return {
                            **basic_diagnosis,
                            "issue_type": FailureType.DATA_SKEW,
                            "reasoning": f"Potential data skew detected. Worker memory usage varies widely from {min(worker_mems):.1f}% to {max(worker_mems):.1f}%, suggesting imbalanced data processing.",
                            "metrics": metrics.get("metrics"),
                            "method": "metrics_enhanced"
                        }
                
                # Add metrics to the diagnosis even if they don't change the issue type
                return {
                    **basic_diagnosis,
                    "metrics": metrics.get("metrics"),
                    "method": "metrics_enhanced"
                }
        except Exception as e:
            logger.error(f"Error fetching metrics for diagnosis: {e}", exc_info=True)
    
    # Return basic diagnosis if no metrics available
    return basic_diagnosis

def diagnose(logs_data: Dict[str, Any], cluster_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Diagnose issues in Databricks logs.
    This function now attempts to use AI first, then falls back to pattern matching.
    If cluster_id is provided, it will also attempt to enhance the diagnosis with metrics.
    
    Args:
        logs_data: The logs data to analyze
        cluster_id: Optional cluster ID to get metrics for
        
    Returns:
        A diagnosis result with issue_type and reasoning
    """
    import asyncio
    
    try:
        # Try to diagnose with AI
        loop = asyncio.get_event_loop()
    except RuntimeError:
        # If no event loop is set, create a new one
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
    try:
        # Use AI diagnosis if possible
        ai_diagnosis = loop.run_until_complete(diagnose_with_ai(logs_data))
        
        # If cluster_id is provided, try to enhance with metrics
        if cluster_id:
            return enhance_with_metrics(ai_diagnosis, cluster_id)
        
        return ai_diagnosis
    except Exception as e:
        logger.error(f"Error in AI diagnosis: {e}", exc_info=True)
        
        # Fall back to pattern matching with metrics if possible
        if cluster_id:
            return diagnose_with_metrics(logs_data, cluster_id)
        
        # Fall back to simple pattern matching
        return diagnose_pattern_matching(logs_data)

def enhance_with_metrics(diagnosis: Dict[str, Any], cluster_id: str) -> Dict[str, Any]:
    """
    Enhance an existing diagnosis with cluster metrics.
    
    Args:
        diagnosis: The existing diagnosis
        cluster_id: Cluster ID to get metrics for
        
    Returns:
        Enhanced diagnosis with metrics
    """
    try:
        # Create Databricks API client
        client = DatabricksAPIClient()
        
        # Get cluster metrics
        metrics = client.get_cluster_metrics(cluster_id)
        
        if metrics.get("status") == "success":
            # Add metrics to the diagnosis
            diagnosis["metrics"] = metrics.get("metrics")
            diagnosis["method"] = f"{diagnosis.get('method', 'unknown')}_with_metrics"
    except Exception as e:
        logger.error(f"Error enhancing diagnosis with metrics: {e}", exc_info=True)
    
    return diagnosis

def simulate_run(failure_type: Optional[str] = None) -> Dict[str, Any]:
    """
    Generate simulated logs for testing.
    
    Args:
        failure_type: The type of failure to simulate
        
    Returns:
        Simulated logs data
    """
    logger.info(f"Simulating run with failure type: {failure_type}")
    
    # Generate a random run ID
    run_id = f"run_{random.randint(10000, 99999)}"
    job_id = f"job_{random.randint(1000, 9999)}"
    
    # Default to a random failure type if none specified
    if not failure_type:
        failure_types = list(FailureType)
        failure_type = random.choice(failure_types)
    
    # Ensure failure_type is a string
    if isinstance(failure_type, FailureType):
        failure_type = failure_type.value
    
    # Generate appropriate logs for the failure type
    stdout = "Starting Databricks job execution...\n"
    stdout += "Loading data...\n"
    stdout += "Processing data...\n"
    
    stderr = ""
    
    if failure_type == FailureType.MEMORY_EXCEEDED:
        stdout += "Processing large dataset...\n"
        stderr += "WARNING: Memory usage is high\n"
        stderr += "ERROR: java.lang.OutOfMemoryError: Java heap space\n"
        stderr += "  at org.apache.spark.sql.execution.aggregate.HashAggregateExec.doExecute(HashAggregateExec.scala:115)\n"
        stderr += "  at org.apache.spark.sql.execution.SparkPlan.execute(SparkPlan.scala:180)\n"
    
    elif failure_type == FailureType.DISK_SPACE_EXCEEDED:
        stdout += "Writing results to disk...\n"
        stderr += "WARNING: Disk usage is high\n"
        stderr += "ERROR: java.io.IOException: No space left on device\n"
        stderr += "  at java.io.FileOutputStream.writeBytes(Native Method)\n"
        stderr += "  at java.io.FileOutputStream.write(FileOutputStream.java:326)\n"
    
    elif failure_type == FailureType.DEPENDENCY_ERROR:
        stdout += "Importing libraries...\n"
        stderr += "ERROR: ModuleNotFoundError: No module named 'pandas'\n"
        stderr += "  at <frozen importlib._bootstrap>(219)._call_with_frames_removed\n"
        stderr += "  at <frozen importlib._bootstrap_external>(728).exec_module\n"
        stderr += "Library resolution failed"
    
    elif failure_type == FailureType.QUOTA_EXCEEDED:
        stdout += "Allocating Azure resources...\n"
        stderr += "ERROR: AZURE_QUOTA_EXCEEDED_EXCEPTION: Azure quota exceeded\n"
        stderr += "  at com.microsoft.azure.databricks.quota.QuotaManager.allocateResource\n"
        
    elif failure_type == FailureType.TIMEOUT:
        stdout += "Connecting to remote service...\n"
        stderr += "WARNING: Connection attempt timed out\n"
        stderr += "ERROR: RPC timed out after 60000 ms\n"
        stderr += "  at org.apache.spark.rpc.RpcTimeout.org$apache$spark$rpc$RpcTimeout\n"
        
    elif failure_type == FailureType.DATA_SKEW:
        stdout += "Executing join operation...\n"
        stdout += "Stage 3: [====>                                                ] 5/42 tasks\n"
        stderr += "WARNING: Possible skew in join detected. Partition 3 has 10x more data.\n"
        stderr += "  at org.apache.spark.sql.execution.ShuffledHashJoinExec.doExecute\n"
    
    else:
        stdout += "Executing job...\n"
        stderr += "ERROR: Unknown error occurred\n"
        stderr += "  at com.databricks.backend.common.rpc.InternalDriverConnectionProvider.lambda$getOrCreate$1(InternalDriverConnectionProvider.scala:102)\n"
    
    # Simulate job status
    status = "FAILED"
    
    # Simulate run duration
    duration_seconds = random.randint(60, 3600)
    
    # Return the simulated logs
    logs_data = {
        "run_id": run_id,
        "job_id": job_id,
        "status": status,
        "duration_seconds": duration_seconds,
        "logs": {
            "stdout": stdout,
            "stderr": stderr
        }
    }
    
    logger.info(f"Simulated run {run_id} with failure type {failure_type}")
    
    return logs_data 