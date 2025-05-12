"""
Tools for retrieving and analyzing Databricks logs.
"""

import time
from typing import Dict, Any, Optional

# Import the logging configuration
from src.agent_core.logging_config import get_logger
from src.agent_core.dbx_client import DbxClient, RunStatus

# Get logger for this module
logger = get_logger(__name__)

def get_logs(job_id: str, run_id: Optional[str] = None, simulate: bool = False, simulate_failure_type: Optional[str] = None) -> Dict[str, Any]:
    """
    Retrieves logs and execution details for a Databricks job run.
    
    This tool connects to the Databricks API to fetch stdout, stderr, and run metadata
    for a specific job execution. It serves as the first step in the monitoring 
    and diagnostic process.
    
    When to use:
    - To investigate why a Databricks job failed
    - As the first step before diagnosing issues
    - When needing to analyze execution details for monitoring
    
    Input JSON example:
    {
        "job_id": "123456",      // Required: The Databricks job ID
        "run_id": "987654",      // Optional: Specific run ID to analyze. If omitted, uses latest run.
        "simulate": false,       // Optional: If true, returns simulated logs instead of connecting to Databricks
        "simulate_failure_type": "memory_exceeded" // Optional: Type of failure to simulate (memory_exceeded, disk_space_exceeded, dependency_error)
    }
    
    Output JSON example:
    {
        "job_id": "123456",
        "run_id": "987654",
        "status": "FAILED",      // Could be "SUCCESS", "FAILED", "RUNNING", etc.
        "duration_seconds": 245,
        "start_time": 1625176800000,
        "end_time": 1625177045000,
        "logs": {
            "stdout": "Starting job execution...\nProcessing data...\n",
            "stderr": "ERROR: java.lang.OutOfMemoryError: Java heap space"
        },
        "timestamp": 1625177050.123
    }
    
    Error output example:
    {
        "job_id": "123456",
        "run_id": "987654",
        "status": "ERROR",
        "error": "Failed to connect to Databricks API: 401 Unauthorized",
        "logs": {
            "stdout": "",
            "stderr": "Error retrieving logs: Failed to connect to Databricks API: 401 Unauthorized"
        },
        "timestamp": 1625177050.123
    }
    """
    logger.info(f"Getting logs for job {job_id}" + (f", run {run_id}" if run_id else ""))
    
    # Check if we're in simulation mode
    if simulate:
        return _simulate_logs(job_id, run_id, simulate_failure_type)
    
    # Initialize Databricks client
    dbx_client = DbxClient()
    
    try:
        # Get the run ID if not provided (use latest run)
        if not run_id:
            run_id = dbx_client.get_latest_run_id(job_id)
            logger.info(f"Using latest run ID: {run_id}")
            
        # Get the run information
        run_info = dbx_client.get_run_info(run_id)
        
        # Get the run logs
        logs = dbx_client.get_run_logs(run_id)
        
        # Combine run info and logs
        result = {
            "job_id": job_id,
            "run_id": run_id,
            "status": run_info.get("status", "UNKNOWN"),
            "duration_seconds": run_info.get("duration_seconds", 0),
            "start_time": run_info.get("start_time", 0),
            "end_time": run_info.get("end_time", 0),
            "logs": logs,
            "timestamp": time.time()
        }
        
        logger.info(f"Successfully retrieved logs for job {job_id}, run {run_id}")
        return result
        
    except Exception as e:
        logger.error(f"Error getting logs for job {job_id}, run {run_id}: {e}", exc_info=True)
        # Return a minimal result with the error
        return {
            "job_id": job_id,
            "run_id": run_id,
            "status": "ERROR",
            "error": str(e),
            "logs": {"stdout": "", "stderr": f"Error retrieving logs: {str(e)}"},
            "timestamp": time.time()
        }

def _simulate_logs(job_id: str, run_id: Optional[str] = None, simulate_failure_type: Optional[str] = None) -> Dict[str, Any]:
    """Simulate logs for a Databricks job run."""
    import random
    from src.tools.databricks_monitoring.diagnostic_tools import FailureType
    
    logger.info(f"Simulating logs for job {job_id} with failure type: {simulate_failure_type}")
    
    # Generate a run ID if one wasn't provided
    if not run_id:
        run_id = f"run_{int(time.time())}"
    
    # Determine the failure type
    if simulate_failure_type:
        try:
            failure_type = FailureType(simulate_failure_type)
        except ValueError:
            logger.warning(f"Invalid failure type: {simulate_failure_type}. Using random type.")
            failure_type = random.choice(list(FailureType))
    else:
        failure_type = random.choice(list(FailureType))
    
    # Common log entries that appear in most runs
    common_stdout = [
        f"Starting job execution for {job_id}",
        "Loading dependencies...",
        "Initializing Spark session...",
        "Driver: SparkSession available as 'spark'.",
        "Processing input data...",
        "Reading data from source...",
        "Applying transformations...",
        "Data processing progress: 25%",
        "Data processing progress: 50%",
        "Data processing progress: 75%"
    ]
    
    # Generate detailed logs based on failure type
    if failure_type == FailureType.MEMORY_EXCEEDED:
        # Memory error patterns
        memory_errors = [
            # Java Heap Space
            {
                "stdout": common_stdout + [
                    "Joining large datasets...",
                    "Executing broadcast join operation...",
                    "WARNING: Memory usage is high (85%)",
                    "WARNING: GC overhead increasing",
                    "Processing large join operation..."
                ],
                "stderr": [
                    "WARN  MemoryStore: Not enough memory to cache broadcast in memory! (computed=1024.0 MB, available=512.0 MB)",
                    "ERROR SparkContext: Error initializing broadcast",
                    "java.lang.OutOfMemoryError: Java heap space",
                    "at org.apache.spark.sql.execution.joins.BroadcastHashJoinExec.executeBroadcast(BroadcastHashJoinExec.scala:163)",
                    "at org.apache.spark.sql.execution.joins.BroadcastHashJoinExec.doExecuteBroadcast(BroadcastHashJoinExec.scala:98)",
                    "at org.apache.spark.sql.execution.SparkPlan.executeQuery(SparkPlan.scala:155)",
                    "at org.apache.spark.sql.execution.SparkPlan.execute(SparkPlan.scala:128)",
                    "at org.apache.spark.sql.execution.joins.BroadcastHashJoinExec.doExecute(BroadcastHashJoinExec.scala:107)"
                ]
            },
            # Driver Out of Memory
            {
                "stdout": common_stdout + [
                    "Collecting results to driver...",
                    "WARNING: Large result set being collected to driver",
                    "Attempting to collect results..."
                ],
                "stderr": [
                    "WARN  TaskSetManager: Stage 3 contains a task of very large size (370 KB). The maximum recommended task size is 100 KB.",
                    "ERROR JobProgressReporter: Exception while running job: Driver: Out of Memory",
                    "org.apache.spark.SparkException: Job aborted due to stage failure: Total size of serialized results of 67 tasks (1024.0 MB) is bigger than spark.driver.maxResultSize (1024.0 MB)",
                    "at org.apache.spark.scheduler.DAGScheduler.failJobAndIndependentStages(DAGScheduler.scala:2454)",
                    "at org.apache.spark.scheduler.DAGScheduler.$anonfun$abortStage$2(DAGScheduler.scala:2403)",
                    "at org.apache.spark.scheduler.DAGScheduler.$anonfun$abortStage$2$adapted(DAGScheduler.scala:2402)",
                    "at scala.collection.mutable.ResizableArray.foreach(ResizableArray.scala:62)"
                ]
            },
            # GC Overhead Limit
            {
                "stdout": common_stdout + [
                    "Processing complex transformations...",
                    "WARNING: Garbage collection taking significant time",
                    "WARNING: Memory pressure increasing",
                    "Executing complex string manipulations..."
                ],
                "stderr": [
                    "WARN  TaskSetManager: Lost task 5.3 in stage 2.0: java.lang.OutOfMemoryError: GC overhead limit exceeded",
                    "java.lang.OutOfMemoryError: GC overhead limit exceeded",
                    "at java.util.Arrays.copyOfRange(Arrays.java:3664)",
                    "at java.lang.String.<init>(String.java:207)",
                    "at org.apache.spark.sql.catalyst.expressions.GeneratedClass$GeneratedIteratorForCodegenStage1.processNext(Unknown Source)",
                    "at org.apache.spark.sql.execution.BufferedRowIterator.hasNext(BufferedRowIterator.java:43)",
                    "at org.apache.spark.sql.execution.WholeStageCodegenExec$$anonfun$13.apply(WholeStageCodegenExec.scala:636)"
                ]
            }
        ]
        
        selected_error = random.choice(memory_errors)
        stdout = "\n".join(selected_error["stdout"])
        stderr = "\n".join(selected_error["stderr"])
        status = "FAILED"
        
    elif failure_type == FailureType.DISK_SPACE_EXCEEDED:
        # Disk space error patterns
        disk_errors = [
            # No Space Left on Device
            {
                "stdout": common_stdout + [
                    "Writing shuffle data...",
                    "Executing large shuffle operation...",
                    "WARNING: Disk usage is high",
                    "Attempting to write shuffle output..."
                ],
                "stderr": [
                    "ERROR StorageManager: Error writing data to local disk",
                    "java.io.IOException: No space left on device",
                    "at java.io.FileOutputStream.writeBytes(Native Method)",
                    "at java.io.FileOutputStream.write(FileOutputStream.java:326)",
                    "at org.apache.spark.storage.DiskBlockObjectWriter.write(DiskBlockObjectWriter.scala:233)",
                    "at org.apache.spark.storage.DiskBlockObjectWriter.write$mcB$sp(DiskBlockObjectWriter.scala:501)",
                    "at org.apache.spark.shuffle.sort.ShuffleExternalSorter.writeSortedFile(ShuffleExternalSorter.java:218)"
                ]
            },
            # Disk Quota Exceeded
            {
                "stdout": common_stdout + [
                    "Writing output data...",
                    "Saving partitioned data to storage...",
                    "WARNING: Approaching disk quota limits",
                    "Writing file 43 of 120..."
                ],
                "stderr": [
                    "ERROR FileFormatWriter: Error while writing output files",
                    "java.io.IOException: Disk quota exceeded",
                    "at java.io.UnixFileSystem.createFileExclusively(Native Method)",
                    "at java.io.File.createNewFile(File.java:1012)",
                    "at org.apache.hadoop.fs.RawLocalFileSystem.create(RawLocalFileSystem.java:297)",
                    "at org.apache.hadoop.fs.LocalFileSystem.create(LocalFileSystem.java:90)",
                    "at org.apache.hadoop.fs.FileSystem.create(FileSystem.java:935)"
                ]
            }
        ]
        
        selected_error = random.choice(disk_errors)
        stdout = "\n".join(selected_error["stdout"])
        stderr = "\n".join(selected_error["stderr"])
        status = "FAILED"
        
    elif failure_type == FailureType.DEPENDENCY_ERROR:
        # Dependency error patterns
        dependency_errors = [
            # Class Not Found
            {
                "stdout": common_stdout + [
                    "Initializing Delta Lake...",
                    "Attempting to create DeltaTable instance...",
                    "Loading table schema..."
                ],
                "stderr": [
                    "ERROR SparkContext: Error initializing DeltaTable",
                    "java.lang.ClassNotFoundException: org.apache.spark.sql.delta.DeltaTable",
                    "at java.net.URLClassLoader.findClass(URLClassLoader.java:382)",
                    "at java.lang.ClassLoader.loadClass(ClassLoader.java:418)",
                    "at java.lang.ClassLoader.loadClass(ClassLoader.java:351)",
                    "at org.apache.spark.sql.delta.sources.DeltaDataSource.createRelation(DeltaDataSource.scala:124)",
                    "at org.apache.spark.sql.execution.datasources.DataSource.resolveRelation(DataSource.scala:399)"
                ]
            },
            # Module Not Found
            {
                "stdout": common_stdout + [
                    "Importing Python libraries...",
                    "Setting up data processing pipeline...",
                    "Attempting to use PyArrow filesystem..."
                ],
                "stderr": [
                    "ERROR PythonRunner: Exception while running Python code",
                    "Traceback (most recent call last):",
                    "  File \"<command-4587>\", line 14, in <module>",
                    "    from pyarrow import fs",
                    "ModuleNotFoundError: No module named 'pyarrow.fs'",
                    "at org.apache.spark.api.python.BasePythonRunner$ReaderIterator.handlePythonException(PythonRunner.scala:545)",
                    "at org.apache.spark.api.python.PythonRunner$$anon$3.read(PythonRunner.scala:702)"
                ]
            }
        ]
        
        selected_error = random.choice(dependency_errors)
        stdout = "\n".join(selected_error["stdout"])
        stderr = "\n".join(selected_error["stderr"])
        status = "FAILED"
        
    else:
        # Unknown error types
        unknown_errors = [
            # Generic Failure
            {
                "stdout": common_stdout + [
                    "Processing data...",
                    "Executing job steps...",
                    "WARNING: Unexpected behavior in processing module"
                ],
                "stderr": [
                    "ERROR SparkContext: Job failed with exit code 1",
                    "ERROR: Command failed with non-zero exit code",
                    "at org.apache.spark.deploy.SparkSubmit.doSubmit(SparkSubmit.scala:90)",
                    "at org.apache.spark.deploy.SparkSubmit.submit(SparkSubmit.scala:78)"
                ]
            },
            # Timeout Error
            {
                "stdout": common_stdout + [
                    "Running long-running transformation...",
                    "WARNING: Operation taking longer than expected",
                    "Stage 4 still processing after 45 minutes..."
                ],
                "stderr": [
                    "ERROR Executor: Stage exceeded timeout",
                    "org.apache.spark.SparkException: Job aborted due to stage failure: Stage 4 was cancelled because it ran for longer than the configured timeout of 3600000 ms",
                    "at org.apache.spark.scheduler.DAGScheduler.failJobAndIndependentStages(DAGScheduler.scala:2454)",
                    "at org.apache.spark.scheduler.DAGScheduler.$anonfun$abortStage$2(DAGScheduler.scala:2403)"
                ]
            }
        ]
        
        selected_error = random.choice(unknown_errors)
        stdout = "\n".join(selected_error["stdout"])
        stderr = "\n".join(selected_error["stderr"])
        status = "FAILED"
    
    # Generate a simulated run result
    start_time = int(time.time() * 1000) - 3600000  # 1 hour ago
    end_time = int(time.time() * 1000) - 60000  # 1 minute ago
    duration = (end_time - start_time) // 1000  # in seconds
    
    return {
        "job_id": job_id,
        "run_id": run_id,
        "status": status,
        "duration_seconds": duration,
        "start_time": start_time,
        "end_time": end_time,
        "logs": {
            "stdout": stdout,
            "stderr": stderr
        },
        "timestamp": time.time(),
        "simulated": True,
        "simulate_failure_type": failure_type.value
    } 