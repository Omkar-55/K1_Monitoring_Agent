"""
Azure OpenAI Integration Tools for K1 Monitoring Agent

This module contains tools for interacting with Azure OpenAI services.
"""
from typing import Dict, List, Optional, Any, Union
import os
import logging
import json
import asyncio

# Set up logger
logger = logging.getLogger(__name__)

class AzureOpenAITools:
    """Tools for interacting with Azure OpenAI services."""
    
    def __init__(self, 
                api_key: Optional[str] = None,
                endpoint: Optional[str] = None,
                api_version: Optional[str] = None,
                deployment: Optional[str] = None):
        """Initialize the Azure OpenAI tools.
        
        Args:
            api_key: Azure OpenAI API key
            endpoint: Azure OpenAI endpoint
            api_version: Azure OpenAI API version
            deployment: Azure OpenAI deployment name
        """
        logger.info("Initializing Azure OpenAI tools")
        self.api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
        self.endpoint = endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        self.api_version = api_version or os.getenv("AZURE_OPENAI_API_VERSION", "2023-05-15")
        self.deployment = deployment or os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1")
        
        # Check if required credentials are available
        self._has_credentials = bool(self.api_key and self.endpoint)
        if not self._has_credentials:
            logger.warning("Azure OpenAI credentials not found. Some features will be unavailable.")
        
        # Import Azure OpenAI package
        try:
            from azure.openai import AsyncAzureOpenAI
            self.AsyncAzureOpenAI = AsyncAzureOpenAI
            self._client = None
            self._init_client()
            logger.info("Azure OpenAI client successfully imported")
        except ImportError:
            logger.error("Failed to import Azure OpenAI package. Install with: pip install azure-openai")
            self.AsyncAzureOpenAI = None
            self._client = None
            
    def _init_client(self):
        """Initialize the Azure OpenAI client."""
        if not self._has_credentials or not self.AsyncAzureOpenAI:
            logger.error("Cannot initialize Azure OpenAI client - missing credentials or package")
            return
            
        try:
            self._client = self.AsyncAzureOpenAI(
                api_key=self.api_key,
                api_version=self.api_version,
                azure_endpoint=self.endpoint
            )
            logger.info("Azure OpenAI client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Azure OpenAI client: {e}")
            self._client = None
            
    def is_available(self) -> bool:
        """Check if Azure OpenAI is available.
        
        Returns:
            True if Azure OpenAI is available, False otherwise
        """
        return self._client is not None
        
    async def analyze_logs(self, logs: List[Dict[str, Any]], max_logs: int = 10) -> Dict[str, Any]:
        """
        Analyzes Databricks log entries using Azure OpenAI to identify patterns, anomalies, and issues.
        
        This tool leverages AI to perform deep analysis of log content that may be difficult
        to identify with traditional pattern matching, extracting insights and potential 
        root causes from log data.
        
        When to use:
        - When standard pattern matching doesn't identify clear issues
        - For complex logs that need semantic understanding
        - To get advanced insights beyond simple error code matching
        
        Input JSON example:
        {
            "logs": [                          // Required: List of log entries to analyze
                {
                    "timestamp": "2023-04-15T08:25:31Z",
                    "message": "Starting job execution",
                    "level": "INFO"
                },
                {
                    "timestamp": "2023-04-15T08:25:45Z",
                    "message": "WARNING: Memory usage is high (85%)",
                    "level": "WARN"
                },
                {
                    "timestamp": "2023-04-15T08:26:12Z",
                    "message": "ERROR: java.lang.OutOfMemoryError: Java heap space",
                    "level": "ERROR"
                }
            ],
            "max_logs": 10                     // Optional: Maximum number of logs to include (default: 10)
        }
        
        Output JSON example:
        {
            "analysis": "The logs show a memory utilization issue that led to a Java heap space error. The memory usage was already high (85%) before the crash, indicating insufficient heap allocation for the data processing workload. Consider increasing the driver/executor memory or optimizing the query to reduce memory requirements.",
            "log_count": 3,
            "truncated": false
        }
        
        Error output example:
        {
            "error": "Azure OpenAI client not available"
        }
        """
        logger.info(f"Analyzing {len(logs)} log entries")
        if not self.is_available():
            logger.error("Azure OpenAI client not available")
            return {"error": "Azure OpenAI client not available"}
            
        # Truncate logs if needed
        if len(logs) > max_logs:
            logs = logs[:max_logs]
            
        try:
            # Prepare the logs in a readable format
            log_text = "\n".join([
                f"[{log.get('timestamp', 'Unknown')}] {log.get('message', 'No message')}"
                for log in logs
            ])
            
            # Prepare the prompt
            prompt = [
                {"role": "system", "content": "You are a log analysis expert. Analyze these logs and identify patterns, anomalies, or errors."},
                {"role": "user", "content": f"Please analyze these logs:\n\n{log_text}\n\nWhat patterns, issues, or anomalies do you see?"}
            ]
            
            # Send request to Azure OpenAI
            response = await self._client.chat.completions.create(
                deployment_name=self.deployment,
                messages=prompt,
                temperature=0.7,
                max_tokens=800
            )
            
            # Extract and return the analysis
            if response.choices and len(response.choices) > 0:
                analysis = response.choices[0].message.content
                return {
                    "analysis": analysis,
                    "log_count": len(logs),
                    "truncated": len(logs) < max_logs
                }
            else:
                return {"error": "No response from Azure OpenAI"}
                
        except Exception as e:
            logger.error(f"Error analyzing logs: {e}")
            return {"error": str(e)}
            
    async def summarize_activity(self, activity: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize workspace activity data.
        
        Args:
            activity: Activity data to summarize
            
        Returns:
            Summarized activity data
        """
        logger.info("Summarizing workspace activity")
        if not self.is_available():
            logger.error("Azure OpenAI client not available")
            return {"error": "Azure OpenAI client not available"}
            
        try:
            # Convert activity data to a readable format
            activity_text = (
                f"Total activities: {activity.get('count', 0)}\n"
                f"Time period: Last {activity.get('days', 7)} days\n\n"
                f"Cluster activities: {activity.get('cluster_count', 0)}\n"
                f"Job activities: {activity.get('job_count', 0)}\n"
                f"Notebook activities: {activity.get('notebook_count', 0)}\n"
                f"Warehouse activities: {activity.get('warehouse_count', 0)}\n"
            )
            
            # Add some example activities if available
            for category in ['clusters', 'jobs', 'notebooks', 'warehouses']:
                items = activity.get(category, [])
                if items:
                    activity_text += f"\nExample {category} activities:\n"
                    for i, item in enumerate(items[:3]):
                        activity_text += f"- {item.get('action_type', 'Unknown')} by {item.get('user_name', 'Unknown')}\n"
            
            # Prepare the prompt
            prompt = [
                {"role": "system", "content": "You are a data analyst specializing in cloud workspace activity. Provide insights based on activity data."},
                {"role": "user", "content": f"Please analyze this Databricks workspace activity data and provide insights:\n\n{activity_text}\n\nWhat patterns or notable activity do you see?"}
            ]
            
            # Send request to Azure OpenAI
            response = await self._client.chat.completions.create(
                deployment_name=self.deployment,
                messages=prompt,
                temperature=0.7,
                max_tokens=800
            )
            
            # Extract and return the insights
            if response.choices and len(response.choices) > 0:
                insights = response.choices[0].message.content
                return {
                    "insights": insights,
                    "activity_summary": activity_text
                }
            else:
                return {"error": "No response from Azure OpenAI"}
                
        except Exception as e:
            logger.error(f"Error summarizing activity: {e}")
            return {"error": str(e)}
            
    async def generate_monitoring_report(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generates a comprehensive, AI-enhanced monitoring report for Databricks workspaces.
        
        This tool creates detailed reports that include:
        - Executive summary of findings
        - Key metrics and status overview
        - Identified issues and their severity
        - Actionable recommendations with rationale
        - Technical details in a structured format
        
        When to use:
        - For creating detailed reports at the end of monitoring cycles
        - To communicate technical findings to different stakeholders
        - When you need AI-enhanced insights and recommendations
        
        Input JSON example:
        {
            "data": {                         // Required: Monitoring data to include in report
                "workspace": "databricks-prod-analytics",
                "timestamp": "2023-04-15T15:30:00Z",
                "metrics": {
                    "jobs": {
                        "total": 125,
                        "failed": 3,
                        "success_rate": 97.6
                    },
                    "clusters": {
                        "total": 8,
                        "running": 5,
                        "terminated": 2,
                        "failed": 1
                    }
                },
                "issues": [
                    {
                        "type": "memory_exceeded",
                        "job_id": "123456",
                        "severity": "high",
                        "fix_status": "resolved"
                    }
                ]
            }
        }
        
        Output JSON example:
        {
            "report": "# Databricks Monitoring Report\n\n## Executive Summary\n\nThe monitoring period showed an overall healthy workspace with a 97.6% job success rate. One high-severity memory issue was identified and successfully resolved.\n\n## Metrics Overview\n\n- **Jobs**: 125 total, 3 failed (97.6% success rate)\n- **Clusters**: 8 total, 5 running, 2 terminated, 1 failed\n\n## Issues Identified\n\n### 1. Memory Exceeded (High Severity)\n- **Job ID**: 123456\n- **Status**: Resolved\n- **Details**: Job experienced Java heap space errors due to insufficient memory allocation\n\n## Recommendations\n\n1. Consider implementing memory usage monitoring alerts at 80% threshold\n2. Review the jobs with highest memory usage for optimization opportunities\n3. Schedule regular maintenance windows for cluster restarts\n\n## Technical Details\n\n[Additional technical metrics and details formatted in tables...]",
            "timestamp": "2023-04-15T15:30:00Z"
        }
        
        Error output example:
        {
            "error": "Azure OpenAI client not available"
        }
        """
        logger.info("Generating monitoring report")
        if not self.is_available():
            logger.error("Azure OpenAI client not available")
            return {"error": "Azure OpenAI client not available"}
            
        try:
            # Convert data to a structured format
            report_data = json.dumps(data, indent=2)
            
            # Prepare the prompt
            prompt = [
                {"role": "system", "content": "You are a monitoring report generator. Create well-structured, professional reports from monitoring data."},
                {"role": "user", "content": f"Please generate a comprehensive monitoring report based on this data:\n\n{report_data}\n\nThe report should include summary, key metrics, issues identified, and recommendations."}
            ]
            
            # Send request to Azure OpenAI
            response = await self._client.chat.completions.create(
                deployment_name=self.deployment,
                messages=prompt,
                temperature=0.5,
                max_tokens=1500
            )
            
            # Extract and return the report
            if response.choices and len(response.choices) > 0:
                report = response.choices[0].message.content
                return {
                    "report": report,
                    "timestamp": data.get("timestamp", "N/A")
                }
            else:
                return {"error": "No response from Azure OpenAI"}
                
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return {"error": str(e)} 