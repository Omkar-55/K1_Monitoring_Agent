"""
Databricks Monitoring Agent for the K1 Monitoring Agent platform.

This agent collects logs from Databricks workspaces, analyzes issues,
suggests and implements fixes, and verifies resolutions.
"""

import os
import json
import time
import asyncio
from typing import Dict, Any, List, Optional, Tuple, Union
from pydantic import BaseModel, Field
from enum import Enum, auto
from opentelemetry import trace

# Import the Azure Agents SDK components
try:
    import openai
    from azure.ai.agents import AgentTask, Assistant, OutOfDomainHandler
    from azure.ai.agents.actions import Action, AzureMonitoringAction
    from azure.ai.agents.models import AgentResponse, ResponseValidation
    from azure.ai.agents.safety import SafetyPolicy, ContentFilterResults
    AGENTS_SDK_AVAILABLE = True
except ImportError:
    AGENTS_SDK_AVAILABLE = False
    # Create dummy classes for type hinting
    class Assistant:
        pass
    
    class AgentResponse:
        content = ""
    
    class ResponseValidation:
        is_out_of_domain = False
        out_of_domain_score = 0.0
        reason = ""
    
    class ContentFilterResults:
        has_filtered_content = False
        harmful_content_score = 0.0
        harassment_content_score = 0.0
        hate_content_score = 0.0
        sexual_content_score = 0.0
        self_harm_content_score = 0.0

# Import our agent core components
from src.agent_core.logging_config import get_logger, setup_logging
from src.agent_core.dbx_client import DbxClient, RunStatus
from src.tools.databricks_monitoring import (
    get_logs,
    diagnose, 
    FailureType,
    suggest_fix, 
    apply_fix,
    verify,
    final_report
)

# Enable tracing
from src.agent_core import enable_tracing

# Initialize tracing
enable_tracing(service_name="databricks-monitoring-agent")

# Set up logging
setup_logging(log_level="info")
logger = get_logger(__name__)

# Get tracer for this module
tracer = trace.get_tracer(__name__)

class FixStatus(Enum):
    """Status of fix attempts."""
    NOT_STARTED = auto()
    IN_PROGRESS = auto()
    SUCCESSFUL = auto()
    FAILED = auto()
    VERIFICATION_PENDING = auto()

class MonitoringRequest(BaseModel):
    """Model for Databricks monitoring request."""
    job_id: str = Field(..., description="The Databricks job ID to monitor")
    run_id: Optional[str] = Field(None, description="A specific run ID to analyze (if None, uses latest)")
    workspace_url: Optional[str] = Field(None, description="The Databricks workspace URL")
    max_attempts: int = Field(5, description="Maximum number of fix attempts")
    simulate: bool = Field(False, description="Whether to simulate fix execution")
    simulate_failure_type: Optional[str] = Field(None, description="Simulated failure type for testing")

class MonitoringResponse(BaseModel):
    """Model for Databricks monitoring response."""
    job_id: str = Field(..., description="The Databricks job ID that was monitored")
    run_id: str = Field(..., description="The Databricks run ID that was analyzed")
    issue_detected: bool = Field(..., description="Whether an issue was detected")
    issue_type: Optional[str] = Field(None, description="The type of issue detected, if any")
    fix_attempts: int = Field(0, description="Number of fix attempts made")
    fix_successful: Optional[bool] = Field(None, description="Whether the fix was successful")
    reasoning: List[Dict[str, Any]] = Field(default_factory=list, description="The reasoning steps taken")
    report: Optional[str] = Field(None, description="Final report text")
    hallucination_detected: bool = Field(False, description="Whether hallucination was detected")
    safety_issues: bool = Field(False, description="Whether safety issues were detected")
    confidence: float = Field(0.0, description="Confidence in the solution")

class DatabricksMonitoringAgent:
    """
    Agent for monitoring Databricks workspaces, detecting issues,
    and automatically applying fixes.
    """
    
    def __init__(self):
        """Initialize the agent with necessary clients and configuration."""
        self.dbx_client = DbxClient()
        
        # Initialize Azure Agents SDK components if available
        if AGENTS_SDK_AVAILABLE:
            logger.info("Azure Agents SDK available, initializing components")
            self.assistant = self._setup_assistant()
            self.safety_policy = SafetyPolicy(
                max_harmful_content_score=0.5,
                max_harassment_content_score=0.5,
                max_hate_content_score=0.5,
                max_sexual_content_score=0.5,
                max_self_harm_content_score=0.5
            )
            self.out_of_domain_handler = OutOfDomainHandler()
        else:
            logger.warning("Azure Agents SDK not available, some features will be limited")
            self.assistant = None
            self.safety_policy = None
            self.out_of_domain_handler = None
        
        logger.info("Databricks Monitoring Agent initialized")
        
    def _setup_assistant(self) -> Optional[Any]:
        """Set up the Azure AI Assistant for the agent."""
        if not AGENTS_SDK_AVAILABLE:
            logger.warning("Cannot setup assistant: Azure Agents SDK not available")
            return None
            
        # Create actions for the assistant
        actions = [
            Action(
                name="diagnose_databricks_issue",
                description="Diagnose issues in Databricks logs",
                function=self._diagnose_wrapper
            ),
            Action(
                name="suggest_databricks_fix",
                description="Suggest a fix for a Databricks issue",
                function=self._suggest_fix_wrapper
            ),
            Action(
                name="apply_databricks_fix",
                description="Apply a fix to a Databricks issue",
                function=self._apply_fix_wrapper
            ),
            Action(
                name="verify_databricks_fix",
                description="Verify if a fix resolved the Databricks issue",
                function=self._verify_fix_wrapper
            ),
            Action(
                name="generate_report",
                description="Generate a final report for the monitoring session",
                function=self._generate_report_wrapper
            )
        ]
        
        # Create and configure the assistant
        assistant = Assistant(
            name="Databricks Monitoring Assistant",
            task=AgentTask.WORKSPACE_MANAGEMENT,
            actions=actions,
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4"),
            system_message="""You are a Databricks monitoring agent that helps diagnose and fix issues in Databricks workspaces. 
            You will analyze logs, suggest fixes, implement them, and verify if they resolve the issues.
            Be thorough in your analysis, and explain your reasoning at each step.
            Always verify that your suggestions are safe before applying them."""
        )
        
        return assistant
        
    async def process_request(self, request: MonitoringRequest) -> MonitoringResponse:
        """
        Process a monitoring request for a Databricks job.
        
        Args:
            request: The monitoring request
            
        Returns:
            The monitoring response with results
        """
        with tracer.start_as_current_span("databricks_agent.process_request") as span:
            span.set_attribute("job_id", request.job_id)
            if request.run_id:
                span.set_attribute("run_id", request.run_id)
            
            logger.info(f"Processing monitoring request for job {request.job_id}")
            
            # Initialize the response
            response = MonitoringResponse(
                job_id=request.job_id,
                run_id=request.run_id or "unknown",
                issue_detected=False,
                reasoning=[]
            )
            
            try:
                # Step 1: Get logs for the job
                with tracer.start_as_current_span("step1_get_logs"):
                    reasoning_step = {"step": "log_collection", "timestamp": time.time()}
                    
                    if request.simulate:
                        from src.tools.databricks_monitoring.diagnostic_tools import simulate_run
                        logs_data = simulate_run(failure_type=request.simulate_failure_type or "memory_exceeded")
                        logger.info(f"Using simulated run data for failure type: {request.simulate_failure_type or 'memory_exceeded'}")
                    else:
                        logs_data = get_logs(request.job_id, request.run_id)
                    
                    # Update response with run ID
                    response.run_id = logs_data.get("run_id", "unknown")
                    reasoning_step["result"] = f"Retrieved logs for run ID: {response.run_id}"
                    reasoning_step["logs_summary"] = self._summarize_logs(logs_data)
                    response.reasoning.append(reasoning_step)
                    
                    logger.info(f"Retrieved logs for run ID: {response.run_id}")
                
                # Step 2: Diagnose the issue
                with tracer.start_as_current_span("step2_diagnose"):
                    reasoning_step = {"step": "diagnosis", "timestamp": time.time()}
                    
                    # If available, use the Azure Agents SDK for diagnosis with hallucination detection
                    if self.assistant and AGENTS_SDK_AVAILABLE:
                        diagnosis_result, hallucination_info = await self._diagnose_with_assistant(logs_data)
                        issue_type = diagnosis_result.get("issue_type")
                        reasoning = diagnosis_result.get("reasoning", "")
                        response.hallucination_detected = hallucination_info.get("detected", False)
                        
                        reasoning_step["hallucination_check"] = hallucination_info
                    else:
                        # Use our built-in diagnostic tools
                        diagnosis_result = diagnose(logs_data)
                        issue_type = diagnosis_result.get("issue_type")
                        reasoning = diagnosis_result.get("reasoning", "")
                    
                    if issue_type:
                        response.issue_detected = True
                        response.issue_type = issue_type
                        reasoning_step["result"] = f"Diagnosed issue: {issue_type} - {reasoning}"
                        logger.info(f"Diagnosed issue: {issue_type} - {reasoning}")
                    else:
                        reasoning_step["result"] = "No issues detected"
                        logger.info("No issues detected in the Databricks run")
                    
                    response.reasoning.append(reasoning_step)
                
                # If no issue is detected, return early
                if not response.issue_detected:
                    response.report = "No issues detected in the Databricks job run."
                    return response
                
                # Step 3-5: Attempt fixes until success or max attempts reached
                fix_status = FixStatus.NOT_STARTED
                fix_attempts = 0
                
                while fix_status != FixStatus.SUCCESSFUL and fix_attempts < request.max_attempts:
                    fix_attempts += 1
                    
                    # Step 3: Suggest a fix
                    with tracer.start_as_current_span(f"step3_suggest_fix_attempt_{fix_attempts}"):
                        reasoning_step = {"step": "fix_suggestion", "timestamp": time.time(), "attempt": fix_attempts}
                        
                        # If available, use the Azure Agents SDK for fix suggestion with safety checks
                        if self.assistant and AGENTS_SDK_AVAILABLE:
                            fix_suggestion, safety_info = await self._suggest_fix_with_assistant(
                                response.issue_type, 
                                logs_data
                            )
                            fix_type = fix_suggestion.get("fix_type")
                            fix_params = fix_suggestion.get("parameters", {})
                            confidence = fix_suggestion.get("confidence", 0.0)
                            response.safety_issues = safety_info.get("issues_detected", False)
                            
                            reasoning_step["safety_check"] = safety_info
                            
                            # If safety issues detected, skip this fix
                            if response.safety_issues:
                                reasoning_step["result"] = f"Skipped unsafe fix: {fix_type}"
                                logger.warning(f"Skipped potentially unsafe fix: {fix_type}")
                                response.reasoning.append(reasoning_step)
                                continue
                        else:
                            # Use our built-in fix suggestion tools
                            fix_suggestion = suggest_fix(response.issue_type, logs_data)
                            fix_type = fix_suggestion.get("fix_type")
                            fix_params = fix_suggestion.get("parameters", {})
                            confidence = fix_suggestion.get("confidence", 0.0)
                        
                        reasoning_step["result"] = f"Suggested fix: {fix_type} - {fix_suggestion.get('description', '')}"
                        reasoning_step["confidence"] = confidence
                        response.confidence = confidence
                        response.reasoning.append(reasoning_step)
                        
                        logger.info(f"Suggested fix: {fix_type} - {fix_suggestion.get('description', '')}")
                    
                    # Step 4: Apply the fix
                    with tracer.start_as_current_span(f"step4_apply_fix_attempt_{fix_attempts}"):
                        reasoning_step = {"step": "fix_application", "timestamp": time.time(), "attempt": fix_attempts}
                        
                        # Apply the fix (or simulate it)
                        if request.simulate:
                            fix_result = {
                                "success": True,
                                "new_run_id": f"simulated-fixed-run-{int(time.time())}",
                                "message": f"Simulated fix applied: {fix_type}"
                            }
                            logger.info(f"Simulated applying fix, new run ID: {fix_result['new_run_id']}")
                        else:
                            fix_result = apply_fix(
                                job_id=request.job_id,
                                run_id=response.run_id,
                                fix_type=fix_type,
                                parameters=fix_params
                            )
                        
                        if fix_result.get("success", False):
                            fix_status = FixStatus.VERIFICATION_PENDING
                            reasoning_step["result"] = f"Fix applied: {fix_result.get('success')} - {fix_result.get('message', '')}"
                            new_run_id = fix_result.get('new_run_id')
                            reasoning_step["new_run_id"] = new_run_id
                        else:
                            fix_status = FixStatus.FAILED
                            reasoning_step["result"] = f"Fix failed: {fix_result.get('message', 'Unknown error')}"
                        
                        response.reasoning.append(reasoning_step)
                        logger.info(f"Fix applied: {fix_result.get('success')} - {fix_result.get('message', '')}")
                    
                    # Step 5: Verify the fix if it was applied
                    if fix_status == FixStatus.VERIFICATION_PENDING:
                        with tracer.start_as_current_span(f"step5_verify_fix_attempt_{fix_attempts}"):
                            reasoning_step = {"step": "verification", "timestamp": time.time(), "attempt": fix_attempts}
                            new_run_id = fix_result.get('new_run_id')
                            
                            # Verify the fix (or simulate it)
                            if request.simulate:
                                verification_result = {
                                    "status": "success",
                                    "details": "Simulated verification"
                                }
                                logger.info(f"Simulated verification result: {verification_result['status']}")
                            else:
                                verification_result = verify(
                                    job_id=request.job_id,
                                    run_id=new_run_id
                                )
                            
                            if verification_result.get("status") == "success":
                                fix_status = FixStatus.SUCCESSFUL
                                reasoning_step["result"] = f"Verification result: {verification_result.get('status')}"
                            else:
                                fix_status = FixStatus.FAILED
                                reasoning_step["result"] = f"Verification failed: {verification_result.get('details', 'Unknown error')}"
                            
                            response.reasoning.append(reasoning_step)
                            logger.info(f"Verification result: {verification_result.get('status')}")
                
                # Update response with fix attempts and success status
                response.fix_attempts = fix_attempts
                response.fix_successful = (fix_status == FixStatus.SUCCESSFUL)
                
                # Step 6: Generate final report
                with tracer.start_as_current_span("step6_generate_report"):
                    report_data = {
                        "job_id": request.job_id,
                        "run_id": response.run_id,
                        "issue_type": response.issue_type,
                        "fix_attempts": response.fix_attempts,
                        "fix_successful": response.fix_successful,
                        "steps": response.reasoning
                    }
                    
                    if AGENTS_SDK_AVAILABLE and self.assistant:
                        report_text = await self._generate_report_with_assistant(report_data)
                    else:
                        report_text = final_report(report_data)
                    
                    response.report = report_text
                    logger.info("Final report generated")
                
                return response
                
            except Exception as e:
                logger.error(f"Error processing monitoring request: {e}", exc_info=True)
                span.record_exception(e)
                span.set_status(trace.Status(trace.StatusCode.ERROR))
                
                # Add error to reasoning
                response.reasoning.append({
                    "step": "error",
                    "timestamp": time.time(),
                    "error": str(e)
                })
                
                # Add error report
                response.report = f"Error during monitoring: {str(e)}"
                return response
    
    async def _diagnose_with_assistant(self, logs_data: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Use the Azure Agents SDK to diagnose issues with hallucination detection.
        
        Args:
            logs_data: The logs data to analyze
            
        Returns:
            A tuple of (diagnosis_result, hallucination_info)
        """
        if not AGENTS_SDK_AVAILABLE or not self.assistant:
            logger.warning("Cannot use AI assistant for diagnosis: Azure Agents SDK not available")
            # Fall back to built-in diagnostic tools
            diagnosis_result = diagnose(logs_data)
            hallucination_info = {
                "detected": False,
                "score": 0.0,
                "reason": "Hallucination detection unavailable (Azure Agents SDK not installed)"
            }
            return diagnosis_result, hallucination_info
            
        logs_text = json.dumps(logs_data, indent=2)
        
        prompt = f"""
        Analyze these Databricks logs and identify any issues:
        
        {logs_text}
        
        Provide a diagnosis in the following format:
        1. Issue Type: [specific error type]
        2. Reasoning: [why you identified this issue]
        3. Relevant Log Sections: [sections of the logs that indicate the issue]
        """
        
        # Get response from the assistant
        agent_response: AgentResponse = await self.assistant.handle_message(prompt)
        
        # Check for hallucinations
        validation: ResponseValidation = self.out_of_domain_handler.validate(agent_response.content)
        hallucination_info = {
            "detected": validation.is_out_of_domain,
            "score": validation.out_of_domain_score,
            "reason": validation.reason if validation.is_out_of_domain else None
        }
        
        # Parse the diagnosis from the response
        diagnosis_text = agent_response.content
        issue_type = self._extract_issue_type(diagnosis_text)
        reasoning = self._extract_reasoning(diagnosis_text)
        
        diagnosis_result = {
            "issue_type": issue_type,
            "reasoning": reasoning,
            "full_text": diagnosis_text
        }
        
        return diagnosis_result, hallucination_info
    
    async def _suggest_fix_with_assistant(self, issue_type: str, logs_data: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Use the Azure Agents SDK to suggest fixes with safety checks.
        
        Args:
            issue_type: The type of issue detected
            logs_data: The logs data
            
        Returns:
            A tuple of (fix_suggestion, safety_info)
        """
        if not AGENTS_SDK_AVAILABLE or not self.assistant:
            logger.warning("Cannot use AI assistant for fix suggestion: Azure Agents SDK not available")
            # Fall back to built-in fix suggestion tools
            fix_suggestion = suggest_fix(issue_type, logs_data)
            safety_info = {
                "issues_detected": False,
                "harmful_score": 0.0,
                "harassment_score": 0.0,
                "hate_score": 0.0,
                "sexual_score": 0.0,
                "self_harm_score": 0.0,
                "reason": "Safety check unavailable (Azure Agents SDK not installed)"
            }
            return fix_suggestion, safety_info
            
        logs_text = json.dumps(logs_data, indent=2)
        
        prompt = f"""
        Suggest a fix for this Databricks issue:
        
        Issue Type: {issue_type}
        
        Logs:
        {logs_text}
        
        Provide a fix suggestion in the following format:
        1. Fix Type: [specific fix action]
        2. Parameters: [settings or configurations]
        3. Confidence: [numerical score 0.0-1.0]
        4. Reasoning: [why this fix should work]
        """
        
        # Get response from the assistant
        agent_response: AgentResponse = await self.assistant.handle_message(prompt)
        
        # Check for safety issues
        content_filter_results: ContentFilterResults = self.safety_policy.evaluate(agent_response.content)
        safety_info = {
            "issues_detected": content_filter_results.has_filtered_content,
            "harmful_score": content_filter_results.harmful_content_score,
            "harassment_score": content_filter_results.harassment_content_score,
            "hate_score": content_filter_results.hate_content_score,
            "sexual_score": content_filter_results.sexual_content_score,
            "self_harm_score": content_filter_results.self_harm_content_score
        }
        
        # Parse the fix suggestion from the response
        fix_text = agent_response.content
        fix_type = self._extract_fix_type(fix_text)
        parameters = self._extract_parameters(fix_text)
        confidence = self._extract_confidence(fix_text)
        
        fix_suggestion = {
            "fix_type": fix_type,
            "parameters": parameters,
            "confidence": confidence,
            "full_text": fix_text
        }
        
        return fix_suggestion, safety_info
    
    async def _generate_report_with_assistant(self, report_data: Dict[str, Any]) -> str:
        """
        Use the Azure Agents SDK to generate a final report.
        
        Args:
            report_data: The report data
            
        Returns:
            The formatted report text
        """
        if not AGENTS_SDK_AVAILABLE or not self.assistant:
            logger.warning("Cannot use AI assistant for report generation: Azure Agents SDK not available")
            # Fall back to built-in report generation
            return final_report(report_data)
            
        report_text = json.dumps(report_data, indent=2)
        
        prompt = f"""
        Generate a comprehensive final report for this Databricks monitoring session:
        
        {report_text}
        
        The report should include:
        1. Summary of the issue
        2. Diagnosis details
        3. Fix attempts and their results
        4. Recommendations for future
        
        Format it in Markdown with clear sections.
        """
        
        # Get response from the assistant
        agent_response: AgentResponse = await self.assistant.handle_message(prompt)
        
        return agent_response.content
    
    def _summarize_logs(self, logs_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a summary of the logs data."""
        return {
            "run_id": logs_data.get("run_id"),
            "job_id": logs_data.get("job_id"),
            "status": logs_data.get("status"),
            "duration_seconds": logs_data.get("duration_seconds"),
            "log_lines_count": len(logs_data.get("logs", {}).get("stdout", "").split("\n")) + 
                              len(logs_data.get("logs", {}).get("stderr", "").split("\n"))
        }
    
    def _extract_issue_type(self, text: str) -> str:
        """Extract issue type from assistant response."""
        # Simple extraction logic - in a real implementation, use regex or more robust parsing
        if "memory" in text.lower():
            return "memory_exceeded"
        elif "disk" in text.lower():
            return "disk_space_exceeded"
        elif "dependency" in text.lower():
            return "dependency_error"
        else:
            return "unknown"
    
    def _extract_reasoning(self, text: str) -> str:
        """Extract reasoning from assistant response."""
        # Simple extraction logic
        if "reasoning:" in text.lower():
            parts = text.lower().split("reasoning:")
            if len(parts) > 1:
                return parts[1].split("\n")[0].strip()
        return "No explicit reasoning provided"
    
    def _extract_fix_type(self, text: str) -> str:
        """Extract fix type from assistant response."""
        # Simple extraction logic
        if "memory" in text.lower():
            return "increase_memory"
        elif "disk" in text.lower():
            return "increase_disk_space"
        elif "dependency" in text.lower() or "package" in text.lower():
            return "install_dependencies"
        else:
            return "generic_fix"
    
    def _extract_parameters(self, text: str) -> Dict[str, Any]:
        """Extract parameters from assistant response."""
        # Simple extraction logic - in a real implementation, use regex or more robust parsing
        params = {}
        
        if "memory" in text.lower():
            params["memory_increment"] = "50%"
        elif "disk" in text.lower():
            params["disk_increment"] = "100%"
        elif "dependency" in text.lower() or "package" in text.lower():
            # Try to extract package names
            package_names = []
            lines = text.split("\n")
            for line in lines:
                if "install" in line.lower() and ":" not in line:
                    parts = line.split()
                    for part in parts:
                        if part not in ["install", "package", "dependency", "dependencies"]:
                            package_names.append(part.strip(",.;()[]{}\"'"))
            
            if package_names:
                params["packages"] = package_names
        
        return params
    
    def _extract_confidence(self, text: str) -> float:
        """Extract confidence score from assistant response."""
        # Simple extraction logic
        if "confidence:" in text.lower():
            parts = text.lower().split("confidence:")
            if len(parts) > 1:
                confidence_text = parts[1].split("\n")[0].strip()
                try:
                    # Try to extract a numerical value
                    confidence = float(confidence_text)
                    return min(max(confidence, 0.0), 1.0)  # Ensure it's in the range [0.0, 1.0]
                except ValueError:
                    pass
        
        # Default confidence if not found
        return 0.5
    
    # Action wrapper methods for the Azure Agents SDK
    
    async def _diagnose_wrapper(self, logs_json: str) -> str:
        """Wrapper for diagnose action."""
        try:
            logs_data = json.loads(logs_json)
            result = diagnose(logs_data)
            return json.dumps(result)
        except Exception as e:
            logger.error(f"Error in diagnose action: {e}", exc_info=True)
            return json.dumps({"error": str(e)})
    
    async def _suggest_fix_wrapper(self, issue_type: str, logs_json: str) -> str:
        """Wrapper for suggest_fix action."""
        try:
            logs_data = json.loads(logs_json)
            result = suggest_fix(issue_type, logs_data)
            return json.dumps(result)
        except Exception as e:
            logger.error(f"Error in suggest_fix action: {e}", exc_info=True)
            return json.dumps({"error": str(e)})
    
    async def _apply_fix_wrapper(self, job_id: str, run_id: str, fix_type: str, parameters_json: str) -> str:
        """Wrapper for apply_fix action."""
        try:
            parameters = json.loads(parameters_json)
            result = apply_fix(job_id, run_id, fix_type, parameters)
            return json.dumps(result)
        except Exception as e:
            logger.error(f"Error in apply_fix action: {e}", exc_info=True)
            return json.dumps({"error": str(e)})
    
    async def _verify_fix_wrapper(self, job_id: str, run_id: str) -> str:
        """Wrapper for verify action."""
        try:
            result = verify(job_id, run_id)
            return json.dumps(result)
        except Exception as e:
            logger.error(f"Error in verify action: {e}", exc_info=True)
            return json.dumps({"error": str(e)})
    
    async def _generate_report_wrapper(self, report_data_json: str) -> str:
        """Wrapper for generate_report action."""
        try:
            report_data = json.loads(report_data_json)
            result = final_report(report_data)
            return result
        except Exception as e:
            logger.error(f"Error in generate_report action: {e}", exc_info=True)
            return f"Error generating report: {str(e)}"

# Utility function for easy access
async def monitor_databricks_job(
    job_id: str, 
    run_id: Optional[str] = None,
    simulate: bool = False,
    simulate_failure_type: Optional[str] = None
) -> Dict[str, Any]:
    """
    Monitor a Databricks job for issues and attempt to fix them.
    
    Args:
        job_id: The Databricks job ID
        run_id: Optional specific run ID to analyze
        simulate: Whether to simulate execution
        simulate_failure_type: The failure type to simulate
        
    Returns:
        The monitoring result
    """
    agent = DatabricksMonitoringAgent()
    request = MonitoringRequest(
        job_id=job_id,
        run_id=run_id,
        simulate=simulate,
        simulate_failure_type=simulate_failure_type
    )
    
    response = await agent.process_request(request)
    return response.dict() 