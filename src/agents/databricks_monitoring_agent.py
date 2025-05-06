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

# Import the Azure Agents SDK components
try:
    import openai
    from azure.ai.agents import AgentTask, Assistant, OutOfDomainHandler
    from azure.ai.agents.actions import Action, AzureMonitoringAction
    from azure.ai.agents.models import AgentResponse, ResponseValidation
    from azure.ai.agents.safety import SafetyPolicy, ContentFilterResults
    from azure.ai.agents.guardrails import GuardrailFunctionOutput, InputGuardrailTripwireTriggered, OutputGuardrailTripwireTriggered, RunContextWrapper, input_guardrail, output_guardrail
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
    
    # Dummy guardrail classes
    class GuardrailFunctionOutput:
        def __init__(self, tripwire_triggered=False, output_info=None):
            self.tripwire_triggered = tripwire_triggered
            self.output_info = output_info
    
    class RunContextWrapper:
        def __init__(self, context=None):
            self.context = context
    
    def input_guardrail(func):
        return func
    
    def output_guardrail(func):
        return func

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

# Set up logging
setup_logging(log_level="info")
logger = get_logger(__name__)

class FixStatus(Enum):
    """Status of fix attempts."""
    NOT_STARTED = auto()
    IN_PROGRESS = auto()
    SUCCESSFUL = auto()
    FAILED = auto()
    VERIFICATION_PENDING = auto()

class InputValidationResult(BaseModel):
    """Result of input validation."""
    is_valid: bool
    reason: Optional[str] = None
    severity: float = 0.0

class OutputValidationResult(BaseModel):
    """Result of output validation."""
    is_valid: bool
    reason: Optional[str] = None
    severity: float = 0.0

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
        
    # Add input guardrail function
    @input_guardrail
    async def validate_input(
        self,
        ctx: RunContextWrapper, 
        agent: Assistant, 
        input: str
    ) -> GuardrailFunctionOutput:
        """
        Input guardrail to validate that incoming requests are appropriate for this agent.
        Prevents non-Databricks related queries and potential harmful content.
        
        Args:
            ctx: The run context
            agent: The assistant
            input: The user input to validate
            
        Returns:
            GuardrailFunctionOutput indicating whether the tripwire was triggered
        """
        logger.info(f"Validating input: {input}")
        
        # Skip validation if Agent SDK isn't available
        if not AGENTS_SDK_AVAILABLE:
            logger.warning("Skipping input validation: Azure Agents SDK not available")
            return GuardrailFunctionOutput(
                tripwire_triggered=False,
                output_info=InputValidationResult(is_valid=True)
            )
        
        # Check if input is empty or too short
        if not input or len(input.strip()) < 5:
            logger.warning("Input validation failed: Input too short")
            return GuardrailFunctionOutput(
                tripwire_triggered=True,
                output_info=InputValidationResult(
                    is_valid=False,
                    reason="Input too short to process",
                    severity=0.8
                )
            )
        
        # Check for non-Databricks related queries using the assistant
        if self.assistant:
            prompt = f"""
            Is this query related to Databricks monitoring, diagnostics, or fixes?
            Query: "{input}"
            
            Answer only 'yes' or 'no' and provide a brief reason.
            """
            
            try:
                response = await self.assistant.handle_message(prompt)
                result = response.content.lower()
                
                if "no" in result[:5]:  # Check only beginning of response
                    logger.warning(f"Input validation failed: Non-Databricks query detected: {result}")
                    return GuardrailFunctionOutput(
                        tripwire_triggered=True,
                        output_info=InputValidationResult(
                            is_valid=False,
                            reason="This query is not related to Databricks monitoring",
                            severity=0.9
                        )
                    )
            except Exception as e:
                logger.error(f"Error during input validation: {e}", exc_info=True)
        
        # Check for safety issues using the safety policy
        if self.safety_policy:
            try:
                content_filter_results = self.safety_policy.evaluate(input)
                
                if content_filter_results.has_filtered_content:
                    logger.warning("Input validation failed: Harmful content detected")
                    return GuardrailFunctionOutput(
                        tripwire_triggered=True,
                        output_info=InputValidationResult(
                            is_valid=False,
                            reason="Potentially harmful content detected",
                            severity=0.95
                        )
                    )
            except Exception as e:
                logger.error(f"Error during safety check: {e}", exc_info=True)
        
        logger.info("Input validation passed")
        return GuardrailFunctionOutput(
            tripwire_triggered=False,
            output_info=InputValidationResult(is_valid=True)
        )
    
    # Add output guardrail function
    @output_guardrail
    async def validate_output(
        self,
        ctx: RunContextWrapper, 
        agent: Assistant, 
        output: Any
    ) -> GuardrailFunctionOutput:
        """
        Output guardrail to ensure responses are safe, relevant, and not hallucinations.
        
        Args:
            ctx: The run context
            agent: The assistant
            output: The output to validate
            
        Returns:
            GuardrailFunctionOutput indicating whether the tripwire was triggered
        """
        logger.info("Validating output")
        output_content = getattr(output, "content", str(output))
        
        # Skip validation if Agent SDK isn't available
        if not AGENTS_SDK_AVAILABLE:
            logger.warning("Skipping output validation: Azure Agents SDK not available")
            return GuardrailFunctionOutput(
                tripwire_triggered=False,
                output_info=OutputValidationResult(is_valid=True)
            )
        
        # Check for safety issues using the safety policy
        if self.safety_policy:
            try:
                content_filter_results = self.safety_policy.evaluate(output_content)
                
                if content_filter_results.has_filtered_content:
                    logger.warning("Output validation failed: Harmful content detected")
                    return GuardrailFunctionOutput(
                        tripwire_triggered=True,
                        output_info=OutputValidationResult(
                            is_valid=False,
                            reason="Potentially harmful content detected in output",
                            severity=0.95
                        )
                    )
            except Exception as e:
                logger.error(f"Error during output safety check: {e}", exc_info=True)
        
        # Check for hallucinations using the OutOfDomainHandler
        if self.out_of_domain_handler:
            try:
                validation = self.out_of_domain_handler.validate(output_content)
                
                if validation.is_out_of_domain:
                    logger.warning(f"Output validation failed: Potential hallucination detected with score {validation.out_of_domain_score}")
                    return GuardrailFunctionOutput(
                        tripwire_triggered=True,
                        output_info=OutputValidationResult(
                            is_valid=False,
                            reason=f"Potential hallucination detected: {validation.reason}",
                            severity=validation.out_of_domain_score
                        )
                    )
            except Exception as e:
                logger.error(f"Error during hallucination check: {e}", exc_info=True)
        
        # Check if output is empty or too short
        if not output_content or len(output_content.strip()) < 5:
            logger.warning("Output validation failed: Output too short")
            return GuardrailFunctionOutput(
                tripwire_triggered=True,
                output_info=OutputValidationResult(
                    is_valid=False,
                    reason="Output too short to be useful",
                    severity=0.8
                )
            )
        
        logger.info("Output validation passed")
        return GuardrailFunctionOutput(
            tripwire_triggered=False,
            output_info=OutputValidationResult(is_valid=True)
        )
        
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
            Always verify that your suggestions are safe before applying them.""",
            input_guardrails=[self.validate_input],
            output_guardrails=[self.validate_output]
        )
        
        logger.info("Assistant configured with input and output guardrails")
        return assistant
        
    async def process_request(self, request: MonitoringRequest) -> MonitoringResponse:
        """
        Process a monitoring request for a Databricks job.
        
        Args:
            request: The monitoring request
            
        Returns:
            The monitoring response with results
        """
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
        
            # Step 2: Diagnose issues in the logs
            reasoning_step = {"step": "diagnosis", "timestamp": time.time()}
            
            try:
                # If available, use the Azure Agents SDK for diagnosis with hallucination detection
                if self.assistant and AGENTS_SDK_AVAILABLE:
                    diagnosis, hallucination_info = await self._diagnose_with_assistant(logs_data)
                    response.hallucination_detected = hallucination_info.get("detected", False)
                    
                    # Add hallucination info to reasoning
                    reasoning_step["hallucination_check"] = hallucination_info
                    
                    # If hallucination detected, fallback to basic diagnosis
                    if response.hallucination_detected:
                        logger.warning(f"Hallucination detected in diagnosis with score {hallucination_info.get('score')}: {hallucination_info.get('reason')}")
                        diagnosis = diagnose(logs_data)
                else:
                    # Fallback to basic diagnosis
                    diagnosis = diagnose(logs_data)
            except (InputGuardrailTripwireTriggered, OutputGuardrailTripwireTriggered) as e:
                logger.warning(f"Guardrail triggered during diagnosis: {e}")
                diagnosis = diagnose(logs_data)
                reasoning_step["guardrail_triggered"] = {
                    "message": str(e),
                    "step": "diagnosis"
                }
            
            # Extract diagnosis results
            response.issue_detected = True
            response.issue_type = diagnosis.get("issue_type")
            
            reasoning_step["result"] = f"Diagnosed issue: {response.issue_type}"
            reasoning_step["details"] = diagnosis.get("reasoning", "")
            response.reasoning.append(reasoning_step)
            
            logger.info(f"Diagnosed issue: {response.issue_type}")
        
            # Step 3-5: Attempt fixes until success or max attempts reached
            fix_status = FixStatus.NOT_STARTED
            fix_attempts = 0
            
            while fix_status != FixStatus.SUCCESSFUL and fix_attempts < request.max_attempts:
                fix_attempts += 1
                
                # Step 3: Suggest a fix
                reasoning_step = {"step": "fix_suggestion", "timestamp": time.time(), "attempt": fix_attempts}
                
                try:
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
                        # Fallback to built-in fix suggestion
                        fix_suggestion = suggest_fix(response.issue_type, logs_data)
                        fix_type = fix_suggestion.get("fix_type")
                        fix_params = fix_suggestion.get("parameters", {})
                        confidence = fix_suggestion.get("confidence", 0.0)
                except (InputGuardrailTripwireTriggered, OutputGuardrailTripwireTriggered) as e:
                    logger.warning(f"Guardrail triggered during fix suggestion: {e}")
                    fix_suggestion = suggest_fix(response.issue_type, logs_data)
                    fix_type = fix_suggestion.get("fix_type")
                    fix_params = fix_suggestion.get("parameters", {})
                    confidence = fix_suggestion.get("confidence", 0.0)
                    reasoning_step["guardrail_triggered"] = {
                        "message": str(e),
                        "step": "fix_suggestion"
                    }
                
                response.confidence = confidence
                reasoning_step["result"] = f"Suggested fix: {fix_type} (confidence: {confidence})"
                reasoning_step["parameters"] = fix_params
                response.reasoning.append(reasoning_step)
                
                logger.info(f"Suggested fix: {fix_type} (confidence: {confidence})")
            
                # Step 4: Apply the fix
                reasoning_step = {"step": "fix_application", "timestamp": time.time(), "attempt": fix_attempts}
                
                # Apply the fix (or simulate if requested)
                try:
                    fix_result = apply_fix(
                        request.job_id, 
                        response.run_id, 
                        fix_type, 
                        fix_params,
                        simulate=request.simulate
                    )
                    
                    reasoning_step["result"] = fix_result.get("message", "Fix applied")
                    reasoning_step["details"] = fix_result.get("details", {})
                    response.reasoning.append(reasoning_step)
                    
                    logger.info(f"Applied fix: {fix_type}")
                except Exception as e:
                    logger.error(f"Error applying fix: {e}", exc_info=True)
                    reasoning_step["result"] = f"Error applying fix: {str(e)}"
                    response.reasoning.append(reasoning_step)
                    continue
            
                # Step 5: Verify the fix
                reasoning_step = {"step": "fix_verification", "timestamp": time.time(), "attempt": fix_attempts}
                
                # Verify the fix
                try:
                    verification_result = verify(request.job_id, response.run_id, simulate=request.simulate)
                    
                    fix_status = FixStatus[verification_result.get("status", "FAILED").upper()]
                    
                    reasoning_step["result"] = verification_result.get("message", "Fix verification completed")
                    reasoning_step["details"] = verification_result.get("details", {})
                    reasoning_step["status"] = fix_status.name
                    response.reasoning.append(reasoning_step)
                    
                    if fix_status == FixStatus.SUCCESSFUL:
                        logger.info(f"Fix successful on attempt {fix_attempts}")
                        response.fix_successful = True
                    else:
                        logger.warning(f"Fix verification failed on attempt {fix_attempts}: {verification_result.get('message')}")
                        fix_status = FixStatus.FAILED
                except Exception as e:
                    logger.error(f"Error verifying fix: {e}", exc_info=True)
                    reasoning_step["result"] = f"Error verifying fix: {str(e)}"
                    reasoning_step["status"] = FixStatus.FAILED.name
                    response.reasoning.append(reasoning_step)
                    fix_status = FixStatus.FAILED
            
            # Record the number of fix attempts
            response.fix_attempts = fix_attempts
            
            # Generate final report
            try:
                if self.assistant and AGENTS_SDK_AVAILABLE:
                    report, hallucination_info = await self._generate_report_with_assistant(
                        response.issue_type,
                        response.reasoning,
                        response.fix_successful
                    )
                    
                    # If hallucination detected, fallback to basic report
                    if hallucination_info.get("detected", False):
                        logger.warning(f"Hallucination detected in report with score {hallucination_info.get('score')}: {hallucination_info.get('reason')}")
                        response.hallucination_detected = True
                        response.report = final_report(
                            response.issue_type,
                            response.reasoning,
                            response.fix_successful
                        )
                    else:
                        response.report = report
                else:
                    response.report = final_report(
                        response.issue_type,
                        response.reasoning,
                        response.fix_successful
                    )
            except (InputGuardrailTripwireTriggered, OutputGuardrailTripwireTriggered) as e:
                logger.warning(f"Guardrail triggered during report generation: {e}")
                response.report = final_report(
                    response.issue_type,
                    response.reasoning,
                    response.fix_successful
                )
                # Add guardrail info to the response
                response.reasoning.append({
                    "step": "report_generation",
                    "timestamp": time.time(),
                    "guardrail_triggered": {
                        "message": str(e),
                        "step": "report_generation"
                    },
                    "result": "Used fallback report generation due to guardrail trigger"
                })
            
            logger.info("Generated final report")
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing request: {e}", exc_info=True)
            
            # Update response with error details
            error_step = {
                "step": "error",
                "timestamp": time.time(),
                "result": f"Error: {str(e)}"
            }
            response.reasoning.append(error_step)
            
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
    
    async def _generate_report_with_assistant(self, issue_type: str, reasoning: List[Dict[str, Any]], fix_successful: bool) -> Tuple[str, Dict[str, Any]]:
        """
        Use the Azure Agents SDK to generate a final report.
        
        Args:
            issue_type: The type of issue detected
            reasoning: The reasoning steps taken
            fix_successful: Whether the fix was successful
            
        Returns:
            A tuple of (report_text, hallucination_info)
        """
        if not AGENTS_SDK_AVAILABLE or not self.assistant:
            logger.warning("Cannot use AI assistant for report generation: Azure Agents SDK not available")
            # Fall back to built-in report generation
            report_text = final_report(issue_type, reasoning, fix_successful)
            hallucination_info = {
                "detected": False,
                "score": 0.0,
                "reason": "Report generation unavailable (Azure Agents SDK not installed)"
            }
            return report_text, hallucination_info
            
        report_text = json.dumps({
            "issue_type": issue_type,
            "reasoning": reasoning,
            "fix_successful": fix_successful
        }, indent=2)
        
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
        
        # Check for hallucinations
        validation: ResponseValidation = self.out_of_domain_handler.validate(agent_response.content)
        hallucination_info = {
            "detected": validation.is_out_of_domain,
            "score": validation.out_of_domain_score,
            "reason": validation.reason if validation.is_out_of_domain else None
        }
        
        return agent_response.content, hallucination_info
    
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
    
    async def _generate_report_wrapper(self, issue_type: str, reasoning_json: str, fix_successful: bool) -> str:
        """Wrapper for generate_report action."""
        try:
            reasoning = json.loads(reasoning_json)
            result = final_report(issue_type, reasoning, fix_successful)
            return result
        except Exception as e:
            logger.error(f"Error in generate_report action: {e}", exc_info=True)
            return json.dumps({"error": str(e)})

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