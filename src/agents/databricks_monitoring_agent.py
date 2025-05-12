"""
Databricks Monitoring Agent for the Monitoring Agent platform.

This agent collects logs from Databricks workspaces, analyzes issues,
suggests fixes, waits for the human approval, implements fixes, verifies resolutions and updates the user of the summary
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
    approved_fix: Optional[str] = Field(None, description="ID of the fix that was approved by the user")
    conversation_id: Optional[str] = Field(None, description="Unique ID for the conversation")

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
    pending_approval: bool = Field(False, description="Whether user approval is pending")
    suggested_fix: Optional[Dict[str, Any]] = Field(None, description="The fix suggestion awaiting approval")
    conversation_id: Optional[str] = Field(None, description="Unique ID for the conversation")
    fix_history: List[Dict[str, Any]] = Field(default_factory=list, description="History of suggested fixes")

class DatabricksMonitoringAgent:
    """
    Agent for monitoring Databricks workspaces, detecting issues,
    applying fixes after human approval, and generating a summary for the user
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
        Prevents job monitoring related queries and potential harmful content.
        
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
            Is this query related to Jobs, Pieplines or Clusters monitoring, diagnostics, or fixes?
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
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1"),
            system_message="""You are a **Databricks Support Engineer** agent (using GPT-4.1) whose mission is to:

            1. **Diagnose** Databricks pipelines/jobs  
            2. **Analyze** and **pull logs** using Databricks tools  
            3. **Propose** and—upon user approval—**apply fixes**  
            4. **Verify** that fixes resolve the issue  
            5. **Answer** user queries about pipeline/job status  

            > **Important:**  
            > - Only end your turn once you're certain the user's problem or query is fully resolved.  
            > - Be thorough and explain your reasoning **step by step**.  
            > - **Do not guess**—use tools to inspect logs or config.  



            ---

            ## High-Level Strategy

            1. **Understand**  
            - Read the user's query carefully.  
            - Clarify ambiguities by asking follow-up questions if needed.  

            2. **Gather Context**  
            - Use Databricks APIs/tools to **pull logs**, workspace metadata, job configurations, etc.  

            3. **Root Cause Analysis**  
            - Read logs, error messages, and config.  
            - Identify whether the issue stems from OOM, cluster limits, timeouts, dependencies, permissions, etc.  

            4. **Plan**  
            - Break the fix into **small, verifiable steps**.  
            - Outline each step before acting.  

            5. **Execute**  
            - After user approval, apply patches or configuration changes.  
            - If a patch fails to apply, retry or choose an alternative.  

            6. **Verify & Test**  
            - Re-run the pipeline/job after each change.  
            - Confirm errors are resolved or iterate again.  

            7. **Reflect & Summarize**  
            - Once resolved, summarize:  
                - Issue type & root cause  
                - Steps you took  
                - Confirmation that the fix worked  
            - For **status-only** queries, simply report current pipeline/job health.  

            ---

            ## Diagnostic Workflow

            - **When fixing:**  
            1. Pull relevant logs.  
            2. Analyze for root cause.  
            3. Plan & seek user approval.  
            4. Apply fix.  
            5. Test & verify.  
            6. Iterate until green.  

            - **When monitoring:**  
            1. Pull status/logs.  
            2. Summarize current state.  
            3. No fixes unless requested.  

            ---

            ## Reasoning & Tools

            - **Plan thoroughly** before any function/tool call.  
            - **Reflect** on every tool output—do not chain calls blindly.  
            - **Ensure safety** of suggestions before applying.  

            ---
            If you are not sure about file content or codebase structure pertaining to the user's request, 
            use your tools to read files and gather the relevant information: do NOT guess or make up an answer.

            ---

            ## **CONFLICTS & REDUNDANCIES**  
            - **"You MUST iterate and keep going…"** appears twice—consolidated above.  
            - "Only terminate your turn when you are sure…" was repeated—now single bullet.  
            - "Pipeline and job mean the same." → **Removed**, since terminology is consistent now.  

            ---

            ## Examples

            ### 1. Monitoring Status  
            **User**: "What's the status of pipeline `daily_ingest_v2`?"  
            **Agent**:  
            1. Pulls the latest run logs/config via `get_pipeline_status('daily_ingest_v2')`.  
            2. Summarizes: "Last run succeeded at 2025-05-11 02:15 UTC; no errors in recent logs. No further action needed."

            ### 2. Diagnosing & Fixing  
            **User**: "`customer_etl` is failing with an OOM error."  
            **Agent**:  
            1. **Plan**:  
            - Pull job logs.  
            - Check driver/executor memory usage.  
            - Propose increasing cluster driver memory by 2 GB.  
            2. **Log Analysis**:  
            - Finds executor peaks at 14 GB / 12 GB limit.  
            3. **User Approval**:  
            - "I recommend bumping executor memory to 16 GB. Approve?"  
            4. **Apply Fix**:  
            - Calls `update_cluster_config(...)`.  
            5. **Verify**:  
            - Reruns pipeline; confirms success.  
            6. **Summary**:  
            - "Issue: OOM due to 12 GB executor limit; increased to 16 GB; pipeline now succeeds."

            ---
            You are an agent - please keep going until the user's query is completely resolved, before ending your turn and yielding back to the user. 
            Only terminate your turn when you are sure that the problem is solved.
            You MUST plan extensively before each function call, and reflect extensively on the outcomes of the previous function calls. 
            DO NOT do this entire process by making function calls only, as this can impair your ability to solve the problem and think insightfully.


            *End prompt.*
            .""",
            input_guardrails=[validate_input],
            output_guardrails=[validate_output]
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
            reasoning=[],
            conversation_id=request.conversation_id
        )
        
        try:
            # Step 1: Get logs for the job
            reasoning_step = {"step": "log_collection", "timestamp": time.time()}
            
            if request.simulate:
                logs_data = get_logs(
                    request.job_id, 
                    request.run_id, 
                    simulate=True, 
                    simulate_failure_type=request.simulate_failure_type or "memory_exceeded"
                )
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
            
            # Check if a previously suggested fix was approved
            if request.approved_fix is not None:
                # Find the approved fix in the fix history
                approved_fix = None
                for fix in response.fix_history:
                    if fix.get("fix_id") == request.approved_fix:
                        approved_fix = fix
                        break
                
                if approved_fix is None:
                    # This should be the first request with an approved fix
                    # Continue with the fix suggestion flow to regenerate it
                    logger.info(f"Approved fix {request.approved_fix} not found in history, regenerating fix suggestion")
                else:
                    # Apply the approved fix
                    logger.info(f"Applying approved fix {request.approved_fix}")
                    return await self._apply_approved_fix(request, response, logs_data, approved_fix)
            
            # Step 3: Suggest a fix (but don't apply it yet, wait for user approval)
            fix_attempts = response.fix_attempts
            reasoning_step = {"step": "fix_suggestion", "timestamp": time.time(), "attempt": fix_attempts + 1}
            
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
                        reasoning_step["result"] = f"Unsafe fix detected: {fix_type}"
                        logger.warning(f"Unsafe fix detected: {fix_type}")
                        response.reasoning.append(reasoning_step)
                        
                        # Return the response with the unsafe fix warning
                        return response
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
            
            # Generate a unique ID for this fix suggestion
            import uuid
            fix_id = str(uuid.uuid4())
            
            # Create structured fix suggestion for user approval
            suggested_fix = {
                "fix_id": fix_id,
                "fix_type": fix_type,
                "parameters": fix_params,
                "confidence": confidence,
                "timestamp": time.time(),
                "attempt": fix_attempts + 1,
                "description": self._generate_fix_description(fix_type, fix_params)
            }
            
            # Update response
            response.confidence = confidence
            response.pending_approval = True
            response.suggested_fix = suggested_fix
            
            # Add to fix history
            response.fix_history.append(suggested_fix)
            
            reasoning_step["result"] = f"Suggested fix: {fix_type} (confidence: {confidence})"
            reasoning_step["parameters"] = fix_params
            reasoning_step["fix_id"] = fix_id
            reasoning_step["description"] = suggested_fix["description"]
            reasoning_step["status"] = "awaiting_approval"
            response.reasoning.append(reasoning_step)
            
            logger.info(f"Suggested fix: {fix_type} (confidence: {confidence}) - awaiting user approval")
            
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
            
    async def _apply_approved_fix(self, request: MonitoringRequest, response: MonitoringResponse, 
                               logs_data: Dict[str, Any], approved_fix: Dict[str, Any]) -> MonitoringResponse:
        """
        Apply a fix that has been approved by the user.
        
        Args:
            request: The monitoring request
            response: The current monitoring response
            logs_data: The logs data for context
            approved_fix: The approved fix details
            
        Returns:
            The updated monitoring response
        """
        fix_attempts = response.fix_attempts
        fix_type = approved_fix.get("fix_type")
        fix_params = approved_fix.get("parameters", {})
        
        # Step 4: Apply the fix
        reasoning_step = {"step": "fix_application", "timestamp": time.time(), "attempt": fix_attempts + 1}
        reasoning_step["fix_id"] = approved_fix.get("fix_id")
        
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
            reasoning_step["status"] = "error"
            response.reasoning.append(reasoning_step)
            
            # Increment fix attempts and return for another suggestion
            response.fix_attempts = fix_attempts + 1
            response.pending_approval = False  # Ready for a new suggestion
            return response
    
        # Step 5: Verify the fix
        reasoning_step = {"step": "fix_verification", "timestamp": time.time(), "attempt": fix_attempts + 1}
        reasoning_step["fix_id"] = approved_fix.get("fix_id")  # Track which fix is being verified
        
        # Verify the fix
        try:
            verification_result = verify(
                job_id=request.job_id, 
                run_id_or_fix_details={
                    "run_id": response.run_id,
                    "fix_type": approved_fix.get("fix_type"),
                    "parameters": approved_fix.get("parameters", {}),
                    "attempt": fix_attempts + 1
                }, 
                simulate=request.simulate
            )
            
            # Process verification result
            status = verification_result.get("status", "failed").upper()
            success = verification_result.get("issue_resolved", False)
            
            try:
                fix_status = FixStatus[status]
            except KeyError:
                # Handle case where status string doesn't match enum exactly
                fix_status = FixStatus.SUCCESSFUL if success else FixStatus.FAILED
            
            # Update the response based on verification results
            response.fix_successful = success
            
            reasoning_step["result"] = verification_result.get("message", "Fix verification completed")
            reasoning_step["details"] = verification_result.get("details", {})
            reasoning_step["status"] = fix_status.name
            response.reasoning.append(reasoning_step)
            
            # Generate final report if the fix was successful
            if success:
                logger.info(f"Fix successful on attempt {fix_attempts + 1}")
                
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
                except Exception as e:
                    logger.error(f"Error generating report: {e}", exc_info=True)
                    response.report = final_report(
                        response.issue_type,
                        response.reasoning,
                        response.fix_successful
                    )
            else:
                logger.warning(f"Fix verification failed on attempt {fix_attempts + 1}: {verification_result.get('message')}")
                # Increment fix attempts for the next suggestion
                response.fix_attempts = fix_attempts + 1
                response.pending_approval = False  # Ready for a new suggestion
            
        except Exception as e:
            logger.error(f"Error verifying fix: {e}", exc_info=True)
            reasoning_step["result"] = f"Error verifying fix: {str(e)}"
            reasoning_step["status"] = FixStatus.FAILED.name
            response.reasoning.append(reasoning_step)
            
            # Increment fix attempts for the next suggestion
            response.fix_attempts = fix_attempts + 1
            response.pending_approval = False  # Ready for a new suggestion
        
        return response
    
    def _generate_fix_description(self, fix_type: str, parameters: Dict[str, Any]) -> str:
        """
        Generate a human-readable description of the fix.
        
        Args:
            fix_type: The type of fix
            parameters: The fix parameters
            
        Returns:
            A human-readable description
        """
        if fix_type == "increase_memory":
            return f"Increase memory allocation by {parameters.get('memory_increment', '50%')}."
        
        elif fix_type == "increase_disk_space":
            return f"Increase disk space allocation by {parameters.get('disk_increment', '100%')}."
        
        elif fix_type == "install_dependencies":
            packages = parameters.get("packages", [])
            if packages:
                return f"Install missing dependencies: {', '.join(packages)}."
            else:
                return "Install missing dependencies."
        
        elif fix_type == "restart_cluster":
            return "Restart the Databricks cluster to resolve transient issues."
        
        elif fix_type == "update_configuration":
            return f"Update configuration parameters: {json.dumps(parameters)}."
        
        else:
            return f"Apply {fix_type} fix with parameters: {json.dumps(parameters)}."
    
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
            result = verify(job_id=job_id, run_id_or_fix_details=run_id)
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