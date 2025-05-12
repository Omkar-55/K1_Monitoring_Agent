"""
Databricks Monitoring Agent for the Monitoring Agent platform.

This agent collects logs from Databricks workspaces, analyzes issues,
suggests fixes, waits for the human approval, implements fixes, verifies resolutions and updates the user of the summary
"""

import os
import json
import time
import asyncio
import uuid
from typing import Dict, Any, List, Optional, Tuple, Union
from enum import Enum, auto
from pydantic import BaseModel, Field

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
        self.tools = DatabricksTools()
        self.memory = FixMemoryStore()
        
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
        Process the monitoring request and return a monitoring response.
        """
        logger.info(f"AGENT STATE: Starting to process request: {request}")
        
        # Extract required fields from the request
        job_id = request.job_id
        logger.info(f"AGENT STATE: Processing monitoring request for job {job_id}")
        
        # Initialize the response
        response = MonitoringResponse(
            job_id=job_id,
            run_id="pending",  # Default value while we wait to get the real run_id
            issue_detected=False,
            issue_type=None,
            confidence=0.0,
            reasoning=[],
            suggested_fix=None,
            fix_successful=None,
            report=None,
        )
        
        # Check if the user has approved a fix
        if request.approved_fix:
            logger.info(f"AGENT STATE: User approved fix: {request.approved_fix}")
            # Attempt to find the fix in the memory store
            fix_details = self.memory.get_fix_details(request.approved_fix)
            if fix_details:
                logger.info(f"AGENT STATE: Found fix details: {fix_details}")
                # Apply the fix and update the response
                fix_result = await self._apply_fix(fix_details, job_id)
                response.fix_successful = fix_result.get("successful", False)
                response.report = fix_result.get("report", "")
                logger.info(f"AGENT STATE: Applied fix with result: {fix_result}")
                return response
            else:
                logger.warning(f"AGENT STATE: Approved fix {request.approved_fix} not found in history, regenerating fix suggestion")
                # Fall through to diagnosis since we couldn't find the fix
        
        # If not applying a fix, proceed with diagnosis
        logger.info(f"AGENT STATE: Proceeding with diagnosis for job {job_id}")
        
        # Get log data for diagnosis
        logs_data, run_id = await self._get_logs(job_id, request.simulate, request.simulate_failure_type)
        response.run_id = run_id
        logger.info(f"AGENT STATE: Retrieved logs for run ID: {run_id}")
        
        # Diagnose the issue
        diagnosis = await self._diagnose_issue(logs_data, run_id)
        logger.info(f"AGENT STATE: Diagnosis complete: {diagnosis}")
        
        # Check if an issue was detected
        if diagnosis.get("issue_detected", False):
            issue_type = diagnosis.get("issue_type", "unknown")
            confidence = diagnosis.get("confidence", 0.0)
            logger.info(f"AGENT STATE: Diagnosed issue: {issue_type} with confidence {confidence}")
            
            # Update the response with the diagnosis results
            response.issue_detected = True
            response.issue_type = issue_type
            response.confidence = confidence
            response.reasoning = diagnosis.get("reasoning", [])
            
            # Generate a fix suggestion if an issue was detected
            if not request.approved_fix:  # Only suggest a fix if we're not already applying one
                logger.info(f"AGENT STATE: Generating fix suggestion for issue: {issue_type}")
                fix_suggestion = await self._suggest_fix(issue_type, logs_data)
                if fix_suggestion:
                    logger.info(f"AGENT STATE: Suggested fix: {fix_suggestion.get('fix_type')}")
                    # Generate a unique ID for this fix
                    fix_id = str(uuid.uuid4())
                    fix_suggestion["fix_id"] = fix_id
                    
                    # Store the fix details for later retrieval
                    self.memory.store_fix_details(fix_id, fix_suggestion)
                    
                    # Add the fix suggestion to the response
                    response.suggested_fix = fix_suggestion
                    
                    # Log the suggested fix for tracking
                    logger.info(f"AGENT STATE: Suggested fix: {fix_suggestion.get('fix_type')} (confidence: {fix_suggestion.get('confidence', 0.0)}) - awaiting user approval")
                else:
                    logger.warning("AGENT STATE: No fix could be suggested for the diagnosed issue")
            else:
                logger.info(f"AGENT STATE: User approved fix, but details weren't found. In normal operation, regenerating fix")
        else:
            logger.info("AGENT STATE: No issues detected in the logs")
        
        logger.info(f"AGENT STATE: Returning response: {response}")
        return response
    
    async def _get_logs(self, job_id: str, simulate: bool = False, simulate_failure_type: Optional[str] = None) -> Tuple[Dict[str, Any], str]:
        """
        Get logs for the specified job.
        
        Args:
            job_id: The ID of the job to get logs for.
            simulate: Whether to simulate logs for testing.
            simulate_failure_type: The type of failure to simulate.
            
        Returns:
            A tuple of (logs_data, run_id).
        """
        logger.info(f"AGENT STATE: Getting logs for job {job_id}")
        
        if simulate:
            logger.info(f"AGENT STATE: Simulating logs for job {job_id} with failure type: {simulate_failure_type}")
            # Generate a unique run ID with timestamp to avoid collisions
            timestamp = int(time.time())
            run_id = f"run_{timestamp}"
            logs_data = {
                "run_id": run_id,
                "job_id": job_id,
                "simulated": True,
                "simulated_failure_type": simulate_failure_type,
                "logs": []  # We'll generate fake logs based on the failure type
            }
            logger.info(f"AGENT STATE: Using simulated run data for failure type: {simulate_failure_type}")
        else:
            logger.info(f"AGENT STATE: Getting real logs for job {job_id}")
            # Get the most recent run for the job
            try:
                run_id = await self.dbx_client.get_most_recent_run_id(job_id)
                logger.info(f"AGENT STATE: Found most recent run ID: {run_id}")
            except Exception as e:
                logger.error(f"Error getting most recent run ID: {e}")
                # Generate a placeholder run ID
                timestamp = int(time.time())
                run_id = f"error_run_{timestamp}"
            
            # Get the logs for the run
            try:
                logs_data = await self.tools.get_logs(job_id, run_id)
                logger.info(f"AGENT STATE: Retrieved {len(logs_data.get('logs', []))} log entries for run {run_id}")
            except Exception as e:
                logger.error(f"Error getting logs: {e}")
                logs_data = {
                    "run_id": run_id,
                    "job_id": job_id,
                    "error": str(e),
                    "logs": []
                }
        
        logger.info(f"AGENT STATE: Retrieved logs for run ID: {run_id}")
        return logs_data, run_id
    
    async def _diagnose_issue(self, logs_data: Dict[str, Any], run_id: str) -> Dict[str, Any]:
        """
        Diagnose issues in the logs.
        
        Args:
            logs_data: The logs data to diagnose.
            run_id: The run ID.
            
        Returns:
            A dictionary with the diagnosis results.
        """
        logger.info(f"AGENT STATE: Diagnosing logs for run {run_id}")
        
        try:
            diagnosis = await self.tools.diagnose(logs_data, run_id)
            
            # If we have an issue_type but no issue_detected flag, set it to True
            if 'issue_type' in diagnosis and diagnosis['issue_type'] and 'issue_detected' not in diagnosis:
                diagnosis['issue_detected'] = True
                
            logger.info(f"AGENT STATE: Diagnosis complete with result: {diagnosis.get('issue_type', 'no issue')}")
        except Exception as e:
            logger.error(f"Error during diagnosis: {e}")
            diagnosis = {
                "issue_detected": False,
                "issue_type": "unknown",
                "confidence": 0.0,
                "reasoning": [
                    {"step": "error", "description": f"Error during diagnosis: {str(e)}"}
                ]
            }
        
        return diagnosis
    
    async def _suggest_fix(self, issue_type: str, logs_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Suggest a fix for the diagnosed issue.
        
        Args:
            issue_type: The type of issue to suggest a fix for.
            logs_data: The logs data.
            
        Returns:
            A dictionary with the fix suggestion or None if no fix could be suggested.
        """
        logger.info(f"AGENT STATE: Suggesting fix for issue: {issue_type}")
        
        try:
            fix_suggestion = await self.tools.suggest_fix(issue_type, logs_data)
            logger.info(f"AGENT STATE: Generated fix suggestion: {fix_suggestion.get('fix_type')}")
            return fix_suggestion
        except Exception as e:
            logger.error(f"Error suggesting fix: {e}")
            return None
    
    async def _apply_fix(self, fix_details: Dict[str, Any], job_id: str) -> Dict[str, Any]:
        """
        Apply the suggested fix.
        
        Args:
            fix_details: The details of the fix to apply.
            job_id: The job ID.
            
        Returns:
            A dictionary with the result of applying the fix.
        """
        fix_type = fix_details.get("fix_type")
        logger.info(f"AGENT STATE: Applying fix: {fix_type} for job {job_id}")
        
        try:
            fix_result = await self.tools.apply_fix(fix_details, job_id)
            logger.info(f"AGENT STATE: Fix application result: {fix_result}")
            
            # Check if the fix was successful
            fix_success = False
            if isinstance(fix_result, dict):
                fix_success = fix_result.get("status") == "success"
            
            # Generate a report using positional arguments
            # Argument order: issue_type, reasoning, fix_successful, job_id
            report = await self.tools.final_report(
                fix_details.get("fix_type", "unknown"), 
                [], 
                fix_success,
                job_id
            )
            logger.info(f"AGENT STATE: Generated report of length {len(report)}")
            
            result = {
                "successful": fix_success,
                "report": report
            }
        except Exception as e:
            logger.error(f"Error applying fix: {e}")
            result = {
                "successful": False,
                "report": f"Error applying fix: {str(e)}"
            }
        
        logger.info(f"AGENT STATE: Fix application complete with result: {result}")
        return result
    
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

# Helper class for tools
class DatabricksTools:
    """Tools for Databricks monitoring agent."""
    
    async def get_logs(self, job_id: str, run_id: str = None) -> Dict[str, Any]:
        """Get logs for the specified job and run."""
        from src.tools.databricks_monitoring import get_logs
        return get_logs(job_id, run_id)
    
    async def diagnose(self, logs_data: Dict[str, Any], run_id: str = None) -> Dict[str, Any]:
        """Diagnose issues in the logs."""
        from src.tools.databricks_monitoring import diagnose
        return diagnose(logs_data, run_id)
    
    async def suggest_fix(self, issue_type: str, logs_data: Dict[str, Any]) -> Dict[str, Any]:
        """Suggest a fix for the diagnosed issue."""
        from src.tools.databricks_monitoring import suggest_fix
        return suggest_fix(issue_type, logs_data)
    
    async def apply_fix(self, fix_details: Dict[str, Any], job_id: str) -> Dict[str, Any]:
        """Apply the suggested fix."""
        from src.tools.databricks_monitoring import apply_fix
        
        # Extract values from fix_details
        fix_type = fix_details.get("fix_type")
        parameters = fix_details.get("parameters", {})
        
        return apply_fix(job_id, "unknown", fix_type, parameters, simulate=True)
    
    async def final_report(self, issue_type: Union[str, Dict[str, Any]], reasoning: List[Dict[str, Any]] = None, fix_successful: bool = False, job_id: str = "unknown") -> str:
        """Generate a final report."""
        from src.tools.databricks_monitoring import final_report
        
        return final_report(issue_type, reasoning, fix_successful, job_id)

# Helper class for fix memory
class FixMemoryStore:
    """Memory store for fix suggestions."""
    
    def __init__(self):
        """Initialize the memory store."""
        self.fixes = {}
    
    def store_fix_details(self, fix_id: str, fix_details: Dict[str, Any]) -> None:
        """Store fix details for later retrieval."""
        self.fixes[fix_id] = fix_details
    
    def get_fix_details(self, fix_id: str) -> Optional[Dict[str, Any]]:
        """Get fix details by ID."""
        return self.fixes.get(fix_id) 