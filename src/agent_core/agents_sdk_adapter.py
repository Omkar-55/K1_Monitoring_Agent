"""
OpenAI Agents integration for the K1 Monitoring Agent.
"""

import os
import time
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

# Import OpenAI Agents
try:
    import openai
    from openai import OpenAI
    from openai_agents import AgentMessage, MessageRole, Tool, AgentState, AssistantId, Thread
    # Try to import Azure Agents SDK components
    try:
        from azure.ai.agents.guardrails import RunContextWrapper
        AGENTS_SDK_AVAILABLE = True
    except ImportError:
        # Still consider it available even if just basic OpenAI agents work
        AGENTS_SDK_AVAILABLE = True
        # Define a minimal RunContextWrapper
        class RunContextWrapper:
            def __init__(self, context=None):
                self.context = context
except ImportError:
    AGENTS_SDK_AVAILABLE = False
    # Define a minimal RunContextWrapper for compatibility when SDK not available
    class RunContextWrapper:
        def __init__(self, context=None):
            self.context = context

# Import our existing components
from .logging_config import get_logger
from .core_logic import Agent, AgentInput, AgentResponse

# Initialize logger
logger = get_logger(__name__)

class AgentsSdkRequest(BaseModel):
    """Model for an Agents request."""
    messages: List[Dict[str, Any]] = Field(..., description="List of messages in the conversation")
    config: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Configuration for the agent")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")


class AgentsSdkAdapter:
    """Adapter for integrating with OpenAI Agents."""
    
    def __init__(self, agent: Optional[Agent] = None):
        """
        Initialize the OpenAI Agents adapter.
        
        Args:
            agent: An optional Agent instance to use. If not provided, one will be created.
        """
        if not AGENTS_SDK_AVAILABLE:
            logger.warning("OpenAI Agents not available. Please install with 'pip install openai openai-agents'")
        
        self.agent = agent or Agent()
        
        # Initialize OpenAI client if available
        if AGENTS_SDK_AVAILABLE:
            # Get API key from environment variable
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                logger.warning("OPENAI_API_KEY environment variable not set")
                self.client = None
            else:
                self.client = OpenAI(api_key=api_key)
                logger.info("OpenAI client initialized")
        else:
            self.client = None
            logger.warning("OpenAI client not initialized due to missing dependencies")
    
    async def process_agents_sdk_request(self, request: AgentsSdkRequest) -> Dict[str, Any]:
        """
        Process a request using our agent and return in a format compatible with OpenAI Agents.
        
        Args:
            request: The request containing messages and config
            
        Returns:
            A response formatted for the OpenAI Agents ecosystem
        """
        logger.info("Processing Agents SDK request")
        start_time = time.time()
        
        if not AGENTS_SDK_AVAILABLE:
            logger.error("Cannot process Agents request: OpenAI Agents not available")
            return {
                "type": "error",
                "error": {
                    "message": "OpenAI Agents is not installed"
                }
            }
        
        logger.info("Processing request attributes")
        
        try:
            # Extract the latest user message
            last_user_message = None
            for message in reversed(request.messages):
                if message.get("role") == "user":
                    last_user_message = message.get("content", "")
                    break
            
            if not last_user_message:
                logger.warning("No user message found in the conversation")
                last_user_message = "Hello"
            
            # Convert to our internal format
            agent_input = AgentInput(
                query=last_user_message,
                context=request.metadata,
                conversation_id=request.metadata.get("conversation_id")
            )
            
            # Process with our agent
            agent_response = await self.agent.process_query(agent_input)
            
            # Format the response in a way compatible with OpenAI Agents
            response = {
                "type": "agent_response",
                "messages": [
                    {
                        "role": "assistant",
                        "content": agent_response.response
                    }
                ],
                "metadata": {
                    "confidence": agent_response.confidence,
                    "sources": agent_response.sources,
                    "processing_time": time.time() - start_time
                }
            }
            
            logger.info(f"Agent response created with confidence: {agent_response.confidence}, took {time.time() - start_time:.2f}s")
            return response
            
        except Exception as e:
            logger.error(f"Error processing Agents request: {e}", exc_info=True)
            logger.error(f"Request processing failed after {time.time() - start_time:.2f}s")
            
            # Return error response
            return {
                "type": "error",
                "error": {
                    "message": str(e)
                }
            }
    
    def is_available(self) -> bool:
        """Check if OpenAI Agents is available."""
        return AGENTS_SDK_AVAILABLE and self.client is not None
    
    async def create_thread_response(self, thread_id: str, user_message: str) -> Dict[str, Any]:
        """
        Process a message using the OpenAI Assistants API through a thread.
        
        Args:
            thread_id: The thread ID to use
            user_message: The user message to process
            
        Returns:
            The assistant's response
        """
        if not self.is_available():
            logger.error("Cannot create thread response: OpenAI client not available")
            return {"error": "OpenAI client not available"}
        
        try:
            logger.info(f"Creating thread response for thread {thread_id}")
            start_time = time.time()
            
            # Add a message to the thread
            message = self.client.beta.threads.messages.create(
                thread_id=thread_id,
                role="user",
                content=user_message
            )
            
            # Run the assistant on the thread
            run = self.client.beta.threads.runs.create(
                thread_id=thread_id,
                assistant_id=os.getenv("OPENAI_ASSISTANT_ID")  # You need to create an assistant in OpenAI
            )
            
            # Wait for the run to complete
            while run.status in ["queued", "in_progress"]:
                # Sleep a bit to avoid polling too frequently
                time.sleep(1)
                
                # Retrieve the updated run
                run = self.client.beta.threads.runs.retrieve(
                    thread_id=thread_id,
                    run_id=run.id
                )
            
            if run.status == "completed":
                # Get the latest message from the assistant
                messages = self.client.beta.threads.messages.list(
                    thread_id=thread_id,
                    order="desc"
                )
                
                # Format the response
                response_time = time.time() - start_time
                logger.info(f"Thread response created in {response_time:.2f} seconds")
                
                return {
                    "type": "agent_response",
                    "messages": [
                        {
                            "role": "assistant",
                            "content": messages.data[0].content[0].text.value
                        }
                    ],
                    "metadata": {
                        "thread_id": thread_id,
                        "run_id": run.id,
                        "processing_time": response_time
                    }
                }
            else:
                logger.error(f"Run failed with status: {run.status}")
                return {
                    "type": "error",
                    "error": {
                        "message": f"Run failed with status: {run.status}"
                    }
                }
                
        except Exception as e:
            logger.error(f"Error creating thread response: {e}", exc_info=True)
            return {
                "type": "error",
                "error": {
                    "message": str(e)
                }
            } 