"""
Core logic for the K1 Monitoring Agent.
"""

import os
import asyncio
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, validator
from opentelemetry import trace

# Import our logging configuration
from .logging_config import get_logger

# Initialize logger
logger = get_logger(__name__)

# Get tracer for this module
tracer = trace.get_tracer(__name__)

class AgentInput(BaseModel):
    """Input data model for the agent."""
    query: str = Field(..., description="The user's query or request")
    context: Optional[Dict[str, Any]] = Field(
        default_factory=dict, 
        description="Additional context for the agent"
    )
    conversation_id: Optional[str] = Field(
        None, 
        description="Identifier for the conversation"
    )

class AgentResponse(BaseModel):
    """Response data model from the agent."""
    response: str = Field(..., description="The agent's response text")
    confidence: float = Field(
        ..., 
        description="Confidence score between 0.0 and 1.0",
        ge=0.0,
        le=1.0
    )
    sources: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Sources used to generate the response"
    )
    
    @validator('confidence')
    def validate_confidence(cls, v):
        """Ensure confidence is between 0.0 and 1.0."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
        return v

class Agent:
    """The K1 Monitoring Agent for processing queries."""
    
    def __init__(self, azure_openai_deployment: Optional[str] = None):
        """
        Initialize the agent.
        
        Args:
            azure_openai_deployment: The Azure OpenAI deployment name to use
        """
        self.deployment_name = azure_openai_deployment or os.getenv(
            "AZURE_OPENAI_DEPLOYMENT", 
            "gpt-4"
        )
        self._initialize_client()
        logger.info(f"Agent initialized with deployment: {self.deployment_name}")
    
    def _initialize_client(self):
        """Initialize the Azure OpenAI client."""
        try:
            from azure.openai import AsyncAzureOpenAI
            
            self.client = AsyncAzureOpenAI(
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2023-05-15"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
            )
            logger.debug("Azure OpenAI client initialized successfully")
        except ImportError:
            logger.warning("Azure OpenAI package not installed. Using mock client.")
            self.client = None
        except Exception as e:
            logger.error(f"Error initializing Azure OpenAI client: {e}", exc_info=True)
            self.client = None
    
    async def process_query(self, input_data: AgentInput) -> AgentResponse:
        """
        Process a query using the agent.
        
        Args:
            input_data: The input data containing the query and context
            
        Returns:
            The agent's response
        """
        with tracer.start_as_current_span("agent.process_query") as span:
            span.set_attribute("query", input_data.query)
            if input_data.conversation_id:
                span.set_attribute("conversation_id", input_data.conversation_id)
            
            logger.info(f"Processing query: {input_data.query}")
            
            if not self.client:
                logger.warning("Using mock response (Azure OpenAI client not available)")
                return AgentResponse(
                    response="I'm sorry, but I'm not fully initialized yet.",
                    confidence=0.5,
                    sources=[]
                )
            
            try:
                # Prepare the messages for the chat completion
                messages = [
                    {"role": "system", "content": "You are the K1 Monitoring Agent, designed to help with monitoring tasks."},
                    {"role": "user", "content": input_data.query}
                ]
                
                # Add context if available
                if input_data.context:
                    context_str = str(input_data.context)
                    messages.insert(1, {"role": "system", "content": f"Context: {context_str}"})
                
                # Create a span for the API call
                with tracer.start_as_current_span("azure_openai.chat.completions") as api_span:
                    api_span.set_attribute("deployment_name", self.deployment_name)
                    
                    # Make the API call
                    response = await self.client.chat.completions.create(
                        deployment_name=self.deployment_name,
                        messages=messages,
                        temperature=0.7,
                        max_tokens=800
                    )
                
                # Extract the response
                response_text = response.choices[0].message.content
                
                # Create the agent response
                agent_response = AgentResponse(
                    response=response_text,
                    confidence=0.85,  # For a real agent, this would be dynamically calculated
                    sources=[]  # In a real agent, this would include sources
                )
                
                logger.info(f"Generated response with {len(response_text)} characters")
                return agent_response
                
            except Exception as e:
                logger.error(f"Error processing query: {e}", exc_info=True)
                span.record_exception(e)
                span.set_status(trace.Status(trace.StatusCode.ERROR))
                
                # Return a fallback response
                return AgentResponse(
                    response="I'm sorry, but I encountered an error processing your request.",
                    confidence=0.0,
                    sources=[]
                )

# For simple usage
async def process_query(query: str, context: Optional[Dict[str, Any]] = None) -> str:
    """
    Simplified function to process a query.
    
    Args:
        query: The user query
        context: Optional context dictionary
        
    Returns:
        The response text
    """
    agent = Agent()
    input_data = AgentInput(query=query, context=context or {})
    response = await agent.process_query(input_data)
    return response.response
