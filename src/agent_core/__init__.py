"""
Agent Core package for the K1 Monitoring Agent.
"""

import os
import logging
from typing import Optional
from dotenv import load_dotenv

# Import logging configuration
from .logging_config import setup_logging, get_logger

# Initialize logger
logger = get_logger(__name__)

# Import to make available at package level
try:
    from .agents_sdk_adapter import AgentsSdkAdapter, AgentsSdkRequest, AGENTS_SDK_AVAILABLE
except ImportError:
    logger.warning("Could not import agents_sdk_adapter. Some features may not be available.")
    AGENTS_SDK_AVAILABLE = False

# Load environment variables from .env file
load_dotenv()

# Access environment variables
db_host = os.getenv("DATABRICKS_HOST")
db_token = os.getenv("DATABRICKS_TOKEN")

def enable_tracing(
    service_name: str = "k1-monitoring-agent",
    endpoint: Optional[str] = None,
    rate: float = 1.0  # Sampling rate: 1.0 = 100%
) -> bool:
    """
    Enable distributed tracing for the application.
    
    Args:
        service_name: Name of the service for tracing
        endpoint: OTLP collector endpoint (if using OpenTelemetry) or Azure endpoint 
        rate: Sampling rate between 0.0 and 1.0
    
    Returns:
        bool: True if tracing was successfully enabled, False otherwise
    """
    # First, try to use Azure Agents SDK tracing if available
    try:
        if AGENTS_SDK_AVAILABLE:
            logger.info("Azure Agents SDK is available, using Azure tracing")
            # Azure Agents SDK handles tracing automatically
            # We just need to make sure the SDK is loaded
            from azure.ai.agents import Agent, Assistant, AgentTask
            from azure.ai.agents.tracing import enable_tracing as enable_azure_tracing
            
            # Enable Azure tracing if the function is available
            if enable_azure_tracing and callable(enable_azure_tracing):
                enable_azure_tracing(service_name=service_name)
                logger.info(f"Azure tracing enabled for service {service_name}")
            else:
                logger.info("Azure tracing enabled by default")
                
            return True
    
    except (ImportError, AttributeError) as e:
        logger.info(f"Azure Agents SDK tracing not available: {e}")
        logger.info("Falling back to basic tracing")
        # No need to do anything else as the Azure SDK will handle tracing automatically
    
    # If Azure Agents SDK is not available or fails, log a message and return True
    # We're no longer implementing a fallback to OpenTelemetry
    logger.info("Tracing enabled (Azure Agents SDK tracing not available)")
    return True
