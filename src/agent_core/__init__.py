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
        endpoint: OpenTelemetry collector endpoint (None uses default) 
        rate: Sampling rate between 0.0 and 1.0
    
    Returns:
        bool: True if tracing was successfully enabled, False otherwise
    """
    try:
        # Import optional dependencies only when needed
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.trace.sampling import TraceIdRatioBased
        
        # Configure the tracer provider
        resource = Resource.create({"service.name": service_name})
        
        # Set up sampling
        sampler = TraceIdRatioBased(rate)
        
        # Create tracer provider
        provider = TracerProvider(resource=resource, sampler=sampler)
        trace.set_tracer_provider(provider)
        
        # Configure exporter to send spans to collector
        if endpoint is None:
            # Try to get from environment
            endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
        
        # Create exporter and processor
        otlp_exporter = OTLPSpanExporter(endpoint=endpoint)
        span_processor = BatchSpanProcessor(otlp_exporter)
        provider.add_span_processor(span_processor)
        
        # Create a test span to verify setup
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("Tracing started") as span:
            span.set_attribute("initialization.success", True)
            logger.info("Tracing initialized successfully")
            
        return True
    
    except ImportError as e:
        logger.warning(f"Tracing not enabled due to missing dependencies: {e}")
        logger.info("Install OpenTelemetry packages to enable tracing")
        return False
    except Exception as e:
        logger.error(f"Failed to initialize tracing: {e}", exc_info=True)
        return False
