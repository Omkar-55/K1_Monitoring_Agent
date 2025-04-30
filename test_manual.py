"""
Manual test script for logging and tracing functionality.
"""

import os
import time
from pathlib import Path

# Import the agent core package
import agent_core
from agent_core.logging_config import get_logger, setup_logging


def test_logging():
    """Test basic logging functionality."""
    print("\n=== Testing Logging ===")
    
    # Get logger for this module
    logger = get_logger(__name__)
    
    # Log at different levels
    logger.debug("This is a DEBUG message")
    logger.info("This is an INFO message")
    logger.warning("This is a WARNING message")
    logger.error("This is an ERROR message")
    
    # Check if log file exists
    log_path = Path(".") / "logs" / "agent.log"
    print(f"Log file should be at: {log_path.absolute()}")
    
    if log_path.exists():
        print(f"Log file exists: {log_path}")
        print("\nLast 5 lines of log file:")
        with open(log_path, 'r') as f:
            lines = f.readlines()
            for line in lines[-5:]:
                print(f"  {line.strip()}")
    else:
        print(f"Log file does not exist: {log_path}")
    
    return True


def test_tracing():
    """Test basic tracing functionality."""
    print("\n=== Testing Tracing ===")
    
    # Enable tracing
    tracing_enabled = agent_core.enable_tracing(
        service_name="manual-test",
        endpoint="http://localhost:4317"  # This is the default OTLP endpoint
    )
    
    if tracing_enabled:
        print("Tracing enabled successfully!")
        
        # Try to create a span
        try:
            from opentelemetry import trace
            tracer = trace.get_tracer(__name__)
            
            print("Creating test span...")
            with tracer.start_as_current_span("manual_test_span") as span:
                span.set_attribute("test_attribute", "test_value")
                print("Span created with attribute: test_attribute=test_value")
                
                # Simulate some work
                time.sleep(0.5)
                print("Work done inside span")
            
            print("Span completed")
            return True
            
        except ImportError as e:
            print(f"Could not create span due to missing dependencies: {e}")
            return False
        except Exception as e:
            print(f"Error creating span: {e}")
            return False
    else:
        print("Failed to enable tracing. Check if OpenTelemetry packages are installed.")
        return False


if __name__ == "__main__":
    # Set up logging
    print("Setting up logging...")
    setup_logging(log_level="debug")  # Set to debug to see all log levels
    
    # Run tests
    logging_result = test_logging()
    tracing_result = test_tracing()
    
    # Print summary
    print("\n=== Test Summary ===")
    print(f"Logging test: {'PASSED' if logging_result else 'FAILED'}")
    print(f"Tracing test: {'PASSED' if tracing_result else 'FAILED'}") 