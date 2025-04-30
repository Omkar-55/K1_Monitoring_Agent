"""
Tests for the tracing functionality of the K1 Monitoring Agent.
"""

import pytest
from unittest.mock import patch, MagicMock
import agent_core
from opentelemetry import trace


class TestTracing:
    """Tests for the tracing functionality."""
    
    @patch("agent_core.trace.get_tracer")
    @patch("agent_core.trace.set_tracer_provider")
    @patch("agent_core.OTLPSpanExporter")
    @patch("agent_core.BatchSpanProcessor")
    @patch("agent_core.TracerProvider")
    @patch("agent_core.Resource.create")
    def test_enable_tracing_success(
        self, mock_resource_create, mock_tracer_provider, 
        mock_batch_processor, mock_exporter, mock_set_provider,
        mock_get_tracer
    ):
        """Test that enable_tracing initializes tracing successfully."""
        # Set up mocks
        mock_resource = MagicMock()
        mock_resource_create.return_value = mock_resource
        
        mock_provider = MagicMock()
        mock_tracer_provider.return_value = mock_provider
        
        mock_span_processor = MagicMock()
        mock_batch_processor.return_value = mock_span_processor
        
        mock_otlp_exporter = MagicMock()
        mock_exporter.return_value = mock_otlp_exporter
        
        mock_tracer = MagicMock()
        mock_get_tracer.return_value = mock_tracer
        
        mock_span = MagicMock()
        mock_context_manager = MagicMock()
        mock_context_manager.__enter__.return_value = mock_span
        mock_tracer.start_as_current_span.return_value = mock_context_manager
        
        # Call the function
        result = agent_core.enable_tracing(service_name="test-service")
        
        # Verify results
        assert result is True, "enable_tracing should return True on success"
        
        # Verify that trace provider was set up correctly
        mock_resource_create.assert_called_once_with({"service.name": "test-service"})
        mock_tracer_provider.assert_called_once()
        mock_set_provider.assert_called_once_with(mock_provider)
        
        # Verify that exporter was set up
        mock_exporter.assert_called_once()
        mock_batch_processor.assert_called_once_with(mock_otlp_exporter)
        mock_provider.add_span_processor.assert_called_once_with(mock_span_processor)
        
        # Verify that a test span was created
        mock_get_tracer.assert_called_once_with("agent_core")
        mock_tracer.start_as_current_span.assert_called_once_with("Tracing started")
        mock_span.set_attribute.assert_called_once_with("initialization.success", True)

    @patch("agent_core.trace")
    def test_enable_tracing_import_error(self, mock_trace):
        """Test that enable_tracing handles import errors gracefully."""
        # Simulate ImportError by raising it when trace is accessed
        mock_trace.get_tracer.side_effect = ImportError("Module not found")
        
        # Call the function
        result = agent_core.enable_tracing()
        
        # Verify results
        assert result is False, "enable_tracing should return False on ImportError"

    @patch("agent_core.trace.get_tracer")
    @patch("agent_core.trace.set_tracer_provider")
    def test_enable_tracing_general_exception(self, mock_set_provider, mock_get_tracer):
        """Test that enable_tracing handles general exceptions gracefully."""
        # Simulate an exception during setup
        mock_set_provider.side_effect = Exception("General error")
        
        # Call the function
        result = agent_core.enable_tracing()
        
        # Verify results
        assert result is False, "enable_tracing should return False on Exception"


@pytest.mark.integration
class TestTracingIntegration:
    """Integration tests for tracing that require the OpenTelemetry SDK to be installed."""
    
    def test_enable_tracing_integration(self):
        """
        Test that enable_tracing works with the actual OpenTelemetry SDK.
        
        This test is marked as integration because it requires the OpenTelemetry
        packages to be installed and configured.
        """
        try:
            # Import required packages to check if they're available
            from opentelemetry import trace
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
            
            # If we get here, the required packages are available
            # Enable tracing
            result = agent_core.enable_tracing(
                service_name="test-integration",
                # Use in-memory exporter for testing
                endpoint="http://localhost:4317"
            )
            
            # Verify result
            assert result is True, "enable_tracing should return True with OTel SDK installed"
            
            # Verify that we can get a tracer and create spans
            tracer = trace.get_tracer("test_integration")
            with tracer.start_as_current_span("test_span") as span:
                span.set_attribute("test", True)
                assert span is not None, "Failed to create span"
        
        except ImportError:
            pytest.skip("OpenTelemetry SDK not installed")
        except Exception as e:
            pytest.fail(f"Unexpected error in integration test: {e}")


@pytest.mark.parametrize("service_name,endpoint,rate", [
    ("test-service", None, 1.0),
    ("custom-service", "http://custom:4317", 0.5),
    ("minimal-service", None, 0.1),
])
def test_enable_tracing_parameters(service_name, endpoint, rate):
    """Test enable_tracing with different parameters."""
    with patch("agent_core.trace") as mock_trace, \
         patch("agent_core.TracerProvider") as mock_provider, \
         patch("agent_core.OTLPSpanExporter") as mock_exporter, \
         patch("agent_core.Resource.create") as mock_resource_create:
        
        # Set up mocks
        mock_tracer = MagicMock()
        mock_trace.get_tracer.return_value = mock_tracer
        
        mock_span = MagicMock()
        mock_context_manager = MagicMock()
        mock_context_manager.__enter__.return_value = mock_span
        mock_tracer.start_as_current_span.return_value = mock_context_manager
        
        # Call the function with parameters
        result = agent_core.enable_tracing(
            service_name=service_name,
            endpoint=endpoint,
            rate=rate
        )
        
        # Verify results
        assert result is True, "enable_tracing should return True on success"
        
        # Verify service name was used
        mock_resource_create.assert_called_once_with({"service.name": service_name})
        
        # Verify endpoint was passed correctly if specified
        if endpoint:
            mock_exporter.assert_called_once_with(endpoint=endpoint)
        
        # For rate, we need to check the sampler which is trickier to verify
        # Here we just ensure the function ran successfully
