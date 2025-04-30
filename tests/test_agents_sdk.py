"""
Tests for the OpenAI Agents integration of the K1 Monitoring Agent.
"""

import os
import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
from typing import Dict, Any, List

from agent_core.agents_sdk_adapter import AgentsSdkAdapter, AgentsSdkRequest, AGENTS_SDK_AVAILABLE


# Skip all tests if OpenAI Agents is not available
pytestmark = pytest.mark.skipif(not AGENTS_SDK_AVAILABLE, 
                              reason="OpenAI Agents not installed")


@pytest.fixture
def mock_agent():
    """Fixture to create a mock agent."""
    mock = MagicMock()
    mock.process_query = AsyncMock()
    mock.process_query.return_value = MagicMock(
        response="This is a test response", 
        confidence=0.95,
        sources=[]
    )
    return mock


@pytest.fixture
def mock_openai_client():
    """Fixture to create a mock OpenAI client."""
    mock = MagicMock()
    return mock


@pytest.fixture
def sample_request():
    """Fixture to create a sample Agents request."""
    return AgentsSdkRequest(
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello, can you help me?"}
        ],
        config={},
        metadata={"conversation_id": "test-conversation-123"}
    )


class TestAgentsSdkAdapter:
    """Tests for the Agents adapter."""

    def test_adapter_initialization(self, mock_agent):
        """Test that the adapter initializes correctly."""
        # Skip test if OpenAI Agents is not available
        if not AGENTS_SDK_AVAILABLE:
            pytest.skip("OpenAI Agents not installed")
        
        # Patch os.getenv to return a mock API key
        with patch('os.getenv', return_value="test-api-key"):
            adapter = AgentsSdkAdapter(agent=mock_agent)
            assert adapter.agent == mock_agent
            assert adapter.is_available() == True
            assert adapter.client is not None

    def test_adapter_initialization_no_api_key(self, mock_agent):
        """Test that the adapter handles missing API key."""
        # Skip test if OpenAI Agents is not available
        if not AGENTS_SDK_AVAILABLE:
            pytest.skip("OpenAI Agents not installed")
        
        # Patch os.getenv to return None for API key
        with patch('os.getenv', return_value=None):
            adapter = AgentsSdkAdapter(agent=mock_agent)
            assert adapter.agent == mock_agent
            assert not adapter.is_available()
            assert adapter.client is None

    @pytest.mark.asyncio
    async def test_process_agents_sdk_request(self, mock_agent, sample_request):
        """Test processing an Agents request."""
        # Skip test if OpenAI Agents is not available
        if not AGENTS_SDK_AVAILABLE:
            pytest.skip("OpenAI Agents not installed")
        
        # Patch os.getenv to return a mock API key
        with patch('os.getenv', return_value="test-api-key"):
            adapter = AgentsSdkAdapter(agent=mock_agent)
            
            # Process request
            response = await adapter.process_agents_sdk_request(sample_request)
            
            # Verify agent was called with correct parameters
            mock_agent.process_query.assert_called_once()
            call_args = mock_agent.process_query.call_args[0][0]
            assert call_args.query == "Hello, can you help me?"
            assert call_args.conversation_id == "test-conversation-123"
            
            # Verify response format
            assert isinstance(response, dict)
            assert response.get("type") == "agent_response"
            
            # Verify messages in response
            messages = response.get("messages", [])
            assert len(messages) == 1
            assert messages[0].get("role") == "assistant"
            assert messages[0].get("content") == "This is a test response"
            
            # Verify metadata
            metadata = response.get("metadata", {})
            assert metadata.get("confidence") == 0.95

    @pytest.mark.asyncio
    async def test_error_handling(self, mock_agent, sample_request):
        """Test handling errors during request processing."""
        # Skip test if OpenAI Agents is not available
        if not AGENTS_SDK_AVAILABLE:
            pytest.skip("OpenAI Agents not installed")
        
        # Make the agent raise an exception
        mock_agent.process_query.side_effect = ValueError("Test error")
        
        # Patch os.getenv to return a mock API key
        with patch('os.getenv', return_value="test-api-key"):
            adapter = AgentsSdkAdapter(agent=mock_agent)
            
            # Process request that will cause an error
            response = await adapter.process_agents_sdk_request(sample_request)
            
            # Verify error response format
            assert isinstance(response, dict)
            assert response.get("type") == "error"
            assert "error" in response
            assert response["error"].get("message") == "Test error"

    def test_unavailable_agents_sdk(self):
        """Test behavior when OpenAI Agents is not available."""
        with patch("agent_core.agents_sdk_adapter.AGENTS_SDK_AVAILABLE", False):
            adapter = AgentsSdkAdapter()
            assert not adapter.is_available()
            assert adapter.client is None

    @pytest.mark.asyncio
    async def test_create_thread_response(self, mock_agent):
        """Test create_thread_response method."""
        # Skip test if OpenAI Agents is not available
        if not AGENTS_SDK_AVAILABLE:
            pytest.skip("OpenAI Agents not installed")
        
        # Create mock objects for OpenAI API responses
        mock_message = MagicMock()
        mock_run = MagicMock()
        mock_run.status = "completed"
        mock_run.id = "run-123"
        
        mock_messages_list = MagicMock()
        mock_messages_list.data = [
            MagicMock(
                content=[
                    MagicMock(
                        text=MagicMock(
                            value="This is the assistant's response"
                        )
                    )
                ]
            )
        ]
        
        # Set up mock client
        mock_client = MagicMock()
        mock_client.beta.threads.messages.create.return_value = mock_message
        mock_client.beta.threads.runs.create.return_value = mock_run
        mock_client.beta.threads.runs.retrieve.return_value = mock_run
        mock_client.beta.threads.messages.list.return_value = mock_messages_list
        
        # Create adapter with mock client
        with patch('os.getenv', return_value="test-api-key"):
            adapter = AgentsSdkAdapter(agent=mock_agent)
            adapter.client = mock_client
            
            # Call the method
            response = await adapter.create_thread_response(
                thread_id="thread-123",
                user_message="Hello, assistant"
            )
            
            # Verify the response
            assert isinstance(response, dict)
            assert response.get("type") == "agent_response"
            assert len(response.get("messages", [])) == 1
            assert response["messages"][0]["content"] == "This is the assistant's response"
            assert response["metadata"]["thread_id"] == "thread-123"
            assert response["metadata"]["run_id"] == "run-123"


@pytest.mark.integration
class TestAgentsIntegration:
    """Integration tests for the OpenAI Agents."""
    
    @pytest.mark.asyncio
    async def test_real_agents_adapter(self):
        """
        Test with the real OpenAI Agents adapter.
        
        This test requires the OpenAI Agents to be installed and an API key to be set.
        """
        # Skip if OpenAI Agents not available or no API key
        if not AGENTS_SDK_AVAILABLE:
            pytest.skip("OpenAI Agents not installed")
        
        # Skip if no API key
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")
        
        # Create a real adapter (without mocks)
        adapter = AgentsSdkAdapter()
        
        # Skip if client is not available
        if not adapter.client:
            pytest.skip("OpenAI client not available")
        
        # Create a simple request
        request = AgentsSdkRequest(
            messages=[
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": "What is the current status?"}
            ],
            metadata={"conversation_id": "integration-test-123"}
        )
        
        # Process request
        response = await adapter.process_agents_sdk_request(request)
        
        # Basic verification
        assert isinstance(response, dict)
        assert response.get("type") == "agent_response"
        
        # Verify messages exist
        messages = response.get("messages", [])
        assert len(messages) > 0
        assert "content" in messages[0]
        
        # Should have confidence in metadata
        metadata = response.get("metadata", {})
        assert "confidence" in metadata


def test_imports():
    """Test that all necessary components can be imported."""
    # Skip if OpenAI Agents not available
    if not AGENTS_SDK_AVAILABLE:
        pytest.skip("OpenAI Agents not installed")
    
    # Try importing the main components from OpenAI Agents
    import openai
    from openai import OpenAI
    from openai_agents import AgentMessage, MessageRole, Tool, AgentState, AssistantId, Thread
    
    # Verify the imports worked
    assert openai is not None
    assert OpenAI is not None
    assert AgentMessage is not None
    assert MessageRole is not None
    
    # Import our adapter
    from agent_core.agents_sdk_adapter import AgentsSdkAdapter, AgentsSdkRequest
    assert AgentsSdkAdapter is not None
    assert AgentsSdkRequest is not None 