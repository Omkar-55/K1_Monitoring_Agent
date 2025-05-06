"""
Tests for the logging functionality of the K1 Monitoring Agent.
"""

import os
import sys
import pytest
import logging
from unittest.mock import patch, MagicMock

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import the logging configuration
from src.agent_core.logging_config import setup_logging, get_logger

class TestLogging:
    """Tests for the logging functionality."""
    
    def test_setup_logging(self):
        """Test that setup_logging configures logging with the correct handlers."""
        # Call setup_logging with a test log level
        result = setup_logging(log_level="INFO")
        
        # Verify the result
        assert result is True, "setup_logging should return True on success"
        
        # Get the root logger to inspect handlers
        root_logger = logging.getLogger()
        
        # Check handlers
        assert len(root_logger.handlers) >= 1, "Root logger should have at least one handler"
        
        # Check log level
        assert root_logger.level <= logging.INFO, "Root logger should have level INFO or lower"
    
    def test_get_logger(self):
        """Test that get_logger returns a properly configured logger."""
        # Get a test logger
        test_logger = get_logger("test_module")
        
        # Verify logger properties
        assert test_logger.name == "test_module", "Logger should have the correct name"
        assert test_logger.level <= logging.INFO, "Logger should have level INFO or lower"
        
        # Check that the logger can log
        with patch("logging.Logger.info") as mock_info:
            test_logger.info("Test message")
            mock_info.assert_called_once_with("Test message")
    
    def test_log_file_creation(self):
        """Test that log files are created in the correct directory."""
        # Set up logging with a specific file name
        log_dir = "logs"
        log_file = os.path.join(log_dir, "test_log.log")
        
        # Ensure the log directory exists and file doesn't exist yet
        os.makedirs(log_dir, exist_ok=True)
        if os.path.exists(log_file):
            os.remove(log_file)
        
        # Set up logging with the test file
        setup_logging(log_level="INFO", log_file=log_file)
        
        # Get a logger and log a message
        logger = get_logger("test_file_logger")
        logger.info("Test log message for file creation")
        
        # Check if the log file was created
        assert os.path.exists(log_file), f"Log file should exist at {log_file}"
        
        # Check if the message is in the log file
        with open(log_file, "r") as f:
            log_content = f.read()
            assert "Test log message for file creation" in log_content, "Log message should be in the file"
        
        # Clean up
        try:
            os.remove(log_file)
        except:
            pass
    
    def test_log_formatting(self):
        """Test that logs are formatted correctly."""
        # Create a mock handler to capture log records
        mock_handler = MagicMock()
        mock_handler.level = logging.INFO  # Add level attribute
        
        # Get a logger and add the mock handler
        logger = get_logger("test_format_logger")
        logger.addHandler(mock_handler)
        
        # Log a test message
        logger.info("Test format message")
        
        # Check if the handler was called
        assert mock_handler.handle.called, "Handler should be called when logging"
        
        # Get the log record
        call_args = mock_handler.handle.call_args[0]
        assert len(call_args) > 0, "Handler should receive log records"
        
        log_record = call_args[0]
        assert log_record.name == "test_format_logger", "Log record should have correct logger name"
        assert log_record.levelname == "INFO", "Log record should have correct level name"
        assert log_record.message == "Test format message", "Log record should have correct message"

if __name__ == "__main__":
    pytest.main(["-xvs", "test_logging.py"])
