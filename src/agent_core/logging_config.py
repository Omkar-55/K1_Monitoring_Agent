"""
Centralized logging configuration for the K1 Monitoring Agent.
Configures both file and console logging handlers.
"""

import os
import sys
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

# Default log format
DEFAULT_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Log levels
LOG_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL
}

def get_project_root() -> Path:
    """Get the project root directory.
    
    This function handles both running from the project root and from the src directory,
    ensuring logs are always placed in the project root's logs directory.
    """
    # Get the directory of the current file (logging_config.py)
    current_dir = Path(__file__).parent
    
    # If the current directory is agent_core inside src, go up two levels
    if current_dir.name == "agent_core" and current_dir.parent.name == "src":
        return current_dir.parent.parent
    # If the current directory is just agent_core, go up one level
    elif current_dir.name == "agent_core":
        return current_dir.parent
    # Default fallback - use current working directory
    else:
        return Path(os.getcwd())

def setup_logging(
    log_level: str = "info",
    log_format: str = DEFAULT_LOG_FORMAT,
    date_format: str = DEFAULT_DATE_FORMAT,
    log_to_console: bool = True,
    log_file_name: str = "agent.log",
    log_file_max_size: int = 10 * 1024 * 1024,  # 10 MB
    log_file_backup_count: int = 5
) -> None:
    """
    Configure logging with both file and console handlers.
    
    Args:
        log_level: The log level (debug, info, warning, error, critical)
        log_format: The log message format string
        date_format: The date format for log timestamps
        log_to_console: Whether to output logs to console
        log_file_name: The name of the log file
        log_file_max_size: Maximum size of the log file before rotation
        log_file_backup_count: Number of backup log files to keep
    """
    # Convert log level string to logging level
    level = LOG_LEVELS.get(log_level.lower(), logging.INFO)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter(log_format, date_format)
    
    # Create logs directory if it doesn't exist
    project_root = get_project_root()
    logs_dir = project_root / "logs"
    os.makedirs(logs_dir, exist_ok=True)
    
    # Configure file handler for logging to file
    log_file_path = logs_dir / log_file_name
    file_handler = RotatingFileHandler(
        filename=log_file_path,
        maxBytes=log_file_max_size,
        backupCount=log_file_backup_count
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    # Configure console handler for logging to stdout
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # Log that the logging system has been initialized
    logging.info(f"Logging initialized. Log file: {log_file_path}")

# Default logger for this module
logger = logging.getLogger(__name__)

# Convenience function to get a logger for a specific module
def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.
    
    Args:
        name: The name for the logger, typically __name__
        
    Returns:
        A logger instance configured with the appropriate handlers
    """
    # Ensure logging is setup if this is the first logger requested
    if not logging.getLogger().handlers:
        setup_logging()
    
    return logging.getLogger(name)

# Initialize logging with default settings when the module is imported
# Can be reconfigured later if needed
setup_logging()
