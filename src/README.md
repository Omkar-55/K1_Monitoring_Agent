# K1 Monitoring Agent Source Code

This directory contains the source code for the K1 Monitoring Agent.

## Directory Structure

- **agent_core/**: Core functionality of the monitoring agent
  - `__init__.py`: Package initialization and tracing setup
  - `agents_sdk_adapter.py`: Adapter for the Agents SDK protocol
  - `core_logic.py`: Core business logic for the agent
  - `dbx_client.py`: Databricks client integration
  - `logging_config.py`: Centralized logging configuration

- **app/**: Streamlit application frontend
  - `main.py`: Main application entry point
  - `requirements.txt`: Application-specific dependencies

- **tools/**: Tools used by the agent
  - `databricks_tools.py`: Tools for interacting with Databricks

- **agents/**: Agent implementations
  - `monitoring_agent.py`: Main monitoring agent implementation

## Development Guidelines

Follow the guidelines in the project's `cursor.rules` file for:
- Azure OpenAI integration
- AgentsSDK usage
- Tracing and monitoring
- Testing practices
- Documentation standards 