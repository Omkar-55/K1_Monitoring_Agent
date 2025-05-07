# K1 Monitoring Agent

A monitoring agent with robust logging and tracing capabilities specifically designed for Databricks workspaces.

## Project Structure

```
.
├── src/
│   ├── agent_core/           # Core functionality 
│   │   ├── __init__.py       # Contains enable_tracing()
│   │   ├── logging_config.py # Centralizes logging setup
│   │   ├── core_logic.py     # Agent's main functions
│   │   ├── dbx_client.py     # Databricks client
│   │   └── agents_sdk_adapter.py # Agents SDK integration
│   │
│   ├── app/                 # Streamlit UI
│   │   ├── main.py          # Web application interface
│   │   └── requirements.txt
│   │
│   ├── tools/               # Agent tools
│   │   ├── databricks_tools.py # Databricks-specific tools
│   │   ├── azure_openai_tools.py # Azure OpenAI integration
│   │   └── databricks_monitoring/ # Advanced diagnosis tools
│   │
│   └── agents/              # Agent implementations
│       └── databricks_monitoring_agent.py # Primary Databricks agent with AI capabilities
│
├── tests/
│   ├── unit/                # Unit tests
│   └── integration/         # Integration tests
│
├── logs/                    # Auto-created at runtime
├── cursor.rules             # Development guidelines
└── README.md
```

## Features

- **Databricks Monitoring**: Advanced monitoring for Databricks workspaces
- **AI-Powered Diagnosis**: Intelligent diagnosis of Databricks job failures
- **Automated Fixes**: Suggests and applies fixes to common Databricks issues
- **Centralized Logging**: File and console logging with configurable log levels
- **Distributed Tracing**: OpenTelemetry-based tracing for monitoring requests
- **Azure OpenAI Integration**: Connects to Azure OpenAI for intelligent analysis
- **Agents SDK Integration**: Adapter for the Agents SDK protocol
- **Streamlit UI**: Web interface for configuration and usage

## Getting Started

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/k1-monitoring-agent.git
   cd k1-monitoring-agent
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   ```
   # Azure OpenAI credentials
   AZURE_OPENAI_API_KEY=your-api-key
   AZURE_OPENAI_ENDPOINT=your-endpoint
   AZURE_OPENAI_API_VERSION=2023-05-15
   AZURE_OPENAI_DEPLOYMENT=gpt-4
   
   # Databricks credentials
   DATABRICKS_HOST=your-databricks-workspace
   DATABRICKS_TOKEN=your-databricks-token
   ```

### Running the Application

To run the Streamlit application:

```bash
streamlit run src/app/main.py
```

To run the command-line interface:

```bash
python -m src.cli monitor --job-id JOB_ID
```

### Using the Monitoring Agent

The K1 Monitoring Agent is focused on intelligent monitoring and issue resolution for Databricks workspaces. It uses the DatabricksMonitoringAgent to provide:

- Automated diagnosis of job failures
- Intelligent fix suggestions
- Implementation of approved fixes
- Verification of fix effectiveness
- Comprehensive reporting

The agent can be accessed through the Streamlit UI or the command-line interface.

### Testing

To run the automated tests:

```bash
pytest tests/
```

## Logging

The logging system automatically creates log files in the `logs` directory. By default, logs are sent to both the console and a rotating file handler.

To use the logger in your code:

```python
from src.agent_core.logging_config import get_logger

logger = get_logger(__name__)
logger.info("This is an informational message")
logger.error("This is an error message")
```

## Tracing

Distributed tracing is provided through OpenTelemetry:

```python
from src.agent_core import enable_tracing

# Enable tracing
enable_tracing(service_name="my-service")

# Create spans in your code
from opentelemetry import trace
tracer = trace.get_tracer(__name__)

with tracer.start_as_current_span("operation_name") as span:
    span.set_attribute("attribute_name", "value")
    # Your code here
```

## Agents SDK Integration

The K1 Monitoring Agent integrates with the Agents SDK protocol:

```python
from src.agent_core.agents_sdk_adapter import AgentsSdkAdapter, AgentsSdkRequest

# Create an adapter
adapter = AgentsSdkAdapter()

# Create a request
request = AgentsSdkRequest(
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "What is the status of the Databricks job?"}
    ],
    metadata={"conversation_id": "test-123"}
)

# Process the request asynchronously
response = await adapter.process_agents_sdk_request(request)
```

## Requirements

- Python 3.8+
- OpenTelemetry packages
- Azure OpenAI access
- Agents SDK (for protocol integration)
- Databricks API access

## License

[Your License Information]

## Architecture

The K1 Monitoring Agent is structured around a powerful agent for Databricks workspace monitoring and issue resolution:

1. **DatabricksMonitoringAgent (Primary)** - An advanced agent with:
   - AI-powered diagnostics
   - Automated fix generation and application
   - Safety checks and guardrails
   - Verification of fix effectiveness
   - Comprehensive reporting

For production scenarios, the DatabricksMonitoringAgent is the primary agent used in the application workflow.
