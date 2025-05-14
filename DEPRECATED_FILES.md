# Deprecated Files

This file tracks files that are deprecated and should be removed from the codebase.

## Recently Removed Files

The following files have been removed in the latest cleanup (Production Release):

- `test_streamlit.py` - Test file for UI components (now implemented in main.py)
- `fix_streamlit.py` - Test file for UI fixes (no longer needed)
- `debug_reasoning.json` - Debug file (not needed in production)
- `simple_agent_cli.py` - CLI version (superseded by Streamlit app, but core CLI functionality preserved in `src/cli.py`)
- `agent_visualizer.py` - Visualization tool (integrated into main app)
- `monitor_job.py` - Standalone monitoring script (integrated into main app)

## Retained Core Files

The following files were initially considered for removal but are retained for specific purposes:

- `src/cli.py` - Core command-line interface for programmatic access and automation (not deprecated)

## Previously Removed Files

These files were previously removed as their functionality was migrated:

- `get_databricks_logs.py` - Functionality moved to `src/tools/databricks_api_tools.py`
- `src/tools/databricks_monitoring/reporting_tools.py.bak` - Backup file removed
- `agent_core/` - Moved to `src/agent_core/`
- `app/` - Moved to `src/app/`

## Previously Removed Test Files

The following test files were properly migrated or removed:

- ~~`test_tracing_and_logging.py`~~ - Replaced by `tests/test_logging.py`
- ~~`test_databricks_activity.py`~~ - Removed
- ~~`test_databricks_get_logs.py`~~ - Removed
- ~~`test_databricks_import.py`~~ - Removed
- ~~`test_databricks_logs.py`~~ - Removed
- ~~`test_agents_sdk_install.py`~~ - Removed
- ~~`test_app_startup.py`~~ - Removed
- ~~`test_manual.py`~~ - Removed

## Current Test Structure

All tests should now be organized in the `tests/` directory following this structure:
- `tests/unit/` - Unit tests
- `tests/integration/` - Integration tests (including `test_monitoring_agent.py`)
- `tests/e2e/` - End-to-end tests

## Next Steps

1. Review any remaining files in the `tests/` directory for redundancy
2. Ensure all test files follow the new directory structure
3. Keep this file updated as the codebase evolves

**Note:** When removing files, ensure all necessary functionality has been preserved in the new structure. 