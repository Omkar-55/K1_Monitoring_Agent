# Deprecated Files

This file tracks files that are deprecated and should be removed from the codebase.

## Deprecated Files Ready for Removal

The following files can be safely removed as their functionality has been migrated to the new structure:

- `get_databricks_logs.py` - This functionality is now implemented in `src/tools/databricks_api_tools.py` and the monitoring tools in `src/tools/databricks_monitoring/`.
- `src/tools/databricks_monitoring/reporting_tools.py.bak` - This is a backup file of reporting_tools.py that is no longer needed.

## Previously Moved Files 

These directories were previously moved to the new structure and have already been removed:

- `agent_core/` - Moved to `src/agent_core/`
- `app/` - Moved to `src/app/`

## Test Files 

The following test files were previously in the DEPRECATED_FILES.md and have been properly migrated or deleted:

- ~~`test_tracing_and_logging.py`~~ - Replaced by `tests/test_logging.py`
- ~~`test_databricks_activity.py`~~ - Removed (no longer needed)
- ~~`test_databricks_get_logs.py`~~ - Removed (no longer needed)
- ~~`test_databricks_import.py`~~ - Removed (no longer needed)
- ~~`test_databricks_logs.py`~~ - Removed (no longer needed)
- ~~`test_agents_sdk_install.py`~~ - Removed (no longer needed)
- ~~`test_app_startup.py`~~ - Removed (no longer needed)
- ~~`test_manual.py`~~ - Removed (no longer needed)

## Next Steps

1. Remove `get_databricks_logs.py` from the root directory
2. Regularly review the codebase for other unnecessary or redundant files
3. Keep this file updated as the codebase evolves

**Note:** When removing files, make sure all necessary functionality has been preserved in the new structure. 