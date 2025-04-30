# Deprecated Files

The following files are considered deprecated or unnecessary and can be deleted once the project's new structure is verified to be working properly.

## Original Structure Files

These files have been moved to the new structure in the `src/` directory:

- `agent_core/` - Moved to `src/agent_core/`
- `app/` - Moved to `src/app/`

## Test Files 

The following test files should be evaluated and either moved to the proper test directories or deleted:

- `test_tracing_and_logging.py` - Review and move to `tests/unit/` or `tests/integration/`
- `test_databricks_activity.py` - Review and move to `tests/integration/`
- `test_databricks_get_logs.py` - Review and move to `tests/integration/`
- `test_databricks_import.py` - Review and move to `tests/unit/`
- `test_databricks_logs.py` - Review and move to `tests/integration/`
- `test_agents_sdk_install.py` - Review and move to `tests/unit/`
- `test_app_startup.py` - Review and move to `tests/integration/`
- `test_manual.py` - Review and decide if it should be kept

## Next Steps

1. Verify the new structure works correctly
2. Move any useful test files to the appropriate test directories
3. Delete the original directories and files once everything is confirmed working

**Do not delete these files until the new structure is verified to be working properly!** 