"""
Simple test script to verify that the Databricks SDK can be imported.
"""

import sys
print(f"Python version: {sys.version}")
print(f"Python path: {sys.path}")

print("\nTrying to import Databricks SDK...")
try:
    import databricks
    print(f"✅ databricks module imported successfully: {databricks.__file__}")
    
    from databricks.sdk import WorkspaceClient
    print(f"✅ WorkspaceClient imported successfully")
    
    from databricks.sdk.service import jobs, clusters
    print(f"✅ Service modules imported successfully")
    
    print("\nDatabricks SDK import test passed!")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print(f"Failed to import Databricks SDK") 