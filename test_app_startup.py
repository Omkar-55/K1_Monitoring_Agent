"""
Test script to verify the Streamlit app can start without errors.
This is a simplified test that just imports and checks basic functionality.
"""

import os
import sys
import importlib.util
from pathlib import Path

def test_app_imports():
    """Test that the app imports correctly without errors."""
    print("\n=== Testing App Imports ===")
    
    # Get the path to the app
    app_path = Path(".") / "app" / "main.py"
    
    if not app_path.exists():
        print(f"Error: App file not found at {app_path.absolute()}")
        return False
    
    try:
        # Import the module
        print(f"Importing app from {app_path}")
        spec = importlib.util.spec_from_file_location("app.main", app_path)
        app_module = importlib.util.module_from_spec(spec)
        
        # Patch to avoid Streamlit starting
        import sys
        import unittest.mock
        with unittest.mock.patch.object(sys, 'argv', ['streamlit', 'test']):
            with unittest.mock.patch('streamlit.web.bootstrap.run'):
                spec.loader.exec_module(app_module)
        
        print("App imports successful!")
        
        # Check if the key functions exist
        if hasattr(app_module, 'main'):
            print("Found main() function")
        else:
            print("Warning: main() function not found")
        
        if hasattr(app_module, 'initialize_tracing'):
            print("Found initialize_tracing() function")
        else:
            print("Warning: initialize_tracing() function not found")
            
        if hasattr(app_module, 'configure_logging'):
            print("Found configure_logging() function")
        else:
            print("Warning: configure_logging() function not found")
        
        return True
    
    except ImportError as e:
        print(f"Import error: {e}")
        return False
    except Exception as e:
        print(f"Error testing app imports: {e}")
        return False

if __name__ == "__main__":
    # Run the test
    result = test_app_imports()
    
    # Print summary
    print("\n=== Test Summary ===")
    print(f"App import test: {'PASSED' if result else 'FAILED'}")
    
    # Exit with appropriate code
    sys.exit(0 if result else 1) 