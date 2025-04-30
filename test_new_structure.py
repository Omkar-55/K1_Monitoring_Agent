#!/usr/bin/env python
"""
Test script to verify the new project structure.

This script checks that the important modules can be imported from their
new locations and that basic functionality works.
"""

import os
import sys
import importlib

GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
RESET = '\033[0m'
BOLD = '\033[1m'

def print_status(message, status, color):
    """Print a status message with color."""
    print(f"{message.ljust(60)} [{color}{status}{RESET}]")

def print_section(title):
    """Print a section title."""
    print(f"\n{BOLD}{title}{RESET}")
    print("=" * 80)

def check_import(module_name):
    """Check if a module can be imported."""
    try:
        importlib.import_module(module_name)
        return True
    except ImportError as e:
        print(f"  Error: {e}")
        return False

def main():
    """Run verification tests for the new project structure."""
    print_section("K1 Monitoring Agent - New Structure Verification")
    
    # Check for critical imports
    print_section("1. Checking Core Module Imports")
    
    modules_to_check = [
        "src.agent_core",
        "src.agent_core.logging_config",
        "src.agent_core.core_logic",
        "src.agent_core.agents_sdk_adapter",
        "src.app.main",
        "src.tools.databricks_tools",
        "src.agents.monitoring_agent"
    ]
    
    all_imports_success = True
    for module in modules_to_check:
        success = check_import(module)
        if success:
            print_status(f"Import {module}", "SUCCESS", GREEN)
        else:
            print_status(f"Import {module}", "FAILED", RED)
            all_imports_success = False
    
    # Check directory structure
    print_section("2. Checking Directory Structure")
    
    directories_to_check = [
        "src",
        "src/agent_core",
        "src/app",
        "src/tools",
        "src/agents",
        "tests/unit",
        "tests/integration"
    ]
    
    all_dirs_exist = True
    for directory in directories_to_check:
        if os.path.isdir(directory):
            print_status(f"Directory {directory}", "EXISTS", GREEN)
        else:
            print_status(f"Directory {directory}", "MISSING", RED)
            all_dirs_exist = False
    
    # Summary
    print_section("Verification Summary")
    
    if all_imports_success and all_dirs_exist:
        print(f"{GREEN}✓ New project structure is properly set up.{RESET}")
        print("You can now start implementing tools and agents in their respective directories.")
    else:
        print(f"{RED}✗ There are issues with the new project structure.{RESET}")
        print("Please fix the issues above before proceeding.")
    
    print(f"\n{BOLD}Next Steps:{RESET}")
    print("1. Implement tools in src/tools/")
    print("2. Implement agents in src/agents/")
    print("3. Once everything is working, delete the deprecated files listed in DEPRECATED_FILES.md")

if __name__ == "__main__":
    main() 