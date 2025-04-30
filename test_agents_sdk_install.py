#!/usr/bin/env python
"""
Verification script for OpenAI Agents integration.

This script tests the installation of the OpenAI Agents and the functionality
of the AgentsSdkAdapter in the K1 Monitoring Agent.

Run this script to verify that everything is working correctly:
    python test_agents_sdk_install.py
"""

import os
import sys
from typing import Dict, Any, List, Optional

# Define colors for output
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
RESET = '\033[0m'
BOLD = '\033[1m'

def print_status(message: str, status: str, color: str) -> None:
    """Print a status message with color."""
    print(f"{message.ljust(60)} [{color}{status}{RESET}]")

def print_section(title: str) -> None:
    """Print a section title."""
    print(f"\n{BOLD}{title}{RESET}")
    print("=" * 80)

def check_import(module_name: str) -> bool:
    """Check if a module can be imported."""
    try:
        __import__(module_name)
        return True
    except ImportError:
        return False

def main() -> None:
    """Run verification tests for the OpenAI Agents integration."""
    print_section("K1 Monitoring Agent - OpenAI Agents Integration Verification")
    
    # Check for OpenAI Agents installation
    print_section("1. Checking OpenAI Agents Installation")
    
    openai_installed = check_import("openai")
    if openai_installed:
        print_status("OpenAI Python SDK", "INSTALLED", GREEN)
    else:
        print_status("OpenAI Python SDK", "NOT FOUND", RED)
        print(f"\n{RED}OpenAI SDK not found. Please install it with:{RESET}")
        print("    pip install openai")
        sys.exit(1)
    
    openai_agents_installed = check_import("openai_agents")
    if openai_agents_installed:
        print_status("OpenAI Agents SDK", "INSTALLED", GREEN)
    else:
        print_status("OpenAI Agents SDK", "NOT FOUND", RED)
        print(f"\n{RED}OpenAI Agents SDK not found. Please install it with:{RESET}")
        print("    pip install openai-agents")
        sys.exit(1)
    
    # Check API key
    print_section("2. Checking OpenAI API Key")
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        print_status("OpenAI API Key", "FOUND", GREEN)
    else:
        print_status("OpenAI API Key", "NOT FOUND", YELLOW)
        print(f"\n{YELLOW}OpenAI API Key not found. Please set the OPENAI_API_KEY environment variable.{RESET}")
        print("You can still use the adapter with a mock agent, but real OpenAI integration will not work.")
    
    # Check adapter imports
    print_section("3. Checking Adapter Integration")
    try:
        from agent_core.agents_sdk_adapter import AgentsSdkAdapter, AgentsSdkRequest, AGENTS_SDK_AVAILABLE
        print_status("Adapter Imports", "SUCCESS", GREEN)
        
        if AGENTS_SDK_AVAILABLE:
            print_status("OpenAI Agents Available Flag", "TRUE", GREEN)
        else:
            print_status("OpenAI Agents Available Flag", "FALSE", RED)
            print(f"\n{RED}OpenAI Agents integration is not available. Check the imports above.{RESET}")
            sys.exit(1)
        
        # Test basic adapter functionality
        adapter = AgentsSdkAdapter()
        print_status("Adapter Initialization", "SUCCESS", GREEN)
        
        if adapter.is_available():
            if api_key:
                print_status("Adapter Status", "READY", GREEN)
            else:
                print_status("Adapter Status", "UNAVAILABLE (No API key)", YELLOW)
        else:
            print_status("Adapter Status", "UNAVAILABLE", RED)
        
        # Create a sample request to test serialization
        request = AgentsSdkRequest(
            messages=[{"role": "user", "content": "Hello"}],
            config={"model": "gpt-4"},
            metadata={"test": True}
        )
        print_status("Request Creation", "SUCCESS", GREEN)
        
    except ImportError as e:
        print_status("Adapter Imports", "FAILED", RED)
        print(f"\n{RED}Error importing adapter components: {e}{RESET}")
        sys.exit(1)
    except Exception as e:
        print_status("Adapter Functionality", "FAILED", RED)
        print(f"\n{RED}Error during adapter testing: {e}{RESET}")
        sys.exit(1)
    
    # Summary
    print_section("Verification Summary")
    if api_key:
        print(f"{GREEN}✓ OpenAI Agents integration is properly installed and configured.{RESET}")
        print("You can now use the AgentsSdkAdapter in your application.")
    else:
        print(f"{YELLOW}⚠ OpenAI Agents integration is installed but not fully configured.{RESET}")
        print("Please set the OPENAI_API_KEY environment variable to enable full functionality.")
    
    print(f"\n{BOLD}Next Steps:{RESET}")
    print("1. Run the tests to verify functionality:")
    print("   pytest tests/test_agents_sdk.py")
    print("2. Integrate the adapter into your application:")
    print("   from agent_core.agents_sdk_adapter import AgentsSdkAdapter")
    print("   adapter = AgentsSdkAdapter()")
    print("   # Use adapter.process_agents_sdk_request() to handle messages")

if __name__ == "__main__":
    main() 