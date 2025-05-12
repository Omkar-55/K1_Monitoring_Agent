#!/usr/bin/env python
"""
Agent State Visualizer - Display the state transitions of the Databricks Monitoring Agent
"""

import os
import re
import sys
import time
import argparse
from datetime import datetime
from colorama import init, Fore, Style

# Initialize colorama for colored terminal output
init()

# Regular expression to match state transitions in logs
STATE_REGEX = re.compile(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) \| INFO \| src\.agents\.databricks_monitoring_agent \| AGENT STATE: (.+)')
FIX_REGEX = re.compile(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) \| INFO \| src\.agents\.databricks_monitoring_agent \| Suggested fix: (.+) \(confidence: ([\d\.]+)\) - awaiting user approval')

def parse_logs(log_file):
    """Parse log file and extract agent state transitions."""
    states = []
    
    try:
        with open(log_file, 'r') as f:
            for line in f:
                # Check for state transitions
                state_match = STATE_REGEX.search(line)
                if state_match:
                    timestamp = state_match.group(1)
                    state_msg = state_match.group(2)
                    states.append((timestamp, state_msg))
                    continue
                
                # Also check for fix suggestions (for backward compatibility)
                fix_match = FIX_REGEX.search(line)
                if fix_match:
                    timestamp = fix_match.group(1)
                    fix_type = fix_match.group(2)
                    confidence = fix_match.group(3)
                    states.append((timestamp, f"Suggested fix: {fix_type} with confidence {confidence}"))
    except FileNotFoundError:
        print(f"Error: Log file {log_file} not found.")
        sys.exit(1)
    
    return states

def get_color_for_state(state_msg):
    """Get appropriate color for state message."""
    if "Starting" in state_msg:
        return Fore.GREEN
    elif "Diagnosing" in state_msg:
        return Fore.BLUE
    elif "Diagnosed issue" in state_msg:
        return Fore.YELLOW
    elif "Suggesting fix" in state_msg or "Suggested fix" in state_msg:
        return Fore.MAGENTA
    elif "Applying fix" in state_msg:
        return Fore.CYAN
    elif "Applied fix" in state_msg:
        return Fore.GREEN
    elif "Generated report" in state_msg:
        return Fore.GREEN
    elif "Error" in state_msg:
        return Fore.RED
    else:
        return Fore.WHITE

def display_state_flow(states, follow=False):
    """Display state transitions in a visually appealing way."""
    if not states:
        print("No state transitions found in the log file.")
        return
    
    # Print header
    print("\n" + "=" * 80)
    print(f"{Fore.CYAN}DATABRICKS MONITORING AGENT - STATE FLOW{Style.RESET_ALL}")
    print("=" * 80)
    
    last_index = -1
    while True:
        # Display any new states
        for i in range(last_index + 1, len(states)):
            timestamp, state_msg = states[i]
            color = get_color_for_state(state_msg)
            dt = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
            time_str = dt.strftime('%H:%M:%S')
            
            # Print the state with appropriate formatting
            print(f"{Fore.WHITE}[{time_str}] {color}{state_msg}{Style.RESET_ALL}")
            
            # Add arrow for flow visualization
            if i < len(states) - 1:
                print(f"{Fore.WHITE}   ↓{Style.RESET_ALL}")
        
        last_index = len(states) - 1
        
        # If not in follow mode, break after displaying all states
        if not follow:
            break
        
        # In follow mode, wait and check for new states
        time.sleep(1)
        states = parse_logs(args.log_file)
        
        # If no new states, continue waiting
        if last_index == len(states) - 1:
            continue

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Visualize Databricks Monitoring Agent state transitions')
    parser.add_argument('--log-file', '-f', type=str, default='logs/agent.log',
                        help='Path to the agent log file (default: logs/agent.log)')
    parser.add_argument('--follow', '-F', action='store_true',
                        help='Follow the log file for new state transitions')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    states = parse_logs(args.log_file)
    display_state_flow(states, args.follow) 