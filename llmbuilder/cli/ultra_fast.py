#!/usr/bin/env python3
"""
Ultra-fast CLI wrapper for LLMBuilder.
This bypasses all heavy imports until absolutely necessary.
"""

import sys
import os

def main():
    """Ultra-fast main entry point."""
    # Quick help for common commands
    if len(sys.argv) == 1 or (len(sys.argv) == 2 and sys.argv[1] in ['-h', '--help']):
        print("""
LLMBuilder - Fast CLI

Quick Commands:
  llmb status          - Show system status
  llmb init <name>     - Create new project  
  llmb data prepare    - Process data
  llmb train start     - Start training
  llmb --version       - Show version

For full help: llmb help
        """)
        return
    
    # Version check
    if len(sys.argv) == 2 and sys.argv[1] in ['--version', '-V']:
        print("LLMBuilder 1.0.2")
        return
    
    # Status command (ultra-fast)
    if len(sys.argv) == 2 and sys.argv[1] == 'status':
        print("LLMBuilder Status: Ready")
        print("For detailed status: llmb monitor status")
        return
    
    # For all other commands, load the full CLI
    try:
        from llmbuilder.cli.fast_main import main as full_main
        full_main()
    except ImportError:
        # Fallback to regular CLI
        from llmbuilder.cli.main import main as regular_main
        regular_main()

if __name__ == "__main__":
    main()