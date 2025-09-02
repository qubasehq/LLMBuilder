"""
Entry point for running LLMBuilder as a module.

Usage: python -m llmbuilder [command] [options]
"""

from llmbuilder.cli.main import cli

if __name__ == "__main__":
    cli()