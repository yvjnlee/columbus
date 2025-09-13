#!/usr/bin/env python3
"""
Columbus Agent - Main entry point

A comprehensive AI agent framework with:
- DSPy-optimized planning
- CUA tool integration
- mem0 memory management
- Human-in-the-loop support
- HUD/OS-World benchmarking

Usage:
    uv run columbus                    # Interactive mode
    uv run columbus benchmark          # Run HUD benchmarks
    uv run columbus help-cmd          # Show help
    Visit http://localhost:8080        # Dashboard monitoring
"""

import sys
from columbus.cli import main

if __name__ == "__main__":
    main()
