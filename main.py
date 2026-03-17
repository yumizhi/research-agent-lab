"""Entry point for the multi‑agent research assistant prototype.

This module demonstrates how to use the ``Orchestrator`` to process a
user's research idea. Running this script will execute the pipeline
defined in ``orchestrator.py`` and print the resulting plan and code.
"""

from __future__ import annotations

import json
from pprint import pprint

import os
import sys

# Ensure that the parent directory is on sys.path when this file is executed
# directly (i.e. ``python3 multi_agent_flow/main.py``). Without this,
# relative imports fail because ``multi_agent_flow`` isn't treated as a
# package. When the module is run with ``python -m multi_agent_flow.main``
# this hack isn't necessary.
PACKAGE_ROOT = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(PACKAGE_ROOT)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from multi_agent_flow.orchestrator import Orchestrator


def demo():
    """Run a demonstration with a sample user input."""
    orchestrator = Orchestrator()
    # Example user input; replace with actual idea or existing results
    user_input = (
        "Exploring transformer architectures for energy consumption forecasting in smart grids, "
        "with emphasis on temporal attention mechanisms and comparison to traditional methods."
    )
    final_state = orchestrator.run(user_input)
    print("\n=== Research Plan ===")
    print(final_state.get("plan", "No plan generated."))
    print("\n=== Generated Code ===")
    print(final_state.get("code", "No code generated."))


if __name__ == "__main__":
    demo()