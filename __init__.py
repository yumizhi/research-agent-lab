"""Top‑level package for the multi‑agent research assistant prototype.

This package contains a set of modular agents, utility functions and an
orchestrator used to build a research workflow. Each agent is responsible
for a specific part of the pipeline: keyword extraction, literature search,
summarization, critique, trend analysis, research planning and code
generation. The orchestrator coordinates the agents and handles shared
state. See the accompanying documentation for details.
"""

__all__ = [
    "agents",
    "utils",
    "orchestrator",
]