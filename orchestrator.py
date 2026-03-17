"""Orchestrator for the multi‑agent research assistant prototype.

The ``Orchestrator`` class coordinates the execution of individual agents
defined in ``agents.py``. It maintains a ``state`` dictionary that is
passed through each agent in sequence. The orchestrator could be
extended to support more complex control flow using a framework like
LangGraph, but here we implement a simple sequential pipeline for
demonstration purposes.
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional

from .agents import (
    IdeaAnalyzerAgent,
    FetcherAgent,
    SummarizerAgent,
    CriticAgent,
    TrendAgent,
    PlanAgent,
    CodeGenAgent,
)


class Orchestrator:
    """Coordinates the multi‑agent workflow."""

    def __init__(self):
        # Instantiate agents. Parameters can be adjusted as needed.
        self.agents: List = [
            IdeaAnalyzerAgent(),
            FetcherAgent(max_results=10),
            SummarizerAgent(),
            CriticAgent(),
            TrendAgent(n_clusters=5),
            PlanAgent(top_n=3),
            CodeGenAgent(),
        ]

    def run(self, user_input: str, initial_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Run the full pipeline on a user input.

        Parameters
        ----------
        user_input: str
            The initial research idea or description provided by the user.
        initial_state: Optional[Dict[str, Any]]
            An optional state dictionary with pre‑existing data. If none
            is provided, a new dictionary is created.

        Returns
        -------
        Dict[str, Any]
            The final state after running all agents.
        """
        state = initial_state.copy() if initial_state is not None else {}
        state["user_input"] = user_input
        for agent in self.agents:
            state = agent.run(state)
        return state