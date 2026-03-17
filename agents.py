"""Agent definitions backed by service-layer abstractions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from models import ResearchState
from services import ServiceBundle
from utils import extract_keywords


class Agent(Protocol):
    """Protocol implemented by all agents."""

    name: str

    def run(self, state: ResearchState) -> ResearchState:
        ...


@dataclass
class IdeaAnalyzerAgent:
    """Extract keywords from the user's input idea."""

    services: ServiceBundle
    max_keywords: int = 10
    name: str = "IdeaAnalyzer"

    def run(self, state: ResearchState) -> ResearchState:
        state["keywords"] = extract_keywords(state["user_input"], max_keywords=self.max_keywords)
        return state


@dataclass
class FetcherAgent:
    """Retrieve literature across multiple sources."""

    services: ServiceBundle
    max_results: int = 12
    name: str = "Fetcher"

    def run(self, state: ResearchState) -> ResearchState:
        papers, metadata = self.services.retrieval.search(state, max_results=self.max_results)
        state["papers"] = papers
        state["retrieval_sources"] = metadata
        state["run_metrics"]["papers_retrieved"] = len(papers)
        return state


@dataclass
class SummarizerAgent:
    """Produce structured summaries for retrieved papers."""

    services: ServiceBundle
    name: str = "Summarizer"

    def run(self, state: ResearchState) -> ResearchState:
        self.services.review.summarize_papers(state)
        return state


@dataclass
class CriticAgent:
    """Score and critique structured summaries."""

    services: ServiceBundle
    name: str = "Critic"

    def run(self, state: ResearchState) -> ResearchState:
        self.services.review.critique_summaries(state)
        return state


@dataclass
class TrendAgent:
    """Select candidate topics from the reviewed literature set."""

    services: ServiceBundle
    top_n: int = 3
    name: str = "TrendAnalyzer"

    def run(self, state: ResearchState) -> ResearchState:
        self.services.review.select_candidate_topics(state, top_n=self.top_n)
        return state


@dataclass
class PlanAgent:
    """Generate a research plan from the selected topic."""

    services: ServiceBundle
    name: str = "Planner"

    def run(self, state: ResearchState) -> ResearchState:
        self.services.planning.build_plan(state)
        return state


@dataclass
class CodeGenAgent:
    """Generate a project scaffold from the research plan."""

    services: ServiceBundle
    name: str = "CodeGenerator"

    def run(self, state: ResearchState) -> ResearchState:
        self.services.planning.build_project_files(state)
        self.services.planning.build_report(state)
        return state
