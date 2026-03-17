"""Agent definitions for the research multi-agent MVP."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from models import CandidateTopic, ResearchState
from utils import (
    build_arxiv_query,
    build_candidate_topics,
    build_code_scaffold,
    build_plan_markdown,
    cluster_texts,
    criticize_summary,
    extract_keywords,
    search_arxiv,
    summarize_paper,
    vectorize_texts,
)


class Agent(Protocol):
    """Base class for all agents.

    Subclasses must implement the ``run`` method, which takes a ``state``
    dictionary and returns the updated state. The state may contain
    arbitrary keys and values; agents should avoid modifying keys that
    they do not own.
    """

    name: str = "BaseAgent"

    def run(self, state: ResearchState) -> ResearchState:
        ...


@dataclass
class IdeaAnalyzerAgent(Agent):
    """Extract keywords from the user's input idea or existing results."""

    name: str = "IdeaAnalyzer"

    max_keywords: int = 8

    def run(self, state: ResearchState) -> ResearchState:
        user_input = state["user_input"]
        keywords = extract_keywords(user_input, max_keywords=self.max_keywords)
        state["keywords"] = keywords
        return state


@dataclass
class FetcherAgent(Agent):
    """Search for papers based on extracted keywords."""

    name: str = "Fetcher"
    max_results: int = 10

    def run(self, state: ResearchState) -> ResearchState:
        keywords = state["keywords"]
        if not keywords:
            state["papers"] = []
            return state
        query = build_arxiv_query(keywords)
        papers = search_arxiv(query, max_results=self.max_results)
        state["papers"] = papers
        return state


@dataclass
class SummarizerAgent(Agent):
    """Summarize papers using an LLM."""

    name: str = "Summarizer"
    live_llm: bool = False

    def run(self, state: ResearchState) -> ResearchState:
        summaries = []
        for paper in state["papers"]:
            summaries.append(summarize_paper(paper, live=self.live_llm))
        state["summaries"] = summaries
        return state


@dataclass
class CriticAgent(Agent):
    """Critique summaries and assign scores."""

    name: str = "Critic"
    live_llm: bool = False

    def run(self, state: ResearchState) -> ResearchState:
        critiques = []
        for item in state["summaries"]:
            critiques.append(criticize_summary(item, state["keywords"], live=self.live_llm))
        state["critiques"] = critiques
        return state


@dataclass
class TrendAgent(Agent):
    """Identify candidate topics from scored summaries."""

    name: str = "TrendAnalyzer"
    n_clusters: int = 5
    top_n: int = 3

    def run(self, state: ResearchState) -> ResearchState:
        critiques = state["critiques"]
        candidate_topics: list[CandidateTopic] = []
        texts = [item["summary"] for item in critiques if item["summary"]]
        cluster_labels = None
        if len(texts) >= 3:
            vectors = vectorize_texts(texts)
            cluster_labels = cluster_texts(vectors, n_clusters=min(self.n_clusters, len(texts)))
        candidate_topics = build_candidate_topics(critiques, cluster_labels=cluster_labels, top_n=self.top_n)
        if not candidate_topics:
            title = " / ".join(keyword.title() for keyword in state["keywords"][:3]) or "Research Direction"
            candidate_topics = [
                {
                    "title": title,
                    "rationale": "Fallback topic derived from the user idea because there were too few scored papers.",
                    "source_papers": [paper["title"] for paper in state["papers"][:3]],
                    "score": 0.0,
                }
            ]
        state["candidate_topics"] = candidate_topics
        return state


@dataclass
class PlanAgent(Agent):
    """Generate a detailed research plan for the selected topic."""

    name: str = "Planner"
    live_llm: bool = False

    def run(self, state: ResearchState) -> ResearchState:
        state["plan_markdown"] = build_plan_markdown(state, live=self.live_llm)
        return state


@dataclass
class CodeGenAgent(Agent):
    """Generate code skeletons based on the research plan."""

    name: str = "CodeGenerator"
    live_llm: bool = False

    def run(self, state: ResearchState) -> ResearchState:
        state["generated_code"] = build_code_scaffold(state, live=self.live_llm)
        return state
