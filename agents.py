"""Definitions of agent classes for the research assistant prototype.

Each agent encapsulates a single step in the research workflow. Agents
receive a mutable ``state`` dictionary, perform their processing and
update the state with their outputs. The orchestration layer (see
``orchestrator.py``) is responsible for instantiating agents and
invoking them in the correct order.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

from .utils import Paper, search_arxiv, call_llm, vectorize_texts, cluster_texts


class Agent:
    """Base class for all agents.

    Subclasses must implement the ``run`` method, which takes a ``state``
    dictionary and returns the updated state. The state may contain
    arbitrary keys and values; agents should avoid modifying keys that
    they do not own.
    """

    name: str = "BaseAgent"

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError


@dataclass
class IdeaAnalyzerAgent(Agent):
    """Extract keywords from the user's input idea or existing results."""

    name: str = "IdeaAnalyzer"

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        user_input: str = state.get("user_input", "")
        # Simple keyword extraction: split by spaces and filter short words
        words = [w.strip(".,:;!?()[]{}") for w in user_input.lower().split()]
        keywords = [w for w in words if len(w) > 3]
        # Store keywords in state
        state["keywords"] = keywords
        return state


@dataclass
class FetcherAgent(Agent):
    """Search for papers based on extracted keywords."""

    name: str = "Fetcher"
    max_results: int = 10

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        keywords: List[str] = state.get("keywords", [])
        if not keywords:
            return state
        query = "+".join(keywords)
        papers = search_arxiv(query, max_results=self.max_results)
        # Store papers in state
        state.setdefault("papers", [])
        state["papers"] = papers
        return state


@dataclass
class SummarizerAgent(Agent):
    """Summarize papers using an LLM."""

    name: str = "Summarizer"
    summary_prompt_template: str = (
        "Summarize the following research paper in 5 sentences and suggest 3 future improvement ideas:\n\n{title}\n\n"
        "If available, here is an abstract:\n{abstract}\n\nSummary:"
    )

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        papers: List[Paper] = state.get("papers", [])
        summaries: List[Dict[str, Any]] = []
        for paper in papers:
            prompt = self.summary_prompt_template.format(
                title=paper.title,
                abstract=paper.summary or "",
            )
            summary = call_llm(prompt)
            summaries.append({"paper": paper, "summary": summary})
        state["summaries"] = summaries
        return state


@dataclass
class CriticAgent(Agent):
    """Critique summaries and assign scores."""

    name: str = "Critic"
    critique_prompt_template: str = (
        "Evaluate the following summary on three criteria: novelty, methodological soundness, and relevance to the topic.\n"
        "Provide a 0–10 score for each criterion and a short one‑sentence critique.\n\nSummary:\n{summary}\n\nJSON Output:"
    )

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        summaries: List[Dict[str, Any]] = state.get("summaries", [])
        critiques: List[Dict[str, Any]] = []
        for item in summaries:
            summary_text = item.get("summary", "")
            prompt = self.critique_prompt_template.format(summary=summary_text)
            response = call_llm(prompt)
            # For demonstration, produce a dummy score dictionary
            # In practice, the LLM would return JSON; parse accordingly
            critique = {
                "novelty": 7,
                "methodology": 6,
                "relevance": 8,
                "critique": response,
                "paper": item["paper"],
                "summary": summary_text,
            }
            critiques.append(critique)
        state["critiques"] = critiques
        return state


@dataclass
class TrendAgent(Agent):
    """Identify trending topics or clusters from summaries and scores."""

    name: str = "TrendAnalyzer"
    n_clusters: int = 5

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        critiques: List[Dict[str, Any]] = state.get("critiques", [])
        # Use the summary texts for clustering
        texts = [item["summary"] for item in critiques if item.get("summary")]
        if not texts:
            # Nothing to cluster; return state unchanged
            state["clusters"] = None
            state["critiques"] = critiques
            return state
        vectors = vectorize_texts(texts)
        labels = cluster_texts(vectors, n_clusters=self.n_clusters)
        # If clustering failed (e.g. scikit‑learn missing), return without labels
        if labels is None:
            state["clusters"] = None
            state["critiques"] = critiques
            return state
        # Assign cluster labels to the first len(labels) critiques; extra items remain unlabeled
        for idx, label in enumerate(labels):
            critiques[idx]["cluster"] = label
        state["critiques"] = critiques
        state["clusters"] = labels
        return state


@dataclass
class PlanAgent(Agent):
    """Generate a detailed research plan for the selected topic."""

    name: str = "Planner"
    top_n: int = 3

    plan_prompt_template: str = (
        "You are a research planner. Based on the following top papers and critiques, propose a detailed research plan.\n"
        "List the specific research question, required datasets, methods, evaluation metrics, and a timeline divided into phases.\n\n"
        "Papers and critiques:\n{content}\n\nResearch plan:"
    )

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        critiques: List[Dict[str, Any]] = state.get("critiques", [])
        # Select top papers based on combined score (simple sum here)
        scored = []
        for item in critiques:
            score_sum = item.get("novelty", 0) + item.get("methodology", 0) + item.get("relevance", 0)
            scored.append((score_sum, item))
        scored.sort(reverse=True, key=lambda x: x[0])
        top_items = [item for _, item in scored[: self.top_n]]
        content_lines = []
        for i, item in enumerate(top_items, start=1):
            paper: Paper = item["paper"]
            line = f"{i}. {paper.title} — scores(novelty={item.get('novelty')}, methodology={item.get('methodology')}, relevance={item.get('relevance')})."
            content_lines.append(line)
        content = "\n".join(content_lines)
        prompt = self.plan_prompt_template.format(content=content)
        plan_text = call_llm(prompt)
        state["plan"] = plan_text
        return state


@dataclass
class CodeGenAgent(Agent):
    """Generate code skeletons based on the research plan."""

    name: str = "CodeGenerator"
    code_prompt_template: str = (
        "You are a helpful coding assistant. Write Python code to implement the following research plan.\n"
        "The code should set up the experimental environment, including data loading, preprocessing, model definition, training loop, and evaluation.\n"
        "Assume necessary libraries (e.g., PyTorch or TensorFlow) are installed.\n\n"
        "Research plan:\n{plan}\n\nPython code:"
    )

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        plan: str = state.get("plan", "")
        prompt = self.code_prompt_template.format(plan=plan)
        code_text = call_llm(prompt, max_tokens=1024)
        # For the prototype, we simply return the generated code string.
        # In a full implementation, this code could be written to files.
        state["code"] = code_text
        return state