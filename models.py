"""Typed state and record definitions for the research assistant."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal, NotRequired, TypedDict

RunStatus = Literal["running", "completed", "failed"]


def utc_now_iso() -> str:
    """Return a stable UTC timestamp string for persistence."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


class PaperRecord(TypedDict):
    """Normalized metadata for a research paper."""

    title: str
    authors: list[str]
    summary: str
    pdf_url: str | None
    published: str | None
    doi: str | None


class SummaryRecord(TypedDict):
    """Summary produced for a paper."""

    paper: PaperRecord
    summary: str
    improvement_ideas: list[str]


class CritiqueRecord(TypedDict):
    """Evaluation record for a paper summary."""

    paper: PaperRecord
    summary: str
    novelty: int
    methodology: int
    relevance: int
    overall: float
    critique: str
    cluster: NotRequired[int]


class CandidateTopic(TypedDict):
    """Potential research direction selected from the review set."""

    title: str
    rationale: str
    source_papers: list[str]
    score: float
    cluster: NotRequired[int]


class ArtifactRecord(TypedDict):
    """Artifact emitted during orchestration."""

    stage: str
    kind: str
    file_path: str | None
    created_at: str
    payload: dict[str, object] | list[object] | str | None


class ErrorRecord(TypedDict):
    """Structured pipeline error."""

    stage: str
    type: str
    message: str
    created_at: str


class ResearchState(TypedDict):
    """Shared state passed between agents."""

    run_id: str
    status: RunStatus
    user_input: str
    keywords: list[str]
    papers: list[PaperRecord]
    summaries: list[SummaryRecord]
    critiques: list[CritiqueRecord]
    candidate_topics: list[CandidateTopic]
    plan_markdown: str
    generated_code: str
    artifacts: list[ArtifactRecord]
    errors: list[ErrorRecord]
    started_at: str
    updated_at: str


def create_research_state(run_id: str, user_input: str) -> ResearchState:
    """Create a fresh pipeline state object."""
    now = utc_now_iso()
    return {
        "run_id": run_id,
        "status": "running",
        "user_input": user_input,
        "keywords": [],
        "papers": [],
        "summaries": [],
        "critiques": [],
        "candidate_topics": [],
        "plan_markdown": "",
        "generated_code": "",
        "artifacts": [],
        "errors": [],
        "started_at": now,
        "updated_at": now,
    }
