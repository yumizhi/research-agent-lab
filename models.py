"""Typed state and record definitions for the research agent system."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal, NotRequired, TypedDict

RunStatus = Literal["queued", "running", "completed", "failed"]
StageStatus = Literal["pending", "running", "completed", "failed", "skipped"]


def utc_now_iso() -> str:
    """Return a stable UTC timestamp string for persistence."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


class PaperRecord(TypedDict):
    """Normalized metadata for a retrieved paper."""

    title: str
    authors: list[str]
    abstract: str
    source: str
    external_id: str | None
    pdf_url: str | None
    published: str | None
    doi: str | None
    citation_count: int
    url: str | None
    score: float
    snippets: list[str]
    raw: dict[str, Any]


class SummaryRecord(TypedDict):
    """Structured summary of a paper."""

    paper: PaperRecord
    problem: str
    method: str
    datasets: list[str]
    metrics: list[str]
    strengths: list[str]
    weaknesses: list[str]
    reproducibility_risks: list[str]
    improvement_ideas: list[str]
    source_excerpt: str
    summary_markdown: str


class CritiqueRecord(TypedDict):
    """Structured critique of a paper summary."""

    paper: PaperRecord
    summary_markdown: str
    novelty: int
    methodology: int
    relevance: int
    reproducibility: int
    overall: float
    critique: str
    reasons: list[str]
    cluster: NotRequired[int]


class CandidateTopic(TypedDict):
    """Potential research direction selected from the review set."""

    title: str
    rationale: str
    differentiation: list[str]
    failure_modes: list[str]
    source_papers: list[str]
    score: float
    cluster: NotRequired[int]


class GeneratedFileRecord(TypedDict):
    """Generated project file."""

    path: str
    content: str
    description: str


class RetrievalSourceRecord(TypedDict):
    """Per-source retrieval metadata."""

    source: str
    query: str
    returned_count: int
    used_cache: bool
    latency_ms: float


class PromptCallRecord(TypedDict):
    """Logged prompt execution."""

    task: str
    stage: str
    prompt_name: str
    prompt_version: str
    model: str
    latency_ms: float
    input_tokens: int
    output_tokens: int
    success: bool
    response_preview: str
    created_at: str


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


class StageMetric(TypedDict):
    """Runtime metrics for a stage."""

    stage: str
    status: StageStatus
    started_at: str
    ended_at: str
    duration_ms: float


class RunMetrics(TypedDict):
    """Aggregated metrics for a run."""

    stage_metrics: list[StageMetric]
    llm_calls: int
    retrieval_cache_hits: int
    retrieval_cache_misses: int
    papers_retrieved: int
    prompt_calls: list[PromptCallRecord]


class ResearchState(TypedDict):
    """Shared state passed between agents."""

    run_id: str
    status: RunStatus
    current_stage: str
    completed_stages: list[str]
    user_input: str
    keywords: list[str]
    papers: list[PaperRecord]
    summaries: list[SummaryRecord]
    critiques: list[CritiqueRecord]
    candidate_topics: list[CandidateTopic]
    selected_topic: CandidateTopic | None
    plan_markdown: str
    generated_code: str
    generated_files: list[GeneratedFileRecord]
    report_markdown: str
    artifacts: list[ArtifactRecord]
    errors: list[ErrorRecord]
    retrieval_sources: list[RetrievalSourceRecord]
    config_snapshot: dict[str, Any]
    prompt_versions: dict[str, str]
    run_metrics: RunMetrics
    started_at: str
    updated_at: str


def create_research_state(run_id: str, user_input: str, config_snapshot: dict[str, Any] | None = None) -> ResearchState:
    """Create a fresh pipeline state object."""
    now = utc_now_iso()
    return {
        "run_id": run_id,
        "status": "queued",
        "current_stage": "bootstrap",
        "completed_stages": [],
        "user_input": user_input,
        "keywords": [],
        "papers": [],
        "summaries": [],
        "critiques": [],
        "candidate_topics": [],
        "selected_topic": None,
        "plan_markdown": "",
        "generated_code": "",
        "generated_files": [],
        "report_markdown": "",
        "artifacts": [],
        "errors": [],
        "retrieval_sources": [],
        "config_snapshot": config_snapshot or {},
        "prompt_versions": {},
        "run_metrics": {
            "stage_metrics": [],
            "llm_calls": 0,
            "retrieval_cache_hits": 0,
            "retrieval_cache_misses": 0,
            "papers_retrieved": 0,
            "prompt_calls": [],
        },
        "started_at": now,
        "updated_at": now,
    }
