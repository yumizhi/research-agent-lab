"""Service layer for retrieval, LLM calls, review, planning, and reporting."""

from __future__ import annotations

import json
import logging
import re
import textwrap
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable

try:
    import requests  # type: ignore
except ImportError:
    requests = None

from config import AppConfig
from logging_utils import StructuredLogger
from models import CandidateTopic, GeneratedFileRecord, PaperRecord, PromptCallRecord, ResearchState
from prompting import PromptManager
from repository import ResearchRepository
from utils import (
    cluster_texts,
    dedupe_papers,
    estimate_tokens,
    extract_keywords,
    first_sentence,
    keyword_overlap_score,
    normalize_text,
    parse_arxiv_response,
    parse_json_object,
    stable_hash,
    vectorize_texts,
)

logger = logging.getLogger(__name__)


class LLMConfigurationError(RuntimeError):
    """Raised when a live LLM call cannot be completed."""


class LLMService:
    """Prompt-driven LLM service with routing, caching, and call logging."""

    def __init__(
        self,
        config: AppConfig,
        prompts: PromptManager,
        repository: ResearchRepository,
        run_logger: StructuredLogger,
    ):
        self.config = config
        self.prompts = prompts
        self.repository = repository
        self.run_logger = run_logger

    def _call_live_model(self, prompt: str, model: str) -> str:
        if requests is None:
            raise LLMConfigurationError("Live LLM mode requires the requests package.")

        api_key = __import__("os").environ.get("OPENAI_API_KEY")
        if not api_key:
            raise LLMConfigurationError("Live LLM mode requires OPENAI_API_KEY.")

        base_url = __import__("os").environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        endpoint = f"{base_url}/chat/completions"
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a precise research workflow assistant."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
            "max_tokens": 1200,
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        last_error: Exception | None = None
        for _ in range(max(1, self.config.llm_retries)):
            try:
                response = requests.post(endpoint, headers=headers, json=payload, timeout=self.config.llm_timeout_seconds)
                response.raise_for_status()
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                if isinstance(content, str):
                    return content.strip()
                if isinstance(content, list):
                    return "\n".join(part.get("text", "") for part in content if isinstance(part, dict)).strip()
                raise LLMConfigurationError("Unsupported content format returned by the live model.")
            except Exception as exc:
                last_error = exc
        raise LLMConfigurationError(f"Live LLM request failed: {last_error}")

    def _record_prompt_call(
        self,
        state: ResearchState,
        *,
        task: str,
        stage: str,
        prompt_name: str,
        prompt_version: str,
        model: str,
        latency_ms: float,
        input_text: str,
        output_text: str,
        success: bool,
    ) -> None:
        record: PromptCallRecord = {
            "task": task,
            "stage": stage,
            "prompt_name": prompt_name,
            "prompt_version": prompt_version,
            "model": model,
            "latency_ms": latency_ms,
            "input_tokens": estimate_tokens(input_text),
            "output_tokens": estimate_tokens(output_text),
            "success": success,
            "response_preview": normalize_text(output_text)[:200],
            "created_at": state["updated_at"],
        }
        state["run_metrics"]["prompt_calls"].append(record)
        state["run_metrics"]["llm_calls"] += 1
        self.repository.record_prompt_call(state["run_id"], record)

    def generate_text(
        self,
        state: ResearchState,
        *,
        stage: str,
        task: str,
        prompt_name: str,
        variables: dict[str, object],
        fallback_builder: Callable[[], str],
    ) -> str:
        prompt, version = self.prompts.render(prompt_name, **variables)
        model = self.config.model_for(task)
        cache_key = stable_hash(f"text::{task}::{model}::{version}::{prompt}")
        cached = self.repository.cache_get("llm", cache_key) if self.config.enable_cache else None
        if cached is not None:
            state["run_metrics"]["retrieval_cache_hits"] += 1
            response_text = str(cached)
            self._record_prompt_call(
                state,
                task=task,
                stage=stage,
                prompt_name=prompt_name,
                prompt_version=version,
                model=model,
                latency_ms=0.0,
                input_text=prompt,
                output_text=response_text,
                success=True,
            )
            return response_text

        start = time.perf_counter()
        if self.config.live_llm:
            response_text = self._call_live_model(prompt, model)
        else:
            response_text = fallback_builder()
        latency_ms = round((time.perf_counter() - start) * 1000, 3)
        if self.config.enable_cache:
            self.repository.cache_set("llm", cache_key, response_text)
        self._record_prompt_call(
            state,
            task=task,
            stage=stage,
            prompt_name=prompt_name,
            prompt_version=version,
            model=model,
            latency_ms=latency_ms,
            input_text=prompt,
            output_text=response_text,
            success=True,
        )
        return response_text

    def generate_json(
        self,
        state: ResearchState,
        *,
        stage: str,
        task: str,
        prompt_name: str,
        variables: dict[str, object],
        required_keys: list[str],
        fallback_builder: Callable[[], dict[str, Any]],
    ) -> dict[str, Any]:
        prompt, version = self.prompts.render(prompt_name, **variables)
        model = self.config.model_for(task)
        cache_key = stable_hash(f"json::{task}::{model}::{version}::{prompt}")
        cached = self.repository.cache_get("llm", cache_key) if self.config.enable_cache else None
        if cached is not None:
            state["run_metrics"]["retrieval_cache_hits"] += 1
            output_text = json.dumps(cached, ensure_ascii=False)
            self._record_prompt_call(
                state,
                task=task,
                stage=stage,
                prompt_name=prompt_name,
                prompt_version=version,
                model=model,
                latency_ms=0.0,
                input_text=prompt,
                output_text=output_text,
                success=True,
            )
            return cached

        start = time.perf_counter()
        if self.config.live_llm:
            response_text = self._call_live_model(prompt, model)
            parsed = parse_json_object(response_text)
            missing = [key for key in required_keys if key not in parsed]
            if missing:
                raise LLMConfigurationError(f"Missing required keys in model response: {missing}")
            output = parsed
        else:
            output = fallback_builder()
            response_text = json.dumps(output, ensure_ascii=False)
        latency_ms = round((time.perf_counter() - start) * 1000, 3)
        if self.config.enable_cache:
            self.repository.cache_set("llm", cache_key, output)
        self._record_prompt_call(
            state,
            task=task,
            stage=stage,
            prompt_name=prompt_name,
            prompt_version=version,
            model=model,
            latency_ms=latency_ms,
            input_text=prompt,
            output_text=response_text,
            success=True,
        )
        return output


class RetrievalService:
    """Multi-source paper retrieval with caching, deduplication, and reranking."""

    def __init__(self, config: AppConfig, repository: ResearchRepository, run_logger: StructuredLogger):
        self.config = config
        self.repository = repository
        self.run_logger = run_logger

    def _search_arxiv(self, query: str, max_results: int) -> list[PaperRecord]:
        if requests is None:
            return []
        endpoint = "http://export.arxiv.org/api/query"
        params = {
            "search_query": query,
            "start": 0,
            "max_results": max_results,
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
        try:
            response = requests.get(endpoint, params=params, timeout=15)
            response.raise_for_status()
        except Exception:
            return []
        return parse_arxiv_response(response.text)

    def _search_semantic_scholar(self, query: str, max_results: int) -> list[PaperRecord]:
        if requests is None:
            return []
        endpoint = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {
            "query": query,
            "limit": max_results,
            "fields": "title,abstract,authors,citationCount,year,url,externalIds,openAccessPdf",
        }
        try:
            response = requests.get(endpoint, params=params, timeout=15)
            response.raise_for_status()
            payload = response.json()
        except Exception:
            return []

        papers = []
        for item in payload.get("data", []):
            title = normalize_text(item.get("title", ""))
            abstract = normalize_text(item.get("abstract", ""))
            authors = [normalize_text(author.get("name", "")) for author in item.get("authors", []) if author.get("name")]
            open_access_pdf = item.get("openAccessPdf") or {}
            source_id = item.get("paperId")
            papers.append(
                {
                    "title": title,
                    "authors": authors,
                    "abstract": abstract,
                    "source": "semantic_scholar",
                    "external_id": source_id,
                    "pdf_url": open_access_pdf.get("url"),
                    "published": str(item.get("year")) if item.get("year") else None,
                    "doi": ((item.get("externalIds") or {}).get("DOI")),
                    "citation_count": int(item.get("citationCount") or 0),
                    "url": item.get("url"),
                    "score": 0.0,
                    "snippets": [first_sentence(abstract)] if abstract else [],
                    "raw": item,
                }
            )
        return papers

    def _search_crossref(self, query: str, max_results: int) -> list[PaperRecord]:
        if requests is None:
            return []
        endpoint = "https://api.crossref.org/works"
        params = {"query": query, "rows": max_results}
        try:
            response = requests.get(endpoint, params=params, timeout=15)
            response.raise_for_status()
            payload = response.json()
        except Exception:
            return []

        papers = []
        for item in payload.get("message", {}).get("items", []):
            title = normalize_text(" ".join(item.get("title", [])))
            abstract = normalize_text(re.sub(r"<[^>]+>", " ", item.get("abstract", ""))) if item.get("abstract") else ""
            authors = []
            for author in item.get("author", []):
                name = " ".join(part for part in [author.get("given"), author.get("family")] if part)
                if name:
                    authors.append(normalize_text(name))
            papers.append(
                {
                    "title": title,
                    "authors": authors,
                    "abstract": abstract,
                    "source": "crossref",
                    "external_id": item.get("DOI"),
                    "pdf_url": None,
                    "published": str(((item.get("published-print") or item.get("published-online") or {}).get("date-parts") or [[None]])[0][0])
                    if ((item.get("published-print") or item.get("published-online") or {}).get("date-parts"))
                    else None,
                    "doi": item.get("DOI"),
                    "citation_count": int(item.get("is-referenced-by-count") or 0),
                    "url": item.get("URL"),
                    "score": 0.0,
                    "snippets": [first_sentence(abstract)] if abstract else [],
                    "raw": item,
                }
            )
        return papers

    def _offline_seed_papers(self, user_input: str, keywords: list[str]) -> list[PaperRecord]:
        lead = " ".join(keyword.title() for keyword in keywords[:3]) or "Research Direction"
        secondary = " ".join(keyword.title() for keyword in keywords[3:6]) or "Evaluation"
        return [
            {
                "title": f"{lead}: A Baseline Literature Anchor",
                "authors": ["Offline Seed"],
                "abstract": f"Synthetic placeholder paper derived from user intent: {user_input}",
                "source": "offline_seed",
                "external_id": None,
                "pdf_url": None,
                "published": None,
                "doi": None,
                "citation_count": 0,
                "url": None,
                "score": 0.9,
                "snippets": [f"Offline seed generated for {lead}."],
                "raw": {"synthetic": True},
            },
            {
                "title": f"{secondary}: Experimental Design Considerations",
                "authors": ["Offline Seed"],
                "abstract": "Synthetic placeholder paper focused on methods, baselines, and evaluation design.",
                "source": "offline_seed",
                "external_id": None,
                "pdf_url": None,
                "published": None,
                "doi": None,
                "citation_count": 0,
                "url": None,
                "score": 0.7,
                "snippets": [f"Offline design anchor for {secondary}."],
                "raw": {"synthetic": True},
            },
        ]

    def _rerank(self, papers: list[PaperRecord], user_input: str, keywords: list[str], max_results: int) -> list[PaperRecord]:
        for paper in papers:
            abstract = paper.get("abstract", "")
            overlap = keyword_overlap_score(f"{paper['title']} {abstract}", keywords)
            citation_bonus = min(paper.get("citation_count", 0) / 200, 1.0)
            abstract_bonus = min(len(abstract.split()) / 120, 1.0)
            paper["score"] = round(overlap * 0.6 + citation_bonus * 0.2 + abstract_bonus * 0.1 + 0.1, 4)
            if not paper.get("snippets"):
                paper["snippets"] = [first_sentence(abstract) or first_sentence(user_input)]
        papers.sort(key=lambda item: item["score"], reverse=True)
        return papers[:max_results]

    def search(self, state: ResearchState, max_results: int | None = None) -> tuple[list[PaperRecord], list[dict[str, Any]]]:
        query = " ".join(state["keywords"]) if state["keywords"] else state["user_input"]
        results: list[PaperRecord] = []
        metadata: list[dict[str, Any]] = []
        max_results = max_results or self.config.max_results
        per_source = max(1, max_results // max(len(self.config.retrieval_sources), 1))
        for source in self.config.retrieval_sources:
            cache_key = stable_hash(f"{source}:{query}:{per_source}")
            cached = self.repository.cache_get("retrieval", cache_key) if self.config.enable_cache else None
            started = time.perf_counter()
            used_cache = cached is not None
            if cached is not None:
                papers = cached
                state["run_metrics"]["retrieval_cache_hits"] += 1
            else:
                state["run_metrics"]["retrieval_cache_misses"] += 1
                if source == "arxiv":
                    papers = self._search_arxiv(query, per_source)
                elif source == "semantic_scholar":
                    papers = self._search_semantic_scholar(query, per_source)
                elif source == "crossref":
                    papers = self._search_crossref(query, per_source)
                else:
                    papers = []
                if self.config.enable_cache:
                    self.repository.cache_set("retrieval", cache_key, papers)
            latency_ms = round((time.perf_counter() - started) * 1000, 3)
            results.extend(papers)
            metadata.append(
                {
                    "source": source,
                    "query": query,
                    "returned_count": len(papers),
                    "used_cache": used_cache,
                    "latency_ms": latency_ms,
                }
            )
        results = dedupe_papers(results)
        results = self._rerank(results, state["user_input"], state["keywords"], max_results=max_results)
        if not results:
            results = self._offline_seed_papers(state["user_input"], state["keywords"])
            metadata.append(
                {
                    "source": "offline_seed",
                    "query": query,
                    "returned_count": len(results),
                    "used_cache": False,
                    "latency_ms": 0.0,
                }
            )
        return results, metadata


class ReviewService:
    """Structured summarization, critique, and topic selection."""

    def __init__(self, llm: LLMService):
        self.llm = llm

    def _fallback_summary(self, paper: PaperRecord, keywords: list[str]) -> dict[str, Any]:
        abstract = paper.get("abstract", "")
        summary_keywords = extract_keywords(f"{paper['title']} {abstract}", max_keywords=5)
        return {
            "problem": first_sentence(abstract) or f"{paper['title']} addresses a problem relevant to the user query.",
            "method": f"The paper appears to focus on {', '.join(summary_keywords[:2]) or 'a domain-specific method'}.",
            "datasets": ["Dataset not explicit in the metadata; inspect full text if needed."],
            "metrics": ["Task-specific benchmark metrics"],
            "strengths": [f"Strong topical match around {summary_keywords[0]}" if summary_keywords else "Relevant title signal"],
            "weaknesses": ["Abstract-only view may hide evaluation gaps."],
            "reproducibility_risks": ["Code and data availability are unknown from metadata alone."],
            "improvement_ideas": [
                "Add stronger baselines and ablation studies.",
                "Clarify dataset splits and evaluation protocol.",
                "Stress-test generalization and robustness.",
            ],
            "source_excerpt": first_sentence(abstract) or paper["title"],
            "summary_markdown": (
                f"### {paper['title']}\n"
                f"- Problem: {first_sentence(abstract) or 'Needs deeper reading.'}\n"
                f"- Signal: {', '.join(summary_keywords[:3]) or 'metadata-derived relevance'}"
            ),
        }

    def summarize_papers(self, state: ResearchState) -> None:
        summaries = []
        for paper in state["papers"]:
            fallback = lambda paper=paper: self._fallback_summary(paper, state["keywords"])
            payload = {
                "user_input": state["user_input"],
                "keywords": ", ".join(state["keywords"]),
                "paper_json": json.dumps(paper, ensure_ascii=False, indent=2),
            }
            result = self.llm.generate_json(
                state,
                stage="Summarizer",
                task="summary",
                prompt_name="summary",
                variables=payload,
                required_keys=[
                    "problem",
                    "method",
                    "datasets",
                    "metrics",
                    "strengths",
                    "weaknesses",
                    "reproducibility_risks",
                    "improvement_ideas",
                    "source_excerpt",
                    "summary_markdown",
                ],
                fallback_builder=fallback,
            )
            result["paper"] = paper
            summaries.append(result)
        state["summaries"] = summaries

    def _fallback_critique(self, summary: dict[str, Any], keywords: list[str]) -> dict[str, Any]:
        text = summary["summary_markdown"]
        overlap = keyword_overlap_score(text, keywords)
        novelty = min(10, 5 + int(overlap * 4))
        methodology = 6
        relevance = min(10, 6 + int(overlap * 4))
        reproducibility = 5
        overall = round((novelty + methodology + relevance + reproducibility) / 4, 2)
        return {
            "novelty": novelty,
            "methodology": methodology,
            "relevance": relevance,
            "reproducibility": reproducibility,
            "overall": overall,
            "critique": "Relevant starting point, but needs stronger evidence and clearer experimental grounding.",
            "reasons": [
                "The paper has a visible topical match to the user query.",
                "Metadata alone limits confidence in methodology quality.",
                "Full-text inspection is still needed for rigorous comparison.",
            ],
        }

    def critique_summaries(self, state: ResearchState) -> None:
        critiques = []
        for summary in state["summaries"]:
            fallback = lambda summary=summary: self._fallback_critique(summary, state["keywords"])
            payload = {
                "user_input": state["user_input"],
                "keywords": ", ".join(state["keywords"]),
                "summary_json": json.dumps(summary, ensure_ascii=False, indent=2),
            }
            result = self.llm.generate_json(
                state,
                stage="Critic",
                task="critique",
                prompt_name="critique",
                variables=payload,
                required_keys=[
                    "novelty",
                    "methodology",
                    "relevance",
                    "reproducibility",
                    "overall",
                    "critique",
                    "reasons",
                ],
                fallback_builder=fallback,
            )
            critique = {
                "paper": summary["paper"],
                "summary_markdown": summary["summary_markdown"],
                **result,
            }
            critiques.append(critique)
        state["critiques"] = critiques

    def select_candidate_topics(self, state: ResearchState, top_n: int = 3) -> None:
        critiques = state["critiques"]
        if not critiques:
            state["candidate_topics"] = [
                {
                    "title": " / ".join(keyword.title() for keyword in state["keywords"][:3]) or "Research Direction",
                    "rationale": "Fallback topic derived directly from the input idea.",
                    "differentiation": ["Use the user input as the first hypothesis anchor."],
                    "failure_modes": ["Insufficient literature evidence."],
                    "source_papers": [],
                    "score": 0.0,
                }
            ]
            state["selected_topic"] = state["candidate_topics"][0]
            return

        texts = [item["summary_markdown"] for item in critiques]
        labels = None
        if len(texts) >= 3:
            vectors = vectorize_texts(texts)
            labels = cluster_texts(vectors, n_clusters=min(3, len(texts)))

        if labels and len(set(labels)) > 1:
            grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
            for critique, label in zip(critiques, labels):
                critique["cluster"] = label
                grouped[label].append(critique)
            topics = []
            for label, items in grouped.items():
                combined_titles = " ".join(item["paper"]["title"] for item in items)
                title_keywords = extract_keywords(combined_titles, max_keywords=4)
                topics.append(
                    {
                        "title": " / ".join(keyword.title() for keyword in title_keywords[:3]) or items[0]["paper"]["title"],
                        "rationale": f"Cluster {label} groups {len(items)} related papers with strong internal topical overlap.",
                        "differentiation": [
                            "Use clustered papers to derive a cleaner problem framing.",
                            "Compare cluster-level consensus with the user’s original intuition.",
                        ],
                        "failure_modes": [
                            "Cluster may reflect keyword overlap rather than real methodological novelty.",
                        ],
                        "source_papers": [item["paper"]["title"] for item in items[:4]],
                        "score": round(sum(item["overall"] for item in items) / len(items), 2),
                        "cluster": label,
                    }
                )
            topics.sort(key=lambda item: item["score"], reverse=True)
        else:
            topics = []
            for critique in sorted(critiques, key=lambda item: item["overall"], reverse=True)[:top_n]:
                topics.append(
                    {
                        "title": critique["paper"]["title"],
                        "rationale": critique["critique"],
                        "differentiation": [
                            "Use this paper as the comparison anchor.",
                            "Extract the strongest claim and test where it breaks.",
                        ],
                        "failure_modes": critique["reasons"][:2],
                        "source_papers": [critique["paper"]["title"]],
                        "score": critique["overall"],
                    }
                )
        state["candidate_topics"] = topics[:top_n]
        state["selected_topic"] = state["candidate_topics"][0] if state["candidate_topics"] else None


class PlanningService:
    """Planning, code generation, and reporting service."""

    def __init__(self, llm: LLMService):
        self.llm = llm

    def _infer_datasets(self, keywords: list[str]) -> list[str]:
        joined = " ".join(keywords)
        if any(token in joined for token in {"energy", "load", "forecast", "temporal", "time"}):
            return [
                "ETTh1 / ETTm1",
                "ElectricityLoadDiagrams20112014",
                "ISO-NE or another grid operator dataset",
            ]
        return [
            "One public benchmark aligned with the topic",
            "One secondary validation dataset for generalization checks",
        ]

    def _infer_metrics(self, keywords: list[str]) -> list[str]:
        joined = " ".join(keywords)
        if any(token in joined for token in {"energy", "load", "forecast", "temporal", "time"}):
            return ["MAE", "RMSE", "MAPE", "latency"]
        return ["Primary task metric", "generalization gap", "runtime cost"]

    def _fallback_plan(self, state: ResearchState) -> dict[str, Any]:
        topic = state["selected_topic"] or {
            "title": "Research Direction",
            "rationale": "Fallback plan derived from the user idea.",
            "source_papers": [],
        }
        datasets = self._infer_datasets(state["keywords"])
        metrics = self._infer_metrics(state["keywords"])
        return {
            "research_question": f"How can {topic['title']} be turned into a robust, testable research project?",
            "hypotheses": [
                "A carefully selected baseline set will reveal whether the proposed direction adds real value.",
                "Most gains will come from better experiment design, not just a more complex model.",
            ],
            "datasets": datasets,
            "baselines": ["Strong classical baseline", "Existing neural baseline", "Ablated proposed method"],
            "methods": [
                "Define preprocessing and split strategy first.",
                "Run baseline experiments before implementing the full proposal.",
                "Add ablations and error analysis before final conclusions.",
            ],
            "experiment_matrix": [
                "Baseline vs proposed method",
                "Ablation on the core architectural choice",
                "Robustness under domain shift or noisy inputs",
            ],
            "metrics": metrics,
            "timeline": [
                "Week 1: literature consolidation and benchmark selection",
                "Week 2: baseline implementation and data pipeline",
                "Week 3: proposed method and ablations",
                "Week 4: error analysis and report writing",
            ],
            "risks": [
                "Retrieved literature may be too shallow for strong novelty claims.",
                "Evaluation may overfit one benchmark without a secondary validation setting.",
            ],
        }

    def build_plan(self, state: ResearchState) -> None:
        fallback = lambda: self._fallback_plan(state)
        payload = {
            "user_input": state["user_input"],
            "keywords": ", ".join(state["keywords"]),
            "selected_topic_json": json.dumps(state["selected_topic"], ensure_ascii=False, indent=2),
            "paper_titles": "\n".join(f"- {paper['title']}" for paper in state["papers"][:10]),
        }
        plan_data = self.llm.generate_json(
            state,
            stage="Planner",
            task="planning",
            prompt_name="plan",
            variables=payload,
            required_keys=[
                "research_question",
                "hypotheses",
                "datasets",
                "baselines",
                "methods",
                "experiment_matrix",
                "metrics",
                "timeline",
                "risks",
            ],
            fallback_builder=fallback,
        )
        sections = [
            "# Research Plan",
            "",
            "## Research Question",
            plan_data["research_question"],
            "",
            "## Hypotheses",
            *[f"- {item}" for item in plan_data["hypotheses"]],
            "",
            "## Datasets",
            *[f"- {item}" for item in plan_data["datasets"]],
            "",
            "## Baselines",
            *[f"- {item}" for item in plan_data["baselines"]],
            "",
            "## Methods",
            *[f"- {item}" for item in plan_data["methods"]],
            "",
            "## Experiment Matrix",
            *[f"- {item}" for item in plan_data["experiment_matrix"]],
            "",
            "## Metrics",
            *[f"- {item}" for item in plan_data["metrics"]],
            "",
            "## Timeline",
            *[f"- {item}" for item in plan_data["timeline"]],
            "",
            "## Risks",
            *[f"- {item}" for item in plan_data["risks"]],
        ]
        state["plan_markdown"] = "\n".join(sections).strip()

    def build_project_files(self, state: ResearchState) -> None:
        topic_title = state["selected_topic"]["title"] if state["selected_topic"] else "Research Experiment"
        metrics = self._infer_metrics(state["keywords"])
        config_content = json.dumps(
            {
                "topic": topic_title,
                "metrics": metrics,
                "keywords": state["keywords"],
                "papers": [paper["title"] for paper in state["papers"][:5]],
            },
            ensure_ascii=False,
            indent=2,
        )
        train_code = textwrap.dedent(
            f"""
            \"\"\"Training entrypoint for {topic_title}.\"\"\"

            from __future__ import annotations

            import json
            from pathlib import Path


            def load_config(path: Path) -> dict:
                return json.loads(path.read_text(encoding="utf-8"))


            def train() -> None:
                config = load_config(Path("generated_project/configs/default.json"))
                print("Training topic:", config["topic"])
                print("Tracked metrics:", ", ".join(config["metrics"]))
                print("TODO: replace stub training loop with the real experiment.")


            if __name__ == "__main__":
                train()
            """
        ).strip()
        evaluate_code = textwrap.dedent(
            """
            \"\"\"Evaluation entrypoint for the generated project.\"\"\"

            from __future__ import annotations


            def evaluate() -> None:
                print("TODO: implement quantitative evaluation and reporting.")


            if __name__ == "__main__":
                evaluate()
            """
        ).strip()
        experiment_readme = textwrap.dedent(
            f"""
            # Generated Experiment Scaffold

            Topic: {topic_title}

            This scaffold is generated from the current research plan. It is intentionally minimal:

            - `configs/default.json` stores the experiment snapshot
            - `src/train.py` is the training entrypoint
            - `src/evaluate.py` is the evaluation entrypoint

            The next step is to replace the stubs with a real data pipeline and model implementation.
            """
        ).strip()
        generated_files: list[GeneratedFileRecord] = [
            {
                "path": "generated_experiment.py",
                "content": train_code,
                "description": "Single-file quickstart training scaffold.",
            },
            {
                "path": "generated_project/README_experiment.md",
                "content": experiment_readme,
                "description": "Project-level generated experiment README.",
            },
            {
                "path": "generated_project/configs/default.json",
                "content": config_content,
                "description": "Experiment configuration snapshot.",
            },
            {
                "path": "generated_project/src/train.py",
                "content": train_code,
                "description": "Generated training entrypoint.",
            },
            {
                "path": "generated_project/src/evaluate.py",
                "content": evaluate_code,
                "description": "Generated evaluation entrypoint.",
            },
        ]
        state["generated_files"] = generated_files
        state["generated_code"] = train_code

    def build_report(self, state: ResearchState) -> None:
        topic_title = state["selected_topic"]["title"] if state["selected_topic"] else "Research Direction"
        report = [
            "# Run Report",
            "",
            f"- Run ID: {state['run_id']}",
            f"- Status: {state['status']}",
            f"- Selected Topic: {topic_title}",
            f"- Keywords: {', '.join(state['keywords'])}",
            "",
            "## Retrieved Papers",
            *[f"- {paper['title']} ({paper['source']}, score={paper['score']})" for paper in state["papers"][:10]],
            "",
            "## Candidate Topics",
            *[f"- {topic['title']}: {topic['rationale']}" for topic in state["candidate_topics"]],
            "",
            "## Plan Preview",
            state["plan_markdown"][:1500],
        ]
        state["report_markdown"] = "\n".join(report).strip()


@dataclass
class ServiceBundle:
    """Convenience container for runtime services."""

    retrieval: RetrievalService
    review: ReviewService
    planning: PlanningService
    llm: LLMService


def build_service_bundle(
    config: AppConfig,
    repository: ResearchRepository,
    prompts: PromptManager,
    run_logger: StructuredLogger,
) -> ServiceBundle:
    """Build the service graph for a run."""
    llm = LLMService(config=config, prompts=prompts, repository=repository, run_logger=run_logger)
    return ServiceBundle(
        retrieval=RetrievalService(config=config, repository=repository, run_logger=run_logger),
        review=ReviewService(llm=llm),
        planning=PlanningService(llm=llm),
        llm=llm,
    )
