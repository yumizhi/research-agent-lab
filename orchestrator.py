"""Resumable orchestrator for the research agent workflow."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

from agents import CodeGenAgent, CriticAgent, FetcherAgent, IdeaAnalyzerAgent, PlanAgent, SummarizerAgent, TrendAgent
from config import AppConfig, load_config
from logging_utils import StructuredLogger, configure_logging
from models import ArtifactRecord, ErrorRecord, ResearchState, StageMetric, create_research_state, utc_now_iso
from prompting import PromptManager
from repository import ResearchRepository
from services import ServiceBundle, build_service_bundle


class Orchestrator:
    """Coordinate agents, persistence, artifacts, and resumable execution."""

    def __init__(
        self,
        *,
        config: AppConfig | None = None,
        repository: ResearchRepository | None = None,
        prompts: PromptManager | None = None,
    ):
        self.base_config = config
        self.base_repository = repository
        self.base_prompts = prompts

    def _resolve_runtime(
        self,
        *,
        settings_path: str | None,
        db_path: str | None,
        output_dir: str | None,
        max_results: int | None,
        live_llm: bool | None,
    ) -> tuple[AppConfig, ResearchRepository, PromptManager, StructuredLogger, ServiceBundle]:
        config = self.base_config or load_config(
            settings_path=settings_path,
            overrides={
                key: value
                for key, value in {
                    "db_path": db_path,
                    "output_dir": output_dir,
                    "max_results": max_results,
                    "live_llm": live_llm,
                }.items()
                if value is not None
            },
        )
        repository = self.base_repository or ResearchRepository(config.db_path)
        prompts = self.base_prompts or PromptManager(config.prompt_dir)
        configure_logging(config.log_level, json_logs=config.json_logs)
        run_logger = StructuredLogger(repository=repository)
        services = build_service_bundle(config=config, repository=repository, prompts=prompts, run_logger=run_logger)
        return config, repository, prompts, run_logger, services

    def _build_agents(self, services: ServiceBundle, max_results: int) -> list[object]:
        return [
            IdeaAnalyzerAgent(services=services),
            FetcherAgent(services=services, max_results=max_results),
            SummarizerAgent(services=services),
            CriticAgent(services=services),
            TrendAgent(services=services, top_n=3),
            PlanAgent(services=services),
            CodeGenAgent(services=services),
        ]

    def _append_artifact(
        self,
        state: ResearchState,
        repository: ResearchRepository,
        *,
        stage: str,
        kind: str,
        payload: dict[str, object] | list[object] | str | None = None,
        file_path: str | None = None,
    ) -> None:
        created_at = utc_now_iso()
        artifact: ArtifactRecord = {
            "stage": stage,
            "kind": kind,
            "file_path": file_path,
            "created_at": created_at,
            "payload": payload,
        }
        state["artifacts"].append(artifact)
        repository.add_artifact(
            state["run_id"],
            stage=stage,
            kind=kind,
            payload=payload,
            file_path=file_path,
            created_at=created_at,
        )

    def _write_text_file(
        self,
        state: ResearchState,
        repository: ResearchRepository,
        run_dir: Path,
        stage: str,
        relative_path: str,
        content: str,
        description: str,
    ) -> None:
        path = (run_dir / relative_path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        self._append_artifact(
            state,
            repository,
            stage=stage,
            kind="file",
            payload={"description": description, "bytes": len(content)},
            file_path=str(path),
        )

    def _write_state_snapshot(self, state: ResearchState, repository: ResearchRepository, run_dir: Path, stage: str) -> None:
        snapshot = json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True)
        self._write_text_file(state, repository, run_dir, stage, "state.json", snapshot, "Latest run state snapshot.")

    def _write_outputs(self, state: ResearchState, repository: ResearchRepository, run_dir: Path, stage: str) -> None:
        self._write_state_snapshot(state, repository, run_dir, stage)
        if state["plan_markdown"]:
            self._write_text_file(
                state,
                repository,
                run_dir,
                stage,
                "research_plan.md",
                state["plan_markdown"],
                "Generated research plan.",
            )
        if state["generated_files"]:
            for generated in state["generated_files"]:
                self._write_text_file(
                    state,
                    repository,
                    run_dir,
                    stage,
                    generated["path"],
                    generated["content"],
                    generated["description"],
                )
        if state["report_markdown"]:
            self._write_text_file(
                state,
                repository,
                run_dir,
                stage,
                "report.md",
                state["report_markdown"],
                "Run summary report.",
            )

    def _record_stage_metric(self, state: ResearchState, metric: StageMetric) -> None:
        stage_metrics = [item for item in state["run_metrics"]["stage_metrics"] if item["stage"] != metric["stage"]]
        stage_metrics.append(metric)
        state["run_metrics"]["stage_metrics"] = stage_metrics

    def _load_or_create_state(
        self,
        repository: ResearchRepository,
        prompts: PromptManager,
        *,
        user_input: str,
        run_id: str | None,
        config: AppConfig,
        resume: bool,
        force: bool,
    ) -> ResearchState:
        if run_id:
            existing = repository.get_run_state(run_id)
            if existing is not None:
                if resume:
                    existing["status"] = "running"
                    existing["updated_at"] = utc_now_iso()
                    existing["config_snapshot"] = config.to_snapshot()
                    existing["prompt_versions"] = prompts.versions()
                    return existing
                if not force:
                    return existing
        state = create_research_state(run_id or uuid4().hex[:12], user_input, config_snapshot=config.to_snapshot())
        state["status"] = "running"
        state["prompt_versions"] = prompts.versions()
        return state

    def run(
        self,
        user_input: str,
        *,
        run_id: str | None = None,
        db_path: str | None = None,
        output_dir: str | None = None,
        max_results: int | None = None,
        live_llm: bool | None = None,
        settings_path: str | None = None,
        resume: bool = False,
        force: bool = False,
    ) -> ResearchState:
        """Run the full workflow, with optional resume support."""
        config, repository, prompts, run_logger, services = self._resolve_runtime(
            settings_path=settings_path,
            db_path=db_path,
            output_dir=output_dir,
            max_results=max_results,
            live_llm=live_llm,
        )
        max_results = max_results if max_results is not None else config.max_results
        state = self._load_or_create_state(
            repository,
            prompts,
            user_input=user_input,
            run_id=run_id,
            config=config,
            resume=resume,
            force=force,
        )
        run_logger.bind(state["run_id"])
        run_dir = Path(config.output_dir).resolve() / state["run_id"]
        run_dir.mkdir(parents=True, exist_ok=True)

        if state["status"] == "completed" and not resume and not force:
            return state

        agents = self._build_agents(services, max_results=max_results)
        total_started = time.perf_counter()
        repository.save_run(state)
        self._write_outputs(state, repository, run_dir, "bootstrap")

        for agent in agents:
            stage_name = agent.name
            if resume and stage_name in state["completed_stages"] and not force:
                metric: StageMetric = {
                    "stage": stage_name,
                    "status": "skipped",
                    "started_at": utc_now_iso(),
                    "ended_at": utc_now_iso(),
                    "duration_ms": 0.0,
                }
                self._record_stage_metric(state, metric)
                continue

            stage_started_at = utc_now_iso()
            started = time.perf_counter()
            state["current_stage"] = stage_name
            state["updated_at"] = stage_started_at
            run_logger.event("info", f"Running stage {stage_name}", stage=stage_name)
            repository.save_run(state)
            try:
                state = agent.run(state)
                duration_ms = round((time.perf_counter() - started) * 1000, 3)
                state["updated_at"] = utc_now_iso()
                if stage_name not in state["completed_stages"]:
                    state["completed_stages"].append(stage_name)
                metric = {
                    "stage": stage_name,
                    "status": "completed",
                    "started_at": stage_started_at,
                    "ended_at": state["updated_at"],
                    "duration_ms": duration_ms,
                }
                self._record_stage_metric(state, metric)
                self._write_outputs(state, repository, run_dir, stage_name)
                repository.save_run(state)
            except Exception as exc:
                state["status"] = "failed"
                state["updated_at"] = utc_now_iso()
                error: ErrorRecord = {
                    "stage": stage_name,
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "created_at": state["updated_at"],
                }
                state["errors"].append(error)
                metric = {
                    "stage": stage_name,
                    "status": "failed",
                    "started_at": stage_started_at,
                    "ended_at": state["updated_at"],
                    "duration_ms": round((time.perf_counter() - started) * 1000, 3),
                }
                self._record_stage_metric(state, metric)
                run_logger.event("error", f"Stage failed: {stage_name}", stage=stage_name, payload=error)
                self._append_artifact(state, repository, stage=stage_name, kind="error", payload=error)
                self._write_outputs(state, repository, run_dir, stage_name)
                repository.save_run(state)
                return state

        state["status"] = "completed"
        state["current_stage"] = "completed"
        state["updated_at"] = utc_now_iso()
        total_duration_ms = round((time.perf_counter() - total_started) * 1000, 3)
        run_logger.event("info", "Run completed", stage="orchestrator", payload={"duration_ms": total_duration_ms})
        self._append_artifact(
            state,
            repository,
            stage="orchestrator",
            kind="summary",
            payload={"duration_ms": total_duration_ms, "papers": len(state["papers"])},
        )
        self._write_outputs(state, repository, run_dir, "orchestrator")
        repository.save_run(state)
        return state

    def list_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        """Return a list of recent runs."""
        repository = self.base_repository or ResearchRepository((self.base_config or load_config()).db_path)
        return repository.list_runs(limit=limit)
