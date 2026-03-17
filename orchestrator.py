"""Sequential orchestrator with persistence and artifact generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional
from uuid import uuid4

from agents import (
    Agent,
    CodeGenAgent,
    CriticAgent,
    FetcherAgent,
    IdeaAnalyzerAgent,
    PlanAgent,
    SummarizerAgent,
    TrendAgent,
)
from models import ArtifactRecord, ResearchState, create_research_state, utc_now_iso
from storage import SQLitePersistence


class Orchestrator:
    """Coordinates the research multi-agent workflow."""

    def __init__(self, agents: Optional[list[Agent]] = None):
        self._agents = agents

    def _build_agents(self, *, max_results: int, live_llm: bool) -> list[Agent]:
        return self._agents or [
            IdeaAnalyzerAgent(),
            FetcherAgent(max_results=max_results),
            SummarizerAgent(live_llm=live_llm),
            CriticAgent(live_llm=live_llm),
            TrendAgent(n_clusters=5, top_n=3),
            PlanAgent(live_llm=live_llm),
            CodeGenAgent(live_llm=live_llm),
        ]

    def _append_artifact(
        self,
        state: ResearchState,
        persistence: SQLitePersistence,
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
        persistence.add_artifact(
            state["run_id"],
            stage=stage,
            kind=kind,
            payload=payload,
            file_path=file_path,
            created_at=created_at,
        )

    def _write_state_snapshot(self, state: ResearchState, run_dir: Path) -> str:
        snapshot_path = (run_dir / "state.json").resolve()
        snapshot_path.write_text(json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        return str(snapshot_path)

    def _write_runtime_outputs(
        self,
        state: ResearchState,
        persistence: SQLitePersistence,
        run_dir: Path,
        stage: str,
    ) -> None:
        if state["plan_markdown"]:
            plan_path = (run_dir / "research_plan.md").resolve()
            if not plan_path.exists():
                plan_path.write_text(state["plan_markdown"], encoding="utf-8")
                self._append_artifact(
                    state,
                    persistence,
                    stage=stage,
                    kind="file",
                    payload={"bytes": len(state["plan_markdown"])},
                    file_path=str(plan_path),
                )
        if state["generated_code"]:
            code_path = (run_dir / "generated_experiment.py").resolve()
            if not code_path.exists():
                code_path.write_text(state["generated_code"], encoding="utf-8")
                self._append_artifact(
                    state,
                    persistence,
                    stage=stage,
                    kind="file",
                    payload={"bytes": len(state["generated_code"])},
                    file_path=str(code_path),
                )

    def _mark_updated(self, state: ResearchState, status: str | None = None) -> None:
        if status is not None:
            state["status"] = status
        state["updated_at"] = utc_now_iso()

    def run(
        self,
        user_input: str,
        *,
        run_id: str | None = None,
        db_path: str = "research_agent.db",
        output_dir: str = "runs",
        max_results: int = 10,
        live_llm: bool = False,
    ) -> ResearchState:
        """Run the full pipeline on a user input."""
        state = create_research_state(run_id or uuid4().hex[:12], user_input)
        persistence = SQLitePersistence(db_path=db_path)
        run_dir = Path(output_dir).resolve() / state["run_id"]
        run_dir.mkdir(parents=True, exist_ok=True)
        agents = self._build_agents(max_results=max_results, live_llm=live_llm)

        snapshot_path = self._write_state_snapshot(state, run_dir)
        self._append_artifact(
            state,
            persistence,
            stage="orchestrator",
            kind="state_snapshot",
            payload={"status": state["status"], "keywords": 0, "papers": 0},
            file_path=snapshot_path,
        )
        persistence.save_run(state)

        for agent in agents:
            stage_name = agent.name
            try:
                state = agent.run(state)
                self._mark_updated(state, status="running")
                self._write_runtime_outputs(state, persistence, run_dir, stage_name)
                snapshot_path = self._write_state_snapshot(state, run_dir)
                self._append_artifact(
                    state,
                    persistence,
                    stage=stage_name,
                    kind="state_snapshot",
                    payload={
                        "status": state["status"],
                        "keywords": len(state["keywords"]),
                        "papers": len(state["papers"]),
                        "summaries": len(state["summaries"]),
                        "critiques": len(state["critiques"]),
                        "candidate_topics": len(state["candidate_topics"]),
                    },
                    file_path=snapshot_path,
                )
                persistence.save_run(state)
            except Exception as exc:
                error = {
                    "stage": stage_name,
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "created_at": utc_now_iso(),
                }
                state["errors"].append(error)
                self._mark_updated(state, status="failed")
                snapshot_path = self._write_state_snapshot(state, run_dir)
                self._append_artifact(
                    state,
                    persistence,
                    stage=stage_name,
                    kind="error",
                    payload=error,
                    file_path=snapshot_path,
                )
                persistence.save_run(state)
                return state

        self._mark_updated(state, status="completed")
        snapshot_path = self._write_state_snapshot(state, run_dir)
        self._append_artifact(
            state,
            persistence,
            stage="orchestrator",
            kind="state_snapshot",
            payload={"status": state["status"], "finalized": True},
            file_path=snapshot_path,
        )
        persistence.save_run(state)
        return state
