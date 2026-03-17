"""SQLite-backed repository for runs, artifacts, events, and cache."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

from models import ResearchState, utc_now_iso


class ResearchRepository:
    """Persistence and query layer for research runs."""

    def __init__(self, db_path: str = "research_agent.db"):
        self.db_path = Path(db_path)
        if self.db_path.parent != Path("."):
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with closing(self._connect()) as connection:
            with connection:
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS runs (
                        run_id TEXT PRIMARY KEY,
                        user_input TEXT NOT NULL,
                        status TEXT NOT NULL,
                        current_stage TEXT NOT NULL,
                        state_json TEXT NOT NULL,
                        config_json TEXT NOT NULL,
                        prompt_versions_json TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    )
                    """
                )
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS artifacts (
                        id INTEGER PRIMARY KEY,
                        run_id TEXT NOT NULL,
                        stage TEXT NOT NULL,
                        kind TEXT NOT NULL,
                        payload_json TEXT,
                        file_path TEXT,
                        created_at TEXT NOT NULL
                    )
                    """
                )
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS events (
                        id INTEGER PRIMARY KEY,
                        run_id TEXT NOT NULL,
                        stage TEXT,
                        level TEXT NOT NULL,
                        message TEXT NOT NULL,
                        payload_json TEXT,
                        created_at TEXT NOT NULL
                    )
                    """
                )
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS prompt_calls (
                        id INTEGER PRIMARY KEY,
                        run_id TEXT NOT NULL,
                        stage TEXT NOT NULL,
                        task TEXT NOT NULL,
                        prompt_name TEXT NOT NULL,
                        prompt_version TEXT NOT NULL,
                        model TEXT NOT NULL,
                        latency_ms REAL NOT NULL,
                        input_tokens INTEGER NOT NULL,
                        output_tokens INTEGER NOT NULL,
                        success INTEGER NOT NULL,
                        response_preview TEXT,
                        created_at TEXT NOT NULL
                    )
                    """
                )
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS cache_entries (
                        namespace TEXT NOT NULL,
                        cache_key TEXT NOT NULL,
                        value_json TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        PRIMARY KEY(namespace, cache_key)
                    )
                    """
                )

    def save_run(self, state: ResearchState) -> None:
        payload = json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True)
        config_json = json.dumps(state["config_snapshot"], ensure_ascii=False, sort_keys=True)
        prompt_json = json.dumps(state["prompt_versions"], ensure_ascii=False, sort_keys=True)
        with closing(self._connect()) as connection:
            with connection:
                connection.execute(
                    """
                    INSERT INTO runs (
                        run_id, user_input, status, current_stage, state_json, config_json,
                        prompt_versions_json, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(run_id) DO UPDATE SET
                        user_input = excluded.user_input,
                        status = excluded.status,
                        current_stage = excluded.current_stage,
                        state_json = excluded.state_json,
                        config_json = excluded.config_json,
                        prompt_versions_json = excluded.prompt_versions_json,
                        created_at = excluded.created_at,
                        updated_at = excluded.updated_at
                    """,
                    (
                        state["run_id"],
                        state["user_input"],
                        state["status"],
                        state["current_stage"],
                        payload,
                        config_json,
                        prompt_json,
                        state["started_at"],
                        state["updated_at"],
                    ),
                )

    def get_run_state(self, run_id: str) -> ResearchState | None:
        with closing(self._connect()) as connection:
            row = connection.execute("SELECT state_json FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if row is None:
            return None
        return json.loads(row["state_json"])

    def list_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT run_id, user_input, status, current_stage, created_at, updated_at
                FROM runs
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def add_artifact(
        self,
        run_id: str,
        stage: str,
        kind: str,
        payload: Any = None,
        file_path: str | None = None,
        created_at: str | None = None,
    ) -> None:
        payload_json = None if payload is None else json.dumps(payload, ensure_ascii=False, sort_keys=True)
        with closing(self._connect()) as connection:
            with connection:
                connection.execute(
                    """
                    INSERT INTO artifacts (run_id, stage, kind, payload_json, file_path, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (run_id, stage, kind, payload_json, file_path, created_at or utc_now_iso()),
                )

    def list_artifacts(self, run_id: str) -> list[dict[str, Any]]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT stage, kind, payload_json, file_path, created_at
                FROM artifacts
                WHERE run_id = ?
                ORDER BY id ASC
                """,
                (run_id,),
            ).fetchall()
        artifacts: list[dict[str, Any]] = []
        for row in rows:
            artifact = dict(row)
            if artifact["payload_json"] is not None:
                artifact["payload"] = json.loads(artifact.pop("payload_json"))
            else:
                artifact["payload"] = None
                artifact.pop("payload_json")
            artifacts.append(artifact)
        return artifacts

    def add_event(
        self,
        run_id: str,
        *,
        level: str,
        message: str,
        stage: str | None = None,
        payload: Any = None,
        created_at: str | None = None,
    ) -> None:
        payload_json = None if payload is None else json.dumps(payload, ensure_ascii=False, sort_keys=True)
        with closing(self._connect()) as connection:
            with connection:
                connection.execute(
                    """
                    INSERT INTO events (run_id, stage, level, message, payload_json, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (run_id, stage, level.upper(), message, payload_json, created_at or utc_now_iso()),
                )

    def list_events(self, run_id: str) -> list[dict[str, Any]]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT stage, level, message, payload_json, created_at
                FROM events
                WHERE run_id = ?
                ORDER BY id ASC
                """,
                (run_id,),
            ).fetchall()
        events = []
        for row in rows:
            event = dict(row)
            event["payload"] = json.loads(event.pop("payload_json")) if event["payload_json"] else None
            events.append(event)
        return events

    def record_prompt_call(self, run_id: str, record: dict[str, Any]) -> None:
        with closing(self._connect()) as connection:
            with connection:
                connection.execute(
                    """
                    INSERT INTO prompt_calls (
                        run_id, stage, task, prompt_name, prompt_version, model,
                        latency_ms, input_tokens, output_tokens, success, response_preview, created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        record["stage"],
                        record["task"],
                        record["prompt_name"],
                        record["prompt_version"],
                        record["model"],
                        record["latency_ms"],
                        record["input_tokens"],
                        record["output_tokens"],
                        1 if record["success"] else 0,
                        record["response_preview"],
                        record["created_at"],
                    ),
                )

    def list_prompt_calls(self, run_id: str) -> list[dict[str, Any]]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT stage, task, prompt_name, prompt_version, model, latency_ms,
                       input_tokens, output_tokens, success, response_preview, created_at
                FROM prompt_calls
                WHERE run_id = ?
                ORDER BY id ASC
                """,
                (run_id,),
            ).fetchall()
        records = []
        for row in rows:
            data = dict(row)
            data["success"] = bool(data["success"])
            records.append(data)
        return records

    def cache_get(self, namespace: str, cache_key: str) -> Any | None:
        with closing(self._connect()) as connection:
            row = connection.execute(
                """
                SELECT value_json FROM cache_entries
                WHERE namespace = ? AND cache_key = ?
                """,
                (namespace, cache_key),
            ).fetchone()
        if row is None:
            return None
        return json.loads(row["value_json"])

    def cache_set(self, namespace: str, cache_key: str, value: Any) -> None:
        with closing(self._connect()) as connection:
            with connection:
                connection.execute(
                    """
                    INSERT INTO cache_entries (namespace, cache_key, value_json, updated_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(namespace, cache_key) DO UPDATE SET
                        value_json = excluded.value_json,
                        updated_at = excluded.updated_at
                    """,
                    (
                        namespace,
                        cache_key,
                        json.dumps(value, ensure_ascii=False, sort_keys=True),
                        utc_now_iso(),
                    ),
                )

    def latest_run_for_input(self, user_input: str) -> dict[str, Any] | None:
        with closing(self._connect()) as connection:
            row = connection.execute(
                """
                SELECT run_id, status, current_stage, updated_at
                FROM runs
                WHERE user_input = ?
                ORDER BY updated_at DESC
                LIMIT 1
                """,
                (user_input,),
            ).fetchone()
        return dict(row) if row is not None else None
