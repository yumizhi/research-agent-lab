"""SQLite persistence for research assistant runs and artifacts."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

from models import ResearchState, utc_now_iso


class SQLitePersistence:
    """Simple SQLite-backed persistence layer for run state."""

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
                        state_json TEXT NOT NULL,
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

    def save_run(self, state: ResearchState) -> None:
        """Persist a full state snapshot."""
        payload = json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True)
        with closing(self._connect()) as connection:
            with connection:
                connection.execute(
                    """
                    INSERT INTO runs (run_id, user_input, status, state_json, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(run_id) DO UPDATE SET
                        user_input = excluded.user_input,
                        status = excluded.status,
                        state_json = excluded.state_json,
                        created_at = excluded.created_at,
                        updated_at = excluded.updated_at
                    """,
                    (
                        state["run_id"],
                        state["user_input"],
                        state["status"],
                        payload,
                        state["started_at"],
                        state["updated_at"],
                    ),
                )

    def add_artifact(
        self,
        run_id: str,
        stage: str,
        kind: str,
        payload: Any = None,
        file_path: str | None = None,
        created_at: str | None = None,
    ) -> None:
        """Persist a single artifact row."""
        payload_json = None if payload is None else json.dumps(payload, ensure_ascii=False, sort_keys=True)
        with closing(self._connect()) as connection:
            with connection:
                connection.execute(
                    """
                    INSERT INTO artifacts (run_id, stage, kind, payload_json, file_path, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        stage,
                        kind,
                        payload_json,
                        file_path,
                        created_at or utc_now_iso(),
                    ),
                )
