"""Background job manager for asynchronous runs."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from threading import Lock
from typing import Any
from uuid import uuid4
import time

from config import AppConfig
from models import utc_now_iso
from orchestrator import Orchestrator


@dataclass
class JobRecord:
    """In-memory representation of an async job."""

    job_id: str
    user_input: str
    status: str
    run_id: str | None
    error: str | None
    created_at: str
    updated_at: str
    options: dict[str, Any]


class JobManager:
    """Simple thread-backed async job runner."""

    def __init__(self, orchestrator: Orchestrator, config: AppConfig):
        self.orchestrator = orchestrator
        self.config = config
        self.executor = ThreadPoolExecutor(max_workers=config.worker_threads)
        self.lock = Lock()
        self.jobs: dict[str, JobRecord] = {}

    def submit(self, user_input: str, **options: Any) -> JobRecord:
        """Submit a background run and return the job record."""
        job_id = uuid4().hex[:12]
        now = utc_now_iso()
        record = JobRecord(
            job_id=job_id,
            user_input=user_input,
            status="queued",
            run_id=None,
            error=None,
            created_at=now,
            updated_at=now,
            options=options,
        )
        with self.lock:
            self.jobs[job_id] = record
        self.executor.submit(self._run_job, job_id, user_input, options)
        return record

    def _run_job(self, job_id: str, user_input: str, options: dict[str, Any]) -> None:
        with self.lock:
            record = self.jobs[job_id]
            record.status = "running"
            record.updated_at = utc_now_iso()
        try:
            state = self.orchestrator.run(user_input, **options)
            with self.lock:
                record = self.jobs[job_id]
                record.status = state["status"]
                record.run_id = state["run_id"]
                record.updated_at = utc_now_iso()
                if state["errors"]:
                    record.error = state["errors"][-1]["message"]
        except Exception as exc:
            with self.lock:
                record = self.jobs[job_id]
                record.status = "failed"
                record.error = str(exc)
                record.updated_at = utc_now_iso()

    def get(self, job_id: str) -> dict[str, Any] | None:
        """Return a JSON-safe job payload."""
        with self.lock:
            record = self.jobs.get(job_id)
            if record is None:
                return None
            return {
                "job_id": record.job_id,
                "user_input": record.user_input,
                "status": record.status,
                "run_id": record.run_id,
                "error": record.error,
                "created_at": record.created_at,
                "updated_at": record.updated_at,
                "options": dict(record.options),
            }

    def wait_for(self, job_id: str, timeout_seconds: float = 5.0) -> dict[str, Any] | None:
        """Poll until a job reaches a terminal state or timeout expires."""
        deadline = time.time() + timeout_seconds
        while time.time() < deadline:
            snapshot = self.get(job_id)
            if snapshot is None:
                return None
            if snapshot["status"] in {"completed", "failed"}:
                return snapshot
            time.sleep(0.05)
        return self.get(job_id)

    def shutdown(self) -> None:
        """Stop the worker pool."""
        self.executor.shutdown(wait=False)
