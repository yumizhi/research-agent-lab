"""Minimal WSGI API for runs and background jobs."""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from typing import Any, Callable
from wsgiref.simple_server import make_server

from config import AppConfig
from jobs import JobManager
from orchestrator import Orchestrator
from repository import ResearchRepository


class ResearchAgentAPI:
    """WSGI-compatible JSON API."""

    def __init__(self, orchestrator: Orchestrator, repository: ResearchRepository, jobs: JobManager):
        self.orchestrator = orchestrator
        self.repository = repository
        self.jobs = jobs
        self.web_dir = Path(__file__).resolve().parent / "web"

    def _json_response(self, start_response: Callable, status: str, payload: Any):
        body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        start_response(status, [("Content-Type", "application/json; charset=utf-8"), ("Content-Length", str(len(body)))])
        return [body]

    def _text_response(self, start_response: Callable, status: str, body: str, content_type: str = "text/plain; charset=utf-8"):
        payload = body.encode("utf-8")
        start_response(status, [("Content-Type", content_type), ("Content-Length", str(len(payload)))])
        return [payload]

    def _static_response(self, start_response: Callable, relative_path: str, content_type: str):
        target = (self.web_dir / relative_path).resolve()
        try:
            target.relative_to(self.web_dir.resolve())
        except ValueError:
            return self._json_response(start_response, "403 Forbidden", {"error": "forbidden"})
        if not target.exists() or not target.is_file():
            return self._json_response(start_response, "404 Not Found", {"error": "asset not found"})
        payload = target.read_bytes()
        start_response(status="200 OK", headers=[("Content-Type", content_type), ("Content-Length", str(len(payload)))])
        return [payload]

    def __call__(self, environ: dict[str, Any], start_response: Callable):
        method = environ.get("REQUEST_METHOD", "GET").upper()
        path = environ.get("PATH_INFO", "/")
        body_length = int(environ.get("CONTENT_LENGTH") or 0)
        body_bytes = environ.get("wsgi.input", BytesIO()).read(body_length) if body_length else b""
        try:
            payload = json.loads(body_bytes.decode("utf-8")) if body_bytes else {}
        except json.JSONDecodeError:
            return self._json_response(start_response, "400 Bad Request", {"error": "invalid json body"})

        if method == "GET" and path == "/":
            return self._static_response(start_response, "index.html", "text/html; charset=utf-8")

        if method == "GET" and path == "/static/app.js":
            return self._static_response(start_response, "app.js", "application/javascript; charset=utf-8")

        if method == "GET" and path == "/static/styles.css":
            return self._static_response(start_response, "styles.css", "text/css; charset=utf-8")

        if method == "GET" and path == "/health":
            return self._json_response(start_response, "200 OK", {"status": "ok"})

        if method == "GET" and path == "/runs":
            return self._json_response(start_response, "200 OK", {"runs": self.repository.list_runs()})

        if method == "POST" and path == "/runs":
            state = self.orchestrator.run(
                payload.get("user_input", ""),
                run_id=payload.get("run_id"),
                max_results=payload.get("max_results"),
                live_llm=payload.get("live_llm"),
            )
            return self._json_response(start_response, "200 OK", state)

        if method == "POST" and path == "/jobs":
            job = self.jobs.submit(
                payload.get("user_input", ""),
                run_id=payload.get("run_id"),
                max_results=payload.get("max_results"),
                live_llm=payload.get("live_llm"),
            )
            return self._json_response(start_response, "202 Accepted", job.__dict__)

        if method == "GET" and path.startswith("/jobs/"):
            job_id = path.split("/")[-1]
            job = self.jobs.get(job_id)
            if job is None:
                return self._json_response(start_response, "404 Not Found", {"error": "job not found"})
            return self._json_response(start_response, "200 OK", job)

        if method == "GET" and path.startswith("/runs/"):
            parts = [part for part in path.split("/") if part]
            if len(parts) == 2:
                state = self.repository.get_run_state(parts[1])
                if state is None:
                    return self._json_response(start_response, "404 Not Found", {"error": "run not found"})
                return self._json_response(start_response, "200 OK", state)
            if len(parts) == 3 and parts[2] == "artifacts":
                return self._json_response(
                    start_response,
                    "200 OK",
                    {"artifacts": self.repository.list_artifacts(parts[1])},
                )
            if len(parts) == 3 and parts[2] == "events":
                return self._json_response(
                    start_response,
                    "200 OK",
                    {"events": self.repository.list_events(parts[1])},
                )
            if len(parts) == 3 and parts[2] == "prompt-calls":
                return self._json_response(
                    start_response,
                    "200 OK",
                    {"prompt_calls": self.repository.list_prompt_calls(parts[1])},
                )

        return self._json_response(start_response, "404 Not Found", {"error": "endpoint not found"})


def serve_api(
    config: AppConfig,
    orchestrator: Orchestrator,
    repository: ResearchRepository,
    jobs: JobManager,
) -> None:
    """Start the WSGI API server."""
    app = ResearchAgentAPI(orchestrator=orchestrator, repository=repository, jobs=jobs)
    with make_server(config.server_host, config.server_port, app) as server:
        print(f"Serving API on http://{config.server_host}:{config.server_port}")
        server.serve_forever()
