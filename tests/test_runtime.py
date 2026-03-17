"""Runtime tests for jobs, API, and benchmark evaluation."""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path

from api import ResearchAgentAPI
from config import AppConfig
from evaluation import BenchmarkEvaluator
from jobs import JobManager
from orchestrator import Orchestrator
from repository import ResearchRepository


class RuntimeFeatureTests(unittest.TestCase):
    def test_job_manager_completes_background_run(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = AppConfig(db_path=str(Path(temp_dir) / "jobs.db"), output_dir=str(Path(temp_dir) / "runs"), worker_threads=1)
            orchestrator = Orchestrator(config=config, repository=ResearchRepository(config.db_path))
            jobs = JobManager(orchestrator=orchestrator, config=config)
            job = jobs.submit("多智能体系统用于科研选题")
            snapshot = jobs.wait_for(job.job_id, timeout_seconds=5.0)
            self.assertIsNotNone(snapshot)
            self.assertIn(snapshot["status"], {"completed", "failed"})
            jobs.shutdown()

    def test_api_health_and_runs_endpoints(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = AppConfig(db_path=str(Path(temp_dir) / "api.db"), output_dir=str(Path(temp_dir) / "runs"))
            repository = ResearchRepository(config.db_path)
            orchestrator = Orchestrator(config=config, repository=repository)
            jobs = JobManager(orchestrator=orchestrator, config=config)
            api = ResearchAgentAPI(orchestrator=orchestrator, repository=repository, jobs=jobs)

            captured = {}

            def start_response(status, headers):
                captured["status"] = status
                captured["headers"] = headers

            health_body = b"".join(
                api(
                    {
                        "REQUEST_METHOD": "GET",
                        "PATH_INFO": "/health",
                        "CONTENT_LENGTH": "0",
                        "wsgi.input": io.BytesIO(b""),
                    },
                    start_response,
                )
            )
            self.assertEqual(captured["status"], "200 OK")
            self.assertEqual(json.loads(health_body.decode("utf-8"))["status"], "ok")

            root_body = b"".join(
                api(
                    {
                        "REQUEST_METHOD": "GET",
                        "PATH_INFO": "/",
                        "CONTENT_LENGTH": "0",
                        "wsgi.input": io.BytesIO(b""),
                    },
                    start_response,
                )
            )
            self.assertEqual(captured["status"], "200 OK")
            self.assertIn("Research Agent Lab", root_body.decode("utf-8"))

            css_body = b"".join(
                api(
                    {
                        "REQUEST_METHOD": "GET",
                        "PATH_INFO": "/static/styles.css",
                        "CONTENT_LENGTH": "0",
                        "wsgi.input": io.BytesIO(b""),
                    },
                    start_response,
                )
            )
            self.assertEqual(captured["status"], "200 OK")
            self.assertIn("--bg", css_body.decode("utf-8"))

            runs_body = b"".join(
                api(
                    {
                        "REQUEST_METHOD": "GET",
                        "PATH_INFO": "/runs",
                        "CONTENT_LENGTH": "0",
                        "wsgi.input": io.BytesIO(b""),
                    },
                    start_response,
                )
            )
            self.assertEqual(captured["status"], "200 OK")
            self.assertIn("runs", json.loads(runs_body.decode("utf-8")))

            run_state = orchestrator.run("多智能体系统用于科研选题")
            events_body = b"".join(
                api(
                    {
                        "REQUEST_METHOD": "GET",
                        "PATH_INFO": f"/runs/{run_state['run_id']}/events",
                        "CONTENT_LENGTH": "0",
                        "wsgi.input": io.BytesIO(b""),
                    },
                    start_response,
                )
            )
            self.assertEqual(captured["status"], "200 OK")
            self.assertIn("events", json.loads(events_body.decode("utf-8")))
            jobs.shutdown()

    def test_benchmark_evaluator_returns_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            benchmark_path = Path(temp_dir) / "bench.json"
            benchmark_path.write_text(
                json.dumps(
                    [
                        {
                            "case_id": "simple",
                            "user_input": "多智能体系统用于科研选题",
                            "expected_keywords": ["多智能体", "科研"],
                            "max_results": 4,
                        }
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            config = AppConfig(
                db_path=str(Path(temp_dir) / "bench.db"),
                output_dir=str(Path(temp_dir) / "runs"),
                benchmark_path=str(benchmark_path),
            )
            orchestrator = Orchestrator(config=config, repository=ResearchRepository(config.db_path))
            summary = BenchmarkEvaluator(orchestrator=orchestrator, config=config).run()
            self.assertEqual(summary["cases"], 1)
            self.assertIn("pass_rate", summary)


if __name__ == "__main__":
    unittest.main()
