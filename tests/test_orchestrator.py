"""Integration and persistence tests for the orchestrator."""

from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from agents import Agent
from main import main
from orchestrator import Orchestrator


class FailingAgent:
    name = "FailingAgent"

    def run(self, state):
        raise RuntimeError("boom")


class OrchestratorTests(unittest.TestCase):
    def test_successful_run_persists_outputs_and_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "research_agent.db"
            output_dir = Path(temp_dir) / "runs"

            with mock.patch("agents.search_arxiv", return_value=[]):
                final_state = Orchestrator().run(
                    "Energy forecasting with transformers",
                    db_path=str(db_path),
                    output_dir=str(output_dir),
                    max_results=3,
                    live_llm=False,
                )

            self.assertEqual(final_state["status"], "completed")
            run_dir = output_dir / final_state["run_id"]
            self.assertTrue((run_dir / "research_plan.md").exists())
            self.assertTrue((run_dir / "generated_experiment.py").exists())
            self.assertTrue((run_dir / "state.json").exists())

            connection = sqlite3.connect(db_path)
            try:
                run_row = connection.execute("SELECT status, state_json FROM runs WHERE run_id = ?", (final_state["run_id"],)).fetchone()
                self.assertIsNotNone(run_row)
                self.assertEqual(run_row[0], "completed")
                stored_state = json.loads(run_row[1])
                self.assertEqual(stored_state["run_id"], final_state["run_id"])

                artifact_count = connection.execute(
                    "SELECT COUNT(*) FROM artifacts WHERE run_id = ?",
                    (final_state["run_id"],),
                ).fetchone()[0]
                self.assertGreaterEqual(artifact_count, 3)
            finally:
                connection.close()

    def test_failed_run_is_persisted(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "research_agent.db"
            output_dir = Path(temp_dir) / "runs"
            orchestrator = Orchestrator(agents=[FailingAgent()])

            final_state = orchestrator.run(
                "Any input",
                db_path=str(db_path),
                output_dir=str(output_dir),
                live_llm=False,
            )

            self.assertEqual(final_state["status"], "failed")
            self.assertEqual(final_state["errors"][0]["message"], "boom")

            connection = sqlite3.connect(db_path)
            try:
                run_row = connection.execute("SELECT status FROM runs WHERE run_id = ?", (final_state["run_id"],)).fetchone()
                self.assertEqual(run_row[0], "failed")
                error_artifacts = connection.execute(
                    "SELECT COUNT(*) FROM artifacts WHERE run_id = ? AND kind = 'error'",
                    (final_state["run_id"],),
                ).fetchone()[0]
                self.assertEqual(error_artifacts, 1)
            finally:
                connection.close()

    def test_main_cli_runs_from_repo_root(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "cli.db"
            output_dir = Path(temp_dir) / "cli-runs"
            with mock.patch("agents.search_arxiv", return_value=[]):
                exit_code = main(
                    [
                        "--input",
                        "Energy forecasting with transformers",
                        "--db-path",
                        str(db_path),
                        "--output-dir",
                        str(output_dir),
                    ]
                )
            self.assertEqual(exit_code, 0)


if __name__ == "__main__":
    unittest.main()
