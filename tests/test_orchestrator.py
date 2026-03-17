"""Integration tests for orchestration, resume, CLI, and reporting."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from main import main
from models import create_research_state
from orchestrator import Orchestrator
from repository import ResearchRepository


class OrchestratorIntegrationTests(unittest.TestCase):
    def test_successful_run_persists_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "research_agent.db"
            output_dir = Path(temp_dir) / "runs"
            state = Orchestrator().run(
                "Energy forecasting with transformers",
                db_path=str(db_path),
                output_dir=str(output_dir),
                max_results=4,
                live_llm=False,
            )
            self.assertEqual(state["status"], "completed")
            run_dir = output_dir / state["run_id"]
            self.assertTrue((run_dir / "research_plan.md").exists())
            self.assertTrue((run_dir / "generated_experiment.py").exists())
            self.assertTrue((run_dir / "generated_project" / "src" / "train.py").exists())
            self.assertTrue((run_dir / "report.md").exists())

            repository = ResearchRepository(str(db_path))
            self.assertGreaterEqual(len(repository.list_artifacts(state["run_id"])), 3)
            self.assertGreaterEqual(len(repository.list_prompt_calls(state["run_id"])), 2)

    def test_resume_continues_from_partial_state(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "research_agent.db"
            output_dir = Path(temp_dir) / "runs"
            repository = ResearchRepository(str(db_path))
            partial = create_research_state("resume-run", "多智能体系统用于科研选题", config_snapshot={"db_path": str(db_path)})
            partial["status"] = "failed"
            partial["prompt_versions"] = {"summary": "v1"}
            partial["completed_stages"] = ["IdeaAnalyzer"]
            partial["keywords"] = ["多智能体", "科研", "选题"]
            repository.save_run(partial)

            state = Orchestrator(repository=repository).run(
                "多智能体系统用于科研选题",
                run_id="resume-run",
                db_path=str(db_path),
                output_dir=str(output_dir),
                resume=True,
            )
            self.assertEqual(state["status"], "completed")
            self.assertIn("CodeGenerator", state["completed_stages"])

    def test_main_cli_runs_from_repo_root(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "cli.db"
            output_dir = Path(temp_dir) / "cli-runs"
            exit_code = main(
                [
                    "--input",
                    "多智能体系统用于科研选题",
                    "--db-path",
                    str(db_path),
                    "--output-dir",
                    str(output_dir),
                ]
            )
            self.assertEqual(exit_code, 0)


if __name__ == "__main__":
    unittest.main()
