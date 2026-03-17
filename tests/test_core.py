"""Core unit tests for prompts, parsing, repository, and retrieval."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from config import AppConfig
from logging_utils import StructuredLogger
from models import create_research_state
from prompting import PromptManager
from repository import ResearchRepository
from services import RetrievalService
from utils import extract_keywords, parse_arxiv_response


class CoreBehaviorTests(unittest.TestCase):
    def test_keyword_extraction_filters_and_falls_back(self) -> None:
        self.assertEqual(extract_keywords("Transformer transformer for smart grids and energy forecasting.")[:4], ["transformer", "smart", "grids", "energy"])
        self.assertEqual(extract_keywords("AI", max_keywords=4), ["ai"])
        chinese_keywords = extract_keywords("多智能体系统如何用于科研选题与实验规划", max_keywords=6)
        self.assertIn("多智能体系统", chinese_keywords)

    def test_parse_arxiv_response_dedupes_results(self) -> None:
        xml_text = """
        <feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
          <entry>
            <title>Example Paper</title>
            <summary>First summary.</summary>
            <author><name>Alice</name></author>
            <link title="pdf" href="http://arxiv.org/pdf/1234.5678v1" />
            <published>2024-01-01T00:00:00Z</published>
          </entry>
          <entry>
            <title> Example   Paper </title>
            <summary>Duplicate summary.</summary>
            <author><name>Bob</name></author>
            <link title="pdf" href="http://arxiv.org/pdf/1234.5678v1" />
            <published>2024-01-02T00:00:00Z</published>
          </entry>
        </feed>
        """
        papers = parse_arxiv_response(xml_text)
        self.assertEqual(len(papers), 1)
        self.assertEqual(papers[0]["title"], "Example Paper")
        self.assertEqual(papers[0]["authors"], ["Alice"])

    def test_prompt_manager_loads_versions(self) -> None:
        prompts = PromptManager("prompt_templates")
        versions = prompts.versions()
        self.assertIn("summary", versions)
        rendered, version = prompts.render("summary", user_input="x", keywords="y", paper_json="{}")
        self.assertIn("Return strict JSON", rendered)
        self.assertEqual(version, versions["summary"])

    def test_repository_can_save_runs_artifacts_and_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repository = ResearchRepository(str(Path(temp_dir) / "repo.db"))
            state = create_research_state("repo-run", "test input", config_snapshot={"db_path": "repo.db"})
            state["prompt_versions"] = {"summary": "v1"}
            repository.save_run(state)
            repository.add_artifact(state["run_id"], "test", "file", payload={"ok": True}, file_path="/tmp/example.txt")
            repository.add_event(state["run_id"], level="INFO", message="hello", stage="test")
            repository.cache_set("llm", "k1", {"value": 1})

            loaded = repository.get_run_state(state["run_id"])
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded["run_id"], "repo-run")
            self.assertEqual(len(repository.list_artifacts(state["run_id"])), 1)
            self.assertEqual(len(repository.list_events(state["run_id"])), 1)
            self.assertEqual(repository.cache_get("llm", "k1"), {"value": 1})

    def test_retrieval_service_falls_back_to_offline_seed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = AppConfig(db_path=str(Path(temp_dir) / "repo.db"), output_dir=str(Path(temp_dir) / "runs"))
            repository = ResearchRepository(config.db_path)
            logger = StructuredLogger(repository=repository)
            service = RetrievalService(config=config, repository=repository, run_logger=logger)
            state = create_research_state("retrieval-run", "多智能体系统用于科研选题", config_snapshot=config.to_snapshot())
            state["keywords"] = ["多智能体", "科研", "选题"]
            papers, metadata = service.search(state, max_results=4)
            self.assertGreaterEqual(len(papers), 1)
            self.assertTrue(any(item["source"] == "offline_seed" for item in papers))
            self.assertGreaterEqual(len(metadata), 1)


if __name__ == "__main__":
    unittest.main()
