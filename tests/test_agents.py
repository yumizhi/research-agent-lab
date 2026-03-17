"""Unit tests for agent behaviors and utility fallbacks."""

from __future__ import annotations

import unittest

from agents import CodeGenAgent, CriticAgent, IdeaAnalyzerAgent, PlanAgent, TrendAgent
from models import create_research_state
from utils import parse_arxiv_response


class AgentBehaviorTests(unittest.TestCase):
    def test_keyword_extraction_filters_stopwords_and_dedupes(self) -> None:
        state = create_research_state("run-keywords", "Transformer transformer for smart grids and energy forecasting.")
        updated = IdeaAnalyzerAgent().run(state)
        self.assertEqual(updated["keywords"][:4], ["transformer", "smart", "grids", "energy"])

    def test_keyword_extraction_has_short_input_fallback(self) -> None:
        state = create_research_state("run-short", "AI")
        updated = IdeaAnalyzerAgent().run(state)
        self.assertEqual(updated["keywords"], ["ai"])

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

    def test_critic_agent_shapes_scores_and_overall(self) -> None:
        state = create_research_state("run-critic", "Energy forecasting with transformers")
        state["keywords"] = ["energy", "forecasting", "transformers"]
        state["summaries"] = [
            {
                "paper": {
                    "title": "Transformer Forecasting for Smart Grids",
                    "authors": ["Alice", "Bob"],
                    "summary": "This paper studies transformer models for smart-grid demand forecasting.",
                    "pdf_url": None,
                    "published": "2024-01-01T00:00:00Z",
                    "doi": None,
                },
                "summary": "The paper benchmarks transformer variants for smart-grid energy forecasting.",
                "improvement_ideas": [],
            }
        ]
        updated = CriticAgent().run(state)
        critique = updated["critiques"][0]
        self.assertIn("overall", critique)
        self.assertGreaterEqual(critique["novelty"], 0)
        self.assertLessEqual(critique["novelty"], 10)
        self.assertAlmostEqual(
            critique["overall"],
            round((critique["novelty"] + critique["methodology"] + critique["relevance"]) / 3, 1),
        )

    def test_trend_agent_falls_back_when_too_few_summaries(self) -> None:
        state = create_research_state("run-trend", "Short pipeline")
        state["keywords"] = ["energy", "forecasting"]
        state["critiques"] = [
            {
                "paper": {
                    "title": "Single Paper",
                    "authors": ["Alice"],
                    "summary": "One paper summary.",
                    "pdf_url": None,
                    "published": None,
                    "doi": None,
                },
                "summary": "One paper summary.",
                "novelty": 7,
                "methodology": 6,
                "relevance": 8,
                "overall": 7.0,
                "critique": "Promising but under-evaluated.",
            }
        ]
        updated = TrendAgent().run(state)
        self.assertEqual(len(updated["candidate_topics"]), 1)
        self.assertEqual(updated["candidate_topics"][0]["title"], "Single Paper")

    def test_plan_and_codegen_work_without_fetched_papers(self) -> None:
        state = create_research_state("run-plan", "Energy forecasting with transformers")
        state["keywords"] = ["energy", "forecasting", "transformers"]
        state["candidate_topics"] = [
            {
                "title": "Energy Forecasting Transformers",
                "rationale": "Fallback topic.",
                "source_papers": [],
                "score": 0.0,
            }
        ]
        planned = PlanAgent().run(state)
        self.assertIn("# Research Plan", planned["plan_markdown"])
        generated = CodeGenAgent().run(planned)
        self.assertIn("ExperimentConfig", generated["generated_code"])


if __name__ == "__main__":
    unittest.main()
