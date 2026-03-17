"""Benchmark and heuristic evaluation helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from config import AppConfig
from orchestrator import Orchestrator


class BenchmarkEvaluator:
    """Run a small benchmark suite against the orchestrator."""

    def __init__(self, orchestrator: Orchestrator, config: AppConfig):
        self.orchestrator = orchestrator
        self.config = config

    def _load_cases(self) -> list[dict[str, Any]]:
        path = Path(self.config.benchmark_path)
        if not path.exists():
            return []
        return json.loads(path.read_text(encoding="utf-8"))

    def run(self) -> dict[str, Any]:
        cases = self._load_cases()
        results = []
        passed = 0
        for case in cases:
            state = self.orchestrator.run(
                case["user_input"],
                max_results=case.get("max_results", 6),
                live_llm=False,
            )
            expected_keywords = set(case.get("expected_keywords", []))
            keyword_hits = 0
            for expected in expected_keywords:
                if any(expected in keyword or keyword in expected for keyword in state["keywords"]):
                    keyword_hits += 1
            checks = {
                "completed": state["status"] == "completed",
                "has_keywords": bool(state["keywords"]),
                "has_plan": bool(state["plan_markdown"]),
                "has_code": bool(state["generated_files"]),
                "keyword_hits": keyword_hits,
            }
            if checks["completed"] and checks["has_plan"] and checks["has_code"]:
                passed += 1
            results.append({"case_id": case["case_id"], "checks": checks, "run_id": state["run_id"]})

        summary = {
            "cases": len(cases),
            "passed": passed,
            "pass_rate": round((passed / len(cases)) * 100, 2) if cases else 0.0,
            "results": results,
        }
        return summary
