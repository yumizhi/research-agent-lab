"""CLI entrypoint for the research agent system."""

from __future__ import annotations

import argparse
import json
from typing import Sequence

from api import serve_api
from config import load_config
from evaluation import BenchmarkEvaluator
from jobs import JobManager
from orchestrator import Orchestrator
from repository import ResearchRepository

DEFAULT_USER_INPUT = "用时序 Transformer 做电力负荷预测，并比较传统统计方法和深度学习方法。"


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(description="Research Agent Lab")
    parser.add_argument("--input", dest="user_input", default=DEFAULT_USER_INPUT, help="Research idea or seed prompt.")
    parser.add_argument("--run-id", default=None, help="Optional explicit run id.")
    parser.add_argument("--resume-run-id", default=None, help="Resume an existing run by id.")
    parser.add_argument("--db-path", default=None, help="SQLite database path.")
    parser.add_argument("--output-dir", default=None, help="Directory for run artifacts.")
    parser.add_argument("--max-results", type=int, default=None, help="Maximum number of retrieved papers.")
    parser.add_argument("--live-llm", action="store_true", help="Use an OpenAI-compatible live LLM backend.")
    parser.add_argument("--settings", default=None, help="Path to settings TOML file.")
    parser.add_argument("--list-runs", action="store_true", help="List recent runs and exit.")
    parser.add_argument("--show-run", default=None, help="Print the saved state for a run id and exit.")
    parser.add_argument("--benchmark", action="store_true", help="Run the benchmark suite and exit.")
    parser.add_argument("--serve-api", action="store_true", help="Start the local WSGI API server and built-in web UI.")
    return parser


def _build_runtime(args: argparse.Namespace) -> tuple:
    config = load_config(
        settings_path=args.settings,
        overrides={
            key: value
            for key, value in {
                "db_path": args.db_path,
                "output_dir": args.output_dir,
                "max_results": args.max_results,
                "live_llm": args.live_llm or None,
            }.items()
            if value is not None
        },
    )
    repository = ResearchRepository(config.db_path)
    orchestrator = Orchestrator(config=config, repository=repository)
    return config, repository, orchestrator


def _print_run_summary(state: dict) -> None:
    selected_topic = state["selected_topic"] or (state["candidate_topics"][0] if state["candidate_topics"] else None)
    print(f"Run ID: {state['run_id']}")
    print(f"Status: {state['status']}")
    print(f"Current Stage: {state['current_stage']}")
    print(f"Papers Retrieved: {len(state['papers'])}")
    print(f"Completed Stages: {', '.join(state['completed_stages']) or 'none'}")
    if selected_topic is not None:
        print(f"Selected Topic: {selected_topic['title']}")
        print(f"Topic Rationale: {selected_topic['rationale']}")
    print(f"Plan Generated: {'yes' if state['plan_markdown'] else 'no'}")
    print(f"Generated Files: {len(state['generated_files'])}")
    if state["errors"]:
        print("Errors:")
        for error in state["errors"]:
            print(f"- {error['stage']}: {error['type']} - {error['message']}")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI."""
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    config, repository, orchestrator = _build_runtime(args)

    if args.list_runs:
        print(json.dumps(repository.list_runs(), ensure_ascii=False, indent=2))
        return 0

    if args.show_run:
        state = repository.get_run_state(args.show_run)
        if state is None:
            print("Run not found.")
            return 1
        print(json.dumps(state, ensure_ascii=False, indent=2))
        return 0

    if args.benchmark:
        evaluator = BenchmarkEvaluator(orchestrator=orchestrator, config=config)
        print(json.dumps(evaluator.run(), ensure_ascii=False, indent=2))
        return 0

    if args.serve_api:
        jobs = JobManager(orchestrator=orchestrator, config=config)
        serve_api(config=config, orchestrator=orchestrator, repository=repository, jobs=jobs)
        return 0

    if args.resume_run_id:
        state = orchestrator.run(
            args.user_input,
            run_id=args.resume_run_id,
            db_path=config.db_path,
            output_dir=config.output_dir,
            max_results=config.max_results,
            live_llm=config.live_llm,
            settings_path=args.settings,
            resume=True,
        )
    else:
        state = orchestrator.run(
            args.user_input,
            run_id=args.run_id,
            db_path=config.db_path,
            output_dir=config.output_dir,
            max_results=config.max_results,
            live_llm=config.live_llm,
            settings_path=args.settings,
        )

    _print_run_summary(state)
    return 0 if state["status"] != "failed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
