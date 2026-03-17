"""CLI entrypoint for the research multi-agent MVP."""

from __future__ import annotations

import argparse
from typing import Sequence

from orchestrator import Orchestrator

DEFAULT_USER_INPUT = (
    "Exploring transformer architectures for energy consumption forecasting in smart grids, "
    "with emphasis on temporal attention mechanisms and comparison to traditional methods."
)


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(description="Run the research multi-agent MVP.")
    parser.add_argument("--input", dest="user_input", default=DEFAULT_USER_INPUT, help="Research idea or seed prompt.")
    parser.add_argument("--db-path", default="research_agent.db", help="SQLite database path.")
    parser.add_argument("--output-dir", default="runs", help="Directory for run artifacts.")
    parser.add_argument("--max-results", type=int, default=10, help="Maximum number of arXiv results to fetch.")
    parser.add_argument("--live-llm", action="store_true", help="Use an OpenAI-compatible live LLM backend.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI."""
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    orchestrator = Orchestrator()
    final_state = orchestrator.run(
        args.user_input,
        db_path=args.db_path,
        output_dir=args.output_dir,
        max_results=args.max_results,
        live_llm=args.live_llm,
    )

    selected_topic = final_state["candidate_topics"][0] if final_state["candidate_topics"] else None
    artifact_paths = {
        artifact["file_path"].rsplit("/", 1)[-1]: artifact["file_path"]
        for artifact in final_state["artifacts"]
        if artifact["file_path"]
    }

    print(f"Run ID: {final_state['run_id']}")
    print(f"Fetched Papers: {len(final_state['papers'])}")
    if selected_topic is None:
        print("Selected Topic: unavailable")
    else:
        print(f"Selected Topic: {selected_topic['title']} | {selected_topic['rationale']}")
    print(f"Plan Path: {artifact_paths.get('research_plan.md', 'not generated')}")
    print(f"Code Path: {artifact_paths.get('generated_experiment.py', 'not generated')}")
    print(f"Final Status: {final_state['status']}")

    if final_state["errors"]:
        print("Errors:")
        for error in final_state["errors"]:
            print(f"- {error['stage']}: {error['type']} - {error['message']}")

    return 0 if final_state["status"] != "failed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
