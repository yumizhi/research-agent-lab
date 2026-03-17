"""Application configuration loading and normalization."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class AppConfig:
    """Runtime configuration for the research agent system."""

    db_path: str = "research_agent.db"
    output_dir: str = "runs"
    prompt_dir: str = "prompt_templates"
    benchmark_path: str = "benchmarks/sample_tasks.json"
    max_results: int = 12
    live_llm: bool = False
    retrieval_sources: tuple[str, ...] = ("arxiv", "semantic_scholar", "crossref")
    default_model: str = "gpt-4o-mini"
    task_models: dict[str, str] = field(
        default_factory=lambda: {
            "summary": "gpt-4o-mini",
            "critique": "gpt-4o-mini",
            "topic_selection": "gpt-4o-mini",
            "planning": "gpt-4o-mini",
            "codegen": "gpt-4o-mini",
            "report": "gpt-4o-mini",
        }
    )
    llm_timeout_seconds: int = 60
    llm_retries: int = 2
    enable_cache: bool = True
    worker_threads: int = 4
    server_host: str = "127.0.0.1"
    server_port: int = 8000
    log_level: str = "INFO"
    json_logs: bool = False

    def model_for(self, task: str) -> str:
        """Resolve model routing for a task."""
        return self.task_models.get(task, self.default_model)

    def to_snapshot(self) -> dict[str, Any]:
        """Return a JSON-serializable config snapshot."""
        return {
            "db_path": self.db_path,
            "output_dir": self.output_dir,
            "prompt_dir": self.prompt_dir,
            "benchmark_path": self.benchmark_path,
            "max_results": self.max_results,
            "live_llm": self.live_llm,
            "retrieval_sources": list(self.retrieval_sources),
            "default_model": self.default_model,
            "task_models": dict(self.task_models),
            "llm_timeout_seconds": self.llm_timeout_seconds,
            "llm_retries": self.llm_retries,
            "enable_cache": self.enable_cache,
            "worker_threads": self.worker_threads,
            "server_host": self.server_host,
            "server_port": self.server_port,
            "log_level": self.log_level,
            "json_logs": self.json_logs,
        }


def _deep_merge(base: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in incoming.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(settings_path: str | None = None, overrides: dict[str, Any] | None = None) -> AppConfig:
    """Load configuration from defaults, optional TOML, env, and overrides."""
    config_dict: dict[str, Any] = {
        "db_path": "research_agent.db",
        "output_dir": "runs",
        "prompt_dir": "prompt_templates",
        "benchmark_path": "benchmarks/sample_tasks.json",
        "max_results": 12,
        "live_llm": False,
        "retrieval_sources": ["arxiv", "semantic_scholar", "crossref"],
        "default_model": "gpt-4o-mini",
        "task_models": {
            "summary": "gpt-4o-mini",
            "critique": "gpt-4o-mini",
            "topic_selection": "gpt-4o-mini",
            "planning": "gpt-4o-mini",
            "codegen": "gpt-4o-mini",
            "report": "gpt-4o-mini",
        },
        "llm_timeout_seconds": 60,
        "llm_retries": 2,
        "enable_cache": True,
        "worker_threads": 4,
        "server_host": "127.0.0.1",
        "server_port": 8000,
        "log_level": "INFO",
        "json_logs": False,
    }

    candidate_paths = [settings_path] if settings_path else ["settings.toml", "settings.local.toml"]
    for candidate in candidate_paths:
        if not candidate:
            continue
        path = Path(candidate)
        if path.exists():
            with path.open("rb") as handle:
                parsed = tomllib.load(handle)
            config_dict = _deep_merge(config_dict, parsed.get("app", parsed))
            break

    env_map = {
        "RESEARCH_AGENT_DB_PATH": ("db_path", str),
        "RESEARCH_AGENT_OUTPUT_DIR": ("output_dir", str),
        "RESEARCH_AGENT_PROMPT_DIR": ("prompt_dir", str),
        "RESEARCH_AGENT_BENCHMARK_PATH": ("benchmark_path", str),
        "RESEARCH_AGENT_MAX_RESULTS": ("max_results", int),
        "RESEARCH_AGENT_LIVE_LLM": ("live_llm", lambda value: value.lower() in {"1", "true", "yes"}),
        "RESEARCH_AGENT_DEFAULT_MODEL": ("default_model", str),
        "RESEARCH_AGENT_LLM_TIMEOUT": ("llm_timeout_seconds", int),
        "RESEARCH_AGENT_LLM_RETRIES": ("llm_retries", int),
        "RESEARCH_AGENT_ENABLE_CACHE": ("enable_cache", lambda value: value.lower() in {"1", "true", "yes"}),
        "RESEARCH_AGENT_WORKER_THREADS": ("worker_threads", int),
        "RESEARCH_AGENT_SERVER_HOST": ("server_host", str),
        "RESEARCH_AGENT_SERVER_PORT": ("server_port", int),
        "RESEARCH_AGENT_LOG_LEVEL": ("log_level", str),
        "RESEARCH_AGENT_JSON_LOGS": ("json_logs", lambda value: value.lower() in {"1", "true", "yes"}),
    }
    for env_name, (field_name, parser) in env_map.items():
        if env_name in os.environ:
            config_dict[field_name] = parser(os.environ[env_name])

    sources_env = os.getenv("RESEARCH_AGENT_RETRIEVAL_SOURCES")
    if sources_env:
        config_dict["retrieval_sources"] = [item.strip() for item in sources_env.split(",") if item.strip()]

    if overrides:
        config_dict = _deep_merge(config_dict, overrides)

    return AppConfig(
        db_path=str(config_dict["db_path"]),
        output_dir=str(config_dict["output_dir"]),
        prompt_dir=str(config_dict["prompt_dir"]),
        benchmark_path=str(config_dict["benchmark_path"]),
        max_results=int(config_dict["max_results"]),
        live_llm=bool(config_dict["live_llm"]),
        retrieval_sources=tuple(config_dict["retrieval_sources"]),
        default_model=str(config_dict["default_model"]),
        task_models=dict(config_dict["task_models"]),
        llm_timeout_seconds=int(config_dict["llm_timeout_seconds"]),
        llm_retries=int(config_dict["llm_retries"]),
        enable_cache=bool(config_dict["enable_cache"]),
        worker_threads=int(config_dict["worker_threads"]),
        server_host=str(config_dict["server_host"]),
        server_port=int(config_dict["server_port"]),
        log_level=str(config_dict["log_level"]),
        json_logs=bool(config_dict["json_logs"]),
    )
