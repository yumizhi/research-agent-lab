"""Structured logging helpers."""

from __future__ import annotations

import json
import logging
from typing import Any

from repository import ResearchRepository


def configure_logging(level: str = "INFO", json_logs: bool = False) -> None:
    """Configure process-wide logging."""
    if logging.getLogger().handlers:
        return
    log_level = getattr(logging, level.upper(), logging.INFO)
    if json_logs:
        logging.basicConfig(level=log_level, format="%(message)s")
    else:
        logging.basicConfig(level=log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


class StructuredLogger:
    """Logger that mirrors events to the repository."""

    def __init__(self, repository: ResearchRepository | None = None, run_id: str | None = None):
        self.repository = repository
        self.run_id = run_id
        self.logger = logging.getLogger("research_agent")

    def bind(self, run_id: str) -> None:
        """Attach a run id to the logger."""
        self.run_id = run_id

    def event(self, level: str, message: str, *, stage: str | None = None, payload: Any = None) -> None:
        """Log an event and optionally persist it."""
        level_name = level.upper()
        if isinstance(payload, (dict, list)):
            payload_preview = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        elif payload is None:
            payload_preview = ""
        else:
            payload_preview = str(payload)
        self.logger.log(getattr(logging, level_name, logging.INFO), "%s %s", message, payload_preview)
        if self.repository is not None and self.run_id is not None:
            self.repository.add_event(self.run_id, level=level_name, message=message, stage=stage, payload=payload)
