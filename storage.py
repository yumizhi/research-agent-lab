"""Backward-compatible storage wrapper."""

from __future__ import annotations

from repository import ResearchRepository


class SQLitePersistence(ResearchRepository):
    """Compatibility alias for earlier versions of the project."""

