"""Prompt template loading and rendering."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from utils import stable_hash


@dataclass
class PromptTemplate:
    """A loaded prompt template."""

    name: str
    body: str
    version: str


class PromptManager:
    """Load prompt templates from disk and render them with variables."""

    def __init__(self, prompt_dir: str = "prompt_templates"):
        self.prompt_dir = Path(prompt_dir)
        self._templates: dict[str, PromptTemplate] = {}
        self.reload()

    def reload(self) -> None:
        """Reload all prompt templates from disk."""
        self._templates.clear()
        if not self.prompt_dir.exists():
            return
        for path in self.prompt_dir.glob("*.txt"):
            body = path.read_text(encoding="utf-8")
            version = stable_hash(body)[:12]
            self._templates[path.stem] = PromptTemplate(name=path.stem, body=body, version=version)

    def render(self, name: str, **variables: object) -> tuple[str, str]:
        """Render a prompt and return the text plus version."""
        template = self._templates[name]
        formatted_variables = {key: str(value) for key, value in variables.items()}
        return template.body.format_map(formatted_variables), template.version

    def versions(self) -> dict[str, str]:
        """Return prompt versions keyed by prompt name."""
        return {name: template.version for name, template in self._templates.items()}
