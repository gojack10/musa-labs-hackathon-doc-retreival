"""Shared constants and utilities."""

import os
import re

from rich.console import Console

UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")

console = Console()


def load_prompt(name: str) -> str:
    """Load a prompt file from the prompts/ directory.

    Args:
        name: Prompt filename without extension (e.g., "markdown_triage")

    Returns the prompt text, or raises FileNotFoundError.
    """
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "prompts", f"{name}.md")
    with open(path) as f:
        return f.read()
