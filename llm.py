"""Thin OpenRouter LLM client using OpenAI SDK."""

import os

from openai import AsyncOpenAI

_client: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise SystemExit("Error: Set OPENROUTER_API_KEY environment variable")
        _client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
    return _client


async def complete(
    system: str, user: str, model: str = "openai/gpt-5.2"
) -> str:
    """Single-shot chat completion. Returns assistant message content."""
    resp = await _get_client().chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    if not resp.choices:
        raise RuntimeError(f"LLM returned no choices ({model})")
    choice = resp.choices[0]
    # OpenRouter buries errors in the choice object instead of raising
    err = getattr(choice, "error", None)
    if err or getattr(choice, "finish_reason", None) == "error":
        msg = err.get("message", err) if isinstance(err, dict) else err or "unknown error"
        raise RuntimeError(f"LLM error ({model}): {msg}")
    if not choice.message.content:
        raise RuntimeError(f"LLM returned empty content ({model})")
    return choice.message.content
