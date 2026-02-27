"""Thin OpenRouter LLM client using OpenAI SDK — with perf instrumentation."""

import os
import time

from openai import AsyncOpenAI

from perf import perf

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
    t0 = time.time()
    err_str = None
    ok = True
    tokens_in = 0
    tokens_out = 0
    try:
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

        # Extract token usage if available
        if resp.usage:
            tokens_in = resp.usage.prompt_tokens or 0
            tokens_out = resp.usage.completion_tokens or 0

        return choice.message.content
    except Exception as e:
        err_str = f"{type(e).__name__}: {e}"
        ok = False
        raise
    finally:
        dur = (time.time() - t0) * 1000
        perf.event(
            "llm_call",
            dur,
            success=ok,
            error=err_str,
            model=model,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
        )
