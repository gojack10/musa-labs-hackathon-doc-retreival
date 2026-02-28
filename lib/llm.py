"""Azure OpenAI LLM client — with perf instrumentation."""

import os
import time

from openai import AsyncAzureOpenAI

from lib.perf import perf

AZURE_ENDPOINT = "https://odlu-mm5f7ry7-eastus2.cognitiveservices.azure.com/"
AZURE_API_VERSION = "2024-12-01-preview"
DEFAULT_DEPLOYMENT = "gpt-5.2-chat-main"

_client: AsyncAzureOpenAI | None = None


def _get_client() -> AsyncAzureOpenAI:
    global _client
    if _client is None:
        api_key = os.environ.get("AZURE_OPENAI_KEY")
        if not api_key:
            raise SystemExit("Error: Set AZURE_OPENAI_KEY environment variable")
        _client = AsyncAzureOpenAI(
            azure_endpoint=AZURE_ENDPOINT,
            api_key=api_key,
            api_version=AZURE_API_VERSION,
        )
    return _client


def get_client() -> AsyncAzureOpenAI:
    """Get or create the shared AsyncOpenAI client."""
    return _get_client()


async def complete(
    system: str, user: str, model: str = DEFAULT_DEPLOYMENT
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
