"""Thin OpenRouter LLM client using OpenAI SDK."""

import os

from openai import AsyncOpenAI

api_key = os.environ.get("OPENROUTER_API_KEY")
if not api_key:
    raise SystemExit("Error: Set OPENROUTER_API_KEY environment variable")

client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)


async def complete(
    system: str, user: str, model: str = "openai/gpt-5.2-chat"
) -> str:
    """Single-shot chat completion. Returns assistant message content."""
    resp = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return resp.choices[0].message.content
