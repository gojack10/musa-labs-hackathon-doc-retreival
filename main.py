"""CLI dispatcher: routes to markdown or code document pipeline."""

import asyncio
import argparse
import os
import time

from dotenv import load_dotenv

from lib.config import console
from lib.perf import perf
from lib.sifttext import SiftTextClient
from agents.tree_query_agent.agent import query_agent


def _ts() -> str:
    elapsed = time.time() - perf._t0
    m, s = divmod(int(elapsed), 60)
    return f"[{m:02d}:{s:02d}]"


async def main():
    perf.start()
    load_dotenv()

    for var in ("SIFTTEXT_API_KEY", "AZURE_OPENAI_KEY"):
        if not os.environ.get(var):
            raise SystemExit(f"Error: Set {var} in .env or environment")

    args = parse_args()
    sift = SiftTextClient()

    try:
        if args.query_only:
            tree_id = args.query_only
        else:
            if args.mode == "markdown":
                from agents.markdown_agent.pipeline import run
            elif args.mode == "code":
                from agents.code_document_agent.pipeline import run
            else:
                raise SystemExit(f"Unknown mode: {args.mode}")

            tree_id = await run(args.input, sift, args.model, args.smart_model)

        # Interactive query loop
        pad = max(1, 52 - len("Interactive Query"))
        print(f"\n{_ts()} {'═' * 3} Stage 4: Interactive Query {'═' * pad}")
        print(f"{_ts()}   Tree: {tree_id}")
        print(f"{_ts()}   Ask questions (ctrl+c to exit)\n")
        history: list[dict] | None = None
        prompt_name = "code_query" if args.mode == "code" else "tree_query"
        try:
            while True:
                console.print("> ", style="green", end="")
                q = await asyncio.to_thread(input)
                if not q.strip():
                    continue
                print()
                try:
                    _, history = await query_agent(
                        q, tree_id, sift, args.smart_model,
                        history=history, prompt_name=prompt_name,
                    )
                    print()
                except Exception as e:
                    print(f"\n{_ts()}   ! Error: {type(e).__name__}: {e}. Try again.\n")
        except (KeyboardInterrupt, EOFError):
            print("\n\nDone.")

    finally:
        await sift.close()


def parse_args():
    p = argparse.ArgumentParser(description="Document Analysis Pipeline")
    p.add_argument("--mode", choices=["markdown", "code"], default="markdown",
                   help="Pipeline mode (default: markdown)")
    p.add_argument("--input", default="eu_ai_act.md", help="Path to input file or directory")
    p.add_argument("--model", default="gpt-5.2-chat-main", help="Triage model (Azure deployment name)")
    p.add_argument("--smart-model", default="gpt-5.2-chat-main", help="Linkage/query model (Azure deployment name)")
    p.add_argument("--query-only", metavar="TREE_ID",
                   help="Skip pipeline, jump straight to query loop on an existing tree")
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(main())
