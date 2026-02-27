"""Orchestrator: chunk PDF -> create tree -> parallel triage -> linkage -> query loop."""

import asyncio
import argparse
import os
import re
import time

from chunk import chunk_pdf
from sifttext import SiftTextClient
from agents import triage_agent, linkage_agent, query_agent

_UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")


async def main():
    for var in ("SIFTTEXT_API_KEY", "OPENROUTER_API_KEY"):
        if var not in os.environ:
            raise SystemExit(f"Error: Set {var} environment variable")

    args = parse_args()
    sift = SiftTextClient()

    try:
        # Stage 0: Chunk PDF
        t0 = time.time()
        chunks = chunk_pdf(args.pdf)
        print(f"Chunked into {len(chunks)} sections ({time.time() - t0:.1f}s)")

        # Stage 1: Create tree
        print("Creating tree...")
        tree = await sift.create_tree("EU AI Act Analysis", "Parallel decomposition of EU AI Act")
        tree_id = tree["tree_id"]
        root_id = tree["root_id"]
        print(f"Tree created: {tree_id}")

        # Stage 2: Two-pass hierarchical creation
        level1_chunks = [c for c in chunks if c["level"] == 1]
        level2_chunks = [c for c in chunks if c["level"] == 2]

        # Pass 1 — structural parent nodes (no LLM, fast)
        print(f"Creating {len(level1_chunks)} parent nodes...")
        parent_node_ids: dict[str, str] = {}

        for chunk in level1_chunks:
            result_text = await sift.create_node(
                name=chunk["title"],
                scope=chunk["text"][:200] if chunk["text"] else chunk["title"],
                parent_id=root_id,
                tree_id=tree_id,
            )
            match = _UUID_RE.search(result_text)
            if match:
                parent_node_ids[chunk["title"]] = match.group(0)
            print(f"  {chunk['title']}")

        print(f"Created {len(parent_node_ids)} parent nodes")

        # Pass 2 — triage level-2 chunks under parents (parallel, with LLM)
        sem = asyncio.Semaphore(10)

        async def rate_limited_triage(chunk: dict) -> str:
            parent_id = parent_node_ids.get(chunk["parent_title"], root_id)
            async with sem:
                return await triage_agent(chunk, tree_id, parent_id, sift, args.model)

        print(f"Triaging {len(level2_chunks)} chunks in parallel...")
        t1 = time.time()
        tasks = [rate_limited_triage(c) for c in level2_chunks]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        successes = [r for r in results if not isinstance(r, Exception)]
        failures = [r for r in results if isinstance(r, Exception)]
        print(f"Created {len(successes)} nodes ({time.time() - t1:.1f}s)")
        if failures:
            print(f"Warning: {len(failures)} chunks failed")
            for f in failures[:3]:
                print(f"  {type(f).__name__}: {f}")

        # Stage 3: Linkage pass
        print("Running linkage pass...")
        t2 = time.time()
        try:
            link_count = await linkage_agent(tree_id, sift, args.smart_model)
            print(f"Added {link_count} cross-references ({time.time() - t2:.1f}s)")
        except Exception as e:
            print(f"Warning: linkage pass failed ({type(e).__name__}: {e}), continuing without cross-references")

        # Stage 4: Interactive query loop
        total = time.time() - t0
        print(f"\nTree ready. Total time: {total:.1f}s")
        print("Ask questions about the EU AI Act (ctrl+c to exit):\n")
        try:
            while True:
                q = await asyncio.to_thread(input, "> ")
                if not q.strip():
                    continue
                try:
                    answer = await query_agent(q, tree_id, sift, args.smart_model)
                    print(f"\n{answer}\n")
                except Exception as e:
                    print(f"\nError answering query ({type(e).__name__}: {e}). Try again.\n")
        except (KeyboardInterrupt, EOFError):
            print("\nDone.")

    finally:
        await sift.close()


def parse_args():
    p = argparse.ArgumentParser(description="Enterprise Doc Agent")
    p.add_argument("--pdf", default="eu_ai_act.pdf", help="Path to PDF file")
    p.add_argument("--model", default="openai/gpt-5.2-chat", help="Triage model")
    p.add_argument("--smart-model", default="openai/gpt-5.2", help="Linkage/query model")
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(main())
