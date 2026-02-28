"""Markdown document analysis pipeline: chunk → triage → linkage."""

import asyncio
import time

from lib.config import UUID_RE
from lib.perf import perf
from lib.sifttext import SiftTextClient
from agents.markdown_agent.parser import chunk_markdown
from agents.markdown_agent.ingest import triage_agent, linkage_agent


def _ts() -> str:
    elapsed = time.time() - perf._t0
    m, s = divmod(int(elapsed), 60)
    return f"[{m:02d}:{s:02d}]"


def _banner(stage: int, title: str):
    pad = max(1, 52 - len(title))
    print(f"\n{_ts()} {'═' * 3} Stage {stage}: {title} {'═' * pad}")


async def run(
    input_path: str,
    sift: SiftTextClient,
    model: str,
    smart_model: str,
    tree_name: str | None = None,
    tree_scope: str | None = None,
) -> str:
    """Run the full markdown pipeline. Returns tree_id."""
    import os
    basename = os.path.splitext(os.path.basename(input_path))[0].replace("_", " ").title()
    tree_name = tree_name or f"{basename} Analysis"
    tree_scope = tree_scope or f"Parallel decomposition of {basename}"

    # Stage 0: Chunk markdown
    _banner(0, "Chunking Markdown")
    perf.stage("0_chunking")
    chunks = chunk_markdown(input_path)
    level1 = [c for c in chunks if c["level"] == 1]
    level2_plus = [c for c in chunks if c["level"] >= 2]
    print(f"{_ts()}   {len(chunks)} chunks extracted ({len(level1)} parents, {len(level2_plus)} children)")

    # Stage 1: Create tree
    _banner(1, "Creating Knowledge Tree")
    perf.stage("1_tree_creation")
    tree = await sift.create_tree(tree_name, tree_scope)
    tree_id = tree["tree_id"]
    root_id = tree["root_id"]
    print(f"{_ts()}   Tree ID: {tree_id}")

    # Stage 2: Two-pass hierarchical creation
    _banner(2, "Building Document Structure")

    # Pass 1 — structural parent nodes (parallel, no LLM)
    perf.stage("2a_parent_nodes")
    print(f"{_ts()}   Pass 1: Creating {len(level1)} parent nodes (parallel)...")
    parent_node_ids: dict[str, str] = {}
    p1_sem = asyncio.Semaphore(10)

    async def _create_parent(i: int, chunk: dict) -> tuple[str, str | None]:
        async with p1_sem:
            result_text = await sift.create_node(
                name=chunk["title"],
                scope=chunk["text"][:200] if chunk["text"] else chunk["title"],
                parent_id=root_id,
                tree_id=tree_id,
                pipeline_mode=True,
            )
        match = UUID_RE.search(result_text)
        nid = match.group(0) if match else None
        print(f"{_ts()}     [{i}/{len(level1)}] {chunk['title']}")
        return chunk["title"], nid

    p1_results = await asyncio.gather(
        *[_create_parent(i, c) for i, c in enumerate(level1, 1)],
        return_exceptions=True,
    )
    for r in p1_results:
        if isinstance(r, tuple):
            title, nid = r
            if nid:
                parent_node_ids[title] = nid

    print(f"{_ts()}   Pass 1 complete: {len(parent_node_ids)} parent nodes created")

    # Pass 2 — triage level-2 chunks under parents (parallel, with LLM)
    perf.stage("2b_triage")
    print(f"\n{_ts()}   Pass 2: Triaging {len(level2_plus)} sections with LLM (concurrency=10)...")
    sem = asyncio.Semaphore(10)
    counter = {"done": 0}

    async def rate_limited_triage(chunk: dict) -> str:
        parent_id = parent_node_ids.get(chunk["parent_title"], root_id)
        try:
            async with sem:
                result = await triage_agent(chunk, tree_id, parent_id, sift, model, pipeline_mode=True)
        except Exception:
            counter["done"] += 1
            print(f"{_ts()}     [{counter['done']}/{len(level2_plus)}] {chunk['title']} ✗")
            raise
        counter["done"] += 1
        print(f"{_ts()}     [{counter['done']}/{len(level2_plus)}] {chunk['title']} ✓")
        return result

    tasks = [rate_limited_triage(c) for c in level2_plus]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    failures = [r for r in results if isinstance(r, Exception)]
    successes = len(results) - len(failures)

    print(f"{_ts()}   Pass 2 complete: {successes} nodes created", end="")
    if failures:
        print(f" ({len(failures)} failed)")
        for f in failures[:3]:
            print(f"{_ts()}     ! {type(f).__name__}: {f}")
    else:
        print()

    # Stage 3: Linkage pass
    _banner(3, "Cross-Reference Linkage")
    perf.stage("3_linkage")
    try:
        link_count = await linkage_agent(tree_id, sift, smart_model, pipeline_mode=True)
        print(f"{_ts()}   Linkage complete: {link_count} cross-references added")
    except Exception as e:
        print(f"{_ts()}   ! Linkage failed ({type(e).__name__}: {e}), continuing")

    # Finish profiling
    perf.finish()
    perf.summary()
    perf.save()

    total = time.time() - perf._t0
    m, s = divmod(int(total), 60)
    print(f"\n{'=' * 60}")
    print(f"  Pipeline complete in {m}m {s}s")
    print(f"  {len(parent_node_ids)} chapters | {successes} sections | tree {tree_id[:8]}...")
    print(f"{'=' * 60}")

    return tree_id
