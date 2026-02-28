"""Triage and linkage agents for the markdown document pipeline."""

import asyncio
import re

from lib.llm import complete
from lib.perf import perf
from lib.sifttext import SiftTextClient
from lib.config import UUID_RE, load_prompt

_triage_prompt: str | None = None
_linkage_prompt: str | None = None


def _get_triage_prompt() -> str:
    global _triage_prompt
    if _triage_prompt is None:
        _triage_prompt = load_prompt("markdown_triage")
    return _triage_prompt


def _get_linkage_prompt() -> str:
    global _linkage_prompt
    if _linkage_prompt is None:
        _linkage_prompt = load_prompt("markdown_linkage")
    return _linkage_prompt


async def triage_agent(
    chunk: dict,
    tree_id: str,
    parent_id: str,
    sift: SiftTextClient,
    model: str = "gpt-5.2-chat-main",
    pipeline_mode: bool = False,
) -> str:
    """Analyze a document chunk via LLM and create a SiftText node.

    Returns the created node_id.
    """
    # Skip chunks with no text (e.g., parent-only chapter headers)
    if not chunk["text"].strip():
        result = await sift.create_node(
            name=chunk["title"],
            scope=f"Section header for {chunk['title']}",
            parent_id=parent_id,
            tree_id=tree_id,
            pipeline_mode=pipeline_mode,
        )
        match = UUID_RE.search(result)
        return match.group(0) if match else ""

    llm_output = await complete(
        system=_get_triage_prompt().format(title=chunk["title"]),
        user=chunk["text"],
        model=model,
    )

    # First line = one-line summary for scope
    lines = llm_output.strip().split("\n", 1)
    scope = lines[0].strip()
    crystallization = llm_output.strip()

    result = await sift.create_node(
        name=chunk["title"],
        scope=scope,
        parent_id=parent_id,
        tree_id=tree_id,
        crystallization=crystallization,
        pipeline_mode=pipeline_mode,
    )

    match = UUID_RE.search(result)
    return match.group(0) if match else ""


async def linkage_agent(
    tree_id: str,
    sift: SiftTextClient,
    model: str = "gpt-5.2-chat-main",
    max_nodes: int | None = None,
    llm_concurrency: int = 4,
    link_concurrency: int = 20,
    pipeline_mode: bool = False,
) -> int:
    """Read tree structure and add cross-reference links between nodes.

    Returns count of links added.
    """
    outline = await sift.get_outline(tree_id, max_depth=None)

    # Collect all node IDs from the outline
    node_ids = UUID_RE.findall(outline)
    if max_nodes:
        node_ids = node_ids[:max_nodes]

    # Read all nodes in parallel
    read_sem = asyncio.Semaphore(20)

    async def _read_node(nid: str) -> dict:
        async with read_sem:
            content = await sift.get_node(nid)
        name_match = re.search(r"<name>(.*?)</name>", content)
        name = name_match.group(1) if name_match else nid
        return {"id": nid, "name": name, "content": content}

    node_results = await asyncio.gather(
        *[_read_node(nid) for nid in node_ids], return_exceptions=True
    )
    nodes: list[dict] = [r for r in node_results if isinstance(r, dict)]
    read_failures = [r for r in node_results if isinstance(r, Exception)]
    if read_failures:
        print(f"  Warning: {len(read_failures)} node reads failed")

    perf.event("linkage_read_nodes", 0, nodes_read=len(nodes), failures=len(read_failures))

    # Build name -> id lookup for link resolution
    name_to_id: dict[str, str] = {n["name"]: n["id"] for n in nodes}

    # --- Build all batches upfront ---
    batch_size = 8
    batches: list[tuple[int, list[dict]]] = []
    for i in range(0, len(nodes), batch_size):
        batch = nodes[i : i + batch_size]
        batches.append((i // batch_size + 1, batch))
    total_batches = len(batches)

    # --- Process batches with LLM concurrency ---
    llm_sem = asyncio.Semaphore(llm_concurrency)
    all_pending_links: list[dict] = []
    link_lock = asyncio.Lock()

    async def _process_batch(batch_num: int, batch: list[dict]) -> int:
        """Analyze one batch via LLM, return count of links found."""
        batch_text = ""
        for node in batch:
            cryst_match = re.search(
                r"<crystallization>(.*?)</crystallization>", node["content"], re.DOTALL
            )
            cryst = cryst_match.group(1).strip()[:2000] if cryst_match else ""
            batch_text += f"\n### {node['name']}\n{cryst}\n"

        async with llm_sem:
            try:
                llm_output = await complete(
                    system=_get_linkage_prompt().format(outline=outline),
                    user=f"Analyze these nodes for cross-references:\n{batch_text}",
                    model=model,
                )
            except Exception as e:
                print(f"  Batch {batch_num}/{total_batches}: failed ({e})")
                perf.event("linkage_batch", 0, success=False, batch=batch_num,
                           error=str(e))
                return 0

        # Parse links
        batch_links: list[dict] = []
        for line in llm_output.strip().split("\n"):
            if " -> " not in line or " | " not in line:
                continue
            left, reason = line.rsplit(" | ", 1)
            source_title, target_title = left.split(" -> ", 1)
            source_title = source_title.strip()
            target_title = target_title.strip()

            source_id = name_to_id.get(source_title)
            if not source_id:
                continue

            target_id = name_to_id.get(target_title)
            batch_links.append({
                "source_node_id": source_id,
                "target_name": target_title,
                "description": reason.strip(),
                **({"target_node_id": target_id} if target_id else {}),
            })

        async with link_lock:
            all_pending_links.extend(batch_links)

        count = len(batch_links)
        print(f"  Batch {batch_num}/{total_batches}: {count} links found")
        perf.event("linkage_batch", 0, batch=batch_num, links_found=count)
        return count

    # Fire all batches concurrently (limited by llm_sem)
    batch_results = await asyncio.gather(
        *[_process_batch(num, batch) for num, batch in batches],
        return_exceptions=True,
    )
    total_found = sum(r for r in batch_results if isinstance(r, int))
    print(f"  {total_found} links found across {total_batches} batches")

    # Create all links in parallel
    resolved = sum(1 for l in all_pending_links if "target_node_id" in l)
    print(f"  Creating {len(all_pending_links)} links ({resolved} locally resolved, "
          f"{len(all_pending_links) - resolved} server-side)...")
    link_sem = asyncio.Semaphore(link_concurrency)

    async def _create_link(link: dict) -> bool:
        async with link_sem:
            try:
                await sift.link_by_name(**link, pipeline_mode=pipeline_mode)
                return True
            except Exception:
                return False

    link_results = await asyncio.gather(
        *[_create_link(lnk) for lnk in all_pending_links], return_exceptions=True
    )
    links_added = sum(1 for r in link_results if r is True)

    return links_added
