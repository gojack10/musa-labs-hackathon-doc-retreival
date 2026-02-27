"""Agent functions: triage, linkage, and query over SiftText trees."""

import asyncio
import re

from llm import complete
from perf import perf
from sifttext import SiftTextClient

_UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")

TRIAGE_SYSTEM = """\
You are a legal document analyst. Analyze the following section from the EU AI Act.

Section: {title}

Respond in this exact format — first line MUST be a one-line summary (max 120 chars), then the rest:

<one-line summary of what this section establishes>

## Key Obligations
- <bullet points of requirements, prohibitions, or mandates>

## Entities Affected
- <who this applies to: providers, deployers, importers, etc.>

## Cross-References
- <other Articles, Annexes, or Chapters mentioned in this text>

## Timeline & Status
- <effective dates, transition periods, deadlines if mentioned>

Be precise and cite specific paragraph numbers. If a section is empty or not applicable, write "None identified."
"""

LINKAGE_SYSTEM = """\
You are a document structure analyst. Given a batch of nodes from a structured tree \
representing the EU AI Act, identify cross-references between them.

Here is the full tree outline:
{outline}

For each node in the batch below, identify which OTHER nodes in the tree it should \
link to based on explicit cross-references (e.g., "as defined in Article 3", \
"referred to in Annex III").

Respond with one link per line in this exact format:
SOURCE_TITLE -> TARGET_TITLE | reason

Only output links where the source text explicitly references the target. \
Do not infer implicit connections. If no links exist for a node, skip it.
"""

QUERY_SYSTEM = """\
You are an expert on the EU AI Act. Answer the user's question using ONLY the \
provided context from the structured document tree. Cite specific articles, \
annexes, and recitals by name.

Context from document tree:
{context}

Rules:
- Ground every claim in a specific node from the context
- Quote relevant text where helpful
- If the context doesn't contain enough information, say so explicitly
- Reference nodes by their title (e.g., "According to Article 6...")
"""


async def triage_agent(
    chunk: dict,
    tree_id: str,
    parent_id: str,
    sift: SiftTextClient,
    model: str = "openai/gpt-5.2-chat",
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
        match = _UUID_RE.search(result)
        return match.group(0) if match else ""

    llm_output = await complete(
        system=TRIAGE_SYSTEM.format(title=chunk["title"]),
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

    match = _UUID_RE.search(result)
    return match.group(0) if match else ""


async def linkage_agent(
    tree_id: str,
    sift: SiftTextClient,
    model: str = "openai/gpt-5.2",
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
    node_ids = _UUID_RE.findall(outline)
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
                    system=LINKAGE_SYSTEM.format(outline=outline),
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


async def query_agent(
    question: str,
    tree_id: str,
    sift: SiftTextClient,
    model: str = "openai/gpt-5.2",
) -> str:
    """Search the tree and answer a question with grounded citations."""
    # Search for relevant nodes
    search_results = await sift.search(question, tree_id)

    # Extract node IDs from search results (top 5)
    result_ids = _UUID_RE.findall(search_results)[:5]

    if not result_ids:
        return "No relevant sections found in the document tree for this question."

    # Read matching nodes in parallel
    async def _read(nid: str) -> tuple[str, str]:
        content = await sift.get_node(nid)
        name_match = re.search(r"<name>(.*?)</name>", content)
        cryst_match = re.search(
            r"<crystallization>(.*?)</crystallization>", content, re.DOTALL
        )
        name = name_match.group(1) if name_match else nid
        cryst = cryst_match.group(1).strip() if cryst_match else ""
        return name, cryst

    results = await asyncio.gather(*[_read(nid) for nid in result_ids])
    context_parts = [f"### {name}\n{cryst}" for name, cryst in results]

    context = "\n\n---\n\n".join(context_parts)

    return await complete(
        system=QUERY_SYSTEM.format(context=context),
        user=question,
        model=model,
    )
