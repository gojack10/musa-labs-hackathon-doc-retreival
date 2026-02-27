"""Agent functions: triage, linkage, and query over SiftText trees."""

import asyncio
import json
import re
import time

from llm import complete, get_client
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
You are an expert on the EU AI Act (Regulation (EU) 2024/1689). You have access to \
a structured knowledge tree containing the full text and analysis of the regulation.

Use your tools to research the tree before answering:
1. Search for terms related to the question
2. Read relevant nodes for detailed content and cross-references
3. Follow links to related sections when needed
4. Synthesize a grounded answer

Rules:
- Always use tools to find information — do not rely on prior knowledge
- Cite specific Articles, Annexes, Recitals, and Chapters by name
- Quote relevant text where helpful
- If the tree lacks sufficient information, say so explicitly
"""

QUERY_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_tree",
            "description": "Search the EU AI Act knowledge tree. Returns ranked results with node IDs, names, and relevance snippets.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": 'Search query. Supports prefix*, boolean OR/NOT, and "exact phrases".',
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_node",
            "description": "Read full content of a node: scope, crystallization (structured analysis), cross-reference links, and children summary.",
            "parameters": {
                "type": "object",
                "properties": {
                    "node_id": {
                        "type": "string",
                        "description": "UUID of the node to read",
                    }
                },
                "required": ["node_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_outline",
            "description": "Get tree structure showing node hierarchy with names and IDs. Use to understand document organization or browse into a subtree.",
            "parameters": {
                "type": "object",
                "properties": {
                    "node_id": {
                        "type": "string",
                        "description": "Optional: focus on subtree rooted at this node. Omit for top-level overview.",
                    }
                },
                "required": [],
            },
        },
    },
]


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


async def _execute_query_tool(
    name: str, args: dict, tree_id: str, sift: SiftTextClient
) -> str:
    """Execute a read-only tree tool and return the result text."""
    if name == "search_tree":
        return await sift.search(args["query"], tree_id)
    elif name == "read_node":
        return await sift.get_node(args["node_id"])
    elif name == "get_outline":
        node_id = args.get("node_id")
        return await sift.get_outline(tree_id, max_depth=2, node_id=node_id)
    return f"Unknown tool: {name}"


async def query_agent(
    question: str,
    tree_id: str,
    sift: SiftTextClient,
    model: str = "openai/gpt-5.2",
    max_turns: int = 20,
) -> str:
    """Agentic query with read-only tree tools and streaming output.

    Level 1 streaming: tool call activity printed to stdout.
    Level 2 streaming: final answer tokens streamed to stdout.
    Returns the full response text.
    """
    client = get_client()
    messages: list[dict] = [
        {"role": "system", "content": QUERY_SYSTEM},
        {"role": "user", "content": question},
    ]

    full_response = ""
    t0 = time.time()

    for turn in range(max_turns):
        stream = await client.chat.completions.create(
            model=model,
            messages=messages,
            tools=QUERY_TOOLS,
            stream=True,
        )

        # Accumulate streamed response
        content_parts: list[str] = []
        tool_calls_acc: dict[int, dict] = {}

        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta

            # Level 2: stream content tokens to stdout immediately
            if delta.content:
                print(delta.content, end="", flush=True)
                content_parts.append(delta.content)

            # Accumulate tool call deltas
            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    if idx not in tool_calls_acc:
                        tool_calls_acc[idx] = {"id": "", "name": "", "arguments": ""}
                    if tc_delta.id:
                        tool_calls_acc[idx]["id"] = tc_delta.id
                    if tc_delta.function:
                        if tc_delta.function.name:
                            tool_calls_acc[idx]["name"] = tc_delta.function.name
                        if tc_delta.function.arguments:
                            tool_calls_acc[idx]["arguments"] += tc_delta.function.arguments

        content = "".join(content_parts)

        # ── Tool calls: execute and loop ──
        if tool_calls_acc:
            tc_list = [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {"name": tc["name"], "arguments": tc["arguments"]},
                }
                for tc in (tool_calls_acc[i] for i in sorted(tool_calls_acc))
            ]
            messages.append({
                "role": "assistant",
                "content": content or None,
                "tool_calls": tc_list,
            })

            for tc in tc_list:
                name = tc["function"]["name"]
                args = json.loads(tc["function"]["arguments"])

                # Level 1: show tool activity
                if name == "search_tree":
                    print(f"  [search] \"{args.get('query', '')}\"")
                elif name == "read_node":
                    print(f"  [read]   {args.get('node_id', '')[:12]}...")
                elif name == "get_outline":
                    nid = args.get("node_id", "")
                    print(f"  [outline] {nid[:12] + '...' if nid else '(root)'}")

                result = await _execute_query_tool(name, args, tree_id, sift)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result,
                })

            continue

        # ── Final text response (already streamed above) ──
        full_response = content
        break
    else:
        # Exhausted max_turns — force a final answer without tools
        messages.append({
            "role": "user",
            "content": "(Maximum research steps reached. Please answer now with what you have.)",
        })
        stream = await client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
        )
        parts: list[str] = []
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                text = chunk.choices[0].delta.content
                print(text, end="", flush=True)
                parts.append(text)
        full_response = "".join(parts)

    dur = (time.time() - t0) * 1000
    perf.event("query_agent", dur, turns=turn + 1, model=model)
    print()  # Trailing newline after streamed output
    return full_response
