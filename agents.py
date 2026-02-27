"""Agent functions: triage, linkage, and query over SiftText trees."""

import asyncio
import json
import re
import time

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown

from llm import complete, get_client
from perf import perf
from sifttext import SiftTextClient

_console = Console()

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

## How to use the tree

The tree has EXPLICIT structure — chapters, sections, and cross-reference links \
between nodes. Use this structure to navigate, not shotgun keyword searches.

**Step 1 — Orient:** Call get_outline() FIRST to see the tree's hierarchy. This is \
your map. You'll see chapter names and section names with their IDs. Use this to \
know WHERE to look. It can be scoped to nodes to read subtrees.

**Step 2 — Search & Read:** Do some targeted searches. Read the most relevant \
nodes. IMPORTANT: when you read a node, check its <links> section — forward links, \
see_also, and backlinks point to related nodes. Follow those links instead of \
searching for the same content.

**Step 3 — Answer:** After sufficient rounds of tool calls, STOP and answer.

## Multi-turn conversations

This is a multi-turn conversation. Prior questions, tool results, and answers are \
preserved in context. Use information from earlier turns to inform follow-up answers \
— you don't need to re-fetch what you already have, but you CAN use any tool at any \
time to dig deeper, follow links, or explore new areas of the tree.

## How to write your answers

Write like a sharp colleague explaining something in a conversation — direct, clear, \
and easy to follow. NO numbered lists, NO bullet-point dumps. Use short paragraphs \
(2-4 sentences each). Each paragraph should make ONE point and land it.

Use **bold text** to flag key terms, article references, or the core takeaway of a \
paragraph so a reader scanning gets the shape of your answer before reading in full. \
Think of bold phrases as signposts through the explanation.

Explain substance over structure — the reasoning behind provisions, how parts of the \
regulation interact, practical implications. Don't just state requirements; explain \
what they mean and why they matter.

Cite Articles, Annexes, Recitals, and Chapters inline (e.g., "Under **Article 6(1)**, \
..."). Quote relevant text where it sharpens the point. Let citations support the \
narrative, not structure it.

When a question touches multiple areas, call out where things go deeper — give the \
reader a thread to pull on next. Keep the whole response digestible: if you find \
yourself writing a paragraph longer than 4 sentences, break it up.

If the tree lacks sufficient information on a point, say so explicitly.

## Rules
- NEVER use numbered lists. Avoid bullet points. Write in short paragraphs.
- Use **bold** liberally to make key points scannable.
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
    max_turns: int = 10,
    history: list[dict] | None = None,
) -> tuple[str, list[dict]]:
    """Agentic query with read-only tree tools and streaming output.

    Level 1 streaming: tool call activity printed to stdout.
    Level 2 streaming: final answer tokens streamed to stdout.
    Returns (response_text, conversation_history) for multi-turn context.
    """
    client = get_client()
    if history is not None:
        # Continue prior conversation — append new user question
        messages = history + [{"role": "user", "content": question}]
    else:
        messages = [
            {"role": "system", "content": QUERY_SYSTEM},
            {"role": "user", "content": question},
        ]

    full_response = ""
    t0 = time.time()

    reasoning_kwargs = {"extra_body": {"reasoning": {"effort": "high"}}}

    def _stream_reasoning(rd, state: dict) -> list[dict]:
        """Print reasoning tokens dimmed; return raw details for context preservation."""
        raw: list[dict] = []
        for detail in (rd if isinstance(rd, list) else [rd]):
            text = (detail.get("text") or detail.get("summary", "")) if isinstance(detail, dict) else str(detail)
            if text:
                if not state["in_reasoning"]:
                    _console.print("\\[thinking] ", style="dim", end="", highlight=False)
                    state["in_reasoning"] = True
                _console.print(text, style="dim", end="", highlight=False)
                state["reasoning_parts"].append(text)
            if isinstance(detail, dict):
                raw.append(detail)
        return raw

    def _end_reasoning(state: dict):
        if state["in_reasoning"]:
            _console.print()
            state["in_reasoning"] = False

    async def _collect_stream(stream, state: dict):
        """Consume a streaming response, printing reasoning/tool activity dimmed.

        Content tokens are progressively rendered as rich Markdown via Live.
        Returns (content, tool_calls_list, reasoning_details_raw).
        """
        content_parts: list[str] = []
        reasoning_details_raw: list[dict] = []
        tool_calls_acc: dict[int, dict] = {}
        state["in_reasoning"] = False
        state["reasoning_parts"] = []
        live: Live | None = None

        try:
            async for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta

                # Reasoning traces (dimmed)
                # OpenRouter sends reasoning_details, but the OpenAI SDK's
                # ChoiceDelta Pydantic model doesn't define it — check model_extra.
                extras = getattr(delta, "model_extra", None) or {}
                rd = (
                    extras.get("reasoning_details")
                    or extras.get("reasoning")
                    or getattr(delta, "reasoning_content", None)
                    or getattr(delta, "reasoning", None)
                )
                if rd:
                    reasoning_details_raw.extend(_stream_reasoning(rd, state))

                # Content tokens — progressively render as markdown
                if delta.content:
                    _end_reasoning(state)
                    content_parts.append(delta.content)
                    if live is None:
                        live = Live(Markdown("".join(content_parts)), console=_console, refresh_per_second=8)
                        live.start()
                    else:
                        live.update(Markdown("".join(content_parts)))

                # Tool call deltas
                if delta.tool_calls:
                    _end_reasoning(state)
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

            _end_reasoning(state)
        finally:
            if live is not None:
                live.stop()

        tc_list = [
            {
                "id": tc["id"],
                "type": "function",
                "function": {"name": tc["name"], "arguments": tc["arguments"]},
            }
            for tc in (tool_calls_acc[i] for i in sorted(tool_calls_acc))
        ] if tool_calls_acc else []

        return "".join(content_parts), tc_list, reasoning_details_raw

    for turn in range(max_turns):
        stream = await client.chat.completions.create(
            model=model,
            messages=messages,
            tools=QUERY_TOOLS,
            stream=True,
            **reasoning_kwargs,
        )

        state: dict = {}
        content, tc_list, reasoning_details_raw = await _collect_stream(stream, state)

        # ── Tool calls: execute and loop ──
        if tc_list:
            assistant_msg: dict = {
                "role": "assistant",
                "content": content or None,
                "tool_calls": tc_list,
            }
            if reasoning_details_raw:
                assistant_msg["reasoning_details"] = reasoning_details_raw
            messages.append(assistant_msg)

            for tc in tc_list:
                name = tc["function"]["name"]
                args = json.loads(tc["function"]["arguments"])

                # Tool activity (dimmed, highlight=False to avoid rich markup parsing)
                if name == "search_tree":
                    _console.print(f"  \\[search] \"{args.get('query', '')}\"", style="dim", highlight=False)
                elif name == "read_node":
                    _console.print(f"  \\[read]   {args.get('node_id', '')[:12]}...", style="dim", highlight=False)
                elif name == "get_outline":
                    nid = args.get("node_id", "")
                    _console.print(f"  \\[outline] {nid[:12] + '...' if nid else '(root)'}", style="dim", highlight=False)

                result = await _execute_query_tool(name, args, tree_id, sift)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result,
                })

            continue

        # ── Final text response (already rendered via Live) ──
        full_response = content
        if full_response.strip():
            messages.append({"role": "assistant", "content": full_response})
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
            **reasoning_kwargs,
        )
        state = {}
        content, _, _ = await _collect_stream(stream, state)
        full_response = content
        if full_response.strip():
            messages.append({"role": "assistant", "content": full_response})

    dur = (time.time() - t0) * 1000
    perf.event("query_agent", dur, turns=turn + 1, model=model)
    return full_response, messages
