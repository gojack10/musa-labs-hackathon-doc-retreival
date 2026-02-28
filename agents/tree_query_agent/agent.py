"""Agentic query agent with read-only SiftText tree tools and streaming output."""

import asyncio
import json
import logging
import re
import time

from rich.live import Live
from rich.markdown import Markdown

from lib.config import UUID_RE, console, load_prompt
from lib.llm import get_openrouter_client, OPENROUTER_DEFAULT_MODEL
from lib.perf import perf
from lib.sifttext import SiftTextClient

_dbg = logging.getLogger("query_agent.debug")

QUERY_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_tree",
            "description": "Search the knowledge tree. Returns ranked results with node IDs, names, and relevance snippets.",
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
    model: str = OPENROUTER_DEFAULT_MODEL,
    max_turns: int = 10,
    history: list[dict] | None = None,
    prompt_name: str = "tree_query",
    debug: bool = False,
) -> tuple[str, list[dict]]:
    """Agentic query with read-only tree tools and streaming output.

    Level 1 streaming: tool call activity printed to stdout.
    Level 2 streaming: final answer tokens streamed to stdout.
    Returns (response_text, conversation_history) for multi-turn context.
    """
    if debug and not _dbg.handlers:
        _fh = logging.FileHandler("/tmp/query_agent_debug.log", mode="a")
        _fh.setFormatter(logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S"))
        _dbg.addHandler(_fh)
        _dbg.setLevel(logging.DEBUG)
        _dbg.debug("=== new query: %s ===", question)

    client = get_openrouter_client()
    if history is not None:
        messages = history + [{"role": "user", "content": question}]
    else:
        try:
            system_prompt = load_prompt(prompt_name)
        except FileNotFoundError:
            system_prompt = "Answer questions using the knowledge tree. You MUST read_node before citing content."
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]

    full_response = ""
    t0 = time.time()
    node_names: dict[str, str] = {}   # node_id → name cache
    linked_ids: set[str] = set()      # IDs seen in <links> sections
    cited_ids: list[str] = []         # read_node targets, in order

    reasoning_kwargs = {
        "extra_body": {
            "reasoning": {"type": "enabled", "budget_tokens": 10000},
            "include_reasoning": True,
        },
    }

    def _stream_reasoning(rd, state: dict) -> list[dict]:
        """Print reasoning tokens dimmed; return raw details for context preservation."""
        raw: list[dict] = []
        for detail in (rd if isinstance(rd, list) else [rd]):
            text = (detail.get("text") or detail.get("summary", "")) if isinstance(detail, dict) else str(detail)
            if text:
                if not state["in_reasoning"]:
                    console.print("\\[thinking] ", style="dim", end="", highlight=False)
                    state["in_reasoning"] = True
                console.print(text, style="dim", end="", highlight=False)
                state["reasoning_parts"].append(text)
            if isinstance(detail, dict):
                raw.append(detail)
        return raw

    def _end_reasoning(state: dict):
        if state["in_reasoning"]:
            console.print()
            state["in_reasoning"] = False

    async def _collect_stream(stream, state: dict):
        """Consume a streaming response, printing reasoning/tool activity dimmed.

        Content tokens are progressively rendered as rich Markdown via Live.
        Returns (content, tool_calls_list, reasoning_details_raw, usage_dict).
        """
        content_parts: list[str] = []
        reasoning_details_raw: list[dict] = []
        tool_calls_acc: dict[int, dict] = {}
        usage: dict = {}
        state["in_reasoning"] = False
        state["reasoning_parts"] = []
        live: Live | None = None

        try:
            async for chunk in stream:
                # Capture usage from final chunk
                if chunk.usage:
                    usage = {
                        "input": chunk.usage.prompt_tokens or 0,
                        "output": chunk.usage.completion_tokens or 0,
                    }
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
                        live = Live(Markdown("".join(content_parts)), console=console, refresh_per_second=8, vertical_overflow="visible")
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

        return "".join(content_parts), tc_list, reasoning_details_raw, usage

    total_usage = {"input": 0, "output": 0}

    for turn in range(max_turns):
        stream = await client.chat.completions.create(
            model=model,
            messages=messages,
            tools=QUERY_TOOLS,
            stream=True,
            stream_options={"include_usage": True},
            **reasoning_kwargs,
        )

        state: dict = {}
        content, tc_list, reasoning_details_raw, usage = await _collect_stream(stream, state)
        total_usage["input"] += usage.get("input", 0)
        total_usage["output"] += usage.get("output", 0)

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
                fname = tc["function"]["name"]
                raw_args = tc["function"]["arguments"]
                if not raw_args or not raw_args.strip():
                    console.print(f"  \\[{fname}] skipped — empty arguments from model", style="dim red")
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": "Error: empty arguments. Please retry with valid arguments.",
                    })
                    continue
                args = json.loads(raw_args)
                result = await _execute_query_tool(fname, args, tree_id, sift)

                # Extract node names from results into cache
                if fname in ("read_node", "get_outline", "search_tree"):
                    # XML format: id="UUID"><name>...</name>
                    for nid, nname in re.findall(
                        r'id="([0-9a-f-]{36})">\s*<name>([^<]+)</name>', result
                    ):
                        node_names[nid] = nname
                    # Outline plaintext: UUID status Name (children)
                    for nid, nname in re.findall(
                        r'([0-9a-f-]{36})\s+\S+\s{2,3}(.+?)(?:\s+\(\d+\))?$',
                        result, re.MULTILINE,
                    ):
                        nname = nname.strip()
                        if nname and nid not in node_names:
                            node_names[nid] = nname
                if fname == "read_node":
                    # Collect linked IDs for link-traversal detection
                    # SiftText uses multiple tags for links: <links>, <linked_nodes>,
                    # <see_also>, <backlinks>. Scan all of them.
                    found_lids: list[str] = []
                    matched_tags: list[str] = []
                    for link_tag in ("links", "linked_nodes", "see_also", "backlinks"):
                        tag_match = re.search(rf"<{link_tag}>(.*?)</{link_tag}>", result, re.DOTALL)
                        if tag_match:
                            matched_tags.append(link_tag)
                            for lid in UUID_RE.findall(tag_match.group(1)):
                                if lid != args.get("node_id", ""):  # exclude self-references
                                    found_lids.append(lid)
                                    linked_ids.add(lid)
                    # Also grab inline [[Name|uuid]] links from crystallization/scope
                    for lid in re.findall(r'\[\[[^\]]*\|([0-9a-f-]{36})', result):
                        if lid != args.get("node_id", "") and lid not in linked_ids:
                            found_lids.append(lid)
                            linked_ids.add(lid)
                    if debug:
                        if found_lids:
                            _dbg.debug("read %s: link sources: %s → %d UUIDs",
                                       args.get("node_id", "")[:8], ', '.join(matched_tags) or 'inline only', len(found_lids))
                            for lid in found_lids[:5]:
                                lname = node_names.get(lid, "?")
                                _dbg.debug("  %s… (%s)", lid[:8], lname)
                            if len(found_lids) > 5:
                                _dbg.debug("  …and %d more", len(found_lids) - 5)
                        else:
                            _dbg.debug("read %s: no links found", args.get("node_id", "")[:8])

                # Tool activity display (after execution so we have names)
                if fname == "search_tree":
                    console.print(f"  \\[search] \"{args.get('query', '')}\"", style="dim", highlight=False)
                elif fname == "read_node":
                    nid = args.get("node_id", "")
                    nname = node_names.get(nid, "")
                    label = f"{nname} ({nid[:8]}…)" if nname else f"{nid[:12]}…"
                    via = ""
                    if nid in linked_ids:
                        via = " → via link"
                        if debug:
                            _dbg.debug("MATCH: %s… was in linked_ids", nid[:8])
                    elif debug and linked_ids:
                        _dbg.debug("%s… NOT in linked_ids (%d tracked)", nid[:8], len(linked_ids))
                    console.print(f"  \\[read] {label}", style="dim", end="", highlight=False)
                    if via:
                        console.print(via, style="blue", end="", highlight=False)
                    console.print()
                    if nid not in cited_ids:
                        cited_ids.append(nid)
                elif fname == "get_outline":
                    nid = args.get("node_id", "")
                    nname = node_names.get(nid, "")
                    if nid:
                        label = f"{nname} ({nid[:8]}…)" if nname else f"{nid[:12]}…"
                    else:
                        label = "(root)"
                    console.print(f"  \\[outline] {label}", style="dim", highlight=False)

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
            stream_options={"include_usage": True},
            **reasoning_kwargs,
        )
        state = {}
        content, _, _, usage = await _collect_stream(stream, state)
        total_usage["input"] += usage.get("input", 0)
        total_usage["output"] += usage.get("output", 0)
        full_response = content
        if full_response.strip():
            messages.append({"role": "assistant", "content": full_response})

    # ── Citation footer ──
    if cited_ids:
        console.print()
        console.print("  Sources:", style="dim")
        names = []
        for cid in cited_ids:
            cname = node_names.get(cid, cid[:12] + "…")
            names.append(cname)
        console.print("  " + "  ·  ".join(f"[italic blue]{n}[/]" for n in names))

    # ── Token usage ──
    if total_usage["input"] or total_usage["output"]:
        console.print(
            f"  tokens: {total_usage['input']:,} in · {total_usage['output']:,} out",
            style="dim",
        )

    dur = (time.time() - t0) * 1000
    perf.event("query_agent", dur, turns=turn + 1, model=model)
    return full_response, messages
