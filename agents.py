"""Agent functions: triage, linkage, and query over SiftText trees."""

import re

from llm import complete
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
    model: str = "openai/gpt-4.1-mini",
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
    )

    match = _UUID_RE.search(result)
    return match.group(0) if match else ""


async def linkage_agent(
    tree_id: str,
    sift: SiftTextClient,
    model: str = "openai/gpt-4.1-mini",
    max_nodes: int | None = None,
) -> int:
    """Read tree structure and add cross-reference links between nodes.

    Returns count of links added.
    """
    outline = await sift.get_outline(tree_id, max_depth=None)

    # Collect all node IDs from the outline
    node_ids = _UUID_RE.findall(outline)
    if max_nodes:
        node_ids = node_ids[:max_nodes]

    # Read all nodes and build a lookup
    nodes: list[dict] = []
    for nid in node_ids:
        content = await sift.get_node(nid)
        # Extract name from XML-like response
        name_match = re.search(r"<name>(.*?)</name>", content)
        name = name_match.group(1) if name_match else nid
        nodes.append({"id": nid, "name": name, "content": content})

    # Process in batches of 8
    links_added = 0
    batch_size = 8

    for i in range(0, len(nodes), batch_size):
        batch = nodes[i : i + batch_size]

        batch_text = ""
        for node in batch:
            # Extract just crystallization for brevity
            cryst_match = re.search(
                r"<crystallization>(.*?)</crystallization>", node["content"], re.DOTALL
            )
            cryst = cryst_match.group(1).strip()[:2000] if cryst_match else ""
            batch_text += f"\n### {node['name']}\n{cryst}\n"

        llm_output = await complete(
            system=LINKAGE_SYSTEM.format(outline=outline),
            user=f"Analyze these nodes for cross-references:\n{batch_text}",
            model=model,
        )

        # Parse "SOURCE_TITLE -> TARGET_TITLE | reason" lines
        for line in llm_output.strip().split("\n"):
            if " -> " not in line or " | " not in line:
                continue
            left, reason = line.rsplit(" | ", 1)
            source_title, target_title = left.split(" -> ", 1)
            source_title = source_title.strip()
            target_title = target_title.strip()

            # Find source node ID
            source_id = None
            for node in nodes:
                if node["name"] == source_title:
                    source_id = node["id"]
                    break

            if not source_id:
                continue

            try:
                await sift.link_by_name(
                    source_node_id=source_id,
                    target_name=target_title,
                    description=reason.strip(),
                )
                links_added += 1
            except Exception:
                # Link failures are non-fatal — target name might not match exactly
                continue

    return links_added


async def query_agent(
    question: str,
    tree_id: str,
    sift: SiftTextClient,
    model: str = "openai/gpt-5.2-chat",
) -> str:
    """Search the tree and answer a question with grounded citations."""
    # Search for relevant nodes
    search_results = await sift.search(question, tree_id)

    # Extract node IDs from search results (top 5)
    result_ids = _UUID_RE.findall(search_results)[:5]

    if not result_ids:
        return "No relevant sections found in the document tree for this question."

    # Read each matching node
    context_parts: list[str] = []
    for nid in result_ids:
        content = await sift.get_node(nid)
        name_match = re.search(r"<name>(.*?)</name>", content)
        cryst_match = re.search(
            r"<crystallization>(.*?)</crystallization>", content, re.DOTALL
        )
        name = name_match.group(1) if name_match else nid
        cryst = cryst_match.group(1).strip() if cryst_match else ""
        context_parts.append(f"### {name}\n{cryst}")

    context = "\n\n---\n\n".join(context_parts)

    return await complete(
        system=QUERY_SYSTEM.format(context=context),
        user=question,
        model=model,
    )
