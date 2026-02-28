"""Code document ingestion: parsed AST → SiftText tree nodes + static links."""

import asyncio
import time

from lib.llm import complete
from lib.perf import perf
from lib.config import UUID_RE, load_prompt, console
from lib.sifttext import SiftTextClient

_triage_prompt: str | None = None


def _get_triage_prompt() -> str:
    global _triage_prompt
    if _triage_prompt is None:
        _triage_prompt = load_prompt("code_triage")
    return _triage_prompt


def _ts() -> str:
    elapsed = time.time() - perf._t0
    m, s = divmod(int(elapsed), 60)
    return f"[{m:02d}:{s:02d}]"


def _parse_llm_output(text: str) -> tuple[str, str]:
    """Parse LLM output into (micro_summary, scope).

    Expected format: line with MICRO: prefix, then scope lines.
    Defensive: handles blank lines, missing MICRO prefix, etc.
    """
    lines = text.strip().splitlines()
    micro = ""
    scope_lines: list[str] = []

    for i, line in enumerate(lines):
        if line.strip().startswith("MICRO:"):
            micro = line.strip().removeprefix("MICRO:").strip()
            scope_lines = [l for l in lines[i + 1 :] if l.strip()]
            break

    if not micro:
        # Fallback: first sentence as micro, rest as scope
        micro = lines[0].strip()[:120] if lines else "No summary"
        scope_lines = lines[1:] if len(lines) > 1 else lines

    scope = "\n".join(scope_lines).strip()
    return micro, scope


async def ingest_code(
    tree_id: str,
    root_id: str,
    parsed: dict,
    sift: SiftTextClient,
    model: str,
    concurrency: int = 40,
) -> dict:
    """Ingest parsed C code into a SiftText tree.

    Three stages:
      A. Create file-level parent nodes (LLM scope + raw top-level crystallization)
      B. Create function/struct child nodes (LLM scope + raw code crystallization)
      C. Static linkage from parser cross-reference pairs (no LLM)

    Returns dict with stats: {files, functions, types, links, errors}
    """
    sem = asyncio.Semaphore(concurrency)
    errors: list[str] = []
    file_map: dict[str, str] = {}  # filename → node_id
    name_to_id: dict[str, str] = {}  # all names → node_id

    # ── Stage A: File-level parent nodes ──────────────────────────
    perf.stage("2a_file_nodes")
    total_files = len(parsed["files"])
    print(f"{_ts()}   Stage A: Creating {total_files} file nodes (concurrency={concurrency})...")
    counter_a = {"done": 0}

    async def _create_file_node(file_data: dict) -> tuple[str, str | None]:
        user_msg = f"File: {file_data['name']}\n\nTop-level declarations:\n{file_data['top_level_code']}"
        try:
            async with sem:
                llm_output = await complete(_get_triage_prompt(), user_msg, model)

            micro, scope = _parse_llm_output(llm_output)

            async with sem:
                result = await sift.create_node(
                    name=file_data["name"],
                    scope=scope or f"Source file: {file_data['name']}",
                    crystallization=file_data["top_level_code"],
                    parent_id=root_id,
                    tree_id=tree_id,
                    pipeline_mode=True,
                )

            match = UUID_RE.search(result)
            if not match:
                raise RuntimeError(f"No UUID in create_node response for {file_data['name']}")
            node_id = match.group(0)

            async with sem:
                await sift.set_vitals(node_id=node_id, micro_summary=micro, pipeline_mode=True)

            counter_a["done"] += 1
            print(f"{_ts()}     [{counter_a['done']}/{total_files}] {file_data['name']}")
            return file_data["name"], node_id

        except Exception as e:
            counter_a["done"] += 1
            err = f"File {file_data['name']}: {type(e).__name__}: {e}"
            errors.append(err)
            print(f"{_ts()}     [{counter_a['done']}/{total_files}] {file_data['name']} FAILED")
            return file_data["name"], None

    file_results = await asyncio.gather(
        *[_create_file_node(f) for f in parsed["files"]],
        return_exceptions=True,
    )

    for r in file_results:
        if isinstance(r, Exception):
            errors.append(f"File task exception: {type(r).__name__}: {r}")
            continue
        name, node_id = r
        if node_id:
            file_map[name] = node_id
            name_to_id[name] = node_id

    print(f"{_ts()}   Stage A complete: {len(file_map)}/{total_files} file nodes created")

    # ── Stage B: Function/struct child nodes ──────────────────────
    perf.stage("2b_child_nodes")

    child_tasks: list[tuple[dict, str, str]] = []  # (item, parent_id, kind)
    for file_data in parsed["files"]:
        parent_id = file_map.get(file_data["name"])
        if parent_id is None:
            continue  # parent file failed, skip its children
        for func in file_data["functions"]:
            child_tasks.append((func, parent_id, "Function"))
        for typ in file_data["types"]:
            child_tasks.append((typ, parent_id, typ["kind"].capitalize()))

    total_children = len(child_tasks)
    print(f"\n{_ts()}   Stage B: Creating {total_children} child nodes (concurrency={concurrency})...")
    counter_b = {"done": 0}
    func_count = 0
    type_count = 0

    async def _create_child_node(
        item: dict, parent_id: str, kind: str
    ) -> tuple[str, str | None]:
        nonlocal func_count, type_count
        user_msg = f"{kind}: {item['name']}\n\n{item['code']}"
        try:
            async with sem:
                llm_output = await complete(_get_triage_prompt(), user_msg, model)

            micro, scope = _parse_llm_output(llm_output)

            async with sem:
                result = await sift.create_node(
                    name=item["name"],
                    scope=scope or f"{kind}: {item['name']}",
                    crystallization=item["code"],
                    parent_id=parent_id,
                    tree_id=tree_id,
                    pipeline_mode=True,
                )

            match = UUID_RE.search(result)
            if not match:
                raise RuntimeError(f"No UUID in create_node response for {item['name']}")
            node_id = match.group(0)

            async with sem:
                await sift.set_vitals(node_id=node_id, micro_summary=micro, pipeline_mode=True)

            counter_b["done"] += 1
            if counter_b["done"] % 50 == 0 or counter_b["done"] == total_children:
                print(f"{_ts()}     [{counter_b['done']}/{total_children}] {item['name']}")

            if kind == "Function":
                func_count += 1
            else:
                type_count += 1

            return item["name"], node_id

        except Exception as e:
            counter_b["done"] += 1
            err = f"{kind} {item['name']}: {type(e).__name__}: {e}"
            errors.append(err)
            return item["name"], None

    child_results = await asyncio.gather(
        *[_create_child_node(item, pid, kind) for item, pid, kind in child_tasks],
        return_exceptions=True,
    )

    for r in child_results:
        if isinstance(r, Exception):
            errors.append(f"Child task exception: {type(r).__name__}: {r}")
            continue
        name, node_id = r
        if node_id:
            name_to_id[name] = node_id

    print(f"{_ts()}   Stage B complete: {func_count} functions + {type_count} types created")

    # ── Stage C: Static linkage ───────────────────────────────────
    perf.stage("2c_linkage")
    refs = parsed["references"]
    total_refs = len(refs)
    print(f"\n{_ts()}   Stage C: Creating {total_refs} static links (concurrency={concurrency})...")
    counter_c = {"done": 0}
    links_created = 0

    async def _create_link(ref: dict) -> bool:
        nonlocal links_created
        source_id = name_to_id.get(ref["source_name"])
        if source_id is None:
            counter_c["done"] += 1
            return False

        target_id = name_to_id.get(ref["target_name"])
        description = ref["kind"].replace("_", " ")

        try:
            kwargs: dict = {
                "source_node_id": source_id,
                "target_name": ref["target_name"],
                "description": description,
                "pipeline_mode": True,
            }
            if target_id:
                kwargs["target_node_id"] = target_id

            async with sem:
                await sift.link_by_name(**kwargs)

            links_created += 1
            counter_c["done"] += 1
            if counter_c["done"] % 200 == 0 or counter_c["done"] == total_refs:
                print(f"{_ts()}     [{counter_c['done']}/{total_refs}] links processed")
            return True

        except Exception as e:
            counter_c["done"] += 1
            return False

    link_results = await asyncio.gather(
        *[_create_link(ref) for ref in refs],
        return_exceptions=True,
    )

    print(f"{_ts()}   Stage C complete: {links_created}/{total_refs} links created")

    if errors:
        print(f"{_ts()}   Errors: {len(errors)}")
        for e in errors[:5]:
            print(f"{_ts()}     ! {e}")
        if len(errors) > 5:
            print(f"{_ts()}     ... and {len(errors) - 5} more")

    return {
        "files": len(file_map),
        "functions": func_count,
        "types": type_count,
        "links": links_created,
        "errors": errors,
    }
