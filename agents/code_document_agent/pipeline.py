"""Code document analysis pipeline: parse → file nodes → function nodes → static linkage."""

import time
from pathlib import Path

from lib.perf import perf
from lib.sifttext import SiftTextClient
from agents.code_document_agent.parser import parse_c_directory
from agents.code_document_agent.ingest import ingest_code


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
) -> str:
    """Run the code document pipeline. Returns tree_id."""

    # Stage 0: Parse
    _banner(0, "Parsing C Source")
    perf.stage("0_parse")
    parsed = await parse_c_directory(
        input_path,
        exclude_files=["i_net.c", "d_net.c", "d_englsh.h", "d_french.h"],
        exclude_dirs=["sndserv", "ipx", "sersrc"],
    )
    s = parsed["stats"]
    print(f"{_ts()}   {s['c_files']} .c + {s['h_files']} .h files, "
          f"{s['functions']} functions, {s['types']} types, "
          f"{s['references']} cross-references")

    # Stage 1: Create tree
    _banner(1, "Creating Knowledge Tree")
    perf.stage("1_tree_creation")
    dir_name = Path(input_path).name
    tree_name = f"{dir_name} Code Analysis"
    tree_scope = (
        f"Code decomposition of {dir_name} — filing cabinet hierarchy "
        f"(files → functions/types), static call graph and type-usage links"
    )
    tree = await sift.create_tree(tree_name, tree_scope)
    tree_id = tree["tree_id"]
    root_id = tree["root_id"]
    print(f"{_ts()}   Tree: {tree_id}")

    # Set root warning about known limitation
    await sift.set_vitals(
        node_id=root_id,
        warnings=(
            "Global variable + function pointer blindness: link graph captures "
            "direct call_expression and type_identifier references only. This "
            "codebase communicates heavily through shared globals (250+ externs: "
            "dc_x, dc_yl, dc_source, etc.) and function pointers (colfunc, "
            "spanfunc, thinker->function, P_SetMobjState). When tracing data "
            "flow, SEARCH for variable names to find writers/readers — don't "
            "rely solely on links."
        ),
        pipeline_mode=True,
    )

    # Stages 2-4: Ingest (file nodes → child nodes → static linkage)
    _banner(2, "Ingesting Code")
    perf.stage("2_ingest")
    stats = await ingest_code(
        tree_id=tree_id,
        root_id=root_id,
        parsed=parsed,
        sift=sift,
        model=model,
        concurrency=40,
    )
    print(f"{_ts()}   {stats['files']} file nodes, "
          f"{stats['functions']} function nodes, "
          f"{stats['types']} type nodes, "
          f"{stats['links']} links")
    if stats.get("errors"):
        print(f"{_ts()}   [yellow]{len(stats['errors'])} errors[/]")

    # Finish
    perf.finish()
    perf.summary()
    perf.save()

    total = time.time() - perf._t0
    m, s_sec = divmod(int(total), 60)
    total_nodes = stats["files"] + stats["functions"] + stats["types"]
    print(f"\n{'=' * 60}")
    print(f"  Pipeline complete in {m}m {s_sec}s")
    print(f"  {total_nodes} nodes | {stats['links']} links | tree {tree_id[:8]}...")
    print(f"{'=' * 60}")

    return tree_id
