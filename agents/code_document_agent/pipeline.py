"""Code document analysis pipeline: parse → file nodes → function nodes → static linkage."""

from lib.sifttext import SiftTextClient


async def run(
    input_path: str,
    sift: SiftTextClient,
    model: str,
    smart_model: str,
) -> str:
    """Run the code document pipeline. Returns tree_id.

    Stages:
        0. tree-sitter parse → extract functions + structs + cross-reference pairs
        1. Create tree
        2. Create file-level parent nodes with LLM scope
        3. Create function/struct child nodes (raw code + LLM scope + micro_summary)
        4. Static linkage: create links from reference pairs (no LLM)
    """
    raise NotImplementedError("Code document pipeline not yet implemented")
