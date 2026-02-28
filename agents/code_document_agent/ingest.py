"""Code document ingestion: parsed AST → SiftText tree nodes + static links."""

# TODO: Implement code ingestion pipeline
# - Create file-level parent nodes (crystallization = top-level declarations)
# - Create function/struct child nodes (crystallization = raw code, scope = LLM summary)
# - Create links from static reference pairs (no LLM needed)
# - All operations use pipeline_mode=True for speed


async def ingest_code(tree_id: str, root_id: str, parsed: dict, sift, model: str) -> dict:
    """Ingest parsed C code into a SiftText tree.

    Args:
        tree_id: Target tree ID
        root_id: Root node ID
        parsed: Output from parser.parse_c_directory()
        sift: SiftTextClient instance
        model: LLM model for scope generation

    Returns dict with stats: {files, functions, types, links}
    """
    raise NotImplementedError("Code ingestion not yet implemented")
