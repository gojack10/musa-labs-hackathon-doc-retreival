"""C source code parser using tree-sitter.

Extracts functions, structs, enums, typedefs with exact byte ranges.
Also extracts cross-reference pairs (call_expression, type_identifier).
"""

# TODO: Implement tree-sitter based C parsing
# - Parse .c and .h files
# - Extract function_definition nodes with byte ranges
# - Extract struct/enum/typedef definitions
# - Extract cross-reference pairs (caller→callee, function→type)
# - Handle #ifdef variants (disambiguate with line number suffix)
# - Skip stdlib calls (memcpy, printf, etc.)
# - Return structured data for ingest.py to consume


async def parse_c_directory(path: str) -> dict:
    """Parse all C source files in a directory.

    Returns dict with keys:
        files: list of {path, name, top_level_code, functions, types}
        references: list of {source_name, target_name, kind}  # "calls" or "uses_type"
    """
    raise NotImplementedError("tree-sitter C parser not yet implemented")
