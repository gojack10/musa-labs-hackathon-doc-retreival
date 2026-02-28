"""C source code parser using tree-sitter.

Extracts functions, structs, enums, typedefs with exact byte ranges.
Also extracts cross-reference pairs (call_expression, type_identifier).
"""

from pathlib import Path

import tree_sitter_c as tsc
from tree_sitter import Language, Parser, Query, QueryCursor

C_LANGUAGE = Language(tsc.language())
_parser = Parser(C_LANGUAGE)

# --- Tree-sitter queries ---

# Functions: direct declarator
FUNC_QUERY = Query(
    C_LANGUAGE,
    "(function_definition"
    "  declarator: (function_declarator"
    "    declarator: (identifier) @name)) @func",
)

# Functions: pointer-returning (e.g. void *R_GetColumn(...))
FUNC_PTR_QUERY = Query(
    C_LANGUAGE,
    "(function_definition"
    "  declarator: (pointer_declarator"
    "    declarator: (function_declarator"
    "      declarator: (identifier) @name))) @func",
)

# Types
TYPEDEF_QUERY = Query(
    C_LANGUAGE,
    "(type_definition declarator: (type_identifier) @name) @typedef",
)
STRUCT_QUERY = Query(
    C_LANGUAGE,
    "(struct_specifier"
    "  name: (type_identifier) @name"
    "  body: (field_declaration_list)) @struct",
)
ENUM_QUERY = Query(
    C_LANGUAGE,
    "(enum_specifier"
    "  name: (type_identifier) @name"
    "  body: (enumerator_list)) @enum",
)

# Cross-references
CALL_QUERY = Query(
    C_LANGUAGE,
    "(call_expression function: (identifier) @callee)",
)
TYPE_REF_QUERY = Query(C_LANGUAGE, "(type_identifier) @type_ref")

# stdlib and libc names to skip in cross-references
STDLIB_NAMES = {
    "printf", "fprintf", "sprintf", "snprintf", "vprintf", "vfprintf",
    "memcpy", "memset", "memmove", "memcmp",
    "malloc", "calloc", "realloc", "free",
    "strlen", "strcmp", "strncmp", "strcpy", "strncpy", "strcat", "strncat", "strdup",
    "atoi", "atof", "atol", "strtol", "strtoul",
    "abs", "exit", "abort", "atexit",
    "fopen", "fclose", "fread", "fwrite", "fseek", "ftell", "fgets", "fputs", "feof",
    "sscanf", "fscanf", "scanf",
    "qsort", "bsearch",
    "isdigit", "isalpha", "isspace", "toupper", "tolower",
    "rand", "srand",
    "time", "clock",
    "assert",
    # POSIX / X11 commonly seen in Doom
    "open", "close", "read", "write", "lseek", "ioctl",
    "getenv", "system", "usleep", "signal",
    "XOpenDisplay", "XCreateWindow", "XMapWindow", "XNextEvent",
    "XCreateImage", "XPutImage", "XFlush", "XSync",
    "shmget", "shmat", "shmdt", "shmctl",
    "perror", "errno",
}

# C primitive / builtin type names to skip in type references
BUILTIN_TYPES = {
    "int", "char", "void", "short", "long", "float", "double",
    "unsigned", "signed", "size_t", "FILE",
}


def _query_matches(query: Query, node) -> list[tuple[int, dict]]:
    """Run a query against a node using QueryCursor."""
    cursor = QueryCursor(query)
    return cursor.matches(node)


def _collect_c_files(
    root: Path,
    exclude_files: set[str],
    exclude_dirs: set[str],
) -> list[Path]:
    """Recursively collect .c and .h files, respecting exclusions."""
    files: list[Path] = []
    for item in sorted(root.iterdir()):
        if item.is_dir():
            if item.name not in exclude_dirs:
                files.extend(_collect_c_files(item, exclude_files, exclude_dirs))
        elif item.suffix in (".c", ".h") and item.name not in exclude_files:
            files.append(item)
    return files


def _extract_functions(tree, source: bytes) -> list[dict]:
    """Extract all function definitions from a parsed tree."""
    results: list[dict] = []
    seen_ranges: set[tuple[int, int]] = set()

    for query in (FUNC_QUERY, FUNC_PTR_QUERY):
        for _pattern_idx, captures in _query_matches(query, tree.root_node):
            func_nodes = captures.get("func", [])
            name_nodes = captures.get("name", [])
            if not func_nodes or not name_nodes:
                continue
            func_node = func_nodes[0]
            name_node = name_nodes[0]

            byte_range = (func_node.start_byte, func_node.end_byte)
            if byte_range in seen_ranges:
                continue
            seen_ranges.add(byte_range)

            results.append({
                "name": name_node.text.decode("utf-8"),
                "code": source[func_node.start_byte:func_node.end_byte].decode("utf-8", errors="replace"),
                "line": func_node.start_point[0] + 1,
                "byte_range": byte_range,
                "_node": func_node,  # kept for cross-ref extraction, stripped before return
            })

    # Sort by position in file
    results.sort(key=lambda f: f["byte_range"][0])

    # Disambiguate duplicate names within same file
    name_counts: dict[str, int] = {}
    for func in results:
        name_counts[func["name"]] = name_counts.get(func["name"], 0) + 1

    for func in results:
        n = func["name"]
        if name_counts[n] > 1:
            func["name"] = f"{n}_L{func['line']}"

    return results


def _extract_types(tree, source: bytes) -> list[dict]:
    """Extract struct, enum, and typedef definitions from a parsed tree."""
    results: list[dict] = []
    seen_ranges: set[tuple[int, int]] = set()

    query_kind_pairs = [
        (TYPEDEF_QUERY, "typedef", "typedef"),
        (STRUCT_QUERY, "struct", "struct"),
        (ENUM_QUERY, "enum", "enum"),
    ]

    for query, capture_name, kind in query_kind_pairs:
        for _pattern_idx, captures in _query_matches(query, tree.root_node):
            type_nodes = captures.get(capture_name, [])
            name_nodes = captures.get("name", [])
            if not type_nodes or not name_nodes:
                continue
            type_node = type_nodes[0]
            name_node = name_nodes[0]

            byte_range = (type_node.start_byte, type_node.end_byte)
            if byte_range in seen_ranges:
                continue
            seen_ranges.add(byte_range)

            results.append({
                "name": name_node.text.decode("utf-8"),
                "kind": kind,
                "code": source[type_node.start_byte:type_node.end_byte].decode("utf-8", errors="replace"),
                "line": type_node.start_point[0] + 1,
                "byte_range": byte_range,
            })

    results.sort(key=lambda t: t["byte_range"][0])
    return results


def _extract_top_level_code(
    source: bytes,
    functions: list[dict],
    types: list[dict],
) -> str:
    """Extract code outside function bodies and type definitions."""
    # Collect all occupied byte ranges
    ranges: list[tuple[int, int]] = []
    for f in functions:
        ranges.append(f["byte_range"])
    for t in types:
        ranges.append(t["byte_range"])

    if not ranges:
        return source.decode("utf-8", errors="replace")

    # Sort and merge overlapping ranges
    ranges.sort()
    merged: list[tuple[int, int]] = [ranges[0]]
    for start, end in ranges[1:]:
        if start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))

    # Collect gaps
    parts: list[str] = []
    pos = 0
    for start, end in merged:
        if pos < start:
            gap = source[pos:start].decode("utf-8", errors="replace").strip()
            if gap:
                parts.append(gap)
        pos = end
    # Trailing content after last range
    if pos < len(source):
        gap = source[pos:].decode("utf-8", errors="replace").strip()
        if gap:
            parts.append(gap)

    return "\n\n".join(parts)


def _extract_references(
    func_node,
    func_name: str,
    source_file: str,
    known_symbols: set[str],
) -> list[dict]:
    """Extract cross-references (calls + type usage) from a function's AST subtree."""
    refs: list[dict] = []
    seen: set[tuple[str, str, str]] = set()

    # Function calls
    for _pat, captures in _query_matches(CALL_QUERY, func_node):
        for callee_node in captures.get("callee", []):
            callee = callee_node.text.decode("utf-8")
            if callee in STDLIB_NAMES or callee not in known_symbols:
                continue
            if callee == func_name:
                continue  # skip self-recursion
            key = (func_name, callee, "calls")
            if key not in seen:
                seen.add(key)
                refs.append({
                    "source_file": source_file,
                    "source_name": func_name,
                    "target_name": callee,
                    "kind": "calls",
                })

    # Type references
    for _pat, captures in _query_matches(TYPE_REF_QUERY, func_node):
        for type_node in captures.get("type_ref", []):
            type_name = type_node.text.decode("utf-8")
            if type_name in BUILTIN_TYPES or type_name not in known_symbols:
                continue
            key = (func_name, type_name, "uses_type")
            if key not in seen:
                seen.add(key)
                refs.append({
                    "source_file": source_file,
                    "source_name": func_name,
                    "target_name": type_name,
                    "kind": "uses_type",
                })

    return refs


def _parse_file(filepath: Path, base_dir: Path) -> dict:
    """Parse a single C file and return structured data."""
    source = filepath.read_bytes()
    tree = _parser.parse(source)

    rel_path = str(filepath.relative_to(base_dir))
    name = filepath.name
    is_header = filepath.suffix == ".h"

    functions = _extract_functions(tree, source)
    types = _extract_types(tree, source)
    top_level = _extract_top_level_code(source, functions, types)

    return {
        "path": rel_path,
        "name": name,
        "is_header": is_header,
        "top_level_code": top_level,
        "functions": functions,
        "types": types,
    }


async def parse_c_directory(
    path: str,
    exclude_files: list[str] | None = None,
    exclude_dirs: list[str] | None = None,
) -> dict:
    """Parse all C source files in a directory.

    Returns dict with keys:
        files: list of {path, name, is_header, top_level_code, functions, types}
        references: list of {source_file, source_name, target_name, kind}
        stats: {total_files, c_files, h_files, functions, types, references, skipped_stdlib}
    """
    root = Path(path)
    excl_files = set(exclude_files or [])
    excl_dirs = set(exclude_dirs or [])

    c_files = _collect_c_files(root, excl_files, excl_dirs)

    # Phase 1: parse all files
    parsed_files: list[dict] = []
    for filepath in c_files:
        parsed_files.append(_parse_file(filepath, root))

    # Build set of all known symbols (function names + type names)
    known_symbols: set[str] = set()
    for f in parsed_files:
        for func in f["functions"]:
            known_symbols.add(func["name"])
        for t in f["types"]:
            known_symbols.add(t["name"])

    # Phase 2: extract cross-references
    all_references: list[dict] = []
    skipped_stdlib = 0

    for f in parsed_files:
        for func in f["functions"]:
            func_node = func["_node"]

            # Count stdlib skips for stats
            for _pat, captures in _query_matches(CALL_QUERY, func_node):
                for callee_node in captures.get("callee", []):
                    callee = callee_node.text.decode("utf-8")
                    if callee in STDLIB_NAMES:
                        skipped_stdlib += 1

            refs = _extract_references(
                func_node, func["name"], f["name"], known_symbols
            )
            all_references.extend(refs)

    # Strip internal _node from function dicts before returning
    for f in parsed_files:
        for func in f["functions"]:
            del func["_node"]

    # Stats
    c_count = sum(1 for f in parsed_files if not f["is_header"])
    h_count = sum(1 for f in parsed_files if f["is_header"])
    func_count = sum(len(f["functions"]) for f in parsed_files)
    type_count = sum(len(f["types"]) for f in parsed_files)

    return {
        "files": parsed_files,
        "references": all_references,
        "stats": {
            "total_files": len(parsed_files),
            "c_files": c_count,
            "h_files": h_count,
            "functions": func_count,
            "types": type_count,
            "references": len(all_references),
            "skipped_stdlib": skipped_stdlib,
        },
    }
