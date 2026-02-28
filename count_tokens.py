#!/usr/bin/env python3
"""Quick token/word/char count for doom codebase and EU AI Act."""

import os
import tiktoken

enc = tiktoken.get_encoding("cl100k_base")

TARGETS = {
    "Doom (Python)": {
        "path": "doom",
        "exts": {".py"},
    },
    "Doom (C source)": {
        "path": "doom-src/linuxdoom-1.10",
        "exts": {".c", ".h"},
    },
    "EU AI Act": {
        "path": "eu_ai_act.md",
        "exts": None,  # single file
    },
}

def count(text: str):
    tokens = len(enc.encode(text))
    words = len(text.split())
    chars = len(text)
    return tokens, words, chars

def gather_text(path: str, exts: set[str] | None) -> str:
    if os.path.isfile(path):
        with open(path, "r", errors="replace") as f:
            return f.read()

    parts = []
    for root, _, files in os.walk(path):
        for fn in sorted(files):
            if exts and os.path.splitext(fn)[1] not in exts:
                continue
            fp = os.path.join(root, fn)
            with open(fp, "r", errors="replace") as f:
                parts.append(f.read())
    return "\n".join(parts)

grand_t = grand_w = grand_c = 0

for label, cfg in TARGETS.items():
    text = gather_text(cfg["path"], cfg["exts"])
    t, w, c = count(text)
    grand_t += t
    grand_w += w
    grand_c += c
    print(f"\n{'='*50}")
    print(f"  {label}")
    print(f"{'='*50}")
    print(f"  Tokens:     {t:>10,}")
    print(f"  Words:      {w:>10,}")
    print(f"  Characters: {c:>10,}")

print(f"\n{'='*50}")
print(f"  GRAND TOTAL")
print(f"{'='*50}")
print(f"  Tokens:     {grand_t:>10,}")
print(f"  Words:      {grand_w:>10,}")
print(f"  Characters: {grand_c:>10,}")
