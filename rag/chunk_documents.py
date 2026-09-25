"""
chunk_documents.py
Structure-aware + parent-child chunking.

Step 1 (structure-aware): scans each page's text line by line and detects
heading-like lines (numbered sections, "Chapter X", ALL CAPS headings) to
break the document into topically coherent blocks instead of blindly
cutting every N characters. Blocks are then merged (if too small) or split
(if too large) to land inside a target parent-chunk size range. These
become the "parents" -- saved to parents.json.

Step 2 (parent-child): each parent is further split into smaller,
overlapping "child" chunks. Children are what actually get embedded and
searched (precise matching), while at query time we hand back the parent
text as the surrounding context (better generation quality) -- saved to
chunks.json, keyed to their parent_id.

Usage:
    python chunk_documents.py
"""

import json
import re
from collections import Counter
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
INPUT_FILE = DATA_DIR / "documents.json"
PARENTS_FILE = DATA_DIR / "parents.json"
CHUNKS_FILE = DATA_DIR / "chunks.json"

# a line repeating at least this many times within one source is treated
# as a running header/footer (e.g. a chapter title printed on every page)
# rather than a real heading, and stripped before heading detection
RUNNING_HEADER_MIN_COUNT = 4

# --- parent (structure-aware) sizing ---
MIN_PARENT_CHARS = 400
MAX_PARENT_CHARS = 2200

# --- child (embedding-granularity) sizing ---
CHILD_CHUNK_SIZE = 400
CHILD_CHUNK_OVERLAP = 80

HEADING_PATTERNS = [
    # "Chapter 3", "Part II", "Section 4:" -- keyword MUST be followed by
    # a number/roman numeral, not just any word, or prose like "part of
    # what happened next" gets misdetected as a heading
    re.compile(r"^(chapter|section|part)\s+([0-9]+|[ivxlcdm]+)\b[:.]?", re.IGNORECASE),
    re.compile(r"^\d{1,2}(\.\d{1,2}){0,3}\.?\s+[A-Z]"),   # "1.2 Introduction"
    re.compile(r"^[IVXLC]{1,6}\.\s+[A-Z]"),                # "IV. Discussion"
    re.compile(r"^[A-Z][A-Z0-9 ,:'\-]{3,80}$"),            # ALL CAPS HEADING
]


def is_heading(line: str) -> bool:
    line = line.strip()
    if not line or len(line) > 100:
        return False
    if len(line.split()) > 14:
        return False
    return any(pat.match(line) for pat in HEADING_PATTERNS)

def strip_running_headers(documents: list[dict]) -> list[dict]:
    """Remove lines that repeat verbatim across a run of consecutive
    pages within the same source -- running headers/footers like a
    chapter title printed on every page -- without touching short
    headings (e.g. "SUMMARY") that legitimately recur once per chapter
    at scattered, non-consecutive pages."""
    by_source: dict[str, list[dict]] = {}
    for page in documents:
        by_source.setdefault(page["source"], []).append(page)

    running_by_source: dict[str, set[str]] = {}
    for source, pages in by_source.items():
        pages_sorted = sorted(pages, key=lambda p: p["page"])
        line_pages: dict[str, list[int]] = {}
        for page in pages_sorted:
            for line in set(l.strip() for l in page["text"].splitlines() if l.strip()):
                line_pages.setdefault(line, []).append(page["page"])

        running = set()
        for line, page_nums in line_pages.items():
            page_nums.sort()
            run = best = 1
            for i in range(1, len(page_nums)):
                run = run + 1 if page_nums[i] == page_nums[i - 1] + 1 else 1
                best = max(best, run)
            if best >= RUNNING_HEADER_MIN_COUNT:
                running.add(line)
        running_by_source[source] = running

    cleaned = []
    examples = []
    for page in documents:
        running = running_by_source.get(page["source"], set())
        kept = []
        for line in page["text"].splitlines():
            if line.strip() in running:
                if len(examples) < 8:
                    examples.append((page["source"], line.strip()))
                continue
            kept.append(line)
        cleaned.append({**page, "text": "\n".join(kept)})

    total_stripped = sum(len(v) for v in running_by_source.values())
    if total_stripped:
        print(f"Stripped {total_stripped} running header/footer line(s) "
              f"(appeared on {RUNNING_HEADER_MIN_COUNT}+ consecutive pages), e.g.:")
        for src, line in examples:
            print(f"   [{src}] {line!r}")

    return cleaned


def build_blocks(pages: list[dict]) -> list[dict]:
    """Turn a source's pages into structure-aware blocks, splitting on
    detected headings (and always at page boundaries)."""
    blocks = []
    current = None

    for page in pages:
        for line in page["text"].splitlines():
            line = line.strip()
            if not line:
                continue
            if is_heading(line) or current is None:
                if current is not None:
                    blocks.append(current)
                current = {
                    "source": page["source"],
                    "page_start": page["page"],
                    "page_end": page["page"],
                    "heading": line if is_heading(line) else None,
                    "lines": [line],
                }
            else:
                current["page_end"] = page["page"]
                current["lines"].append(line)

    if current is not None:
        blocks.append(current)

    for b in blocks:
        b["text"] = "\n".join(b["lines"])
        del b["lines"]
    return blocks


def merge_small_blocks(blocks: list[dict]) -> list[dict]:
    """Merge adjacent undersized blocks (same source) so parents aren't
    fragments of a page, while respecting MAX_PARENT_CHARS."""
    merged = []
    for b in blocks:
        if (
            merged
            and merged[-1]["source"] == b["source"]
            and len(merged[-1]["text"]) < MIN_PARENT_CHARS
            and len(merged[-1]["text"]) + len(b["text"]) <= MAX_PARENT_CHARS
        ):
            merged[-1]["text"] += "\n" + b["text"]
            merged[-1]["page_end"] = b["page_end"]
        else:
            merged.append(dict(b))
    return merged


def split_large_block(text: str, max_chars: int) -> list[str]:
    """Split an oversized block on paragraph (line) boundaries first,
    falling back to hard slicing if no good boundary exists."""
    if len(text) <= max_chars:
        return [text]
    lines = text.split("\n")
    pieces, current = [], ""
    for line in lines:
        candidate = (current + "\n" + line) if current else line
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                pieces.append(current)
            current = line
    if current:
        pieces.append(current)
    final = []
    for piece in pieces:
        if len(piece) > max_chars:
            for i in range(0, len(piece), max_chars):
                final.append(piece[i:i + max_chars])
        else:
            final.append(piece)
    return final


def build_parents(documents: list[dict]) -> list[dict]:
    by_source: dict[str, list[dict]] = {}
    for page in documents:
        by_source.setdefault(page["source"], []).append(page)

    parents = []
    parent_id = 0
    for source, pages in by_source.items():
        blocks = build_blocks(pages)
        blocks = merge_small_blocks(blocks)

        for b in blocks:
            for piece in split_large_block(b["text"], MAX_PARENT_CHARS):
                parents.append({
                    "parent_id": parent_id,
                    "source": b["source"],
                    "page_start": b["page_start"],
                    "page_end": b["page_end"],
                    "heading": b["heading"],
                    "text": piece.strip(),
                })
                parent_id += 1
    return parents


def split_child_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    if len(text) <= chunk_size:
        return [text]
    words = text.split(" ")
    chunks, current = [], ""
    for w in words:
        candidate = (current + " " + w) if current else w
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current)
            overlap_text = current[-overlap:] if current else ""
            current = (overlap_text + " " + w) if overlap_text else w
    if current:
        chunks.append(current)
    return chunks


def build_children(parents: list[dict]) -> list[dict]:
    children = []
    chunk_id = 0
    for parent in parents:
        pieces = split_child_text(parent["text"], CHILD_CHUNK_SIZE, CHILD_CHUNK_OVERLAP)
        for piece in pieces:
            piece = piece.strip()
            if not piece:
                continue
            children.append({
                "chunk_id": chunk_id,
                "parent_id": parent["parent_id"],
                "source": parent["source"],
                "page": parent["page_start"],
                "text": piece,
            })
            chunk_id += 1
    return children


def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        documents = json.load(f)
    documents = strip_running_headers(documents) 

    print(f"Building structure-aware parent chunks from {len(documents)} pages ...")
    parents = build_parents(documents)
    print(f"Produced {len(parents)} parent chunks "
          f"(target {MIN_PARENT_CHARS}-{MAX_PARENT_CHARS} chars)")

    children = build_children(parents)
    print(f"Produced {len(children)} child chunks "
          f"(target {CHILD_CHUNK_SIZE} chars, overlap {CHILD_CHUNK_OVERLAP})")

    with open(PARENTS_FILE, "w", encoding="utf-8") as f:
        json.dump(parents, f, ensure_ascii=False, indent=2)
    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        json.dump(children, f, ensure_ascii=False, indent=2)

    print(f"\nSaved parents to {PARENTS_FILE}")
    print(f"Saved children (embedding units) to {CHUNKS_FILE}")


if __name__ == "__main__":
    main()