"""
ELSTER Help Documentation Chunker for RAG
==========================================

Pipeline:
  clean boilerplate -> MarkdownHeaderTextSplitter -> size-enforced child split
  -> inject hierarchical context prefix -> dedup -> link siblings

Setup:
  1. pip install -r requirements.txt
  2. Put your .md files in a folder called "docs/" next to this script
  3. python chunk.py

  Or specify a custom input folder:
     python chunk.py /path/to/your/md/files

Output:
  ./chunksnew.json          — chunks with metadata, ready for vector DB ingest

Design notes:
  - strip_headers=True + a single injected context prefix keeps chunks
    self-contained without duplicating heading text in-body.
  - Size is hard-enforced via a "" fallback separator (character-level split).
  - Exact-body duplicates (the ELSTER docs repeat some Q&As across files) are
    collapsed into one chunk with metadata.also_at listing every origin path.
  - Multi-part chunks expose prev_id / next_id so retrievers can expand context.
  - chunk_id = md5(source :: context_path :: part_index :: body)[:16] — stable
    across breadcrumb-format tweaks, unique across sources.
"""

import hashlib
import json
import re
import sys
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

# CONFIGURATION

HEADERS_TO_SPLIT = [
    ("##", "h2"),
    ("###", "h3"),
    ("####", "h4"),
    ("#####", "h5"),
]

MAX_CHUNK_SIZE = 1200  # hard upper bound per chunk
CHUNK_OVERLAP = 250  # overlap when splitting oversized chunks
MIN_CHUNK_SIZE = 50  # discard fragments smaller than this
HEADING_MAX_WORDS = 12  # truncate long headings (full questions) for breadcrumb
OUTPUT_JSON = Path("./data/chunks.json")


# CLEANING — strip ELSTER boilerplate before splitting

BOILERPLATE_PATTERNS = [
    re.compile(r"^#\s+Hilfe\s*$"),
    re.compile(r"^Suchen\s*$"),
    re.compile(r"^Sucheoder Chat\s*$"),
    re.compile(r"^\[Zurück zur Übersicht\]"),
    re.compile(r"^Seite lädt"),
    re.compile(r"^Daten werden geladen"),
    re.compile(r"^Grafik drehender Stern"),
    re.compile(
        r"^\s*\*\s*\[.*\]\(https://www\.elster\.de/eportal/helpGlobal\?themaGlobal="
    ),
]


def is_boilerplate(line: str) -> bool:
    stripped = line.strip()
    return bool(stripped) and any(p.search(stripped) for p in BOILERPLATE_PATTERNS)


def clean_markdown(text: str) -> str:
    lines = [l for l in text.splitlines() if not is_boilerplate(l)]
    cleaned = re.sub(r"\n{3,}", "\n\n", "\n".join(lines))
    return cleaned.strip()


# SPLITTERS

md_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=HEADERS_TO_SPLIT,
    strip_headers=True,  # headings go into metadata; we inject one clean prefix
)

# "" as the final separator forces character-level splitting if nothing else
# fits, which guarantees MAX_CHUNK_SIZE is actually respected.
child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=MAX_CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""],
    length_function=len,
)


def truncate_heading(text: str, max_words: int = HEADING_MAX_WORDS) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + "..."


def build_context_path(metadata: dict) -> str:
    """Hierarchical breadcrumb: h2 > h3 > h4 > (truncated) h5."""
    parts = []
    for key in ("h2", "h3", "h4"):
        if metadata.get(key):
            parts.append(metadata[key].strip())
    if metadata.get("h5"):
        parts.append(truncate_heading(metadata["h5"].strip()))
    return " > ".join(parts)


def make_chunk_id(source: str, context_path: str, part_index: int, body: str) -> str:
    key = f"{source}::{context_path}::{part_index}::{body}".encode("utf-8")
    return hashlib.md5(key).hexdigest()[:16]


def build_prefix(context_path: str, part_index: int, total_parts: int) -> str:
    """
    Clean breadcrumb line prepended to the body to give the embedder topic
    context. Source filename is intentionally NOT here — it lives in metadata
    and would only add filename-token noise to the vector.
    """
    if total_parts > 1:
        return f"{context_path} (Teil {part_index + 1}/{total_parts})"
    return context_path


# PIPELINE


def split_file(filepath: Path) -> list[Document]:
    """Clean -> header split -> size split. Returns raw Documents (no prefix)."""
    raw = filepath.read_text(encoding="utf-8")
    cleaned = clean_markdown(raw)
    header_chunks = md_splitter.split_text(cleaned)
    sized_chunks = child_splitter.split_documents(header_chunks)
    return [d for d in sized_chunks if len(d.page_content.strip()) >= MIN_CHUNK_SIZE]


def enrich_and_link(docs: list[Document], source: str) -> list[Document]:
    """
    Assign part_index/total_parts among siblings sharing heading metadata,
    inject the context prefix, and wire prev_id / next_id pointers.
    Two passes: compute ids first, then wire neighbors.
    """
    # Pass 1: group contiguous docs by identical heading metadata
    # (LangChain preserves metadata across a single oversized split.)
    groups: list[list[int]] = []
    for i, doc in enumerate(docs):
        if groups and docs[groups[-1][0]].metadata == doc.metadata:
            groups[-1].append(i)
        else:
            groups.append([i])

    # Pass 2: build enriched docs with stable ids
    enriched: list[Document] = []
    for group in groups:
        total = len(group)
        for part_index, idx in enumerate(group):
            raw = docs[idx]
            context_path = build_context_path(raw.metadata)
            prefix = build_prefix(context_path, part_index, total)
            body = raw.page_content.strip()
            content = f"{prefix}\n\n{body}"
            chunk_id = make_chunk_id(source, context_path, part_index, body)
            enriched.append(
                Document(
                    page_content=content,
                    metadata={
                        "source": source,
                        "section": raw.metadata.get("h2", ""),
                        "subsection": raw.metadata.get("h3", ""),
                        "topic": raw.metadata.get("h4", ""),
                        "question": raw.metadata.get("h5", ""),
                        "context_path": context_path,
                        "chunk_id": chunk_id,
                        "part_index": part_index,
                        "total_parts": total,
                    },
                )
            )

    # Pass 3: wire prev_id / next_id using the computed ids
    for i, doc in enumerate(enriched):
        if doc.metadata["total_parts"] > 1:
            if doc.metadata["part_index"] > 0:
                doc.metadata["prev_id"] = enriched[i - 1].metadata["chunk_id"]
            if doc.metadata["part_index"] < doc.metadata["total_parts"] - 1:
                doc.metadata["next_id"] = enriched[i + 1].metadata["chunk_id"]

    return enriched


def _extract_body(content: str) -> str:
    """Strip the first line (breadcrumb prefix) and return the raw body."""
    parts = content.split("\n\n", 1)
    return parts[1].strip() if len(parts) == 2 else content.strip()


def dedupe_by_body(docs: list[Document]) -> tuple[list[Document], int]:
    """
    Collapse chunks with identical body text (ELSTER repeats some Q&As across
    registration paths). Surviving chunk records every origin in metadata.also_at.
    """
    seen: dict[str, Document] = {}
    removed = 0
    for doc in docs:
        body = _extract_body(doc.page_content)
        body_hash = hashlib.md5(body.encode("utf-8")).hexdigest()
        if body_hash in seen:
            kept = seen[body_hash]
            origins = kept.metadata.setdefault("also_at", [])
            entry = f"{doc.metadata['source']} | {doc.metadata['context_path']}"
            if (
                entry not in origins
                and entry
                != f"{kept.metadata['source']} | {kept.metadata['context_path']}"
            ):
                origins.append(entry)
            removed += 1
        else:
            seen[body_hash] = doc
    return list(seen.values()), removed


def process_files(input_dir: Path) -> list[Document]:
    md_files = sorted(input_dir.glob("*.md"))
    if not md_files:
        print(f"\n  ERROR: No .md files found in {input_dir.resolve()}")
        print(f"  Put your markdown files there and try again.\n")
        sys.exit(1)

    print(f"\n{'File':<40} {'Chunks':>8}")
    print("-" * 50)

    all_docs: list[Document] = []
    for fp in md_files:
        raw_docs = split_file(fp)
        enriched = enrich_and_link(raw_docs, fp.name)
        print(f"  {fp.name:<38} {len(enriched):>6}")
        all_docs.extend(enriched)

    deduped, removed = dedupe_by_body(all_docs)

    sizes = [len(d.page_content) for d in deduped]
    avg = sum(sizes) // max(len(deduped), 1)
    oversize = sum(1 for s in sizes if s > MAX_CHUNK_SIZE * 1.05)
    multi = sum(1 for d in deduped if d.metadata.get("total_parts", 1) > 1)
    collapsed = sum(1 for d in deduped if d.metadata.get("also_at"))

    print("-" * 50)
    print(f"  Total chunks:     {len(deduped)}")
    print(f"  Deduped away:     {removed}")
    print(f"  Multi-part parts: {multi}")
    print(f"  Cross-file merges:{collapsed}")
    print(f"  Oversize (>{int(MAX_CHUNK_SIZE * 1.05)}):  {oversize}")
    print(f"  Avg size:         {avg} chars")

    return deduped


def save_json(docs: list[Document], path: Path):
    records = [{"page_content": d.page_content, "metadata": d.metadata} for d in docs]
    path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    size_kb = path.stat().st_size / 1024
    print(f"\n  Saved -> {path.resolve()}  ({size_kb:.1f} KB)")


# ENTRY POINT

if __name__ == "__main__":
    input_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("./docs")

    if not input_dir.exists():
        print(f"\n  Folder '{input_dir}' does not exist.")
        print(f"  Create it and put your .md files inside, then run again.\n")
        sys.exit(1)

    print("=" * 50)
    print("  ELSTER Docs Chunker for RAG")
    print(f"  Input:      {input_dir.resolve()}")
    print(f"  Chunk size: max {MAX_CHUNK_SIZE} chars (overlap {CHUNK_OVERLAP})")
    print("=" * 50)

    docs = process_files(input_dir)
    save_json(docs, OUTPUT_JSON)

    print("\n" + "=" * 50)
    print("  SAMPLE CHUNKS (first 3)")
    print("=" * 50)
    for doc in docs[:3]:
        m = doc.metadata
        print(
            f"\n  [{m['chunk_id']}]  {len(doc.page_content)} chars  "
            f"(part {m['part_index'] + 1}/{m['total_parts']})"
        )
        print(f"  Path: {m.get('context_path', '-')}")
        preview = doc.page_content[:300].replace("\n", "\n    ")
        print(f"    {preview} ...")

    print(f"\n  Done! Load {OUTPUT_JSON.name} into your vector store.\n")
