"""
ELSTER Help Pages Scraper — Optimized for RAG
===================================================
Scrapes the ELSTER forms help index and all nested help pages,
producing structured JSON documents ready for vector-store ingestion.

No LLM needed — uses BeautifulSoup for deterministic, fast, free parsing.

Output:
  - elster_output/elster_forms_index.json   → full hierarchy of categories/forms/links
  - elster_output/elster_rag_documents.json → chunked documents with metadata for RAG
  - elster_output/markdown_pages/           → raw markdown per help page (optional backup)

Usage:
  pip install requests beautifulsoup4 markdownify
  python elster_scraper_rag.py
"""

import hashlib
import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Optional
from urllib.parse import parse_qs, urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as md

# ─── Configuration ───────────────────────────────────────────────────────────

BASE_URL = "https://www.elster.de"
INDEX_URL = f"{BASE_URL}/eportal/helpGlobal?themaGlobal=formulare_eop"
OUTPUT_DIR = "elster_output"
MARKDOWN_DIR = os.path.join(OUTPUT_DIR, "markdown_pages")
REQUEST_DELAY = 1.5  # seconds between requests — be polite
MAX_CHUNK_CHARS = 1500  # target chunk size for RAG documents
OVERLAP_CHARS = 200  # overlap between chunks for context continuity
REQUEST_TIMEOUT = 30  # seconds


# ─── Data Models ─────────────────────────────────────────────────────────────


@dataclass
class FormLink:
    title: str
    url: str
    year: Optional[str] = None
    thema_global: Optional[str] = None


@dataclass
class FormEntry:
    name: str
    category: str
    links: list = field(default_factory=list)


@dataclass
class RAGDocument:
    """A single chunk ready for vector-store ingestion."""

    doc_id: str
    title: str
    category: str
    form_name: str
    year: Optional[str]
    url: str
    chunk_index: int
    total_chunks: int
    content: str
    language: str = "de"
    source: str = "elster.de"


# ─── Helpers ─────────────────────────────────────────────────────────────────


def get_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "de-DE,de;q=0.9,en;q=0.5",
        }
    )
    return session


def fetch_page(session: requests.Session, url: str) -> Optional[BeautifulSoup]:
    try:
        resp = session.get(url, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        resp.encoding = resp.apparent_encoding or "utf-8"
        return BeautifulSoup(resp.text, "html.parser")
    except requests.RequestException as e:
        print(f"  ✗ Failed to fetch {url}: {e}")
        return None


def extract_thema(url: str) -> Optional[str]:
    parsed = urlparse(url)
    params = parse_qs(parsed.query)
    values = params.get("themaGlobal", [])
    return values[0] if values else None


def extract_year(title: str) -> Optional[str]:
    match = re.search(r"\b(20\d{2})\b", title)
    return match.group(1) if match else None


def make_absolute(href: str) -> str:
    if href.startswith("/"):
        return urljoin(BASE_URL, href)
    if href.startswith("http"):
        return href
    return urljoin(BASE_URL + "/", href)


def make_doc_id(url: str, chunk_index: int) -> str:
    url_hash = hashlib.md5(url.encode()).hexdigest()[:10]
    return f"elster_{url_hash}_chunk{chunk_index}"


def is_help_link(href: str) -> bool:
    """Check if a link points to an ELSTER help sub-page."""
    if not href:
        return False
    if href.startswith("#"):
        return False
    full = make_absolute(href)
    thema = extract_thema(full)
    # Accept any themaGlobal that starts with "help_"
    if thema and thema.startswith("help_"):
        return True
    return False


# ─── Step 1: Parse the Index Page ────────────────────────────────────────────


def parse_index(soup: BeautifulSoup) -> list[FormEntry]:
    """
    Robust approach: flatten ALL h3, h4, and <a> tags in document order.
    Track the current category (h3) and form name (h4) as state,
    and assign each help link to the current form entry.

    This works regardless of how the HTML nests divs/accordions/etc.
    """
    entries: list[FormEntry] = []
    current_category = "Allgemein"
    current_entry: Optional[FormEntry] = None

    # Get all h3, h4, and anchor elements in document order
    all_elements = soup.find_all(["h3", "h4", "a"])

    for el in all_elements:
        if el.name == "h3":
            text = el.get_text(strip=True)
            if text:
                current_category = text
                current_entry = None  # reset

        elif el.name == "h4":
            text = el.get_text(strip=True)
            if text:
                current_entry = FormEntry(name=text, category=current_category)
                entries.append(current_entry)

        elif el.name == "a":
            href = el.get("href", "")
            if is_help_link(href):
                full_url = make_absolute(href)
                title = el.get_text(strip=True)
                if not title:
                    continue
                link = FormLink(
                    title=title,
                    url=full_url,
                    year=extract_year(title),
                    thema_global=extract_thema(full_url),
                )
                if current_entry is not None:
                    current_entry.links.append(link)
                else:
                    # Link before any h4 — create a catch-all entry
                    current_entry = FormEntry(
                        name=title,
                        category=current_category,
                    )
                    current_entry.links.append(link)
                    entries.append(current_entry)

    # Remove entries with no links (empty h4s)
    entries = [e for e in entries if e.links]
    return entries


# ─── Step 2: Scrape Individual Help Pages ────────────────────────────────────


def extract_help_content(soup: BeautifulSoup) -> str:
    """Extract the main help content and convert to clean markdown."""
    for tag in soup.find_all(
        ["nav", "header", "footer", "script", "style", "noscript"]
    ):
        tag.decompose()

    content_div = (
        soup.find("div", class_="help-content")
        or soup.find("div", class_="helpContent")
        or soup.find("div", id="content")
        or soup.find("main")
        or soup.find("div", class_="container")
    )
    if not content_div:
        content_div = soup.find("body") or soup

    raw_md = md(
        str(content_div),
        heading_style="ATX",
        strip=["img", "svg", "iframe", "form", "input", "button", "select"],
    )

    raw_md = re.sub(r"\n{3,}", "\n\n", raw_md)
    raw_md = re.sub(r"[ \t]+\n", "\n", raw_md)
    raw_md = raw_md.strip()
    return raw_md


# ─── Step 3: Chunk Content for RAG ──────────────────────────────────────────


def chunk_text(
    text: str, max_chars: int = MAX_CHUNK_CHARS, overlap: int = OVERLAP_CHARS
) -> list[str]:
    if len(text) <= max_chars:
        return [text]

    chunks = []
    sections = re.split(r"(?=\n#{2,3}\s)", text)

    current_chunk = ""
    for section in sections:
        if current_chunk and len(current_chunk) + len(section) > max_chars:
            chunks.append(current_chunk.strip())
            current_chunk = current_chunk[-overlap:] if overlap else ""

        current_chunk += section

        if len(current_chunk) > max_chars:
            paragraphs = current_chunk.split("\n\n")
            current_chunk = ""
            for para in paragraphs:
                if current_chunk and len(current_chunk) + len(para) + 2 > max_chars:
                    chunks.append(current_chunk.strip())
                    current_chunk = current_chunk[-overlap:] if overlap else ""
                current_chunk += para + "\n\n"

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


def build_rag_documents(
    content: str,
    url: str,
    title: str,
    category: str,
    form_name: str,
    year: Optional[str],
) -> list[RAGDocument]:
    chunks = chunk_text(content)
    total = len(chunks)
    docs = []
    for i, chunk in enumerate(chunks):
        doc = RAGDocument(
            doc_id=make_doc_id(url, i),
            title=title,
            category=category,
            form_name=form_name,
            year=year,
            url=url,
            chunk_index=i,
            total_chunks=total,
            content=chunk,
        )
        docs.append(doc)
    return docs


# ─── Main Orchestrator ───────────────────────────────────────────────────────


def main():
    os.makedirs(MARKDOWN_DIR, exist_ok=True)

    session = get_session()
    all_rag_docs: list[RAGDocument] = []

    # ── 1. Fetch and parse the index ──
    print(f"📄 Fetching index page: {INDEX_URL}")
    index_soup = fetch_page(session, INDEX_URL)
    if not index_soup:
        print("Failed to fetch the index page. Exiting.")
        return

    entries = parse_index(index_soup)

    total_links = sum(len(e.links) for e in entries)
    print(f"✅ Found {len(entries)} form entries with {total_links} total links.\n")

    # Debug: show first few entries
    for entry in entries[:5]:
        print(f"  📁 [{entry.category}] {entry.name}")
        for link in entry.links[:3]:
            print(f"       → {link.title}: {link.url}")
        if len(entry.links) > 3:
            print(f"       ... and {len(entry.links) - 3} more")
    if len(entries) > 5:
        print(f"  ... and {len(entries) - 5} more entries\n")

    # Save the structured index
    index_data = []
    for entry in entries:
        index_data.append(
            {
                "category": entry.category,
                "form_name": entry.name,
                "links": [asdict(link) for link in entry.links],
            }
        )

    index_path = os.path.join(OUTPUT_DIR, "elster_forms_index.json")
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index_data, f, ensure_ascii=False, indent=2)
    print(f"💾 Saved form index → {index_path}\n")

    # ── 2. Collect unique URLs ──
    urls_to_scrape: list[dict] = []
    seen_urls = set()

    for entry in entries:
        for link in entry.links:
            if link.url not in seen_urls:
                seen_urls.add(link.url)
                urls_to_scrape.append(
                    {
                        "url": link.url,
                        "title": link.title,
                        "category": entry.category,
                        "form_name": entry.name,
                        "year": link.year,
                    }
                )

    print(f"🔗 {len(urls_to_scrape)} unique help pages to scrape.\n")

    if not urls_to_scrape:
        print("⚠ No URLs found! Dumping first 20 <a> tags for debugging:")
        for a in index_soup.find_all("a", href=True)[:20]:
            print(f"  href={a['href']!r}  text={a.get_text(strip=True)!r}")
        return

    # ── 3. Scrape each help page ──
    failed = 0
    for i, item in enumerate(urls_to_scrape, 1):
        url = item["url"]
        thema = extract_thema(url) or "unknown"
        print(
            f"[{i}/{len(urls_to_scrape)}] {item['category']} → {item['form_name']} → {item['title']}"
        )

        soup = fetch_page(session, url)
        if not soup:
            failed += 1
            continue

        content = extract_help_content(soup)

        if not content or len(content) < 50:
            print(f"  ⚠ Skipped — too little content ({len(content)} chars)")
            continue

        # Save raw markdown with metadata header
        md_filename = f"{thema}.md"
        md_path = os.path.join(MARKDOWN_DIR, md_filename)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(f"# {item['title']}\n\n")
            f.write(f"**Kategorie:** {item['category']}  \n")
            f.write(f"**Formular:** {item['form_name']}  \n")
            f.write(f"**Jahr:** {item['year'] or 'N/A'}  \n")
            f.write(f"**URL:** {url}  \n\n---\n\n")
            f.write(content)

        # Build RAG chunks
        rag_docs = build_rag_documents(
            content=content,
            url=url,
            title=item["title"],
            category=item["category"],
            form_name=item["form_name"],
            year=item["year"],
        )
        all_rag_docs.extend(rag_docs)
        print(f"  ✓ {len(content):,} chars → {len(rag_docs)} chunks")

        time.sleep(REQUEST_DELAY)

    # ── 4. Save RAG documents ──
    rag_path = os.path.join(OUTPUT_DIR, "elster_rag_documents.json")
    with open(rag_path, "w", encoding="utf-8") as f:
        json.dump(
            [asdict(doc) for doc in all_rag_docs],
            ensure_ascii=False,
            indent=2,
            fp=f,
        )

    # ── 5. Summary ──
    print(f"\n{'=' * 60}")
    print(f"✅ Done!")
    print(f"   Form entries parsed      : {len(entries)}")
    print(f"   Unique help pages        : {len(urls_to_scrape)}")
    print(f"   Pages scraped OK         : {len(urls_to_scrape) - failed}")
    print(f"   Pages failed             : {failed}")
    print(f"   Total RAG chunks created : {len(all_rag_docs)}")
    print(f"   Index file               : {index_path}")
    print(f"   RAG documents            : {rag_path}")
    print(f"   Raw markdown pages       : {MARKDOWN_DIR}/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
