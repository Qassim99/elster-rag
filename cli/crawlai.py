import asyncio
import json
import os
from typing import Optional
from urllib.parse import parse_qs, urlparse

os.environ["LD_LIBRARY_PATH"] = "/usr/lib/x86_64-linux-gnu:" + os.environ.get(
    "LD_LIBRARY_PATH", ""
)

from crawl4ai import AsyncWebCrawler


def extract_thema(url: str) -> Optional[str]:
    parsed = urlparse(url)
    params = parse_qs(parsed.query)
    values = params.get("themaGlobal", [])
    return values[0] if values else None


OUTPUT_DIR = "./outputs-english"
markdown_dir = os.path.join(OUTPUT_DIR, "markdowns")
jsonl_dir = os.path.join(OUTPUT_DIR, "jsonl")
os.makedirs(markdown_dir, exist_ok=True)
os.makedirs(jsonl_dir, exist_ok=True)


async def main():
    BASE = "https://www.elster.de/eportal/helpGlobal"
    themas = [
        "help_eop",
        "help_registrierung",
        "help_arbeitnehmer_eop",
        "help_arbeitgeber_eop",
        "help_vollmachten_faq",
        "help_elster_eop",
        "benutzerkonto",
    ]

    async with AsyncWebCrawler(verbose=True) as crawler:
        for thema in themas:
            url = f"{BASE}?themaGlobal={thema}&locale=en_US"
            print(f"\nCrawling {url}...")
            result = await crawler.arun(
                url=url,
                bypass_cache=True,
                exclude_external_links=True,
            )

            if result.success:
                print(f"Crawl successful for {thema}!")

                document = {
                    "source": url,
                    "content": result.markdown,
                    "metadata": {"language": "en", "type": "guide"},
                }

                with open(
                    os.path.join(jsonl_dir, f"{thema}.jsonl"), "a", encoding="utf-8"
                ) as f:
                    f.write(json.dumps(document, ensure_ascii=False) + "\n")

                with open(
                    os.path.join(markdown_dir, f"{thema}.md"), "w", encoding="utf-8"
                ) as f:
                    f.write(result.markdown)

                print(f"Saved {thema}.jsonl and {thema}.md")

            else:
                print(f"Failed to crawl {thema}: {result.error_message}")


if __name__ == "__main__":
    asyncio.run(main())
