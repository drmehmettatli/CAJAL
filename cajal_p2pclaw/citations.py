"""CAJAL citations module — fetch real references from arXiv and CrossRef."""

import logging
import re
import time
from typing import List, Dict, Optional
import requests

logger = logging.getLogger(__name__)

ARXIV_API = "https://export.arxiv.org/api/query"
CROSSREF_API = "https://api.crossref.org/works"
API_RATE_LIMIT_DELAY = 0.5  # seconds to wait between API calls (polite rate limiting)


def _parse_arxiv_entry(entry_text: str) -> Optional[Dict]:
    """Parse a single arXiv Atom entry into a citation dict."""
    title_match = re.search(r"<title>(.*?)</title>", entry_text, re.DOTALL)
    authors_matches = re.findall(r"<name>(.*?)</name>", entry_text)
    year_match = re.search(r"<published>(\d{4})", entry_text)
    id_match = re.search(r"<id>http://arxiv\.org/abs/([^<]+)</id>", entry_text)
    summary_match = re.search(r"<summary>(.*?)</summary>", entry_text, re.DOTALL)

    if not (title_match and year_match and id_match):
        return None

    title = re.sub(r"\s+", " ", title_match.group(1)).strip()
    authors = [a.strip() for a in authors_matches if a.strip()]
    year = year_match.group(1)
    arxiv_id = id_match.group(1).strip()
    abstract = ""
    if summary_match:
        abstract = re.sub(r"\s+", " ", summary_match.group(1)).strip()

    return {
        "title": title,
        "authors": authors,
        "year": year,
        "source": f"arXiv:{arxiv_id}",
        "url": f"https://arxiv.org/abs/{arxiv_id}",
        "abstract": abstract,
    }


def fetch_arxiv(topic: str, count: int = 5) -> List[Dict]:
    """Search arXiv for papers related to *topic*."""
    params = {
        "search_query": f"all:{topic}",
        "start": 0,
        "max_results": count,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
    try:
        response = requests.get(ARXIV_API, params=params, timeout=15)
        response.raise_for_status()
    except requests.RequestException as exc:
        logger.warning("arXiv API request failed: %s", exc)
        return []

    entries = re.findall(r"<entry>(.*?)</entry>", response.text, re.DOTALL)
    results = []
    for entry in entries:
        parsed = _parse_arxiv_entry(entry)
        if parsed:
            results.append(parsed)
    return results


def fetch_crossref(topic: str, count: int = 5) -> List[Dict]:
    """Search CrossRef for papers related to *topic*."""
    params = {
        "query": topic,
        "rows": count,
        "select": "DOI,title,author,published,abstract",
    }
    try:
        response = requests.get(CROSSREF_API, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
    except (requests.RequestException, ValueError) as exc:
        logger.warning("CrossRef API request failed: %s", exc)
        return []

    items = data.get("message", {}).get("items", [])
    results = []
    for item in items:
        titles = item.get("title", [])
        if not titles:
            continue
        title = titles[0]

        raw_authors = item.get("author", [])
        authors = []
        for a in raw_authors:
            family = a.get("family", "")
            given = a.get("given", "")
            name = f"{given} {family}".strip() if given else family
            if name:
                authors.append(name)

        published = item.get("published", {})
        date_parts = published.get("date-parts", [[]])
        year = str(date_parts[0][0]) if date_parts and date_parts[0] else "n.d."

        doi = item.get("DOI", "")
        results.append({
            "title": title,
            "authors": authors,
            "year": year,
            "source": f"DOI:{doi}" if doi else "CrossRef",
            "url": f"https://doi.org/{doi}" if doi else "",
            "abstract": item.get("abstract", ""),
        })
    return results


def find_references(topic: str, count: int = 8) -> List[Dict]:
    """
    Fetch *count* real references for *topic* from arXiv and CrossRef combined.

    Returns a de-duplicated list ordered by source (arXiv first).
    """
    half = max(count // 2, 1)
    arxiv_refs = fetch_arxiv(topic, half)
    time.sleep(API_RATE_LIMIT_DELAY)
    crossref_refs = fetch_crossref(topic, count - len(arxiv_refs))

    combined: List[Dict] = []
    seen_titles = set()
    for ref in arxiv_refs + crossref_refs:
        key = ref["title"].lower()[:60]
        if key not in seen_titles:
            seen_titles.add(key)
            combined.append(ref)
        if len(combined) >= count:
            break

    return combined


def format_reference(ref: Dict, index: int) -> str:
    """Format a reference dict as a numbered citation string."""
    authors = ref.get("authors", [])
    if authors:
        author_str = ", ".join(authors[:3])
        if len(authors) > 3:
            author_str += " et al."
    else:
        author_str = "Unknown Authors"

    title = ref.get("title", "Untitled")
    year = ref.get("year", "n.d.")
    source = ref.get("source", "")
    url = ref.get("url", "")

    citation = f"[{index}] {author_str} ({year}). *{title}*. {source}."
    if url:
        citation += f" {url}"
    return citation
