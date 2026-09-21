"""ArXiv API Search Tool.

Fetches research paper details (title, authors, summary/abstract, published date, PDF link)
using the official free arXiv API with zero external dependencies.
"""
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import json
import os

CACHE_FILE = "data/arxiv_cache.json"
LAST_REQUEST_TIME = 0.0

def _load_cache() -> dict:
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def _save_cache(cache: dict):
    try:
        os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def search_arxiv(query: str, max_results: int = 3) -> list[dict] | None:
    """Search arXiv API for the given query and return a list of parsed results.

    Args:
        query: Search keywords or phrases.
        max_results: Maximum number of papers to retrieve.

    Returns:
        List of dicts, each containing 'title', 'authors', 'summary', 'published', 'pdf_link', 'source'.
        Returns None if request failed due to timeout or rate limiting (to signal fail-fast).
    """
    if not query or not query.strip():
        return []

    query_clean = query.strip().lower()

    # Check local cache first
    cache = _load_cache()
    cache_key = f"{query_clean}_max_{max_results}"
    if cache_key in cache:
        print(f"arXiv cache hit: '{query_clean}'")
        return cache[cache_key]

    escaped_query = urllib.parse.quote(query.strip())
    url = f"https://export.arxiv.org/api/query?search_query=all:{escaped_query}&max_results={max_results}"

    import time
    global LAST_REQUEST_TIME

    xml_data = None
    max_retries = 2
    timeout_seconds = 15  # increased from 7s — arXiv can be slow

    for attempt in range(max_retries):
        elapsed = time.time() - LAST_REQUEST_TIME
        if elapsed < 3.5:
            time.sleep(3.5 - elapsed)

        try:
            LAST_REQUEST_TIME = time.time()
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "ScientificDebateAgent/1.0 (fact-checking-research)"},
            )
            with urllib.request.urlopen(req, timeout=timeout_seconds) as response:
                xml_data = response.read()
            break  # success

        except urllib.error.HTTPError as e:
            if e.code == 429:
                # Rate limited — back off much longer than the default retry
                wait = 25 * (attempt + 1)  # 25s first retry, 50s second
                print(f"arXiv rate limited (429). Waiting {wait}s... (attempt {attempt+1}/{max_retries})")
                time.sleep(wait)
                LAST_REQUEST_TIME = time.time()
            else:
                print(f"arXiv HTTP error {e.code} on attempt {attempt+1}/{max_retries}")
                if attempt == max_retries - 1:
                    return None
                time.sleep(5)
                LAST_REQUEST_TIME = time.time()

        except Exception as e:
            if attempt == max_retries - 1:
                print(f"arXiv unavailable after {max_retries} attempts: {type(e).__name__}")
                return None
            wait = 8 * (attempt + 1)
            print(f"arXiv request failed ({type(e).__name__}). Retry in {wait}s... (attempt {attempt+1}/{max_retries})")
            time.sleep(wait)
            LAST_REQUEST_TIME = time.time()

    if xml_data is None:
        return None

    try:
        root = ET.fromstring(xml_data)
        ns = {'atom': 'http://www.w3.org/2005/Atom'}

        results = []
        for entry in root.findall('atom:entry', ns):
            title_node = entry.find('atom:title', ns)
            title = " ".join(title_node.text.split()) if title_node is not None and title_node.text else "No Title"

            summary_node = entry.find('atom:summary', ns)
            summary = " ".join(summary_node.text.split()) if summary_node is not None and summary_node.text else "No Abstract"

            authors = []
            for author in entry.findall('atom:author', ns):
                name_node = author.find('atom:name', ns)
                if name_node is not None and name_node.text:
                    authors.append(name_node.text.strip())
            authors_str = ", ".join(authors) if authors else "Unknown Authors"

            published_node = entry.find('atom:published', ns)
            published = published_node.text.strip()[:10] if published_node is not None and published_node.text else "Unknown Date"

            pdf_link = ""
            for link in entry.findall("atom:link", ns):
                if link.attrib.get('title') == 'pdf' or link.attrib.get('type') == 'application/pdf':
                    pdf_link = link.attrib.get('href', '')
                    break
            if not pdf_link:
                for link in entry.findall("atom:link", ns):
                    if link.attrib.get('rel') == 'alternate':
                        pdf_link = link.attrib.get('href', '')
                        if "/abs/" in pdf_link:
                            pdf_link = pdf_link.replace("/abs/", "/pdf/") + ".pdf"
                        break

            results.append({
                "title": title,
                "authors": authors_str,
                "summary": summary,
                "published": published,
                "pdf_link": pdf_link,
                "source": "arXiv"
            })

        cache = _load_cache()
        cache[cache_key] = results
        _save_cache(cache)

        return results
    except Exception as e:
        print(f"arXiv parse error: {e}")
        return None
