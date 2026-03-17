"""Utility functions for the multi‑agent research assistant prototype.

This module defines helper functions for interacting with external APIs
and performing text processing tasks. Where possible, the functions are
implemented using widely available Python libraries. For example, the
``search_arxiv`` function queries the arXiv API for papers matching a
given query. The ``call_llm`` function acts as a placeholder for
invoking a large language model and must be replaced with actual API
calls (e.g. to OpenAI or Google Gemini) by the user. There are also
functions for vectorising text and clustering using scikit‑learn.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

try:
    import requests  # type: ignore
except ImportError:
    requests = None  # will be None in environments without requests

try:
    from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
    from sklearn.cluster import KMeans  # type: ignore
except ImportError:
    # Provide fallbacks if scikit‑learn is unavailable
    TfidfVectorizer = None
    KMeans = None

logger = logging.getLogger(__name__)


@dataclass
class Paper:
    """Represents metadata for a research paper."""
    title: str
    authors: List[str]
    summary: Optional[str] = None
    pdf_url: Optional[str] = None
    published: Optional[str] = None
    doi: Optional[str] = None


def search_arxiv(query: str, max_results: int = 10) -> List[Paper]:
    """Search the arXiv API for papers matching a query.

    Parameters
    ----------
    query: str
        The search query (e.g. keywords or topic).
    max_results: int
        Maximum number of papers to return.

    Returns
    -------
    List[Paper]
        A list of Paper objects containing metadata.

    Notes
    -----
    This function uses the arXiv API documented at
    https://info.arxiv.org/help/api/index.html. If the ``requests``
    library is not available, the function returns an empty list and
    logs a warning.
    """
    if requests is None:
        logger.warning("requests library not installed; cannot perform arXiv search")
        return []

    # Build the API endpoint
    base_url = "http://export.arxiv.org/api/query"
    params = {
        "search_query": query,
        "start": 0,
        "max_results": max_results,
        "sortBy": "lastUpdatedDate",
        "sortOrder": "descending",
    }
    try:
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()
    except Exception as e:
        logger.error("Failed to query arXiv API: %s", e)
        return []

    # Parse the Atom feed. We avoid introducing feedparser to keep dependencies minimal.
    # A simple parsing approach extracts title, authors and PDF link from the XML.
    import xml.etree.ElementTree as ET  # Standard library

    try:
        root = ET.fromstring(response.text)
    except ET.ParseError as e:
        logger.error("Failed to parse arXiv response: %s", e)
        return []

    ns = {
        "atom": "http://www.w3.org/2005/Atom",
        "arxiv": "http://arxiv.org/schemas/atom",
    }
    papers: List[Paper] = []
    for entry in root.findall("atom:entry", ns):
        title_el = entry.find("atom:title", ns)
        title = title_el.text.strip() if title_el is not None else ""
        author_els = entry.findall("atom:author", ns)
        authors = []
        for author in author_els:
            name_el = author.find("atom:name", ns)
            if name_el is not None and name_el.text:
                authors.append(name_el.text.strip())
        pdf_url = None
        for link in entry.findall("atom:link", ns):
            if link.get("title") == "pdf":
                pdf_url = link.get("href")
                break
        published_el = entry.find("atom:published", ns)
        published = published_el.text if published_el is not None else None
        doi = None
        for identifier in entry.findall("arxiv:doi", ns):
            doi = identifier.text
            break
        papers.append(Paper(title=title, authors=authors, pdf_url=pdf_url, published=published, doi=doi))
    return papers


def call_llm(prompt: str, model: str = "gpt-4", max_tokens: int = 512) -> str:
    """Call a language model to generate text based on a prompt.

    This function serves as a placeholder for integrating with a real
    language model (e.g. OpenAI's ChatCompletion API or Google Gemini).
    Users should implement the actual API call and return the model's
    response as a string. For demonstration purposes, the function
    returns a canned response.

    Parameters
    ----------
    prompt: str
        The text prompt to send to the model.
    model: str
        The name of the model to use.
    max_tokens: int
        Maximum number of tokens to generate.

    Returns
    -------
    str
        The generated text from the model.
    """
    logger.info("call_llm invoked with model=%s and max_tokens=%d", model, max_tokens)
    # TODO: Replace this placeholder with a real API call.
    return f"[LLM response placeholder for prompt: {prompt[:60]}...]"


def vectorize_texts(texts: List[str]):
    """Convert a list of texts into TF-IDF vectors.

    If scikit‑learn is not available, returns ``None``.
    """
    if TfidfVectorizer is None:
        logger.warning("scikit-learn is not installed; cannot vectorize texts")
        return None
    vectorizer = TfidfVectorizer(stop_words="english")
    vectors = vectorizer.fit_transform(texts)
    return vectors


def cluster_texts(vectors, n_clusters: int = 5) -> Optional[List[int]]:
    """Cluster the provided vectors using k‑means.

    Returns a list of cluster labels corresponding to each vector, or
    ``None`` if scikit‑learn is not available.
    """
    if KMeans is None or vectors is None:
        logger.warning("scikit-learn is not installed or vectors are None; cannot cluster texts")
        return None
    km = KMeans(n_clusters=n_clusters, random_state=42)
    labels = km.fit_predict(vectors)
    return labels.tolist()