"""Shared low-level helpers for the research agent system."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import xml.etree.ElementTree as ET
from typing import Any

try:
    from sklearn.cluster import KMeans  # type: ignore
    from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
except ImportError:
    KMeans = None
    TfidfVectorizer = None

from models import PaperRecord

logger = logging.getLogger(__name__)

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "using",
    "with",
}

ZH_STOPWORDS = {
    "如何",
    "用于",
    "以及",
    "并且",
    "比较",
    "对比",
    "研究",
    "系统",
    "方法",
}


def normalize_text(text: str) -> str:
    """Collapse internal whitespace for stable prompts and outputs."""
    return re.sub(r"\s+", " ", text or "").strip()


def first_sentence(text: str) -> str:
    """Return the first sentence-like chunk from a text."""
    normalized = normalize_text(text)
    if not normalized:
        return ""
    parts = re.split(r"(?<=[.!?。！？])\s*", normalized)
    return parts[0]


def stable_hash(value: str) -> str:
    """Return a stable content hash."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def estimate_tokens(text: str) -> int:
    """Estimate token count from plain text."""
    if not text:
        return 0
    return max(1, len(text) // 4)


def extract_keywords(text: str, max_keywords: int = 10) -> list[str]:
    """Extract ordered, de-duplicated keywords from input."""
    normalized_chinese = re.sub(r"[，。；、（）()【】\[\],.!?:：;]", " ", text)
    normalized_chinese = re.sub(r"(如何|用于|以及|并且|并|和|与|在|中|的|做|比较|对比|面向|基于)", " ", normalized_chinese)
    chinese_tokens = [token.strip() for token in re.findall(r"[\u4e00-\u9fff]{2,12}", normalized_chinese)]
    raw_tokens = [token.strip("-_") for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]*", text.lower())]
    keywords: list[str] = []
    seen: set[str] = set()

    for token in chinese_tokens:
        if not token or token in ZH_STOPWORDS:
            continue
        if token not in seen:
            keywords.append(token)
            seen.add(token)
        if len(keywords) >= max_keywords:
            return keywords

    for token in raw_tokens:
        if not token or token in STOPWORDS:
            continue
        if token not in seen:
            keywords.append(token)
            seen.add(token)
        if len(keywords) >= max_keywords:
            return keywords

    if keywords:
        return keywords

    fallback = []
    for token in chinese_tokens + raw_tokens:
        if token and token not in seen:
            fallback.append(token)
            seen.add(token)
        if len(fallback) >= max_keywords:
            break
    return fallback or ["research", "idea"]


def keyword_overlap_score(text: str, keywords: list[str]) -> float:
    """Score a document based on keyword overlap."""
    lowered = normalize_text(text).lower()
    if not keywords:
        return 0.0
    hits = sum(1 for keyword in keywords if keyword.lower() in lowered)
    return hits / max(len(keywords), 1)


def parse_json_object(text: str) -> dict[str, Any]:
    """Parse a JSON object possibly wrapped in a fenced block."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        cleaned = cleaned[start : end + 1]
    parsed = json.loads(cleaned)
    if not isinstance(parsed, dict):
        raise ValueError("Expected a JSON object.")
    return parsed


def parse_arxiv_response(xml_text: str) -> list[PaperRecord]:
    """Parse arXiv Atom XML into normalized paper records."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        logger.error("Failed to parse arXiv response: %s", exc)
        return []

    namespace = {
        "atom": "http://www.w3.org/2005/Atom",
        "arxiv": "http://arxiv.org/schemas/atom",
    }
    papers: list[PaperRecord] = []
    for entry in root.findall("atom:entry", namespace):
        title_element = entry.find("atom:title", namespace)
        summary_element = entry.find("atom:summary", namespace)
        published_element = entry.find("atom:published", namespace)
        authors = []
        for author in entry.findall("atom:author", namespace):
            name_element = author.find("atom:name", namespace)
            if name_element is not None and name_element.text:
                authors.append(normalize_text(name_element.text))

        pdf_url = None
        for link in entry.findall("atom:link", namespace):
            if link.get("title") == "pdf":
                pdf_url = link.get("href")
                break

        doi = None
        doi_element = entry.find("arxiv:doi", namespace)
        if doi_element is not None and doi_element.text:
            doi = normalize_text(doi_element.text)

        title = normalize_text(title_element.text if title_element is not None and title_element.text else "")
        abstract = normalize_text(summary_element.text if summary_element is not None and summary_element.text else "")
        papers.append(
            {
                "title": title,
                "authors": authors,
                "abstract": abstract,
                "source": "arxiv",
                "external_id": None,
                "pdf_url": pdf_url,
                "published": normalize_text(
                    published_element.text if published_element is not None and published_element.text else ""
                )
                or None,
                "doi": doi,
                "citation_count": 0,
                "url": pdf_url,
                "score": 0.0,
                "snippets": [first_sentence(abstract)] if abstract else [],
                "raw": {},
            }
        )
    return dedupe_papers(papers)


def dedupe_papers(papers: list[PaperRecord]) -> list[PaperRecord]:
    """Remove duplicates by normalized title, DOI, and URL."""
    seen_titles: set[str] = set()
    seen_ids: set[str] = set()
    deduped: list[PaperRecord] = []
    for paper in papers:
        title_key = normalize_text(paper["title"]).lower()
        id_key = (paper.get("doi") or paper.get("external_id") or paper.get("url") or "").lower()
        if title_key and title_key in seen_titles:
            continue
        if id_key and id_key in seen_ids:
            continue
        if title_key:
            seen_titles.add(title_key)
        if id_key:
            seen_ids.add(id_key)
        deduped.append(paper)
    return deduped


def vectorize_texts(texts: list[str]):
    """Convert a list of texts into TF-IDF vectors."""
    if TfidfVectorizer is None:
        logger.warning("scikit-learn is not installed; cannot vectorize texts")
        return None
    vectorizer = TfidfVectorizer(stop_words="english")
    return vectorizer.fit_transform(texts)


def cluster_texts(vectors, n_clusters: int = 5) -> list[int] | None:
    """Cluster the provided vectors using k-means."""
    if KMeans is None or vectors is None:
        logger.warning("scikit-learn is not installed or vectors are None; cannot cluster texts")
        return None
    if getattr(vectors, "shape", None) is not None and vectors.shape[0] < 2:
        return None
    if getattr(vectors, "shape", None) is not None:
        n_clusters = min(n_clusters, vectors.shape[0])
    if n_clusters < 2:
        return None
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    return km.fit_predict(vectors).tolist()
