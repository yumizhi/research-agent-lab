"""Utility functions for the research multi-agent MVP."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import textwrap
import xml.etree.ElementTree as ET
from collections import defaultdict
from typing import Any, Optional

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

from models import CandidateTopic, CritiqueRecord, PaperRecord, ResearchState, SummaryRecord

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


class LLMConfigurationError(RuntimeError):
    """Raised when live LLM mode cannot be satisfied."""


def normalize_text(text: str) -> str:
    """Collapse internal whitespace for stable prompts and outputs."""
    return re.sub(r"\s+", " ", text or "").strip()


def extract_keywords(text: str, max_keywords: int = 8) -> list[str]:
    """Extract ordered, de-duplicated keywords from user input."""
    raw_tokens = [token.strip("-_") for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]*", text.lower())]
    keywords: list[str] = []
    seen: set[str] = set()

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
    for token in raw_tokens:
        if not token:
            continue
        if token not in seen:
            fallback.append(token)
            seen.add(token)
        if len(fallback) >= max_keywords:
            break
    if fallback:
        return fallback
    return ["research", "idea"][:max_keywords]


def build_arxiv_query(keywords: list[str]) -> str:
    """Build an arXiv search query from extracted keywords."""
    if not keywords:
        return ""
    return "+AND+".join(f"all:{keyword}" for keyword in keywords)


def _first_sentence(text: str) -> str:
    normalized = normalize_text(text)
    if not normalized:
        return ""
    parts = re.split(r"(?<=[.!?])\s+", normalized)
    return parts[0]


def _dedupe_papers(papers: list[PaperRecord]) -> list[PaperRecord]:
    deduped: list[PaperRecord] = []
    seen_titles: set[str] = set()
    seen_urls: set[str] = set()
    for paper in papers:
        normalized_title = normalize_text(paper["title"]).lower()
        normalized_url = (paper.get("pdf_url") or "").strip().lower()
        if normalized_title and normalized_title in seen_titles:
            continue
        if normalized_url and normalized_url in seen_urls:
            continue
        seen_titles.add(normalized_title)
        if normalized_url:
            seen_urls.add(normalized_url)
        deduped.append(paper)
    return deduped


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

        papers.append(
            {
                "title": normalize_text(title_element.text if title_element is not None and title_element.text else ""),
                "authors": authors,
                "summary": normalize_text(summary_element.text if summary_element is not None and summary_element.text else ""),
                "pdf_url": pdf_url,
                "published": normalize_text(
                    published_element.text if published_element is not None and published_element.text else ""
                )
                or None,
                "doi": doi,
            }
        )
    return _dedupe_papers(papers)

def search_arxiv(query: str, max_results: int = 10) -> list[PaperRecord]:
    """Search the arXiv API for papers matching a query.

    The function returns an empty list when network access or dependencies
    are unavailable so the offline prototype remains runnable.
    """
    if requests is None:
        logger.warning("requests library not installed; cannot perform arXiv search")
        return []

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
    except Exception as exc:
        logger.error("Failed to query arXiv API: %s", exc)
        return []
    return parse_arxiv_response(response.text)


def call_llm(
    prompt: str,
    *,
    task: str,
    live: bool = False,
    model: str | None = None,
    max_tokens: int = 512,
) -> str:
    """Call a language model to generate text based on a prompt.

    In offline mode this returns a deterministic stub. In live mode it
    uses an OpenAI-compatible chat completions endpoint.
    """
    logger.info("call_llm invoked with task=%s live=%s max_tokens=%d", task, live, max_tokens)
    if not live:
        excerpt = normalize_text(prompt)[:160]
        return f"[stub:{task}] {excerpt}"

    if requests is None:
        raise LLMConfigurationError("Live LLM mode requires the requests package.")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise LLMConfigurationError("Live LLM mode requires OPENAI_API_KEY.")

    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    endpoint = f"{base_url}/chat/completions"
    selected_model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    payload = {
        "model": selected_model,
        "messages": [
            {"role": "system", "content": "You are a precise research workflow assistant."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": max_tokens,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    try:
        response = requests.post(endpoint, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
    except Exception as exc:
        raise LLMConfigurationError(f"Live LLM request failed: {exc}") from exc

    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMConfigurationError("Live LLM response did not include a message content field.") from exc

    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        text_parts = [part.get("text", "") for part in content if isinstance(part, dict)]
        return "\n".join(part for part in text_parts if part).strip()
    raise LLMConfigurationError("Unsupported content shape returned by live LLM.")


def _extract_json_object(text: str) -> dict[str, Any]:
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
        raise ValueError("Expected JSON object.")
    return parsed


def _deterministic_score(seed_text: str, floor: int, ceiling: int) -> int:
    span = ceiling - floor + 1
    digest = hashlib.sha256(seed_text.encode("utf-8")).hexdigest()
    return floor + (int(digest[:8], 16) % span)


def _fallback_improvement_ideas(paper: PaperRecord) -> list[str]:
    topic_keywords = extract_keywords(f"{paper['title']} {paper.get('summary', '')}", max_keywords=3)
    lead = topic_keywords[0] if topic_keywords else "the core method"
    support = topic_keywords[1] if len(topic_keywords) > 1 else "robustness"
    third = topic_keywords[2] if len(topic_keywords) > 2 else "evaluation"
    return [
        f"Add a stronger baseline comparison focused on {lead}.",
        f"Include an ablation study that isolates the effect of {support}.",
        f"Test generalization under distribution shift or noisy {third} settings.",
    ]


def summarize_paper(paper: PaperRecord, *, live: bool = False) -> SummaryRecord:
    """Produce a summary record for a paper."""
    if live:
        prompt = textwrap.dedent(
            f"""
            Summarize the following research paper for a scouting workflow.
            Return JSON with keys "summary" and "improvement_ideas".

            Paper:
            {json.dumps(paper, ensure_ascii=False, indent=2)}
            """
        ).strip()
        raw = call_llm(prompt, task="summarize", live=True, max_tokens=500)
        try:
            parsed = _extract_json_object(raw)
            summary_text = normalize_text(str(parsed.get("summary", "")))
            ideas = [normalize_text(str(item)) for item in parsed.get("improvement_ideas", []) if str(item).strip()]
            if summary_text:
                return {
                    "paper": paper,
                    "summary": summary_text,
                    "improvement_ideas": ideas[:3] or _fallback_improvement_ideas(paper),
                }
        except (ValueError, json.JSONDecodeError, TypeError):
            pass

        return {
            "paper": paper,
            "summary": normalize_text(raw),
            "improvement_ideas": _fallback_improvement_ideas(paper),
        }

    first_sentence = _first_sentence(paper.get("summary", "")) or "The abstract is unavailable, so the prototype infers direction from the title."
    keywords = extract_keywords(f"{paper['title']} {paper.get('summary', '')}", max_keywords=3)
    focus = ", ".join(keywords) if keywords else "the proposed research area"
    summary_text = (
        f"{paper['title']} focuses on {focus}. "
        f"{first_sentence} "
        "The paper is treated as a candidate signal for downstream topic selection and experiment design."
    )
    return {
        "paper": paper,
        "summary": summary_text,
        "improvement_ideas": _fallback_improvement_ideas(paper),
    }


def criticize_summary(
    summary_record: SummaryRecord,
    keywords: list[str],
    *,
    live: bool = False,
) -> CritiqueRecord:
    """Score a summary on novelty, methodology, and relevance."""
    paper = summary_record["paper"]
    summary_text = summary_record["summary"]
    critique_keywords = set(extract_keywords(f"{paper['title']} {summary_text}", max_keywords=10))
    overlap = len(critique_keywords.intersection(keywords))

    heuristic = {
        "novelty": min(10, _deterministic_score(paper["title"], 6, 8) + min(2, overlap)),
        "methodology": min(10, _deterministic_score(summary_text, 5, 7) + min(2, len(summary_text.split()) // 25)),
        "relevance": min(10, 5 + overlap + (1 if keywords else 0)),
    }

    if live:
        prompt = textwrap.dedent(
            f"""
            Evaluate this paper summary for a research scouting workflow.
            Return JSON with keys "novelty", "methodology", "relevance", and "critique".

            User keywords: {json.dumps(keywords, ensure_ascii=False)}
            Summary record:
            {json.dumps(summary_record, ensure_ascii=False, indent=2)}
            """
        ).strip()
        raw = call_llm(prompt, task="critique", live=True, max_tokens=350)
        try:
            parsed = _extract_json_object(raw)
            novelty = int(parsed.get("novelty", heuristic["novelty"]))
            methodology = int(parsed.get("methodology", heuristic["methodology"]))
            relevance = int(parsed.get("relevance", heuristic["relevance"]))
            critique_text = normalize_text(str(parsed.get("critique", "")))
            overall = round((novelty + methodology + relevance) / 3, 1)
            return {
                "paper": paper,
                "summary": summary_text,
                "novelty": max(0, min(10, novelty)),
                "methodology": max(0, min(10, methodology)),
                "relevance": max(0, min(10, relevance)),
                "overall": overall,
                "critique": critique_text or "Model response lacked a critique, so heuristic guidance remains primary.",
            }
        except (ValueError, json.JSONDecodeError, TypeError):
            pass

    novelty = heuristic["novelty"]
    methodology = heuristic["methodology"]
    relevance = heuristic["relevance"]
    overall = round((novelty + methodology + relevance) / 3, 1)
    lead_keyword = keywords[0] if keywords else "the target topic"
    critique_text = (
        f"Strongest signal is relevance to {lead_keyword}; the next step should tighten evaluation design and add clearer ablations."
    )
    return {
        "paper": paper,
        "summary": summary_text,
        "novelty": novelty,
        "methodology": methodology,
        "relevance": relevance,
        "overall": overall,
        "critique": critique_text,
    }


def build_candidate_topics(
    critiques: list[CritiqueRecord],
    *,
    cluster_labels: list[int] | None = None,
    top_n: int = 3,
) -> list[CandidateTopic]:
    """Create candidate research topics from scored papers."""
    if not critiques:
        return []

    if cluster_labels and len(cluster_labels) == len(critiques) and len(set(cluster_labels)) > 1:
        grouped: dict[int, list[CritiqueRecord]] = defaultdict(list)
        for critique, label in zip(critiques, cluster_labels):
            critique["cluster"] = label
            grouped[label].append(critique)

        topics: list[CandidateTopic] = []
        for label, items in grouped.items():
            average_score = round(sum(item["overall"] for item in items) / len(items), 1)
            combined_text = " ".join(item["paper"]["title"] for item in items)
            label_keywords = extract_keywords(combined_text, max_keywords=3)
            title = " / ".join(keyword.title() for keyword in label_keywords) or items[0]["paper"]["title"]
            topics.append(
                {
                    "title": title,
                    "rationale": f"Cluster {label} groups {len(items)} papers with mean score {average_score}.",
                    "source_papers": [item["paper"]["title"] for item in items[:3]],
                    "score": average_score,
                    "cluster": label,
                }
            )
        topics.sort(key=lambda item: item["score"], reverse=True)
        return topics[:top_n]

    ranked = sorted(critiques, key=lambda item: item["overall"], reverse=True)
    topics = []
    for critique in ranked[:top_n]:
        topics.append(
            {
                "title": critique["paper"]["title"],
                "rationale": critique["critique"],
                "source_papers": [critique["paper"]["title"]],
                "score": critique["overall"],
            }
        )
    return topics


def _infer_datasets(keywords: list[str]) -> list[str]:
    joined = " ".join(keywords)
    if any(token in joined for token in {"energy", "load", "forecast", "time", "temporal", "grid"}):
        return [
            "ETTh1 / ETTm1 for public long-horizon forecasting benchmarks.",
            "ElectricityLoadDiagrams20112014 for household and regional demand forecasting.",
            "ISO-NE or a comparable grid operator dataset for real-world deployment checks.",
        ]
    if any(token in joined for token in {"vision", "image", "visual"}):
        return [
            "A primary benchmark such as CIFAR-10, ImageNet subset, or a task-specific public dataset.",
            "One robustness dataset to evaluate domain shift or corrupted inputs.",
        ]
    return [
        "One public benchmark dataset directly aligned with the user problem statement.",
        "A secondary validation dataset for out-of-domain or temporal generalization checks.",
    ]


def _infer_methods(keywords: list[str]) -> list[str]:
    joined = " ".join(keywords)
    if any(token in joined for token in {"energy", "load", "forecast", "time", "temporal", "grid"}):
        return [
            "Establish non-neural baselines such as XGBoost or ARIMA.",
            "Compare sequence baselines such as LSTM or Temporal Fusion Transformer.",
            "Evaluate a transformer variant with explicit temporal attention and ablations on horizon length.",
        ]
    return [
        "Define a simple baseline, a stronger neural baseline, and the proposed method.",
        "Run ablations on the core architectural choice and training configuration.",
        "Document preprocessing, split logic, and failure cases before scaling experiments.",
    ]


def _infer_metrics(keywords: list[str]) -> list[str]:
    joined = " ".join(keywords)
    if any(token in joined for token in {"energy", "load", "forecast", "time", "temporal", "grid"}):
        return ["MAE", "RMSE", "MAPE", "inference latency"]
    if any(token in joined for token in {"classify", "classification", "image"}):
        return ["accuracy", "macro F1", "calibration error", "latency"]
    return ["task-specific primary metric", "generalization gap", "runtime cost"]


def build_plan_markdown(state: ResearchState, *, live: bool = False) -> str:
    """Build a markdown research plan from the current state."""
    if live:
        prompt = textwrap.dedent(
            f"""
            Build a concise but detailed markdown research plan.
            Include sections for research question, datasets, methods, evaluation metrics, timeline, and risks.

            State:
            {json.dumps(state, ensure_ascii=False, indent=2)}
            """
        ).strip()
        response = call_llm(prompt, task="plan", live=True, max_tokens=900)
        if response.strip():
            return response.strip()

    keywords = state["keywords"]
    topic = state["candidate_topics"][0] if state["candidate_topics"] else {
        "title": " / ".join(keyword.title() for keyword in keywords[:3]) or "Research Direction",
        "rationale": "No papers were fetched, so the plan is derived from the user idea and extracted keywords.",
        "source_papers": [],
        "score": 0.0,
    }
    datasets = _infer_datasets(keywords)
    methods = _infer_methods(keywords)
    metrics = _infer_metrics(keywords)
    source_lines = (
        [f"- {paper['title']}" for paper in state["papers"][:5]]
        if state["papers"]
        else ["- No literature was fetched; this plan is derived from the original user idea and keyword analysis."]
    )

    research_question = (
        f"How can the project around '{topic['title']}' improve on current baselines for the user goal: "
        f"{state['user_input']}?"
    )

    sections = [
        "# Research Plan",
        "",
        "## Research Question",
        research_question,
        "",
        "## Candidate Topic",
        f"- Title: {topic['title']}",
        f"- Rationale: {topic['rationale']}",
        "",
        "## Datasets",
        *[f"- {item}" for item in datasets],
        "",
        "## Methods",
        *[f"- {item}" for item in methods],
        "",
        "## Evaluation Metrics",
        *[f"- {item}" for item in metrics],
        "",
        "## Timeline",
        "- Phase 1: finalize task framing, dataset choice, and baseline definitions.",
        "- Phase 2: implement preprocessing, baseline training, and reproducible evaluation.",
        "- Phase 3: run the proposed method, ablations, and error analysis.",
        "- Phase 4: summarize findings, risks, and next experiments.",
        "",
        "## Risks",
        "- Literature relevance may be low if retrieval returns sparse or noisy results.",
        "- Evaluation can overfit to one benchmark unless a secondary validation set is included.",
        "- Model complexity should be justified by gains over simple baselines.",
        "",
        "## Source Papers",
        *source_lines,
    ]
    return "\n".join(sections).strip()


def _strip_code_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return cleaned


def build_code_scaffold(state: ResearchState, *, live: bool = False) -> str:
    """Generate an experiment scaffold from the plan."""
    topic = state["candidate_topics"][0]["title"] if state["candidate_topics"] else "Research Experiment"
    metrics = _infer_metrics(state["keywords"])

    if live:
        prompt = textwrap.dedent(
            f"""
            Write Python code only. Build a runnable research experiment scaffold with configuration,
            dataset loading placeholders, model/training stubs, and evaluation hooks.

            Plan:
            {state['plan_markdown']}
            """
        ).strip()
        response = call_llm(prompt, task="codegen", live=True, max_tokens=1200)
        if response.strip():
            return _strip_code_fences(response)

    metric_repr = ", ".join(metrics)
    return textwrap.dedent(
        f'''
        """Experiment scaffold for: {topic}."""

        from __future__ import annotations

        import argparse
        from dataclasses import dataclass
        from pathlib import Path
        from statistics import mean
        from typing import Iterable


        @dataclass
        class ExperimentConfig:
            dataset_path: Path
            output_dir: Path = Path("artifacts")
            epochs: int = 5
            batch_size: int = 32
            learning_rate: float = 1e-3
            seed: int = 42


        def load_dataset(dataset_path: Path) -> list[dict[str, float]]:
            """TODO: Replace this stub with real preprocessing."""
            if not dataset_path.exists():
                return [{{"feature": float(index), "target": float(index % 5)}} for index in range(128)]
            return [{{"feature": float(index), "target": float(index % 7)}} for index in range(128)]


        def split_dataset(records: list[dict[str, float]]) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
            cutoff = max(1, int(len(records) * 0.8))
            return records[:cutoff], records[cutoff:]


        class BaselineModel:
            """Minimal placeholder model that can be replaced with PyTorch or JAX."""

            def __init__(self) -> None:
                self.bias = 0.0

            def fit_batch(self, batch: Iterable[dict[str, float]], learning_rate: float) -> float:
                batch = list(batch)
                if not batch:
                    return 0.0
                error = mean(item["target"] - self.predict(item["feature"]) for item in batch)
                self.bias += learning_rate * error
                return abs(error)

            def predict(self, feature: float) -> float:
                return feature * 0.1 + self.bias


        def train_one_epoch(model: BaselineModel, records: list[dict[str, float]], learning_rate: float) -> float:
            batch_loss = model.fit_batch(records, learning_rate)
            return batch_loss


        def evaluate(model: BaselineModel, records: list[dict[str, float]]) -> dict[str, float]:
            if not records:
                return {{"primary_metric": 0.0}}
            errors = [abs(item["target"] - model.predict(item["feature"])) for item in records]
            return {{
                "primary_metric": mean(errors),
                "tracked_metrics": "{metric_repr}",
            }}


        def run_experiment(config: ExperimentConfig) -> dict[str, float]:
            config.output_dir.mkdir(parents=True, exist_ok=True)
            records = load_dataset(config.dataset_path)
            train_records, eval_records = split_dataset(records)

            model = BaselineModel()
            for _ in range(config.epochs):
                train_one_epoch(model, train_records, config.learning_rate)

            metrics = evaluate(model, eval_records)
            metrics_path = config.output_dir / "metrics.txt"
            metrics_path.write_text("\\n".join(f"{{key}}={{value}}" for key, value in metrics.items()), encoding="utf-8")
            return metrics


        def parse_args() -> ExperimentConfig:
            parser = argparse.ArgumentParser(description="Run the generated research experiment scaffold.")
            parser.add_argument("--dataset-path", type=Path, default=Path("data"))
            parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
            parser.add_argument("--epochs", type=int, default=5)
            parser.add_argument("--batch-size", type=int, default=32)
            parser.add_argument("--learning-rate", type=float, default=1e-3)
            args = parser.parse_args()
            return ExperimentConfig(
                dataset_path=args.dataset_path,
                output_dir=args.output_dir,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
            )


        def main() -> None:
            config = parse_args()
            metrics = run_experiment(config)
            print("Completed experiment for: {topic}")
            for key, value in metrics.items():
                print(f"{{key}}: {{value}}")


        if __name__ == "__main__":
            main()
        '''
    ).strip()


def vectorize_texts(texts: list[str]):
    """Convert a list of texts into TF-IDF vectors.

    If scikit‑learn is not available, returns ``None``.
    """
    if TfidfVectorizer is None:
        logger.warning("scikit-learn is not installed; cannot vectorize texts")
        return None
    vectorizer = TfidfVectorizer(stop_words="english")
    vectors = vectorizer.fit_transform(texts)
    return vectors


def cluster_texts(vectors, n_clusters: int = 5) -> Optional[list[int]]:
    """Cluster the provided vectors using k‑means.

    Returns a list of cluster labels corresponding to each vector, or
    ``None`` if scikit‑learn is not available.
    """
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
    labels = km.fit_predict(vectors)
    return labels.tolist()
