"""CAJAL tribunal module — multi-judge LLM scoring for generated papers."""

import json
from typing import Dict, List, Optional, Tuple

_MAX_PAPER_TEXT_LENGTH = 6000  # cap paper text sent to each judge to avoid token overflow

DIMENSIONS = [
    "Novelty",
    "Methodological Soundness",
    "Citation Quality",
    "Argument Strength",
    "Reproducibility",
    "Clarity & Precision",
    "Technical Depth",
    "Overall Publishability",
]

_JUDGE_PROMPT_TEMPLATE = """You are a rigorous peer reviewer for a top-tier scientific journal.

Evaluate the following research paper excerpt on each dimension below, assigning a score from 1 (very poor) to 10 (excellent).
Return your evaluation as a valid JSON object with exactly these keys:
{dimensions}
and an additional "comments" key with a brief string of improvement suggestions.

Only return valid JSON. Do not include markdown code fences.

=== PAPER ===
{paper_text}
=== END PAPER ===
"""


def _build_judge_prompt(paper_text: str) -> str:
    dims_json = ", ".join(f'"{d}"' for d in DIMENSIONS)
    return _JUDGE_PROMPT_TEMPLATE.format(
        dimensions=dims_json,
        paper_text=paper_text[:_MAX_PAPER_TEXT_LENGTH],
    )


def _parse_scores(raw: str) -> Optional[Dict]:
    """Extract a JSON object from the LLM response."""
    raw = raw.strip()
    # Strip markdown code fences if present
    raw = raw.lstrip("```json").lstrip("```").rstrip("```").strip()
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return None


def _ollama_judge(paper_text: str, host: str, model: str) -> Optional[Dict]:
    """Run one judge via Ollama and return parsed scores."""
    import requests  # lazy import

    prompt = _build_judge_prompt(paper_text)
    try:
        response = requests.post(
            f"{host}/api/chat",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {"temperature": 0.3, "num_ctx": 8192},
            },
            timeout=120,
        )
        response.raise_for_status()
        content = response.json()["message"]["content"]
        return _parse_scores(content)
    except requests.RequestException as exc:
        logger.warning("Tribunal: Ollama request failed: %s", exc)
        return None
    except (KeyError, ValueError) as exc:
        logger.warning("Tribunal: unexpected response format: %s", exc)
        return None


class Tribunal:
    """
    Multi-judge LLM scoring system for CAJAL-generated papers.

    Uses *num_judges* independent LLM calls and averages the scores.

    Parameters
    ----------
    host : str
        Ollama API base URL.
    model : str
        Ollama model name.
    num_judges : int
        Number of independent judges (default 3).  Increasing to 8–10
        gives more stable scores at the cost of latency.
    """

    def __init__(
        self,
        host: str = "http://localhost:11434",
        model: str = "cajal",
        num_judges: int = 3,
    ):
        self.host = host
        self.model = model
        self.num_judges = num_judges

    def score(self, paper_text: str) -> Dict:
        """
        Score *paper_text* with *num_judges* LLM judges.

        Returns
        -------
        dict with keys:
            scores      — dimension -> average score (float)
            overall     — mean of all dimension averages (float)
            comments    — list of improvement suggestions from each judge
            raw_results — list of per-judge result dicts
        """
        raw_results: List[Dict] = []
        for _ in range(self.num_judges):
            result = _ollama_judge(paper_text, self.host, self.model)
            if result:
                raw_results.append(result)

        if not raw_results:
            return {
                "scores": {d: 0.0 for d in DIMENSIONS},
                "overall": 0.0,
                "comments": ["Tribunal could not obtain scores (check Ollama connection)."],
                "raw_results": [],
            }

        # Average numeric scores across judges
        aggregated: Dict[str, List[float]] = {d: [] for d in DIMENSIONS}
        comments: List[str] = []

        for result in raw_results:
            for dim in DIMENSIONS:
                try:
                    val = float(result.get(dim, 0))
                    aggregated[dim].append(max(0.0, min(10.0, val)))
                except (TypeError, ValueError):
                    pass
            comment = result.get("comments", "")
            if comment:
                comments.append(str(comment).strip())

        avg_scores = {
            d: round(sum(vals) / len(vals), 2) if vals else 0.0
            for d, vals in aggregated.items()
        }
        all_scores = [v for v in avg_scores.values() if v > 0]
        overall = round(sum(all_scores) / len(all_scores), 2) if all_scores else 0.0

        return {
            "scores": avg_scores,
            "overall": overall,
            "comments": comments,
            "raw_results": raw_results,
        }

    def format_report(self, result: Dict) -> str:
        """Format a tribunal result dict as a human-readable string."""
        lines = ["=== CAJAL Tribunal Evaluation ===", ""]
        for dim, score in result.get("scores", {}).items():
            bar = "█" * int(score) + "░" * (10 - int(score))
            lines.append(f"  {dim:<30} {bar}  {score:.1f}/10")
        lines.append("")
        lines.append(f"  Overall Score: {result.get('overall', 0):.1f} / 10")
        lines.append("")
        comments = result.get("comments", [])
        if comments:
            lines.append("Improvement Suggestions:")
            for i, c in enumerate(comments, 1):
                lines.append(f"  {i}. {c}")
        return "\n".join(lines)
