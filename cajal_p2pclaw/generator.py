"""CAJAL generator module — core academic paper generation engine."""

import json
import re
from typing import Dict, List, Optional
import requests

from .citations import find_references, format_reference
from .formats import to_markdown, to_latex, save as save_paper

_DEFAULT_HOST = "http://localhost:11434"
_DEFAULT_MODEL = "cajal"

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are CAJAL, an expert scientific writer at the P2PCLAW Research Network. "
    "You produce rigorous, technically precise academic papers with clear structure, "
    "sound methodology, and well-reasoned arguments. "
    "Write in formal academic English. Never hallucinate references or data."
)

# ---------------------------------------------------------------------------
# Section prompts
# ---------------------------------------------------------------------------

_ABSTRACT_PROMPT = (
    "Write an academic abstract (150–250 words) for a paper on: {topic}\n\n"
    "Cover: background, methods, key results, and conclusion. "
    "Return only the abstract text — no heading."
)

_INTRODUCTION_PROMPT = (
    "Write a detailed Introduction section for an academic paper on: {topic}\n\n"
    "Cover: context, problem statement, objectives, and significance. "
    "Return only the section body — no heading."
)

_RELATED_WORK_PROMPT = (
    "Write a Related Work section for a paper on: {topic}\n\n"
    "Discuss 3–5 relevant research directions and how they relate to the paper. "
    "Use placeholder citation markers like [1], [2], etc. "
    "Return only the section body — no heading."
)

_METHODOLOGY_PROMPT = (
    "Write a detailed Methodology section for a paper on: {topic}\n\n"
    "Provide reproducible, step-by-step procedures including any algorithms, "
    "data collection strategies, or experimental setups. "
    "Return only the section body — no heading."
)

_RESULTS_PROMPT = (
    "Write a Results section for a paper on: {topic}\n\n"
    "Present data-driven findings, use tables or figures described in text, "
    "and highlight the most significant outcomes. "
    "Return only the section body — no heading."
)

_DISCUSSION_PROMPT = (
    "Write a Discussion section for a paper on: {topic}\n\n"
    "Interpret the results, address limitations, compare with related work, "
    "and outline future research directions. "
    "Return only the section body — no heading."
)

_CONCLUSION_PROMPT = (
    "Write a Conclusion section for a paper on: {topic}\n\n"
    "Summarise the key contributions and their implications. "
    "Return only the section body — no heading."
)

_REVIEW_PROMPT = (
    "You are a rigorous peer reviewer. Review the following paper draft and provide "
    "specific, actionable feedback covering: structure, clarity, methodology, "
    "citation quality, and argument strength.\n\n"
    "=== DRAFT ===\n{draft}\n=== END DRAFT ==="
)


# ---------------------------------------------------------------------------
# Low-level Ollama helper
# ---------------------------------------------------------------------------

def _ollama_chat(
    prompt: str,
    host: str,
    model: str,
    system: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 4096,
) -> str:
    """Send a single chat request to Ollama and return the response text."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    response = requests.post(
        f"{host}/api/chat",
        json={
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature, "num_ctx": max_tokens},
        },
        timeout=300,
    )
    response.raise_for_status()
    return response.json()["message"]["content"]


# ---------------------------------------------------------------------------
# PaperGenerator
# ---------------------------------------------------------------------------

class PaperGenerator:
    """
    Generate full publication-ready scientific papers using a local LLM via Ollama.

    Parameters
    ----------
    model : str
        Ollama model name (default ``"cajal"``).
    host : str
        Ollama API base URL (default ``"http://localhost:11434"``).
    temperature : float
        Sampling temperature for generation (default 0.7).
    """

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        host: str = _DEFAULT_HOST,
        temperature: float = 0.7,
    ):
        self.model = model
        self.host = host
        self.temperature = temperature

    # ------------------------------------------------------------------
    # Section generators
    # ------------------------------------------------------------------

    def generate_abstract(self, topic: str) -> str:
        """Generate a 150–250 word abstract for *topic*."""
        prompt = _ABSTRACT_PROMPT.format(topic=topic)
        return _ollama_chat(
            prompt, self.host, self.model,
            system=_SYSTEM_PROMPT,
            temperature=self.temperature,
        ).strip()

    def generate_introduction(self, topic: str) -> str:
        """Generate an Introduction section for *topic*."""
        prompt = _INTRODUCTION_PROMPT.format(topic=topic)
        return _ollama_chat(
            prompt, self.host, self.model,
            system=_SYSTEM_PROMPT,
            temperature=self.temperature,
        ).strip()

    def generate_related_work(self, topic: str) -> str:
        """Generate a Related Work section for *topic*."""
        prompt = _RELATED_WORK_PROMPT.format(topic=topic)
        return _ollama_chat(
            prompt, self.host, self.model,
            system=_SYSTEM_PROMPT,
            temperature=self.temperature,
        ).strip()

    def generate_methods(self, topic: str) -> str:
        """Generate a Methodology section for *topic*."""
        prompt = _METHODOLOGY_PROMPT.format(topic=topic)
        return _ollama_chat(
            prompt, self.host, self.model,
            system=_SYSTEM_PROMPT,
            temperature=self.temperature,
        ).strip()

    def generate_results(self, topic: str) -> str:
        """Generate a Results section for *topic*."""
        prompt = _RESULTS_PROMPT.format(topic=topic)
        return _ollama_chat(
            prompt, self.host, self.model,
            system=_SYSTEM_PROMPT,
            temperature=self.temperature,
        ).strip()

    def generate_discussion(self, topic: str) -> str:
        """Generate a Discussion section for *topic*."""
        prompt = _DISCUSSION_PROMPT.format(topic=topic)
        return _ollama_chat(
            prompt, self.host, self.model,
            system=_SYSTEM_PROMPT,
            temperature=self.temperature,
        ).strip()

    def generate_conclusion(self, topic: str) -> str:
        """Generate a Conclusion section for *topic*."""
        prompt = _CONCLUSION_PROMPT.format(topic=topic)
        return _ollama_chat(
            prompt, self.host, self.model,
            system=_SYSTEM_PROMPT,
            temperature=self.temperature,
        ).strip()

    def find_references(self, topic: str, count: int = 8) -> List[str]:
        """
        Fetch *count* real references for *topic* from arXiv / CrossRef.

        Returns a list of formatted citation strings.
        """
        refs = find_references(topic, count)
        return [format_reference(r, i + 1) for i, r in enumerate(refs)]

    # ------------------------------------------------------------------
    # Full-paper generator
    # ------------------------------------------------------------------

    def generate(
        self,
        topic: str,
        fmt: str = "markdown",
        min_references: int = 8,
        output_path: Optional[str] = None,
    ) -> str:
        """
        Generate a complete academic paper on *topic*.

        Parameters
        ----------
        topic : str
            Research topic / title.
        fmt : str
            Output format: ``"markdown"`` (default), ``"latex"``, or ``"text"``.
        min_references : int
            Minimum number of real references to include.
        output_path : str, optional
            If provided, save the paper to this file path.

        Returns
        -------
        str
            The generated paper in the requested format.
        """
        print(f"[CAJAL] Generating paper: {topic!r}")

        print("[CAJAL]  → abstract …")
        abstract = self.generate_abstract(topic)

        print("[CAJAL]  → introduction …")
        introduction = self.generate_introduction(topic)

        print("[CAJAL]  → related work …")
        related_work = self.generate_related_work(topic)

        print("[CAJAL]  → methodology …")
        methodology = self.generate_methods(topic)

        print("[CAJAL]  → results …")
        results = self.generate_results(topic)

        print("[CAJAL]  → discussion …")
        discussion = self.generate_discussion(topic)

        print("[CAJAL]  → conclusion …")
        conclusion = self.generate_conclusion(topic)

        print(f"[CAJAL]  → fetching {min_references} references …")
        references = self.find_references(topic, min_references)

        paper: Dict = {
            "topic": topic,
            "abstract": abstract,
            "introduction": introduction,
            "related_work": related_work,
            "methodology": methodology,
            "results": results,
            "discussion": discussion,
            "conclusion": conclusion,
            "references": references,
        }

        if output_path:
            save_paper(paper, output_path, fmt)
            print(f"[CAJAL] Paper saved to: {output_path}")

        if fmt in ("latex", "tex"):
            return to_latex(paper)
        return to_markdown(paper)

    # ------------------------------------------------------------------
    # Review helper
    # ------------------------------------------------------------------

    def review(self, draft: str) -> str:
        """
        Review an existing paper *draft* and return structured feedback.

        Parameters
        ----------
        draft : str
            The paper text (Markdown or plain text).

        Returns
        -------
        str
            Peer-review feedback from the LLM.
        """
        prompt = _REVIEW_PROMPT.format(draft=draft[:8000])
        return _ollama_chat(
            prompt, self.host, self.model,
            system=_SYSTEM_PROMPT,
            temperature=0.4,
            max_tokens=4096,
        ).strip()
