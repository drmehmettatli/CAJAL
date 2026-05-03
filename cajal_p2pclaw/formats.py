"""CAJAL formats module — export papers to Markdown, LaTeX, and plain text."""

import re
from typing import Dict, List, Optional


def to_markdown(paper: Dict) -> str:
    """
    Convert a structured paper dict to Markdown string.

    Expected paper keys (all optional except *topic*):
        topic, abstract, introduction, related_work, methodology,
        results, discussion, conclusion, references (list of str)
    """
    lines: List[str] = []

    topic = paper.get("topic", "Untitled")
    lines.append(f"# {topic}\n")

    def _section(heading: str, key: str) -> None:
        content = paper.get(key, "")
        if content:
            lines.append(f"## {heading}\n")
            lines.append(content.strip())
            lines.append("")

    _section("Abstract", "abstract")
    _section("1. Introduction", "introduction")
    _section("2. Related Work", "related_work")
    _section("3. Methodology", "methodology")
    _section("4. Results", "results")
    _section("5. Discussion", "discussion")
    _section("6. Conclusion", "conclusion")

    references: List[str] = paper.get("references", [])
    if references:
        lines.append("## References\n")
        lines.extend(references)
        lines.append("")

    return "\n".join(lines)


def to_latex(paper: Dict) -> str:
    """
    Convert a structured paper dict to a minimal LaTeX document.
    """

    def _escape(text: str) -> str:
        """Escape common LaTeX special characters."""
        replacements = [
            ("\\", "\\textbackslash{}"),
            ("&", "\\&"),
            ("%", "\\%"),
            ("$", "\\$"),
            ("#", "\\#"),
            ("_", "\\_"),
            ("{", "\\{"),
            ("}", "\\}"),
            ("~", "\\textasciitilde{}"),
            ("^", "\\textasciicircum{}"),
        ]
        for old, new in replacements:
            text = text.replace(old, new)
        return text

    def _section(heading: str, key: str) -> str:
        content = paper.get(key, "")
        if not content:
            return ""
        return f"\\section{{{heading}}}\n{_escape(content.strip())}\n\n"

    topic = _escape(paper.get("topic", "Untitled"))

    preamble = (
        "\\documentclass[12pt]{article}\n"
        "\\usepackage[utf8]{inputenc}\n"
        "\\usepackage{hyperref}\n"
        "\\usepackage{geometry}\n"
        "\\geometry{margin=1in}\n"
        f"\\title{{{topic}}}\n"
        "\\author{CAJAL — P2PCLAW Research}\n"
        "\\date{\\today}\n"
        "\\begin{document}\n"
        "\\maketitle\n\n"
    )

    body = ""
    abstract = paper.get("abstract", "")
    if abstract:
        body += f"\\begin{{abstract}}\n{_escape(abstract.strip())}\n\\end{{abstract}}\n\n"

    body += _section("Introduction", "introduction")
    body += _section("Related Work", "related_work")
    body += _section("Methodology", "methodology")
    body += _section("Results", "results")
    body += _section("Discussion", "discussion")
    body += _section("Conclusion", "conclusion")

    references: List[str] = paper.get("references", [])
    if references:
        body += "\\section*{References}\n\\begin{enumerate}\n"
        for ref in references:
            clean = re.sub(r"^\[\d+\]\s*", "", ref)
            body += f"  \\item {_escape(clean)}\n"
        body += "\\end{enumerate}\n\n"

    postamble = "\\end{document}\n"

    return preamble + body + postamble


def to_text(paper: Dict) -> str:
    """
    Convert a structured paper dict to plain text (strips Markdown markers).
    """
    md = to_markdown(paper)
    # Remove Markdown headings
    text = re.sub(r"^#{1,6}\s*", "", md, flags=re.MULTILINE)
    # Remove bold/italic markers
    text = re.sub(r"\*{1,3}(.*?)\*{1,3}", r"\1", text)
    # Remove link syntax [text](url) -> text
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    return text


def save_paper(paper: Dict, path: str, fmt: str = "markdown") -> str:
    """
    Save *paper* to *path* in the requested *fmt* (markdown | latex | text).

    Returns the path written to.
    """
    fmt = fmt.lower()
    if fmt in ("markdown", "md"):
        content = to_markdown(paper)
    elif fmt in ("latex", "tex"):
        content = to_latex(paper)
    else:
        content = to_text(paper)

    with open(path, "w", encoding="utf-8") as fh:
        fh.write(content)
    return path
