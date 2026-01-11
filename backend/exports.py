from __future__ import annotations

from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer


def render_markdown_to_pdf(markdown_text: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    styles = getSampleStyleSheet()
    normal = styles["BodyText"]
    h1 = ParagraphStyle("H1", parent=styles["Heading1"], spaceAfter=10)
    h2 = ParagraphStyle("H2", parent=styles["Heading2"], spaceAfter=8)
    h3 = ParagraphStyle("H3", parent=styles["Heading3"], spaceAfter=6)

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54,
        title="qEEG Council Final Report",
    )

    story: list = []
    for raw_line in markdown_text.splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            story.append(Spacer(1, 10))
            continue

        if line.startswith("### "):
            story.append(Paragraph(_escape(line[4:]), h3))
            continue
        if line.startswith("## "):
            story.append(Paragraph(_escape(line[3:]), h2))
            continue
        if line.startswith("# "):
            story.append(Paragraph(_escape(line[2:]), h1))
            continue

        story.append(Paragraph(_escape(line), normal))

    if not story:
        story = [Paragraph("(empty)", normal)]

    doc.build(story)


def _escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\n", "<br/>")
    )

