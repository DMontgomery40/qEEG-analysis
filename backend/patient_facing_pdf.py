"""
Beautiful patient-facing PDF renderer using HTML/CSS and WeasyPrint.

This creates actual beautiful documents, not corporate garbage.
"""
from __future__ import annotations

import base64
import re
from datetime import datetime
from pathlib import Path

import markdown
from weasyprint import HTML, CSS

# Markdown extensions for tables
MD_EXTENSIONS = ["tables", "smarty"]

# Logo path - use the repo assets directory
LOGO_PATH = Path(__file__).parent.parent / "assets" / "neuro-luminance-logo.png"


def render_patient_facing_markdown_to_pdf(
    markdown_text: str,
    output_path: Path,
    *,
    title: str = "Your qEEG Brain Assessment",
    patient_label: str | None = None,
) -> None:
    """
    Render patient-friendly markdown to a beautifully designed PDF using HTML/CSS.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert markdown to HTML
    md = markdown.Markdown(extensions=MD_EXTENSIONS)
    content_html = md.convert(markdown_text or "")

    # Build full HTML document
    html = _build_html_document(
        content_html=content_html,
        title=title,
        patient_label=patient_label,
    )

    # Render to PDF
    HTML(string=html).write_pdf(str(output_path), stylesheets=[CSS(string=_get_css())])


def _get_logo_base64() -> str:
    """Get the logo as a base64 data URI."""
    if LOGO_PATH.exists():
        with open(LOGO_PATH, "rb") as f:
            data = base64.b64encode(f.read()).decode("utf-8")
        return f"data:image/png;base64,{data}"
    return ""


def _build_html_document(
    content_html: str,
    title: str,
    patient_label: str | None,
) -> str:
    """Build the full HTML document."""
    logo_uri = _get_logo_base64()

    # Insert appendix cover page before the Technical Appendix section
    appendix_cover = f"""
    <div class="appendix-cover">
        <div class="appendix-cover-top"></div>
        <div class="appendix-cover-middle">
            <h1 class="appendix-cover-title">Technical<br>Appendix</h1>
            <p class="appendix-cover-subtitle">Detailed Clinical Data</p>
            <p class="appendix-cover-note">
                For specialist consultations and medical records
            </p>
        </div>
        <div class="appendix-cover-bottom">
            {f'<img class="appendix-cover-logo" src="{logo_uri}" alt="">' if logo_uri else ''}
        </div>
    </div>
    """

    # Replace the Technical Appendix h1 with cover + h1
    content_html = re.sub(
        r'<h1>Technical Appendix</h1>',
        appendix_cover + '<h1 class="appendix-h1">Technical Appendix</h1>',
        content_html,
        flags=re.IGNORECASE
    )
    content_html = re.sub(
        r'<h1>\s*Technical\s+Appendix\s*</h1>',
        appendix_cover + '<h1 class="appendix-h1">Technical Appendix</h1>',
        content_html,
        flags=re.IGNORECASE
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
</head>
<body>
    <div class="cover-page">
        <div class="cover-top"></div>
        <div class="cover-middle">
            <h1 class="cover-title">Your qEEG<br>Brain Assessment</h1>
            <p class="cover-subtitle">Summary &amp; Findings</p>
        </div>
        <div class="cover-bottom">
            {f'<img class="cover-logo" src="{logo_uri}" alt="">' if logo_uri else ''}
        </div>
    </div>

    <div class="content">
        {content_html}
    </div>
</body>
</html>"""


def _get_css() -> str:
    """Return the CSS for beautiful PDF rendering."""
    logo_uri = _get_logo_base64()
    
    # Build watermark CSS if logo exists
    watermark_css = ""
    if logo_uri:
        watermark_css = f"""
    .content::after {{
        content: "";
        position: fixed;
        bottom: 0.4in;
        left: 50%;
        transform: translateX(-50%);
        width: 140px;
        height: 50px;
        background-image: url("{logo_uri}");
        background-size: contain;
        background-repeat: no-repeat;
        background-position: center;
        opacity: 0.18;
        z-index: -1;
        pointer-events: none;
    }}
"""
    
    return """
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:ital,wght@0,400;0,600;0,700;1,400&family=Inter:wght@400;500;600;700&display=swap');

@page {
    size: letter;
    margin: 0.9in 1in 1in 1in;
    @bottom-right {
        content: "Page " counter(page);
        font-family: 'Inter', -apple-system, sans-serif;
        font-size: 9pt;
        color: #6b7280;
    }
}

@page :first {
    margin: 0;
    @bottom-right {
        content: none;
    }
}

* {
    box-sizing: border-box;
}
""" + watermark_css + """
body {
    font-family: 'Source Serif 4', Georgia, 'Times New Roman', serif;
    font-size: 11pt;
    line-height: 1.6;
    color: #1a202c;
    margin: 0;
    padding: 0;
}

/* ==================== COVER PAGE ==================== */

.cover-page {
    page: cover;
    page-break-after: always;
    height: 100vh;
    width: 100vw;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    align-items: center;
    text-align: center;
    background: #ffffff;
    padding: 1.5in 1in;
}

.cover-top {
    height: 60px;
}

.cover-middle {
    display: flex;
    flex-direction: column;
    align-items: center;
}

.cover-title {
    font-family: 'Inter', -apple-system, sans-serif;
    font-size: 38pt;
    font-weight: 700;
    color: #0f172a;
    margin: 0 0 16px 0;
    line-height: 1.15;
    letter-spacing: -0.02em;
    border-bottom: 3px solid #1e3a5f;
    padding-bottom: 16px;
}

.cover-subtitle {
    font-family: 'Inter', -apple-system, sans-serif;
    font-size: 14pt;
    font-weight: 400;
    color: #64748b;
    margin: 0;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

.cover-bottom {
    display: flex;
    justify-content: center;
    align-items: center;
    padding-bottom: 40px;
}

.cover-logo {
    width: 380px;
    height: auto;
}

.appendix-cover-logo {
    width: 280px;
    height: auto;
    margin-top: 40px;
}

/* ==================== MAIN CONTENT ==================== */

.content {
    padding-top: 0;
}

/* Headings */
h1 {
    font-family: 'Inter', -apple-system, sans-serif;
    font-size: 22pt;
    font-weight: 700;
    color: #0f172a;
    margin: 0 0 8px 0;
    padding-bottom: 12px;
    border-bottom: 3px solid #1e3a5f;
    page-break-after: avoid;
}

h2 {
    font-family: 'Inter', -apple-system, sans-serif;
    font-size: 14pt;
    font-weight: 600;
    color: #1e3a5f;
    margin: 28px 0 12px 0;
    page-break-after: avoid;
}

h3 {
    font-family: 'Inter', -apple-system, sans-serif;
    font-size: 12pt;
    font-weight: 600;
    color: #334155;
    margin: 20px 0 8px 0;
    page-break-after: avoid;
}

/* Paragraphs */
p {
    margin: 0 0 14px 0;
    text-align: left;
}

/* Lead paragraph (first paragraph after h1) */
h1 + p {
    font-size: 12pt;
    line-height: 1.7;
    color: #1e293b;
}

/* Lists */
ul, ol {
    margin: 0 0 16px 0;
    padding-left: 24px;
}

li {
    margin-bottom: 8px;
    line-height: 1.5;
}

li strong {
    color: #0f172a;
}

/* Tables */
table {
    width: 100%;
    border-collapse: collapse;
    margin: 16px 0 20px 0;
    font-size: 10pt;
    page-break-inside: avoid;
}

thead {
    background: linear-gradient(180deg, #f1f5f9 0%, #e2e8f0 100%);
}

th {
    font-family: 'Inter', -apple-system, sans-serif;
    font-weight: 600;
    text-align: left;
    padding: 10px 12px;
    color: #1e293b;
    border-bottom: 2px solid #1e3a5f;
    font-size: 9.5pt;
}

td {
    padding: 9px 12px;
    border-bottom: 1px solid #e2e8f0;
    vertical-align: top;
}

tr:nth-child(even) {
    background-color: #f8fafc;
}

/* Horizontal rules - section breaks */
hr {
    border: none;
    border-top: 1px solid #e2e8f0;
    margin: 32px 0;
}

/* Strong/emphasis */
strong {
    font-weight: 600;
    color: #0f172a;
}

em {
    font-style: italic;
    color: #475569;
}

/* Technical Appendix styling */
h1:contains("Appendix"), h1:contains("Technical") {
    font-size: 18pt;
    color: #475569;
    border-bottom-color: #94a3b8;
    margin-top: 40px;
}

/* Page breaks */
h1 {
    page-break-before: auto;
}

h2, h3 {
    page-break-after: avoid;
}

table, ul, ol {
    page-break-inside: avoid;
}

/* Avoid orphans/widows */
p {
    orphans: 3;
    widows: 3;
}

/* ==================== APPENDIX COVER PAGE ==================== */

.appendix-cover {
    page-break-before: always;
    page-break-after: always;
    height: 100vh;
    width: 100vw;
    margin: -0.9in -1in;
    padding: 1.5in 1in;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    align-items: center;
    text-align: center;
    background: #f8fafc;
}

.appendix-cover-top {
    height: 60px;
}

.appendix-cover-middle {
    display: flex;
    flex-direction: column;
    align-items: center;
}

.appendix-cover-title {
    font-family: 'Inter', -apple-system, sans-serif;
    font-size: 30pt;
    font-weight: 600;
    color: #475569;
    margin: 0 0 12px 0;
    line-height: 1.15;
    letter-spacing: -0.01em;
    border: none;
    padding: 0;
}

.appendix-cover-subtitle {
    font-family: 'Inter', -apple-system, sans-serif;
    font-size: 12pt;
    font-weight: 400;
    color: #64748b;
    margin: 0 0 20px 0;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

.appendix-cover-note {
    font-family: 'Source Serif 4', Georgia, serif;
    font-size: 11pt;
    color: #94a3b8;
    font-style: italic;
    margin: 0;
}

.appendix-cover-bottom {
    display: flex;
    justify-content: center;
    align-items: center;
    padding-bottom: 40px;
}

/* ==================== APPENDIX CONTENT STYLING ==================== */

.appendix-h1 {
    display: none; /* Hide the duplicate h1 since we have a cover */
}

/* Detect appendix sections and style them differently */
.appendix-cover ~ h2,
.appendix-cover ~ h3,
.appendix-cover ~ p,
.appendix-cover ~ ul,
.appendix-cover ~ ol,
.appendix-cover ~ table {
    /* These will be styled more compactly */
}

/* Appendix tables - tighter, more clinical */
.appendix-cover ~ table,
h1.appendix-h1 ~ table,
h1.appendix-h1 ~ * table {
    font-size: 8.5pt;
    margin: 10px 0 14px 0;
}

.appendix-cover ~ table th,
h1.appendix-h1 ~ table th,
h1.appendix-h1 ~ * table th {
    padding: 6px 8px;
    font-size: 8pt;
    background: #e2e8f0;
}

.appendix-cover ~ table td,
h1.appendix-h1 ~ table td,
h1.appendix-h1 ~ * table td {
    padding: 5px 8px;
    font-size: 8.5pt;
}

/* Appendix headings - smaller */
.appendix-cover ~ h2,
h1.appendix-h1 ~ h2 {
    font-size: 12pt;
    margin: 20px 0 8px 0;
    color: #475569;
}

.appendix-cover ~ h3,
h1.appendix-h1 ~ h3 {
    font-size: 10pt;
    margin: 14px 0 6px 0;
}

/* Appendix paragraphs - smaller, tighter */
.appendix-cover ~ p,
h1.appendix-h1 ~ p {
    font-size: 9.5pt;
    line-height: 1.5;
    margin-bottom: 10px;
}

/* Appendix lists - tighter */
.appendix-cover ~ ul,
.appendix-cover ~ ol,
h1.appendix-h1 ~ ul,
h1.appendix-h1 ~ ol {
    font-size: 9.5pt;
    margin: 8px 0 12px 0;
    padding-left: 20px;
}

.appendix-cover ~ li,
h1.appendix-h1 ~ li {
    margin-bottom: 4px;
    line-height: 1.4;
}

/* Italic subtitle in appendix */
.appendix-cover ~ p em:first-child,
h1.appendix-h1 + p em {
    display: none; /* Hide the "For patients who wish to share..." since it's on the cover now */
}
"""
