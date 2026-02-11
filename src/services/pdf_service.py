"""
PDF Generation Service (v2 — WeasyPrint + Mermaid)

Generates enterprise-grade PDF documents for PRD and QA Test Plans.

Uses WeasyPrint (HTML/CSS → PDF) for professional output with:
    - Full CSS styling and typography
    - Native SVG/PNG diagram embedding
    - Mermaid diagram rendering (flowcharts, sequences, ER diagrams)
    - Auto-generated architecture & flow diagrams from structured data
    - Code syntax highlighting (via Pygments)
    - Professional headers/footers via @page CSS

System Requirements:
    - WeasyPrint: pip install weasyprint
    - System deps (cairo, pango, gdk-pixbuf):
        macOS:  brew install cairo pango gdk-pixbuf libffi
        Ubuntu: apt install libcairo2 libpango-1.0-0 libgdk-pixbuf2.0-0
    - Mermaid CLI (optional, for diagrams):
        npm install -g @mermaid-js/mermaid-cli
"""

import base64
import html as html_mod
import json
import os
import re
import shutil
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from weasyprint import HTML

from src.core.logger import get_logger

# Optional: Pygments for syntax highlighting
try:
    from pygments import highlight as pygments_highlight
    from pygments.lexers import get_lexer_by_name, guess_lexer, TextLexer
    from pygments.formatters import HtmlFormatter

    HAS_PYGMENTS = True
except ImportError:
    HAS_PYGMENTS = False

logger = get_logger(__name__)

# Output directory for generated PDFs
PDF_OUTPUT_DIR = Path(__file__).parent.parent.parent / "generated_pdfs"
PDF_OUTPUT_DIR.mkdir(exist_ok=True)

# Mermaid CLI cache
_MERMAID_CLI_PATH: Optional[str] = None
_MERMAID_CLI_CHECKED = False


# ===========================================================================
# Mermaid Rendering Utilities
# ===========================================================================

def _find_mermaid_cli() -> Optional[str]:
    """Find the mermaid CLI (mmdc) binary."""
    global _MERMAID_CLI_PATH, _MERMAID_CLI_CHECKED
    if _MERMAID_CLI_CHECKED:
        return _MERMAID_CLI_PATH
    _MERMAID_CLI_CHECKED = True

    mmdc_path = shutil.which("mmdc")
    if mmdc_path:
        _MERMAID_CLI_PATH = mmdc_path
        return mmdc_path

    npx_path = shutil.which("npx")
    if npx_path:
        _MERMAID_CLI_PATH = "npx"
        return "npx"

    logger.warning(
        "Mermaid CLI (mmdc) not found — diagrams will render as code blocks. "
        "Install with: npm install -g @mermaid-js/mermaid-cli"
    )
    return None


def _get_puppeteer_config_args() -> List[str]:
    """
    Return extra mmdc CLI args for Puppeteer config if running in Docker.
    The Dockerfile sets PUPPETEER_CONFIG_PATH to a JSON file with
    --no-sandbox flags needed for Chromium inside containers.
    """
    config_path = os.environ.get("PUPPETEER_CONFIG_PATH", "")
    if config_path and os.path.exists(config_path):
        return ["-p", config_path]
    return []


def _render_mermaid_to_svg(mermaid_code: str) -> Optional[str]:
    """Render Mermaid diagram code to an SVG string. Returns None on failure."""
    cli = _find_mermaid_cli()
    if not cli:
        return None

    mmd_path = svg_path = None
    try:
        with tempfile.NamedTemporaryFile(
            suffix=".mmd", mode="w", delete=False
        ) as fh:
            fh.write(mermaid_code.strip())
            mmd_path = fh.name

        svg_path = mmd_path.replace(".mmd", ".svg")
        puppeteer_args = _get_puppeteer_config_args()
        if cli == "npx":
            cmd = [
                "npx", "-y", "@mermaid-js/mermaid-cli",
                "-i", mmd_path, "-o", svg_path,
                "-t", "neutral", "-b", "transparent", "--quiet",
            ] + puppeteer_args
        else:
            cmd = [
                cli, "-i", mmd_path, "-o", svg_path,
                "-t", "neutral", "-b", "transparent", "--quiet",
            ] + puppeteer_args

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0 and os.path.exists(svg_path):
            with open(svg_path, "r", encoding="utf-8") as f:
                return f.read()
        else:
            logger.warning(
                "Mermaid render failed",
                returncode=result.returncode,
                stderr=(result.stderr or "")[:300],
            )
            return None
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
        logger.warning(f"Mermaid rendering error: {exc}")
        return None
    finally:
        for p in (mmd_path, svg_path):
            if p:
                try:
                    os.unlink(p)
                except OSError:
                    pass


def _render_mermaid_to_base64_png(mermaid_code: str) -> Optional[str]:
    """Render Mermaid to a base64-encoded PNG (fallback for SVG issues)."""
    cli = _find_mermaid_cli()
    if not cli:
        return None

    mmd_path = png_path = None
    try:
        with tempfile.NamedTemporaryFile(
            suffix=".mmd", mode="w", delete=False
        ) as fh:
            fh.write(mermaid_code.strip())
            mmd_path = fh.name

        png_path = mmd_path.replace(".mmd", ".png")
        puppeteer_args = _get_puppeteer_config_args()
        if cli == "npx":
            cmd = [
                "npx", "-y", "@mermaid-js/mermaid-cli",
                "-i", mmd_path, "-o", png_path,
                "-t", "neutral", "-b", "white",
                "-w", "1600", "-H", "1200", "--quiet",
            ] + puppeteer_args
        else:
            cmd = [
                cli, "-i", mmd_path, "-o", png_path,
                "-t", "neutral", "-b", "white",
                "-w", "1600", "-H", "1200", "--quiet",
            ] + puppeteer_args

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0 and os.path.exists(png_path):
            with open(png_path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        return None
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return None
    finally:
        for p in (mmd_path, png_path):
            if p:
                try:
                    os.unlink(p)
                except OSError:
                    pass


# ===========================================================================
# Mermaid Auto-Diagram Generators
# ===========================================================================

def _mermaid_escape(text: str) -> str:
    """Escape text for safe use inside Mermaid diagram labels."""
    return (
        text.replace('"', "'")
        .replace("#", "")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("{", "(")
        .replace("}", ")")
    )


def _generate_architecture_diagram(design: Dict[str, Any]) -> Optional[str]:
    """Auto-generate a Mermaid architecture graph from technical design."""
    files_create = design.get("files_to_create", [])
    files_modify = design.get("files_to_modify", [])
    api_contracts = design.get("api_contracts", [])

    if not files_create and not files_modify and not api_contracts:
        return None

    lines = ["graph TB"]
    nid = 0
    create_ids: List[str] = []
    modify_ids: List[str] = []
    api_ids: List[str] = []

    if files_create:
        lines.append('    subgraph NEW["New Components"]')
        lines.append("    style NEW fill:#d4edda,stroke:#28a745,color:#155724")
        for f in files_create[:10]:
            tag = f"NC{nid}"
            name = Path(f.get("path", f"new_{nid}")).name
            purpose = f.get("purpose", "")[:40]
            label = name + (f"\\n{purpose}" if purpose else "")
            lines.append(f'        {tag}["{_mermaid_escape(label)}"]')
            create_ids.append(tag)
            nid += 1
        lines.append("    end")

    if files_modify:
        lines.append('    subgraph MOD["Modified Components"]')
        lines.append("    style MOD fill:#fff3cd,stroke:#ffc107,color:#856404")
        for f in files_modify[:10]:
            tag = f"MC{nid}"
            name = Path(f.get("path", f"mod_{nid}")).name
            changes = f.get("changes", "")[:40]
            label = name + (f"\\n{changes}" if changes else "")
            lines.append(f'        {tag}["{_mermaid_escape(label)}"]')
            modify_ids.append(tag)
            nid += 1
        lines.append("    end")

    if api_contracts:
        lines.append('    subgraph API["API Endpoints"]')
        lines.append("    style API fill:#cce5ff,stroke:#004085,color:#004085")
        for api in api_contracts[:8]:
            tag = f"AP{nid}"
            method = api.get("method", "")
            path = api.get("path", "")
            lines.append(f'        {tag}["{_mermaid_escape(f"{method} {path}")}"]')
            api_ids.append(tag)
            nid += 1
        lines.append("    end")

    # Edges
    if create_ids and api_ids:
        for i, cid in enumerate(create_ids):
            if i < len(api_ids):
                lines.append(f"    {cid} --> {api_ids[i]}")
    if modify_ids and api_ids:
        for i, mid in enumerate(modify_ids):
            lines.append(f"    {mid} -.-> {api_ids[min(i, len(api_ids) - 1)]}")

    return "\n".join(lines) if nid > 0 else None


def _generate_api_sequence_diagram(api_contracts: List[Dict]) -> Optional[str]:
    """Auto-generate a Mermaid sequence diagram from API contracts."""
    if not api_contracts:
        return None

    lines = [
        "sequenceDiagram",
        "    participant C as Client",
        "    participant API as API Server",
        "    participant S as Service Layer",
        "    participant DB as Database",
    ]

    for idx, api in enumerate(api_contracts[:6]):
        method = api.get("method", "GET")
        path = api.get("path", "/")
        lines.append(f"    C->>API: {_mermaid_escape(f'{method} {path}')}")
        lines.append("    API->>S: Process request")
        lines.append("    S->>DB: Query / Update")
        lines.append("    DB-->>S: Result")
        lines.append("    S-->>API: Response")
        lines.append("    API-->>C: 200 OK")
        if idx < len(api_contracts[:6]) - 1:
            lines.append("    Note over C,DB: ---")

    return "\n".join(lines) if len(lines) > 5 else None


def _generate_test_distribution_diagram(qa_data: Dict[str, Any]) -> Optional[str]:
    """Auto-generate a Mermaid pie chart of test-type distribution."""
    type_keys = {
        "Unit Tests": [
            "unit_testing", "unit_testing_plan", "unit_tests",
        ],
        "Integration Tests": [
            "integration_testing", "integration_testing_plan", "integration_tests",
        ],
        "System Tests": [
            "system_testing", "system_testing_plan", "system_tests", "e2e_testing",
        ],
        "Regression Tests": [
            "regression_testing", "regression_testing_plan", "regression_tests",
        ],
        "Performance Tests": [
            "performance_testing", "performance_testing_plan", "performance_tests",
        ],
        "Security Tests": [
            "security_testing", "security_testing_plan", "security_tests",
        ],
    }
    counts: Dict[str, int] = {}
    for label, keys in type_keys.items():
        for key in keys:
            data = qa_data.get(key)
            if data:
                if isinstance(data, dict):
                    tc = data.get(
                        "test_cases",
                        data.get("cases", data.get("scenarios", [])),
                    )
                    counts[label] = len(tc) if isinstance(tc, list) else 1
                elif isinstance(data, list):
                    total = sum(
                        len(item.get("test_cases", []))
                        for item in data
                        if isinstance(item, dict)
                    )
                    counts[label] = max(total, len(data))
                else:
                    counts[label] = 1
                break

    if len(counts) < 2:
        return None

    lines = ['pie title Test Case Distribution']
    for label, count in counts.items():
        lines.append(f'    "{label}" : {count}')
    return "\n".join(lines)


def _generate_component_diagram(codebase: Dict[str, Any]) -> Optional[str]:
    """Auto-generate a component map from codebase analysis."""
    files = codebase.get("relevant_files", [])
    components = codebase.get("reusable_components", [])
    if not files and not components:
        return None

    lines = ["graph LR"]
    nid = 0

    if files:
        lines.append('    subgraph FILES["Relevant Files"]')
        for f in files[:8]:
            name = Path(f.get("file", f"file_{nid}")).name
            purpose = f.get("purpose", "")[:30]
            label = name + (f"\\n{purpose}" if purpose else "")
            lines.append(f'        F{nid}["{_mermaid_escape(label)}"]')
            nid += 1
        lines.append("    end")

    if components:
        lines.append('    subgraph REUSE["Reusable Components"]')
        for c in components[:6]:
            name = c.get("component", f"comp_{nid}")[:25]
            lines.append(f'        R{nid}["{_mermaid_escape(name)}"]')
            nid += 1
        lines.append("    end")

    return "\n".join(lines) if nid > 0 else None


# ===========================================================================
# CSS Stylesheet
# ===========================================================================

DOCUMENT_CSS = """
/* ===== Page Layout ===== */
@page {
    size: A4;
    margin: 2.5cm 2cm 2.5cm 2cm;
    @top-left {
        content: string(doc-info);
        font-family: 'Helvetica Neue', Arial, sans-serif;
        font-size: 7.5pt;
        color: #999;
    }
    @top-right {
        content: string(doc-date);
        font-family: 'Helvetica Neue', Arial, sans-serif;
        font-size: 7.5pt;
        color: #999;
    }
    @bottom-center {
        content: "Page " counter(page) " of " counter(pages);
        font-family: 'Helvetica Neue', Arial, sans-serif;
        font-size: 7.5pt;
        color: #aaa;
    }
    @bottom-left {
        content: "Code Intelligence Platform";
        font-family: 'Helvetica Neue', Arial, sans-serif;
        font-size: 7pt;
        color: #bbb;
    }
}
@page :first {
    @top-left { content: ""; }
    @top-right { content: ""; }
    @bottom-center { content: ""; }
    @bottom-left { content: ""; }
}

/* ===== Base Typography ===== */
body {
    font-family: 'Helvetica Neue', 'Segoe UI', Arial, sans-serif;
    font-size: 10pt;
    line-height: 1.65;
    color: #1a1a2e;
}

/* Running header values (hidden in body) */
.running-str { height: 0; overflow: hidden; font-size: 0; margin: 0; padding: 0; }
.doc-info  { string-set: doc-info content(); }
.doc-date  { string-set: doc-date content(); }

/* ===== Cover Page ===== */
.cover-page {
    page-break-after: always;
    padding-top: 180px;
    text-align: center;
}
.cover-page .accent-line {
    width: 80px; height: 4px;
    background: #2d2d7f;
    margin: 0 auto 40px auto;
}
.cover-page h1 {
    font-size: 30pt; font-weight: 700; color: #141450;
    margin: 0 0 12px 0; letter-spacing: -0.5px;
}
.cover-page .subtitle {
    font-size: 17pt; color: #505080;
    margin: 0 0 30px 0; font-weight: 400;
}
.cover-page .meta-line {
    font-size: 10.5pt; color: #888; margin: 4px 0;
}
.cover-page .version-badge {
    display: inline-block; background: #e8e8f5; color: #2d2d7f;
    padding: 4px 16px; border-radius: 14px;
    font-size: 10pt; font-weight: 600; margin-top: 12px;
}

/* ===== Table of Contents ===== */
.toc-page { page-break-after: always; }
.toc-page h2 {
    font-size: 18pt; color: #141450;
    border-bottom: 2px solid #2d2d7f;
    padding-bottom: 6px; margin-bottom: 16px;
}
.toc-page ul { list-style: none; padding: 0; margin: 0; }
.toc-page li {
    padding: 6px 0; border-bottom: 1px dotted #d0d0d0; font-size: 10.5pt;
}
.toc-page li a { text-decoration: none; color: #1a1a2e; }
.toc-page .toc-num {
    display: inline-block; width: 30px;
    font-weight: 700; color: #2d2d7f;
}

/* ===== Sections ===== */
h2.section-title {
    font-size: 16pt; font-weight: 700; color: #141450;
    border-bottom: 2.5px solid #2d2d7f;
    padding-bottom: 5px; margin-top: 30px; margin-bottom: 12px;
    page-break-after: avoid;
}
h2.section-title .section-num { color: #5b5bd6; margin-right: 6px; }
h3.sub-section {
    font-size: 12pt; font-weight: 600; color: #28286e;
    margin-top: 18px; margin-bottom: 6px;
    page-break-after: avoid;
}
h3.sub-section .sub-num { color: #6b6bb0; margin-right: 4px; }
h4.sub-sub-section {
    font-size: 10.5pt; font-weight: 600; color: #3a3a80;
    margin-top: 12px; margin-bottom: 4px;
}

/* ===== Body Text ===== */
p.body-text { margin: 5px 0 8px 0; text-align: justify; }

/* ===== Lists ===== */
ul.content-list { padding-left: 22px; margin: 6px 0 10px 0; }
ul.content-list li { margin: 3px 0; line-height: 1.55; }

/* ===== Key-Value ===== */
.kv-pair { margin: 3px 0; line-height: 1.55; }
.kv-label { font-weight: 700; color: #1a1a2e; }
.kv-value  { color: #333; }

/* ===== Tables ===== */
table.data-table {
    width: 100%; border-collapse: collapse;
    margin: 14px 0; font-size: 8.5pt;
    page-break-inside: auto;
}
table.data-table thead th {
    background-color: #28286e; color: #fff;
    padding: 8px 10px; text-align: left;
    font-weight: 600; font-size: 8.5pt; letter-spacing: 0.3px;
}
table.data-table tbody td {
    padding: 7px 10px; border-bottom: 1px solid #e2e2ea;
    vertical-align: top; word-wrap: break-word;
}
table.data-table tbody tr:nth-child(even) { background-color: #f4f4fb; }
table.data-table tbody tr:nth-child(odd)  { background-color: #ffffff; }

/* ===== Code Blocks ===== */
pre.code-block {
    background-color: #f8f8f8; border: 1px solid #e0e0e0;
    border-left: 3px solid #5b5bd6; border-radius: 3px;
    padding: 10px 14px;
    font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
    font-size: 8pt; line-height: 1.5;
    overflow-wrap: break-word; white-space: pre-wrap;
    margin: 10px 0; color: #333;
}
.highlight pre {
    background-color: #f8f8f8; border: 1px solid #e0e0e0;
    border-left: 3px solid #5b5bd6; border-radius: 3px;
    padding: 10px 14px; font-size: 8pt; line-height: 1.5;
    overflow-wrap: break-word; white-space: pre-wrap; margin: 10px 0;
}

/* ===== Diagrams ===== */
.diagram-container {
    text-align: center; margin: 20px 0;
    page-break-inside: avoid;
    padding: 16px; background: #fafafe;
    border: 1px solid #e8e8f0; border-radius: 6px;
}
.diagram-container img, .diagram-container svg { max-width: 100%; height: auto; }
.diagram-caption {
    font-size: 8.5pt; color: #666; font-style: italic;
    margin-top: 8px; text-align: center;
}
.diagram-fallback {
    background-color: #f0f0f5; border: 1px dashed #b0b0c0;
    border-radius: 4px; padding: 12px; margin: 12px 0;
    page-break-inside: avoid;
}
.diagram-fallback .fallback-label {
    font-size: 8pt; color: #888; margin-bottom: 4px; font-style: italic;
}

/* ===== Accent Boxes ===== */
.accent-box {
    border-left: 4px solid #2d2d7f; background-color: #f0f0fa;
    padding: 10px 16px; margin: 12px 0;
    border-radius: 0 4px 4px 0; page-break-inside: avoid;
}
.accent-box.warning { border-left-color: #e6a817; background-color: #fef9e7; }
.accent-box.danger  { border-left-color: #c0392b; background-color: #fdf2f2; }
.accent-box.success { border-left-color: #27ae60; background-color: #eafaf1; }

/* ===== Badges ===== */
.badge {
    display: inline-block; padding: 2px 10px; border-radius: 10px;
    font-size: 7.5pt; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.3px;
}
.badge-high, .badge-critical { background: #fee2e2; color: #991b1b; }
.badge-medium { background: #fef3c7; color: #92400e; }
.badge-low { background: #d1fae5; color: #065f46; }

/* ===== Utilities ===== */
.page-break { page-break-before: always; }
"""


# ===========================================================================
# HTML Document Builder
# ===========================================================================

class HTMLDocumentBuilder:
    """
    Builds an enterprise HTML document and renders it to PDF via WeasyPrint.

    Drop-in replacement for the old EnterprisePDF class, offering richer
    formatting, native diagram embedding, and full CSS control.
    """

    def __init__(self, title: str = "", doc_type: str = "PRD"):
        self.title = title
        self.doc_type = doc_type
        self._parts: List[str] = []
        self._toc: List[Tuple[str, str, str]] = []  # (num, id, label)
        self._figure_counter = 0

    # --- internal helpers ---------------------------------------------------

    def _e(self, text: str) -> str:
        """HTML-escape text."""
        if not text:
            return ""
        return html_mod.escape(str(text))

    def _add(self, fragment: str):
        self._parts.append(fragment)

    # --- structural elements ------------------------------------------------

    def cover_page(self, title: str, subtitle: str = "", version: str = "v1.0"):
        now = datetime.utcnow()
        self._add(f"""
<div class="cover-page">
    <div class="accent-line"></div>
    <h1>{self._e(title)}</h1>
    {"<p class='subtitle'>" + self._e(subtitle) + "</p>" if subtitle else ""}
    <p class="meta-line">Generated: {now.strftime("%B %d, %Y %H:%M UTC")}</p>
    <p class="meta-line">Code Intelligence Platform</p>
    <span class="version-badge">{self._e(version)}</span>
</div>
""")

    def table_of_contents(self):
        """Insert a placeholder; actual TOC is built in render_html()."""
        self._add("<!-- TOC_PLACEHOLDER -->")

    def page_break(self):
        self._add('<div class="page-break"></div>')

    # --- headings -----------------------------------------------------------

    def section_title(self, number: str, title: str):
        anchor = f"sec-{number}"
        self._toc.append((number, anchor, title))
        self._add(
            f'<h2 class="section-title" id="{anchor}">'
            f'<span class="section-num">{self._e(number)}.</span> '
            f"{self._e(title)}</h2>"
        )

    def sub_section(self, number: str, title: str):
        num_html = (
            f'<span class="sub-num">{self._e(number)}</span> ' if number else ""
        )
        self._add(f'<h3 class="sub-section">{num_html}{self._e(title)}</h3>')

    def sub_sub_section(self, title: str):
        self._add(f'<h4 class="sub-sub-section">{self._e(title)}</h4>')

    # --- content blocks -----------------------------------------------------

    def body_text(self, text: str):
        if text:
            self._add(f'<p class="body-text">{self._e(text)}</p>')

    def bullet_list(self, items: List[str]):
        if not items:
            return
        li = "\n".join(f"    <li>{self._e(item)}</li>" for item in items if item)
        self._add(f'<ul class="content-list">\n{li}\n</ul>')

    def bullet_point(self, text: str):
        """Single bullet — auto-wraps into a 1-item list."""
        if text:
            self._add(
                f'<ul class="content-list"><li>{self._e(text)}</li></ul>'
            )

    def bold_text(self, label: str, value: str):
        self._add(
            f'<p class="kv-pair">'
            f'<span class="kv-label">{self._e(label)}:</span> '
            f'<span class="kv-value">{self._e(value)}</span></p>'
        )

    def code_block(self, code: str, language: str = ""):
        """Syntax-highlighted code block (Pygments if available)."""
        if not code:
            return
        if HAS_PYGMENTS and language:
            try:
                lexer = get_lexer_by_name(language, stripall=True)
            except Exception:
                lexer = TextLexer()
            formatter = HtmlFormatter(noclasses=True, style="friendly")
            highlighted = pygments_highlight(code, lexer, formatter)
            self._add(highlighted)
        else:
            self._add(f'<pre class="code-block">{self._e(code)}</pre>')

    def add_table(
        self,
        headers: List[str],
        rows: List[List[str]],
        col_widths: Optional[List[int]] = None,
    ):
        if not headers:
            return
        # Build colgroup if widths provided
        colgroup = ""
        if col_widths:
            total = sum(col_widths)
            cols = "".join(
                f'<col style="width:{w * 100 / total:.1f}%">'
                for w in col_widths
            )
            colgroup = f"<colgroup>{cols}</colgroup>"

        th = "".join(f"<th>{self._e(h)}</th>" for h in headers)
        body_rows = []
        for row in rows:
            tds = "".join(f"<td>{self._e(str(c))}</td>" for c in row)
            body_rows.append(f"<tr>{tds}</tr>")

        self._add(
            f'<table class="data-table">{colgroup}'
            f"<thead><tr>{th}</tr></thead>"
            f"<tbody>{''.join(body_rows)}</tbody></table>"
        )

    def accent_box(self, text: str, variant: str = ""):
        """Callout / accent box. variant: '', 'warning', 'danger', 'success'."""
        cls = f"accent-box {variant}".strip()
        self._add(f'<div class="{cls}"><p>{self._e(text)}</p></div>')

    def badge(self, text: str, level: str = "medium"):
        """Inline priority/status badge."""
        level_cls = f"badge badge-{level.lower()}"
        return f'<span class="{level_cls}">{self._e(text)}</span>'

    # --- diagrams -----------------------------------------------------------

    def mermaid_diagram(self, mermaid_code: str, caption: str = ""):
        """
        Render a Mermaid diagram and embed it.
        Falls back to a styled code block if the Mermaid CLI is unavailable.
        """
        self._figure_counter += 1
        fig_num = self._figure_counter

        # Try SVG first, then PNG
        svg = _render_mermaid_to_svg(mermaid_code)
        if svg:
            cap = (
                f'<p class="diagram-caption">Figure {fig_num}: {self._e(caption)}</p>'
                if caption
                else f'<p class="diagram-caption">Figure {fig_num}</p>'
            )
            self._add(
                f'<div class="diagram-container">{svg}{cap}</div>'
            )
            return

        png_b64 = _render_mermaid_to_base64_png(mermaid_code)
        if png_b64:
            cap = (
                f'<p class="diagram-caption">Figure {fig_num}: {self._e(caption)}</p>'
                if caption
                else f'<p class="diagram-caption">Figure {fig_num}</p>'
            )
            self._add(
                f'<div class="diagram-container">'
                f'<img src="data:image/png;base64,{png_b64}" '
                f'alt="{self._e(caption or "Diagram")}">'
                f"{cap}</div>"
            )
            return

        # Fallback: styled code block
        self._add(
            f'<div class="diagram-fallback">'
            f'<p class="fallback-label">Diagram (install mmdc to render visually):</p>'
            f'<pre class="code-block">{self._e(mermaid_code)}</pre></div>'
        )

    # --- rendering ----------------------------------------------------------

    def _build_toc_html(self) -> str:
        if not self._toc:
            return ""
        items = "\n".join(
            f'<li><span class="toc-num">{self._e(num)}</span>'
            f'<a href="#{aid}">{self._e(label)}</a></li>'
            for num, aid, label in self._toc
        )
        return (
            '<div class="toc-page">'
            "<h2>Table of Contents</h2>"
            f"<ul>{items}</ul></div>"
        )

    def render_html(self) -> str:
        """Assemble the complete HTML document."""
        now = datetime.utcnow()
        body = "\n".join(self._parts)

        # Insert TOC
        toc_html = self._build_toc_html()
        body = body.replace("<!-- TOC_PLACEHOLDER -->", toc_html)

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{self._e(self.doc_type)} — {self._e(self.title)}</title>
<style>{DOCUMENT_CSS}</style>
</head>
<body>
<p class="running-str doc-info">{self._e(self.doc_type)} | {self._e(self.title)}</p>
<p class="running-str doc-date">{now.strftime("%B %d, %Y")}</p>
{body}
</body>
</html>"""

    def render_pdf(self, filepath: str) -> str:
        """Render the document to PDF via WeasyPrint."""
        html_str = self.render_html()
        HTML(string=html_str).write_pdf(filepath)
        logger.info(f"PDF generated via WeasyPrint: {filepath}")
        return filepath


# ===========================================================================
# JSON Parsing / Repair Utilities  (unchanged from v1)
# ===========================================================================

def _try_parse_raw_json(text: str) -> Optional[Any]:
    """
    Try to parse raw text as JSON with aggressive repair strategies.
    Handles: markdown fences, truncation, text before/after JSON, nested objects.
    """
    if not text or len(text.strip()) < 2:
        return None

    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*\n?", "", cleaned)
    cleaned = re.sub(r"\n?```\s*$", "", cleaned)
    cleaned = cleaned.strip()

    # Strategy 1: Direct parse
    try:
        return json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        pass

    # Strategy 2: Find first { or [
    for start_char in ["{", "["]:
        char_start = cleaned.find(start_char)
        if char_start >= 0:
            json_text = cleaned[char_start:]
            try:
                return json.loads(json_text)
            except (json.JSONDecodeError, ValueError):
                pass

            end_char = "}" if start_char == "{" else "]"
            last_end = json_text.rfind(end_char)
            if last_end > 0:
                try:
                    return json.loads(json_text[: last_end + 1])
                except (json.JSONDecodeError, ValueError):
                    pass

            repaired = _repair_json_text(json_text)
            if repaired is not None:
                return repaired

    # Strategy 5: Section-by-section for truncated JSON
    if cleaned.startswith("{"):
        extracted = _extract_sections_from_truncated_json(cleaned)
        if extracted and len(extracted) >= 1:
            return extracted

    return None


def _extract_sections_from_truncated_json(text: str) -> Optional[Dict[str, Any]]:
    """Extract top-level key-value pairs from a truncated JSON object."""
    if not text or not text.strip().startswith("{"):
        return None

    result: Dict[str, Any] = {}
    pos = 1
    text_len = len(text)

    while pos < text_len:
        while pos < text_len and text[pos] in " \t\n\r,":
            pos += 1
        if pos >= text_len:
            break
        if text[pos] != '"':
            break
        key_start = pos + 1
        key_end = text.find('"', key_start)
        if key_end < 0:
            break
        key = text[key_start:key_end]
        pos = key_end + 1

        while pos < text_len and text[pos] in " \t\n\r":
            pos += 1
        if pos >= text_len or text[pos] != ":":
            break
        pos += 1
        while pos < text_len and text[pos] in " \t\n\r":
            pos += 1
        if pos >= text_len:
            break

        value_text, new_pos = _extract_balanced_value(text, pos)
        if value_text is not None:
            try:
                parsed_value = json.loads(value_text)
                result[key] = parsed_value
            except (json.JSONDecodeError, ValueError):
                repaired = _repair_json_text(value_text)
                if repaired is not None:
                    result[key] = repaired
                elif isinstance(value_text, str) and len(value_text) > 2:
                    clean_val = value_text.strip().strip('"')
                    if clean_val:
                        result[key] = clean_val
            pos = new_pos
        else:
            remaining = text[pos:].strip()
            if remaining.startswith('"'):
                end_quote = remaining.find('"', 1)
                if end_quote > 0:
                    result[key] = remaining[1:end_quote]
            break

    if result:
        logger.info(
            "Extracted sections from truncated JSON",
            keys_found=list(result.keys()),
            total_keys=len(result),
        )
    return result if result else None


def _extract_balanced_value(text: str, start: int) -> tuple:
    """Extract a balanced JSON value starting at *start*."""
    if start >= len(text):
        return None, start

    ch = text[start]

    if ch in ("{", "["):
        close = "}" if ch == "{" else "]"
        depth = 0
        in_str = False
        esc = False
        i = start
        while i < len(text):
            c = text[i]
            if esc:
                esc = False
                i += 1
                continue
            if c == "\\" and in_str:
                esc = True
                i += 1
                continue
            if c == '"':
                in_str = not in_str
            elif not in_str:
                if c == ch:
                    depth += 1
                elif c == close:
                    depth -= 1
                    if depth == 0:
                        return text[start: i + 1], i + 1
            i += 1
        return None, start

    if ch == '"':
        esc = False
        i = start + 1
        while i < len(text):
            c = text[i]
            if esc:
                esc = False
                i += 1
                continue
            if c == "\\":
                esc = True
                i += 1
                continue
            if c == '"':
                return text[start: i + 1], i + 1
            i += 1
        return text[start:] + '"', len(text)

    end = start
    while end < len(text) and text[end] not in ",}] \t\n\r":
        end += 1
    if end > start:
        return text[start:end], end

    return None, start


def _repair_json_text(text: str) -> Optional[Any]:
    """Repair truncated / malformed JSON text."""
    if not text:
        return None

    repaired = text.rstrip()
    repaired = re.sub(r",\s*$", "", repaired)

    in_str = False
    esc = False
    ob, obrk = 0, 0
    for ch in repaired:
        if esc:
            esc = False
            continue
        if ch == "\\":
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            ob += 1
        elif ch == "}":
            ob -= 1
        elif ch == "[":
            obrk += 1
        elif ch == "]":
            obrk -= 1

    if in_str:
        repaired += '"'
    repaired = re.sub(r',\s*"[^"]*"?\s*:?\s*$', "", repaired)
    repaired = re.sub(r",\s*$", "", repaired)

    for _ in range(max(0, obrk)):
        repaired += "]"
    for _ in range(max(0, ob)):
        repaired += "}"

    try:
        return json.loads(repaired)
    except (json.JSONDecodeError, ValueError):
        pass

    last_comma = repaired.rfind(",")
    if last_comma > 0:
        truncated = repaired[:last_comma]
        in_str2 = False
        esc2 = False
        ob2, obrk2 = 0, 0
        for ch in truncated:
            if esc2:
                esc2 = False
                continue
            if ch == "\\":
                esc2 = True
                continue
            if ch == '"':
                in_str2 = not in_str2
                continue
            if in_str2:
                continue
            if ch == "{":
                ob2 += 1
            elif ch == "}":
                ob2 -= 1
            elif ch == "[":
                obrk2 += 1
            elif ch == "]":
                obrk2 -= 1
        if in_str2:
            truncated += '"'
        for _ in range(max(0, obrk2)):
            truncated += "]"
        for _ in range(max(0, ob2)):
            truncated += "}"
        try:
            return json.loads(truncated)
        except (json.JSONDecodeError, ValueError):
            pass

    return None


# ===========================================================================
# Content Rendering Helpers  (rewritten for HTMLDocumentBuilder)
# ===========================================================================

def _render_prd_sections_from_dict(
    doc: HTMLDocumentBuilder, data: Dict[str, Any]
):
    """Render a parsed JSON dict as proper PRD sections."""
    section_num = 1

    # Executive Summary
    exec_s = data.get("executive_summary", {})
    if exec_s:
        doc.page_break()
        doc.section_title(str(section_num), "Executive Summary")
        if isinstance(exec_s, str):
            doc.body_text(exec_s)
        else:
            if exec_s.get("problem_statement"):
                doc.sub_section(f"{section_num}.1", "Problem Statement")
                doc.body_text(exec_s["problem_statement"])
            if exec_s.get("objective"):
                doc.sub_section(f"{section_num}.2", "Objective")
                doc.body_text(exec_s["objective"])
            goals = exec_s.get("goals", [])
            if goals:
                doc.sub_section(f"{section_num}.3", "Goals")
                doc.bullet_list(goals)
            non_goals = exec_s.get("non_goals", [])
            if non_goals:
                doc.sub_section(f"{section_num}.4", "Non-Goals")
                doc.bullet_list(non_goals)
        section_num += 1

    # User Personas
    personas = data.get("user_personas", [])
    if personas:
        doc.section_title(str(section_num), "User Personas")
        headers = ["Persona", "Responsibilities", "Key Need"]
        rows = [
            [
                p.get("persona", ""),
                p.get("responsibilities", "")[:60],
                p.get("key_need", "")[:60],
            ]
            for p in personas
        ]
        doc.add_table(headers, rows, [40, 75, 75])
        section_num += 1

    # Functional Requirements
    func_reqs = data.get("functional_requirements", [])
    if func_reqs:
        doc.page_break()
        doc.section_title(str(section_num), "Functional Requirements")
        for i, req in enumerate(func_reqs, 1):
            s_num = req.get("section_number", f"{section_num}.{i}")
            doc.sub_section(s_num, req.get("title", f"Requirement {i}"))
            doc.body_text(req.get("description", ""))
            rules = req.get("business_rules", [])
            if rules:
                doc.sub_sub_section("Business Rules")
                doc.bullet_list(rules)
            behaviors = req.get("system_behavior", [])
            if behaviors:
                doc.sub_sub_section("System Behavior")
                doc.bullet_list(behaviors)
            validations = req.get("validations", [])
            if validations:
                doc.sub_sub_section("Validations")
                doc.bullet_list(validations)
        section_num += 1

    # Non-Functional Requirements
    nfr = data.get("non_functional_requirements", {})
    if nfr and isinstance(nfr, dict):
        doc.section_title(str(section_num), "Non-Functional Requirements")
        for category in ["performance", "security", "scalability", "observability"]:
            items = nfr.get(category, [])
            if items:
                doc.sub_section("", category.title())
                doc.bullet_list(items)
        section_num += 1

    # Existing Codebase Analysis
    codebase = data.get("existing_codebase_analysis", {})
    if codebase:
        doc.page_break()
        doc.section_title(str(section_num), "Existing Codebase Analysis")
        files = codebase.get("relevant_files", [])
        if files:
            doc.sub_section("", "Relevant Files")
            headers = ["File", "Purpose"]
            rows = [[f.get("file", ""), f.get("purpose", "")[:60]] for f in files]
            doc.add_table(headers, rows, [95, 95])
        apis = codebase.get("relevant_apis", [])
        if apis:
            doc.sub_section("", "Relevant APIs")
            headers = ["Method", "Endpoint", "Purpose"]
            rows = [
                [a.get("method", ""), a.get("endpoint", ""), a.get("purpose", "")[:50]]
                for a in apis
            ]
            doc.add_table(headers, rows, [25, 80, 85])
        reusable = codebase.get("reusable_components", [])
        if reusable:
            doc.sub_section("", "Reusable Components")
            doc.bullet_list(
                [
                    f"{r.get('component', '')}: {r.get('reuse_strategy', '')}"
                    for r in reusable
                ]
            )
        # Auto diagram: component map
        comp_mermaid = _generate_component_diagram(codebase)
        if comp_mermaid:
            doc.mermaid_diagram(comp_mermaid, "Existing Codebase Component Map")
        section_num += 1

    # Proposed Technical Design
    design = data.get("proposed_technical_design", {})
    if design:
        doc.page_break()
        doc.section_title(str(section_num), "Proposed Technical Design")
        if design.get("architecture_approach"):
            doc.sub_section("", "Architecture Approach")
            doc.body_text(design["architecture_approach"])
        fc = design.get("files_to_create", [])
        if fc:
            doc.sub_section("", "Files to Create")
            headers = ["Path", "Purpose"]
            rows = [[f.get("path", ""), f.get("purpose", "")] for f in fc]
            doc.add_table(headers, rows, [95, 95])
        fm = design.get("files_to_modify", [])
        if fm:
            doc.sub_section("", "Files to Modify")
            headers = ["Path", "Changes"]
            rows = [[f.get("path", ""), f.get("changes", "")] for f in fm]
            doc.add_table(headers, rows, [95, 95])
        api_contracts = design.get("api_contracts", [])
        if api_contracts:
            doc.sub_section("", "API Contracts")
            for api in api_contracts:
                doc.bold_text(
                    "Endpoint",
                    f"{api.get('method', '')} {api.get('path', '')}",
                )
                if api.get("request"):
                    doc.bold_text("Request", api["request"])
                if api.get("response"):
                    doc.bold_text("Response", api["response"])
        if design.get("data_flow"):
            doc.sub_section("", "Data Flow")
            doc.body_text(design["data_flow"])

        # Auto diagrams
        arch_mermaid = _generate_architecture_diagram(design)
        if arch_mermaid:
            doc.mermaid_diagram(arch_mermaid, "Architecture Overview")
        if api_contracts:
            seq_mermaid = _generate_api_sequence_diagram(api_contracts)
            if seq_mermaid:
                doc.mermaid_diagram(seq_mermaid, "API Request Flow")
        section_num += 1

    # User Stories
    stories = data.get("user_stories", [])
    if stories:
        doc.section_title(str(section_num), "User Stories")
        items = [
            s if isinstance(s, str) else s.get("story", str(s)) for s in stories
        ]
        doc.bullet_list(items)
        section_num += 1

    # Failure Scenarios
    failures = data.get("failure_scenarios", [])
    if failures:
        doc.section_title(str(section_num), "Failure Scenarios")
        headers = ["Scenario", "Impact", "Mitigation"]
        rows = [
            [
                f.get("scenario", "")[:50],
                f.get("impact", "")[:50],
                f.get("mitigation", "")[:50],
            ]
            for f in failures
        ]
        doc.add_table(headers, rows, [63, 63, 64])
        section_num += 1


def _render_dict_recursively(
    doc: HTMLDocumentBuilder,
    data: Any,
    depth: int = 0,
):
    """Recursively render any dict/list structure as HTML content."""
    if isinstance(data, dict):
        for key, value in data.items():
            label = str(key).replace("_", " ").title()
            if isinstance(value, dict):
                if depth == 0:
                    doc.sub_section("", label)
                else:
                    doc.sub_sub_section(label)
                _render_dict_recursively(doc, value, depth + 1)
            elif isinstance(value, list):
                if depth == 0:
                    doc.sub_section("", label)
                else:
                    doc.sub_sub_section(label)
                _render_dict_recursively(doc, value, depth + 1)
            elif value is not None and str(value).strip():
                str_value = str(value).strip()
                if len(str_value) > 200:
                    doc.sub_sub_section(label)
                    doc.body_text(str_value)
                else:
                    doc.bold_text(label, str_value)
    elif isinstance(data, list):
        simple_items: List[str] = []
        for item in data:
            if isinstance(item, dict):
                parts = []
                for k, v in item.items():
                    if isinstance(v, (str, int, float, bool)):
                        parts.append(
                            f"{str(k).replace('_', ' ').title()}: {v}"
                        )
                    elif isinstance(v, list):
                        parts.append(
                            f"{str(k).replace('_', ' ').title()}: "
                            f"{', '.join(str(x) for x in v[:5])}"
                        )
                if parts:
                    simple_items.append("; ".join(parts))
                else:
                    # Flush accumulated simple items first
                    if simple_items:
                        doc.bullet_list(simple_items)
                        simple_items = []
                    _render_dict_recursively(doc, item, depth + 1)
            elif isinstance(item, str) and item.strip():
                simple_items.append(item.strip())
            elif isinstance(item, (int, float)):
                simple_items.append(str(item))
        if simple_items:
            doc.bullet_list(simple_items)
    elif data is not None:
        text = str(data).strip()
        if text:
            doc.body_text(text)


def _looks_like_json_line(line: str) -> bool:
    if not line:
        return True
    json_indicators = ('"', "{", "}", "[", "]", ",", ":")
    return line[0] in json_indicators or line.rstrip().endswith(
        (",", "{", "[", "}", "]")
    )


def _render_json_as_readable_text(doc: HTMLDocumentBuilder, json_text: str):
    """When JSON can't be parsed, render it as readable key-value content."""
    lines = json_text.split("\n")
    items: List[str] = []
    for line in lines:
        stripped = line.strip().rstrip(",")
        if not stripped or stripped in ("{", "}", "[", "]", "{}", "[]"):
            continue
        kv_match = re.match(r'^"([^"]+)"\s*:\s*(.+)$', stripped)
        if kv_match:
            # Flush any accumulated items
            if items:
                doc.bullet_list(items)
                items = []
            key = kv_match.group(1).replace("_", " ").title()
            value = kv_match.group(2).strip().strip('"').rstrip(",").strip('"')
            if value and value not in ("{", "[", "}", "]"):
                doc.bold_text(key, value)
            elif value in ("{", "["):
                doc.sub_sub_section(key)
            continue
        clean_line = stripped.strip('"').strip(",").strip()
        if clean_line and len(clean_line) > 2:
            items.append(clean_line)
    if items:
        doc.bullet_list(items)


def _render_text_analysis(
    doc: HTMLDocumentBuilder, text: str, doc_type: str = "prd"
):
    """
    Render raw text analysis as formatted content.
    Handles: JSON, markdown-formatted text, plain text, mixed content,
    and embedded Mermaid diagram blocks.
    """
    if not text or not text.strip():
        doc.body_text("No analysis content was generated.")
        return

    clean = text.strip()
    clean = re.sub(r"```(?:json)?\s*\n?", "", clean)
    clean = re.sub(r"\n?```", "", clean)
    clean = clean.strip()

    # Strategy 1: JSON → structured rendering
    if clean.startswith("{") or clean.startswith("["):
        parsed = _try_parse_raw_json(clean)
        if parsed and isinstance(parsed, dict):
            if doc_type == "qa":
                _render_qa_sections_from_dict(doc, parsed)
            else:
                _render_prd_sections_from_dict(doc, parsed)
                rendered_keys = {
                    "executive_summary", "user_personas",
                    "functional_requirements", "non_functional_requirements",
                    "existing_codebase_analysis", "proposed_technical_design",
                    "user_stories", "failure_scenarios",
                    "feature_name", "raw_analysis",
                }
                remaining = {
                    k: v for k, v in parsed.items()
                    if k not in rendered_keys and v
                }
                if remaining:
                    _render_dict_recursively(doc, remaining)
            return
        elif parsed and isinstance(parsed, list):
            _render_dict_recursively(doc, parsed)
            return

    # Strategy 2: Markdown-like text with possible embedded mermaid blocks
    json_accumulator: List[str] = []
    is_json_block = False
    lines = clean.split("\n")
    current_paragraph: List[str] = []
    mermaid_accumulator: List[str] = []
    in_mermaid_block = False

    def flush_paragraph():
        if current_paragraph:
            text_block = " ".join(current_paragraph).strip()
            if text_block:
                doc.body_text(text_block)
            current_paragraph.clear()

    def flush_json_accumulator():
        nonlocal json_accumulator, is_json_block
        if not json_accumulator:
            return
        json_text_block = "\n".join(json_accumulator)
        json_accumulator.clear()
        is_json_block = False

        parsed = _try_parse_raw_json(json_text_block)
        if parsed and isinstance(parsed, dict):
            if doc_type == "qa":
                _render_qa_sections_from_dict(doc, parsed)
            else:
                _render_prd_sections_from_dict(doc, parsed)
                rendered_keys = {
                    "executive_summary", "user_personas",
                    "functional_requirements", "non_functional_requirements",
                    "existing_codebase_analysis", "proposed_technical_design",
                    "user_stories", "failure_scenarios",
                    "feature_name", "raw_analysis",
                }
                remaining = {
                    k: v for k, v in parsed.items()
                    if k not in rendered_keys and v
                }
                if remaining:
                    _render_dict_recursively(doc, remaining)
        elif parsed and isinstance(parsed, list):
            _render_dict_recursively(doc, parsed)
        else:
            _render_json_as_readable_text(doc, json_text_block)

    for line in lines:
        stripped = line.strip()

        # Detect mermaid code blocks
        if stripped.startswith("```mermaid"):
            flush_paragraph()
            in_mermaid_block = True
            mermaid_accumulator = []
            continue
        if in_mermaid_block:
            if stripped == "```":
                code = "\n".join(mermaid_accumulator)
                doc.mermaid_diagram(code, "")
                mermaid_accumulator = []
                in_mermaid_block = False
            else:
                mermaid_accumulator.append(line)
            continue

        # JSON block detection
        if not is_json_block and stripped and stripped[0] in ("{", "["):
            flush_paragraph()
            is_json_block = True
            json_accumulator.append(line)
            continue
        if is_json_block:
            if stripped == "" and json_accumulator:
                json_accumulator.append(line)
                continue
            if _looks_like_json_line(stripped):
                json_accumulator.append(line)
                continue
            else:
                flush_json_accumulator()

        if not stripped:
            flush_paragraph()
            continue

        # Markdown headings
        if stripped.startswith("###"):
            flush_paragraph()
            doc.sub_sub_section(stripped.lstrip("#").strip())
        elif stripped.startswith("##"):
            flush_paragraph()
            doc.sub_section("", stripped.lstrip("#").strip())
        elif stripped.startswith("#"):
            flush_paragraph()
            doc.sub_section("", stripped.lstrip("#").strip())
        elif stripped.startswith(("- ", "* ", "+ ")):
            flush_paragraph()
            doc.bullet_point(stripped[2:].strip())
        elif (
            len(stripped) > 2
            and stripped[0].isdigit()
            and ". " in stripped[:5]
        ):
            flush_paragraph()
            doc.bullet_point(stripped)
        elif (
            ":" in stripped
            and len(stripped) < 200
            and not stripped.startswith("{")
            and not stripped.startswith('"')
        ):
            flush_paragraph()
            colon_pos = stripped.index(":")
            key = stripped[:colon_pos].strip().strip("*").strip()
            value = stripped[colon_pos + 1 :].strip()
            if key and value:
                doc.bold_text(key, value)
            elif key:
                doc.sub_sub_section(key)
            else:
                current_paragraph.append(stripped)
        else:
            current_paragraph.append(stripped)

    flush_json_accumulator()
    flush_paragraph()


def _render_qa_sections_from_dict(doc: HTMLDocumentBuilder, data: Dict[str, Any]):
    """Render a parsed QA JSON dict as proper test plan sections."""
    testing_types = [
        ("1", "Unit Testing", ["unit_testing", "unit_testing_plan", "unit_tests"]),
        ("2", "Integration Testing", ["integration_testing", "integration_testing_plan", "integration_tests"]),
        ("3", "System Testing", ["system_testing", "system_testing_plan", "system_tests", "e2e_testing"]),
        ("4", "Regression Testing", ["regression_testing", "regression_testing_plan", "regression_tests"]),
        ("5", "Performance Testing", ["performance_testing", "performance_testing_plan", "performance_tests", "load_testing"]),
        ("6", "Security Testing", ["security_testing", "security_testing_plan", "security_tests"]),
    ]

    rendered_any_section = False

    # Summary first
    summary = data.get("summary", data.get("test_plan_summary", {}))
    if summary and isinstance(summary, dict):
        doc.page_break()
        doc.section_title("0", "Test Plan Summary")
        rendered_any_section = True
        if summary.get("total_test_cases"):
            doc.bold_text("Total Test Cases", str(summary["total_test_cases"]))
        if summary.get("estimated_coverage"):
            doc.bold_text("Estimated Coverage", str(summary["estimated_coverage"]))
        prio = summary.get("priority_tests", summary.get("critical_tests", []))
        if prio:
            doc.bullet_list([f"Priority: {pt}" for pt in prio])
        risks = summary.get("risk_areas", [])
        if risks:
            doc.bullet_list([f"Risk: {ra}" for ra in risks])
        skip = {
            "total_test_cases", "estimated_coverage", "priority_tests",
            "critical_tests", "risk_areas",
        }
        for k, v in summary.items():
            if k not in skip and v:
                doc.bold_text(k.replace("_", " ").title(), str(v)[:200])

    for section_num, title, keys in testing_types:
        section_data = None
        for key in keys:
            section_data = data.get(key)
            if section_data:
                break
        if not section_data:
            continue

        if isinstance(section_data, list):
            section_data = _flatten_qa_plan_list(section_data)
        if isinstance(section_data, str):
            section_data = {"objective": section_data}

        doc.page_break()
        doc.section_title(section_num, title)
        rendered_any_section = True

        if isinstance(section_data, dict):
            if section_data.get("objective"):
                doc.sub_section(f"{section_num}.1", "Objective")
                doc.body_text(str(section_data["objective"]))
            if section_data.get("scope"):
                doc.sub_section(f"{section_num}.2", "Scope")
                doc.body_text(str(section_data["scope"]))
            if section_data.get("strategy"):
                doc.sub_section(f"{section_num}.3", "Strategy")
                doc.body_text(str(section_data["strategy"]))

            test_cases = section_data.get(
                "test_cases",
                section_data.get("cases", section_data.get("scenarios", [])),
            )
            if test_cases and isinstance(test_cases, list):
                doc.sub_section(f"{section_num}.4", "Test Cases")
                headers = [
                    "ID", "Description", "Preconditions",
                    "Expected Result", "Priority",
                ]
                rows = []
                for tc in test_cases:
                    if isinstance(tc, str):
                        rows.append(["", tc[:50], "", "", "medium"])
                        continue
                    if not isinstance(tc, dict):
                        continue
                    pre = tc.get("preconditions", tc.get("setup", ""))
                    if isinstance(pre, list):
                        pre = "; ".join(str(p) for p in pre)
                    exp = (
                        tc.get("expected_result", "")
                        or tc.get("success_criteria", "")
                        or tc.get("expected_output", "")
                    )
                    if isinstance(exp, list):
                        exp = "; ".join(str(e) for e in exp)
                    rows.append([
                        str(tc.get("id", tc.get("name", tc.get("test_id", ""))))[:12],
                        str(tc.get("description", tc.get("scenario", tc.get("title", ""))))[:45],
                        str(pre)[:35],
                        str(exp)[:35],
                        str(tc.get("priority", "medium")),
                    ])
                if rows:
                    doc.add_table(headers, rows, [15, 55, 40, 40, 20])

            failure_key = (
                "failure_scenarios"
                if "failure_scenarios" in section_data
                else "failure_modes"
            )
            failures = section_data.get(failure_key, [])
            if failures:
                doc.bullet_list([
                    f"Failure: {fs}" if isinstance(fs, str)
                    else f"Failure: {json.dumps(fs, default=str)[:100]}"
                    for fs in failures
                ])
            edge = section_data.get("edge_cases", [])
            if edge:
                doc.bullet_list([
                    f"Edge case: {ec}" if isinstance(ec, str)
                    else f"Edge case: {json.dumps(ec, default=str)[:100]}"
                    for ec in edge
                ])

            env_reqs = section_data.get("environment_requirements", [])
            if env_reqs:
                doc.bullet_list([f"Env: {e}" for e in env_reqs])
            data_reqs = section_data.get("data_requirements", [])
            if data_reqs:
                doc.bullet_list([f"Data: {d}" for d in data_reqs])
            tooling = section_data.get("tooling_suggestions", [])
            if tooling:
                doc.bullet_list([f"Tool: {t}" for t in tooling])
            if section_data.get("mocking_strategy"):
                doc.sub_section("", "Mocking Strategy")
                doc.body_text(str(section_data["mocking_strategy"]))

    if not rendered_any_section:
        rendered_keys = {"feature_name", "summary", "test_plan_summary", "raw_analysis"}
        rendered_keys.update(k for _, _, ks in testing_types for k in ks)
        remaining = {k: v for k, v in data.items() if k not in rendered_keys and v}
        if remaining:
            doc.page_break()
            doc.section_title("1", "Test Plan Analysis")
            _render_dict_recursively(doc, remaining)


def _render_extracted_prd(doc: HTMLDocumentBuilder, parsed_raw: Dict[str, Any]):
    """Render extracted / parsed PRD sections from raw_analysis."""
    _render_prd_sections_from_dict(doc, parsed_raw)
    standard_keys = {
        "executive_summary", "user_personas", "functional_requirements",
        "non_functional_requirements", "existing_codebase_analysis",
        "proposed_technical_design", "user_stories", "failure_scenarios",
        "feature_name", "backward_compatibility", "rollout_plan",
        "assumptions_constraints", "open_questions_risks", "summary_matrix",
        "raw_analysis",
    }
    extra = {k: v for k, v in parsed_raw.items() if k not in standard_keys and v}
    if extra:
        doc.page_break()
        doc.section_title("A", "Additional Analysis")
        _render_dict_recursively(doc, extra)


def _flatten_qa_plan_list(plan_list: list) -> dict:
    """Convert qa_prompt.yaml list format to dict format the renderer expects."""
    if not plan_list:
        return {}
    result: Dict[str, Any] = {
        "objective": "",
        "scope": "",
        "strategy": "",
        "tooling_suggestions": [],
        "mocking_strategy": "",
        "test_cases": [],
        "failure_scenarios": [],
        "edge_cases": [],
    }
    for item in plan_list:
        if isinstance(item, dict):
            if item.get("component") or item.get("flow_name"):
                result["scope"] += (
                    (item.get("component", "") or item.get("flow_name", ""))
                    + "; "
                )
            if item.get("tools"):
                result["tooling_suggestions"].extend(item["tools"])
            if item.get("coverage_target"):
                result["strategy"] += (
                    f"Coverage target: {item['coverage_target']}. "
                )
            for tc in item.get("test_cases", []) or item.get(
                "test_scenarios", []
            ):
                if isinstance(tc, dict):
                    pre = tc.get("preconditions", [])
                    if isinstance(pre, list):
                        pre = ", ".join(pre)
                    sc = tc.get(
                        "success_criteria", tc.get("assertions", [])
                    )
                    if isinstance(sc, list):
                        exp = ", ".join(sc)
                    else:
                        exp = str(tc.get("expected_output", ""))
                    result["test_cases"].append({
                        "id": tc.get("name", tc.get("scenario", "")),
                        "description": tc.get(
                            "description", tc.get("scenario", "")
                        ),
                        "preconditions": str(pre),
                        "expected_result": exp,
                        "priority": tc.get("priority", "medium"),
                    })
            for fm in item.get("failure_modes", []):
                result["failure_scenarios"].append(fm)
    return result


# ===========================================================================
# PRD PDF Generator
# ===========================================================================

def generate_prd_pdf(prd_data: Dict[str, Any]) -> str:
    """
    Generate a professional PRD PDF from structured data.

    Args:
        prd_data: Structured PRD dictionary with all sections.

    Returns:
        Path to the generated PDF file.
    """
    feature_name = prd_data.get("feature_name", "Untitled Feature")
    safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", feature_name)[:50]
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = f"PRD_{safe_name}_{timestamp}.pdf"
    filepath = str(PDF_OUTPUT_DIR / filename)

    doc = HTMLDocumentBuilder(title=feature_name, doc_type="Product Requirements Document")

    # --- Cover Page ---
    doc.cover_page(
        title="Product Requirements Document",
        subtitle=feature_name,
    )

    # --- Table of Contents ---
    doc.table_of_contents()

    # --- 1. Executive Summary ---
    exec_summary = prd_data.get("executive_summary", {})
    if isinstance(exec_summary, str):
        exec_summary = {"problem_statement": exec_summary, "objective": ""}

    doc.section_title("1", "Executive Summary")
    doc.sub_section("1.1", "Problem Statement")
    doc.body_text(exec_summary.get("problem_statement", "Not provided."))
    doc.sub_section("1.2", "Objective")
    doc.body_text(exec_summary.get("objective", "Not provided."))

    goals = exec_summary.get("goals", [])
    if goals:
        doc.sub_section("1.3", "Goals")
        doc.bullet_list(goals)
    non_goals = exec_summary.get("non_goals", [])
    if non_goals:
        doc.sub_section("1.4", "Non-Goals")
        doc.bullet_list(non_goals)

    # --- 2. User Personas ---
    personas = prd_data.get("user_personas", [])
    if personas:
        doc.section_title("2", "User Personas")
        headers = ["Persona", "Responsibilities", "Key Need"]
        rows = [
            [p.get("persona", ""), p.get("responsibilities", ""), p.get("key_need", "")]
            for p in personas
        ]
        doc.add_table(headers, rows, [40, 75, 75])

    # --- 3. Functional Requirements ---
    func_reqs = prd_data.get("functional_requirements", [])
    if func_reqs:
        doc.section_title("3", "Functional Requirements")
        for i, req in enumerate(func_reqs, 1):
            section_num = req.get("section_number", f"3.{i}")
            doc.sub_section(section_num, req.get("title", f"Requirement {i}"))
            doc.body_text(req.get("description", ""))

            rules = req.get("business_rules", [])
            if rules:
                doc.sub_sub_section("Business Rules")
                doc.bullet_list(rules)
            behaviors = req.get("system_behavior", [])
            if behaviors:
                doc.sub_sub_section("System Behavior")
                doc.bullet_list(behaviors)
            validations = req.get("validations", [])
            if validations:
                doc.sub_sub_section("Validations")
                doc.bullet_list(validations)

    # --- 4. Non-Functional Requirements ---
    nfr = prd_data.get("non_functional_requirements", {})
    if nfr:
        doc.section_title("4", "Non-Functional Requirements")
        for category in ["performance", "security", "scalability", "observability"]:
            items = nfr.get(category, [])
            if items:
                doc.sub_section("", category.title())
                doc.bullet_list(items)

    # --- 5. Existing Codebase Analysis ---
    codebase = prd_data.get("existing_codebase_analysis", {})
    if codebase:
        doc.section_title("5", "Existing Codebase Analysis")

        files = codebase.get("relevant_files", [])
        if files:
            doc.sub_section("5.1", "Relevant Files")
            doc.add_table(
                ["File", "Purpose"],
                [[f.get("file", ""), f.get("purpose", "")] for f in files],
                [95, 95],
            )

        services = codebase.get("relevant_services", [])
        if services:
            doc.sub_section("5.2", "Relevant Services")
            doc.add_table(
                ["Service", "Usage"],
                [[s.get("service", ""), s.get("usage", "")] for s in services],
                [95, 95],
            )

        apis = codebase.get("relevant_apis", [])
        if apis:
            doc.sub_section("5.3", "Relevant APIs")
            doc.add_table(
                ["Method", "Endpoint", "Purpose"],
                [
                    [a.get("method", ""), a.get("endpoint", ""), a.get("purpose", "")]
                    for a in apis
                ],
                [25, 80, 85],
            )

        reusable = codebase.get("reusable_components", [])
        if reusable:
            doc.sub_section("5.4", "Reusable Components")
            doc.add_table(
                ["Component", "Reuse Strategy"],
                [
                    [r.get("component", ""), r.get("reuse_strategy", "")]
                    for r in reusable
                ],
                [95, 95],
            )

        # Auto diagram: component map
        comp_mermaid = _generate_component_diagram(codebase)
        if comp_mermaid:
            doc.mermaid_diagram(comp_mermaid, "Existing Codebase Component Map")

    # --- 6. Proposed Technical Design ---
    design = prd_data.get("proposed_technical_design", {})
    if design:
        doc.section_title("6", "Proposed Technical Design")

        approach = design.get("architecture_approach", "")
        if approach:
            doc.sub_section("6.1", "Architecture Approach")
            doc.body_text(approach)

        files_create = design.get("files_to_create", [])
        if files_create:
            doc.sub_section("6.2", "Files to Create")
            doc.add_table(
                ["Path", "Purpose"],
                [[f.get("path", ""), f.get("purpose", "")] for f in files_create],
                [95, 95],
            )

        files_modify = design.get("files_to_modify", [])
        if files_modify:
            doc.sub_section("6.3", "Files to Modify")
            doc.add_table(
                ["Path", "Changes"],
                [[f.get("path", ""), f.get("changes", "")] for f in files_modify],
                [95, 95],
            )

        api_contracts = design.get("api_contracts", [])
        if api_contracts:
            doc.sub_section("6.4", "API Contracts")
            for api in api_contracts:
                doc.bold_text(
                    "Endpoint",
                    f"{api.get('method', '')} {api.get('path', '')}",
                )
                if api.get("request"):
                    doc.bold_text("Request", api["request"])
                if api.get("response"):
                    doc.bold_text("Response", api["response"])

        data_flow = design.get("data_flow", "")
        if data_flow:
            doc.sub_section("6.5", "Data Flow")
            doc.body_text(data_flow)

        # Auto diagrams: architecture + API sequence
        arch_mermaid = _generate_architecture_diagram(design)
        if arch_mermaid:
            doc.mermaid_diagram(arch_mermaid, "Architecture Overview")
        if api_contracts:
            seq_mermaid = _generate_api_sequence_diagram(api_contracts)
            if seq_mermaid:
                doc.mermaid_diagram(seq_mermaid, "API Request Flow")

    # --- 7. User Stories ---
    stories = prd_data.get("user_stories", [])
    if stories:
        doc.section_title("7", "User Stories")
        items = [
            s if isinstance(s, str) else s.get("story", str(s)) for s in stories
        ]
        doc.bullet_list(items)

    # --- 8. Failure Scenarios ---
    failures = prd_data.get("failure_scenarios", [])
    if failures:
        doc.section_title("8", "Failure Scenarios")
        doc.add_table(
            ["Scenario", "Impact", "Mitigation"],
            [
                [f.get("scenario", ""), f.get("impact", ""), f.get("mitigation", "")]
                for f in failures
            ],
            [63, 63, 64],
        )

    # --- 9. Backward Compatibility ---
    compat = prd_data.get("backward_compatibility", "")
    if compat:
        doc.section_title("9", "Backward Compatibility")
        doc.body_text(compat)

    # --- 10. Rollout Plan ---
    rollout = prd_data.get("rollout_plan", [])
    if rollout:
        doc.section_title("10", "Rollout Plan")
        for phase in rollout:
            doc.sub_section(
                "",
                f"Phase {phase.get('phase', '?')}: {phase.get('description', '')}",
            )
            tasks = phase.get("tasks", [])
            if tasks:
                doc.bullet_list(tasks)

    # --- 11. Assumptions & Constraints ---
    ac = prd_data.get("assumptions_constraints", {})
    if ac:
        doc.section_title("11", "Assumptions & Constraints")
        assumptions = ac.get("assumptions", [])
        if assumptions:
            doc.sub_section("", "Assumptions")
            doc.bullet_list(assumptions)
        constraints = ac.get("constraints", [])
        if constraints:
            doc.sub_section("", "Constraints")
            doc.bullet_list(constraints)

    # --- 12. Open Questions & Risks ---
    risks = prd_data.get("open_questions_risks", [])
    if risks:
        doc.section_title("12", "Open Questions & Risks")
        doc.add_table(
            ["Question/Risk", "Risk Level", "Owner"],
            [
                [r.get("question", ""), r.get("risk_level", ""), r.get("owner", "")]
                for r in risks
            ],
            [100, 45, 45],
        )

    # --- 13. Summary Matrix ---
    matrix = prd_data.get("summary_matrix", [])
    if matrix:
        doc.section_title("13", "Summary Matrix")
        doc.add_table(
            ["Area", "Scenario", "Impact", "Validation"],
            [
                [
                    m.get("area", ""),
                    m.get("scenario", ""),
                    m.get("impact", ""),
                    m.get("validation", ""),
                ]
                for m in matrix
            ],
            [40, 50, 50, 50],
        )

    # --- 14. Capability Gap Analysis ---
    gap_analysis = prd_data.get("capability_gap_analysis", [])
    if gap_analysis:
        doc.section_title("14", "Capability Gap Analysis")
        doc.add_table(
            ["Capability", "Priority", "Has Existing", "Gap", "Risk", "Effort"],
            [
                [
                    g.get("capability", "")[:40],
                    g.get("priority", ""),
                    "Yes" if g.get("has_existing") else "No",
                    g.get("gap", "")[:40],
                    g.get("risk", ""),
                    g.get("effort", ""),
                ]
                for g in gap_analysis
            ],
            [35, 22, 20, 40, 18, 25],
        )

    # --- Fallback: raw_analysis ---
    raw_analysis = prd_data.get("raw_analysis", "")
    if raw_analysis and len(raw_analysis) > 50:
        structured_sections = sum(
            1
            for key in [
                "user_personas", "functional_requirements",
                "non_functional_requirements", "existing_codebase_analysis",
                "proposed_technical_design", "user_stories",
                "failure_scenarios", "rollout_plan",
            ]
            if prd_data.get(key)
        )
        if structured_sections < 3:
            logger.info(
                "Rendering raw_analysis fallback for PRD",
                raw_length=len(raw_analysis),
                structured_sections=structured_sections,
            )
            parsed_raw = _try_parse_raw_json(raw_analysis)
            if parsed_raw and isinstance(parsed_raw, dict):
                _render_extracted_prd(doc, parsed_raw)
            elif parsed_raw and isinstance(parsed_raw, list):
                doc.page_break()
                doc.section_title("A", "Detailed Analysis")
                _render_dict_recursively(doc, parsed_raw)
            else:
                doc.page_break()
                doc.section_title("A", "Detailed Analysis")
                _render_text_analysis(doc, raw_analysis, doc_type="prd")

    # Output
    doc.render_pdf(filepath)
    logger.info(f"PRD PDF generated: {filepath}")
    return filepath


# ===========================================================================
# QA Test Plan PDF Generator
# ===========================================================================

def generate_qa_pdf(qa_data: Dict[str, Any]) -> str:
    """
    Generate a professional QA Test Plan PDF from structured data.

    Args:
        qa_data: Structured QA test plan dictionary.

    Returns:
        Path to the generated PDF file.
    """
    feature_name = qa_data.get("feature_name", "Untitled Feature")
    safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", feature_name)[:50]
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = f"QA_Test_Plan_{safe_name}_{timestamp}.pdf"
    filepath = str(PDF_OUTPUT_DIR / filename)

    doc = HTMLDocumentBuilder(title=feature_name, doc_type="QA Test Plan")

    # --- Cover Page ---
    doc.cover_page(title="QA Test Plan", subtitle=feature_name)

    # --- Table of Contents ---
    doc.table_of_contents()

    # --- Auto diagram: test distribution ---
    dist_mermaid = _generate_test_distribution_diagram(qa_data)
    if dist_mermaid:
        doc.mermaid_diagram(dist_mermaid, "Test Case Distribution")

    # --- Testing Types ---
    testing_types = [
        ("1", "Unit Testing", "unit_testing", "unit_testing_plan"),
        ("2", "Integration Testing", "integration_testing", "integration_testing_plan"),
        ("3", "System Testing", "system_testing", "system_testing_plan"),
        ("4", "Regression Testing", "regression_testing", "regression_testing_plan"),
        ("5", "Performance / Load Testing", "performance_testing", "performance_testing_plan"),
        ("6", "Security Testing", "security_testing", "security_testing_plan"),
    ]

    for entry in testing_types:
        section_num, section_title_text = entry[0], entry[1]
        section_data = qa_data.get(entry[2], {}) or qa_data.get(entry[3], {})
        if isinstance(section_data, list):
            section_data = _flatten_qa_plan_list(section_data)
        if not section_data:
            continue

        doc.page_break()
        doc.section_title(section_num, section_title_text)

        if not isinstance(section_data, dict):
            doc.body_text(str(section_data))
            continue

        # Objective
        if section_data.get("objective"):
            doc.sub_section(f"{section_num}.1", "Objective")
            doc.body_text(section_data["objective"])

        # Scope
        if section_data.get("scope"):
            doc.sub_section(f"{section_num}.2", "Scope")
            doc.body_text(section_data["scope"])

        # Strategy
        if section_data.get("strategy"):
            doc.sub_section(f"{section_num}.3", "Test Strategy")
            doc.body_text(section_data["strategy"])

        # Tooling
        tooling = section_data.get("tooling_suggestions", [])
        if tooling:
            doc.sub_section(f"{section_num}.4", "Tooling Suggestions")
            doc.bullet_list(tooling)

        # Test Cases
        test_cases = section_data.get("test_cases", [])
        if test_cases:
            doc.sub_section(f"{section_num}.5", "Test Cases")
            headers = [
                "ID", "Description", "Preconditions",
                "Expected Result", "Priority",
            ]
            rows = []
            for tc in test_cases:
                rows.append([
                    tc.get("id", ""),
                    tc.get("description", "")[:50],
                    tc.get("preconditions", "")[:40],
                    tc.get("expected_result", "")[:40],
                    tc.get("priority", "medium"),
                ])
            doc.add_table(headers, rows, [15, 55, 45, 45, 30])

        # Mocking Strategy
        if section_data.get("mocking_strategy"):
            doc.sub_section(f"{section_num}.6", "Mocking Strategy")
            doc.body_text(section_data["mocking_strategy"])

        # Failure Scenarios
        failures = section_data.get("failure_scenarios", [])
        if failures:
            doc.sub_section(f"{section_num}.7", "Failure Scenarios")
            doc.bullet_list(failures)

        # Edge Cases
        edges = section_data.get("edge_cases", [])
        if edges:
            doc.sub_section(f"{section_num}.8", "Edge Cases")
            doc.bullet_list(edges)

    # --- 7. Test Data Requirements ---
    test_data = qa_data.get("test_data_requirements", {})
    if test_data:
        doc.page_break()
        doc.section_title("7", "Test Data Requirements")
        fixtures = test_data.get("fixtures", [])
        if fixtures:
            doc.sub_section("7.1", "Fixtures")
            doc.bullet_list(fixtures)
        seed = test_data.get("seed_data", [])
        if seed:
            doc.sub_section("7.2", "Seed Data")
            doc.bullet_list(seed)
        mocks = test_data.get("mock_services", [])
        if mocks:
            doc.sub_section("7.3", "Mock Services")
            doc.bullet_list(mocks)

    # --- 8. Testing Strategy Summary ---
    strategy_summary = qa_data.get("testing_strategy_summary", {})
    if strategy_summary:
        doc.page_break()
        doc.section_title("8", "Testing Strategy Summary")
        if strategy_summary.get("overall_risk"):
            doc.bold_text("Overall Risk", str(strategy_summary["overall_risk"]))
        if strategy_summary.get("testing_effort_weeks"):
            doc.bold_text(
                "Testing Effort",
                f"{strategy_summary['testing_effort_weeks']} weeks",
            )
        if strategy_summary.get("automation_coverage_target"):
            doc.bold_text(
                "Automation Coverage Target",
                str(strategy_summary["automation_coverage_target"]),
            )
        focus = strategy_summary.get("critical_testing_focus_areas", [])
        if focus:
            doc.sub_section("", "Critical Focus Areas")
            doc.bullet_list(focus)

    # --- 9. Quality Gates ---
    gates = qa_data.get("quality_gates", [])
    if gates:
        doc.section_title("9", "Quality Gates")
        doc.add_table(
            ["Gate", "Threshold", "Enforcement", "Measurement"],
            [
                [
                    g.get("gate", ""),
                    g.get("threshold", ""),
                    g.get("enforcement", ""),
                    g.get("measurement", ""),
                ]
                for g in gates
            ],
            [45, 45, 50, 50],
        )

    # --- 10. Risk-Based Prioritization ---
    risks = qa_data.get("risk_based_prioritization", [])
    if risks:
        doc.section_title("10", "Risk-Based Prioritization")
        doc.add_table(
            ["Risk", "Impact", "Probability", "Testing Focus"],
            [
                [
                    r.get("risk", ""),
                    r.get("impact", ""),
                    r.get("probability", ""),
                    r.get("testing_focus", ""),
                ]
                for r in risks
            ],
            [50, 25, 25, 90],
        )

    # --- 11. Test Plan Summary ---
    summary = qa_data.get("summary", {})
    if summary and summary.get("total_test_cases", 0) > 0:
        doc.page_break()
        doc.section_title("11", "Test Plan Summary")
        doc.bold_text(
            "Total Test Cases", str(summary.get("total_test_cases", "N/A"))
        )
        doc.bold_text(
            "Estimated Coverage", str(summary.get("estimated_coverage", "N/A"))
        )
        risk_areas = summary.get("risk_areas", [])
        if risk_areas:
            doc.sub_section("", "Risk Areas")
            doc.bullet_list(risk_areas)
        priority_tests = summary.get("priority_tests", [])
        if priority_tests:
            doc.sub_section("", "Priority Tests")
            doc.bullet_list(priority_tests)

    # --- Fallback: raw_analysis ---
    raw_analysis = qa_data.get("raw_analysis", "")
    if raw_analysis and len(raw_analysis) > 50:
        has_tests = any(
            qa_data.get(k)
            for k in [
                "unit_testing", "integration_testing", "system_testing",
                "unit_testing_plan", "integration_testing_plan",
                "regression_testing", "performance_testing",
            ]
        )
        if not has_tests:
            logger.info(
                "Rendering raw_analysis fallback for QA",
                raw_length=len(raw_analysis),
            )
            parsed_raw = _try_parse_raw_json(raw_analysis)
            if parsed_raw and isinstance(parsed_raw, dict):
                _render_qa_sections_from_dict(doc, parsed_raw)
            elif parsed_raw and isinstance(parsed_raw, list):
                doc.page_break()
                doc.section_title("A", "QA Analysis")
                _render_dict_recursively(doc, parsed_raw)
            else:
                doc.page_break()
                doc.section_title("A", "Detailed QA Analysis")
                _render_text_analysis(doc, raw_analysis, doc_type="qa")

    # Output
    doc.render_pdf(filepath)
    logger.info(f"QA PDF generated: {filepath}")
    return filepath
