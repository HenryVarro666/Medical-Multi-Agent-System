"""Render static README architecture diagrams without model or network calls.

    uv run --locked --extra plots python -m evaluation.draw_architecture

Each figure has a self-contained HTML preview, accessible SVG, and PNG.
The palette and offline fonts follow the existing evaluation figures.
"""

from __future__ import annotations

import argparse
import html
import io
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon  # noqa: E402
from matplotlib.path import Path as DrawingPath  # noqa: E402

WIDTH, HEIGHT = 1280, 840
INK = "#172B45"
MUTED = "#53657B"
BLUE = "#2563EB"
PAPER = "#FFFFFF"
TINT = "#EFF5FF"
SURFACE = "#F5F7FA"
RULE = "#D8E1ED"
OUTPUT = Path(__file__).resolve().parents[1] / "docs/assets/architecture"


def canvas(title: str, subtitle: str):
    fig = plt.figure(figsize=(WIDTH / 72, HEIGHT / 72), facecolor=PAPER)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set(xlim=(0, WIDTH), ylim=(HEIGHT, 0))
    ax.axis("off")
    text(ax, 40, 56, title, size=32, weight="bold")
    text(ax, 40, 92, subtitle, size=16, color=MUTED)
    return fig, ax


def text(ax, x, y, content, *, size=16, color=INK, weight="normal", mono=False, **kwargs):
    return ax.text(x, y, content, fontsize=size, color=color, weight=weight,
                   fontfamily="DejaVu Sans Mono" if mono else "DejaVu Sans",
                   zorder=5, **kwargs)


def box(ax, x, y, width, height, title, lines, *, tag="", fill=PAPER,
        stroke=INK, radius=6):
    ax.add_patch(FancyBboxPatch((x, y), width, height,
                               boxstyle=f"round,pad=0,rounding_size={radius}",
                               facecolor=fill, edgecolor=stroke, linewidth=1.3, zorder=3))
    if tag:
        text(ax, x + 16, y + 24, tag, size=11, color=MUTED, mono=True, weight="bold")
    title_y = y + (56 if tag else 30)
    text(ax, x + 16, title_y, title, size=20, weight="bold")
    for index, line in enumerate(lines):
        text(ax, x + 16, title_y + 28 + 24 * index, line, size=15, color=MUTED)


def route(ax, points, *, color=MUTED, dashed=False):
    """Draw an orthogonal connector with rounded eight-unit corners."""
    vertices, codes = [points[0]], [DrawingPath.MOVETO]
    for index in range(1, len(points) - 1):
        previous, corner, following = points[index - 1:index + 2]
        before = (corner[0] - previous[0], corner[1] - previous[1])
        after = (following[0] - corner[0], following[1] - corner[1])
        first_length = abs(before[0]) + abs(before[1])
        second_length = abs(after[0]) + abs(after[1])
        radius = min(8, first_length / 2, second_length / 2)
        incoming = (corner[0] - before[0] / first_length * radius,
                    corner[1] - before[1] / first_length * radius)
        outgoing = (corner[0] + after[0] / second_length * radius,
                    corner[1] + after[1] / second_length * radius)
        vertices.extend((incoming, corner, outgoing))
        codes.extend((DrawingPath.LINETO, DrawingPath.CURVE3, DrawingPath.CURVE3))
    vertices.append(points[-1])
    codes.append(DrawingPath.LINETO)
    ax.add_patch(FancyArrowPatch(path=DrawingPath(vertices, codes), arrowstyle="-|>",
                                mutation_scale=15, linewidth=1.7, color=color,
                                linestyle=(0, (5, 4)) if dashed else "solid", zorder=2))


def edge_label(ax, x, y, label, *, color=MUTED):
    text(ax, x, y, label, size=11, color=color, mono=True,
         ha="center", va="center", bbox={"facecolor": PAPER, "edgecolor": "none", "pad": 3})


def legend(ax, *, workflow=False):
    ax.plot((40, 88), (808, 808), color=MUTED, linewidth=1.7)
    text(ax, 100, 813, "Execution order" if workflow else "Call / local data flow", size=13, color=MUTED)
    ax.plot((360, 408), (808, 808), color=BLUE if workflow else MUTED,
            linewidth=1.7, linestyle=(0, (5, 4)))
    text(ax, 420, 813, "Retry the same narrative" if workflow else "JSON response", size=13, color=MUTED)


def architecture():
    fig, ax = canvas("System architecture",
                     "Current Python implementation · FastAPI + LangGraph · local or hosted model backend")
    ax.add_patch(FancyBboxPatch((288, 144), 672, 560,
                               boxstyle="round,pad=0,rounding_size=8",
                               facecolor="#FBFCFE", edgecolor=RULE, linewidth=1, zorder=0))
    text(ax, 320, 168, "PYTHON APPLICATION", size=11, color=MUTED, mono=True, weight="bold")
    route(ax, [(248, 224), (320, 224)])
    route(ax, [(320, 256), (248, 256)], dashed=True)
    route(ax, [(460, 288), (460, 368)])
    route(ax, [(600, 236), (688, 236)])
    route(ax, [(704, 408), (640, 408)])
    route(ax, [(704, 504), (640, 504)])
    route(ax, [(460, 544), (460, 600)])
    route(ax, [(616, 368), (616, 320), (1124, 320), (1124, 368)], color=BLUE)
    edge_label(ax, 284, 204, "POST")
    edge_label(ax, 284, 276, "RESULT")
    edge_label(ax, 508, 328, "INVOKE")
    edge_label(ax, 872, 300, "MODEL CALLS", color=BLUE)
    edge_label(ax, 500, 572, "STATE")
    box(ax, 40, 184, 208, 104, "API client", ["Synthetic narrative", "Structured JSON result"], fill=SURFACE)
    box(ax, 320, 184, 280, 104, "FastAPI", ["Analysis + offline routes", "api/routes.py"])
    box(ax, 688, 184, 240, 104, "ICD-10 / DRG tools", ["Bundled lookup tables", "Standalone API endpoints"])
    box(ax, 320, 368, 320, 176, "LangGraph workflow",
        ["Intake · Diagnosis · Treatment", "Coding · Audit (rules)", "LLM calls via get_llm()", "Sequential ClinicalState updates"],
        tag="ORCHESTRATION", fill=TINT, stroke=BLUE)
    box(ax, 704, 368, 224, 80, "Symptom map", ["Diagnosis retrieval hints"])
    box(ax, 704, 464, 224, 80, "DDI / allergy rules", ["Treatment reconciliation"])
    box(ax, 320, 600, 608, 80, "ClinicalState · one invocation",
        ["Patient, diagnosis, plan, coding, audit, errors · checkpoints opt-in"], fill=SURFACE)
    box(ax, 1008, 368, 232, 176, "LLM backend",
        ["OpenAI-compatible API", "or Codex CLI (ChatGPT)", "Intake / Diagnosis /", "Treatment / Coding"],
        tag="MODEL SERVICE", fill=SURFACE)
    text(ax, 40, 736, "Coding proposes codes through the LLM; deterministic ICD-10/DRG lookups serve separate endpoints.",
         size=14, color=MUTED)
    text(ax, 40, 764, "Neo4j and FHIR helpers sit outside this default workflow. No database is required for the default run.",
         size=14, color=MUTED)
    legend(ax)
    return fig


def workflow():
    fig, ax = canvas("How the five Agents collaborate",
                     "Sequential graph nodes exchange one shared ClinicalState · four LLM stages, then deterministic Audit")
    route(ax, [(280, 288), (384, 288)])
    route(ax, [(624, 288), (688, 288)])
    route(ax, [(848, 288), (960, 288)])
    route(ax, [(768, 232), (768, 136), (160, 136), (160, 200)], color=BLUE, dashed=True)
    route(ax, [(1100, 376), (1100, 512)])
    route(ax, [(960, 596), (624, 596)])
    route(ax, [(384, 596), (280, 596)])
    edge_label(ax, 904, 264, "OTHERWISE")
    edge_label(ax, 820, 180, "RETRY", color=BLUE)
    edge_label(ax, 1172, 444, "PLAN")
    edge_label(ax, 792, 572, "CODING_RESULT")
    box(ax, 40, 200, 240, 176, "Intake",
        ["Read raw_input", "Validate patient fields", "Write patient_info"], tag="01 · LLM")
    box(ax, 384, 200, 240, 176, "Diagnosis",
        ["Read patient_info", "Use bundled symptom map", "Write diagnosis + trace", "Update retry flags"],
        tag="02 · LLM + RETRIEVAL")
    ax.add_patch(Polygon([(768, 232), (848, 288), (768, 344), (688, 288)],
                         facecolor=TINT, edgecolor=BLUE, linewidth=1.5, zorder=3))
    text(ax, 768, 280, "More info", size=15, weight="bold", ha="center")
    text(ax, 768, 302, "+ retries left?", size=13, ha="center")
    box(ax, 960, 200, 280, 176, "Treatment",
        ["Read patient + diagnosis", "Reconcile DDI / allergies", "Write treatment_plan", "Record ddi_engine stamp"],
        tag="03 · LLM THEN RULES")
    box(ax, 960, 512, 280, 168, "Coding",
        ["Read diagnosis + plan", "Propose ICD-10 / DRG", "Write coding_result"], tag="04 · LLM")
    box(ax, 384, 512, 240, 168, "Audit",
        ["Post-model PHI scan", "Controls stay unverified", "Write audit_result"],
        tag="05 · DETERMINISTIC", fill=SURFACE)
    box(ax, 40, 528, 240, 136, "API response",
        ["Five output sections", "+ accumulated errors"], tag="END", fill=SURFACE, radius=20)
    text(ax, 40, 728, "Retry only if needs_more_info = true AND diagnosis_attempts <= max_retries.",
         size=14, color=MUTED, mono=True)
    text(ax, 40, 756, "Default: 2 loopbacks, up to 3 diagnosis attempts. A retry reuses the same narrative; otherwise proceed.",
         size=14, color=MUTED)
    text(ax, 40, 780, "Audit runs after model calls and reports findings; it does not redact the returned patient data.",
         size=14, color=MUTED)
    legend(ax, workflow=True)
    return fig


def save(fig, output_dir: Path, slug: str, title: str, description: str):
    buffer = io.StringIO()
    fig.savefig(buffer, format="svg", metadata={"Date": None}, facecolor=PAPER)
    svg = buffer.getvalue()
    opening = re.search(r"<svg\b[^>]*>", svg).group()
    accessible = opening[:-1] + (
        f' id="{slug}-root" role="img" aria-labelledby="{slug}-title {slug}-desc">'
        f'<title id="{slug}-title">{html.escape(title)}</title>'
        f'<desc id="{slug}-desc">{html.escape(description)}</desc>'
    )
    svg = svg.replace(opening, accessible, 1)
    svg = svg[svg.index("<svg"):]
    page = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f'<title>{html.escape(title)}</title><style>'
        'body{margin:0;background:#fff}.diagram-container{width:100%;overflow-x:auto}'
        f'svg{{width:100%;min-width:{WIDTH}px;display:block}}'
        '@media print{.diagram-container{overflow-x:visible}svg{min-width:0}}'
        '</style></head><body><div class="diagram-container">'
        + svg + '</div></body></html>\n'
    )
    (output_dir / f"{slug}.html").write_text(page, encoding="utf-8")
    (output_dir / f"{slug}.svg").write_text(svg, encoding="utf-8")
    fig.savefig(output_dir / f"{slug}.png", dpi=144, facecolor=PAPER)
    plt.close(fig)
    print(f"Rendered {slug}: HTML, SVG, PNG")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.hashsalt": "medical-architecture"})
    save(architecture(), args.output_dir, "system-architecture", "System architecture",
         "FastAPI invokes a LangGraph workflow with per-invocation state, a configured LLM backend, "
         "local retrieval and treatment rules, plus independent ICD-10 and DRG API tools.")
    save(workflow(), args.output_dir, "agent-workflow", "Five-Agent collaboration workflow",
         "Intake, Diagnosis, Treatment, Coding and Audit run sequentially through shared state, "
         "with a bounded Diagnosis-to-Intake retry and a final structured API response.")


if __name__ == "__main__":
    main()
