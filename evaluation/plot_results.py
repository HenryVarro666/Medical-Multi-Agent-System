"""Render README figures from the preserved evaluation JSON, without model calls.

Run from the repository root:
    uv run --locked --extra plots python -m evaluation.plot_results
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from textwrap import fill

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "evaluation/results/historical-2026-07-13/eval_results.json"
DEFAULT_OUTPUT = ROOT / "docs/assets/evaluation"
INK = "#172B45"
MUTED = "#53657B"
BLUE = "#2563EB"
PURPLE = "#7954C8"
SLATE = "#64748B"
GRID = "#E7EDF4"


def style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 12,
        "text.color": INK,
        "axes.labelcolor": MUTED,
        "xtick.color": MUTED,
        "ytick.color": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.spines.bottom": False,
        "axes.axisbelow": True,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "svg.hashsalt": "medical-evaluation",
    })


def heading(fig, record: dict, title: str) -> None:
    meta = record["meta"]
    fig.text(0.04, 0.95, title, size=21, weight="bold", va="top")
    subtitle = (
        f"Historical run · {meta['timestamp'][:10]} · {meta['model']} · "
        f"{meta['n_cases']} synthetic cases × {meta['repeats']} repeats"
    )
    fig.text(0.04, 0.89, subtitle, size=11.5, color=MUTED, va="top")


def save(fig, output_dir: Path, name: str) -> None:
    for extension in ("svg", "png"):
        metadata = {"Date": None} if extension == "svg" else {}
        path = output_dir / f"{name}.{extension}"
        fig.savefig(path, dpi=180, metadata=metadata, facecolor="white")
        print(path)
    plt.close(fig)


def plot_metrics(record: dict, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 8.6))
    fig.subplots_adjust(left=0.34, right=0.94, top=0.78, bottom=0.17)
    heading(fig, record, "Recorded evaluation results")
    metrics = [
        ("Top-1 diagnosis · strict alias", "diagnosis_top1_strict", BLUE, 0),
        ("Top-1 diagnosis · either alias", "diagnosis_top1_any", BLUE, 1),
        ("Top-3 diagnosis · either alias", "diagnosis_top3_any", BLUE, 2),
        ("ICD-10 · exact label", "icd10_exact", PURPLE, 3),
        ("ICD-10 · category label", "icd10_category", PURPLE, 4),
        ("Treatment structure complete", "treatment_ok", SLATE, 6),
        ("DDI engine executed", "ddi_engine_checked", SLATE, 7),
        ("Audit report produced", "audit_completed", SLATE, 8),
    ]
    for label, key, color, y in metrics:
        lookup_key = "treatment_structure_complete" if (
            key == "treatment_ok" and key not in record["aggregate"]
        ) else key
        result = record["aggregate"][lookup_key]
        fraction = result["hits"] / result["n"]
        ax.barh(y, 1, height=0.58, color="#F0F4F8")
        ax.barh(y, fraction, height=0.58, color=color)
        ax.text(
            fraction - 0.02 if fraction >= 0.3 else fraction + 0.02, y,
            f"{result['hits']}/{result['n']}  ·  {fraction:.0%}",
            ha="right" if fraction >= 0.3 else "left", va="center",
            color="white" if fraction >= 0.3 else INK, weight="bold", size=12,
        )
    ax.set_yticks([item[3] for item in metrics], [item[0] for item in metrics])
    ax.tick_params(axis="both", length=0, pad=10)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(8.7, -1.1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(axis="x", color=GRID)
    ax.set_xlabel("Share of recorded runs meeting each check", labelpad=15)
    for y, text in [(-0.8, "LABEL MATCHING"), (5.25, "OUTPUT STRUCTURE & STAGE EXECUTION")]:
        ax.text(-0.47, y, text, transform=ax.get_yaxis_transform(), size=11,
                weight="bold", color=MUTED, va="center", clip_on=False)
    fig.text(0.04, 0.075, "Diagnosis uses substring alias rules; ICD-10 uses fixture labels.",
             size=10.5, color=MUTED)
    fig.text(0.04, 0.045, "Completion checks do not establish treatment safety or HIPAA compliance.",
             size=10.5, color=MUTED)
    save(fig, output_dir, "historical-metrics")


def plot_latency(record: dict, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.subplots_adjust(left=0.08, right=0.96, top=0.77, bottom=0.29)
    heading(fig, record, "End-to-end latency by case")
    rows = record["per_case"]
    case_ids = list(dict.fromkeys(row["case_id"] for row in rows))
    repeats = sorted({row["repeat"] for row in rows})
    width = 0.72 / len(repeats)
    colors = [BLUE, "#91B4F8"]
    for index, repeat in enumerate(repeats):
        values = [next(row["latency_seconds"] for row in rows
                       if row["case_id"] == case_id and row["repeat"] == repeat)
                  for case_id in case_ids]
        positions = [x + (index - (len(repeats) - 1) / 2) * width for x in range(len(case_ids))]
        bars = ax.bar(positions, values, width=width * 0.88,
                      color=colors[index % len(colors)], label=f"Repeat {repeat}")
        ax.bar_label(bars, labels=[f"{value:.2f}" for value in values], padding=5,
                     size=10.5, bbox={"facecolor": "white", "edgecolor": "none", "pad": 1})
    mean = sum(row["latency_seconds"] for row in rows) / len(rows)
    ax.axhline(mean, color=SLATE, linewidth=1.4, linestyle=(0, (4, 3)),
               label=f"Overall mean · {mean:.2f} s")
    labels = [case_id + "\n" + fill(next(row["expected_diagnosis"] for row in rows
                                        if row["case_id"] == case_id), width=23)
              for case_id in case_ids]
    ax.set_xticks(range(len(case_ids)), labels, size=10)
    ax.tick_params(axis="both", length=0, pad=9)
    ax.set_ylim(0, max(row["latency_seconds"] for row in rows) * 1.21)
    ax.set_ylabel("Pipeline latency (seconds)", labelpad=12)
    ax.grid(axis="y", color=GRID)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), frameon=False,
              ncol=len(repeats) + 1, fontsize=11, borderaxespad=0)
    fig.text(0.04, 0.065, "Each pair reuses the same synthetic case. Two repeats do not estimate uncertainty.",
             size=10.5, color=MUTED)
    fig.text(0.04, 0.035, "Recorded backend timings; no clinical validation or throughput claim.",
             size=10.5, color=MUTED)
    save(fig, output_dir, "historical-latency")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    record = json.loads(args.results.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    style()
    plot_metrics(record, args.output_dir)
    plot_latency(record, args.output_dir)


if __name__ == "__main__":
    main()
