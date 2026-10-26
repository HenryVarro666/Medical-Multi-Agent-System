"""Plot two matched, two-case Codex evaluation summaries without model calls.

The --results option takes exactly two JSON paths in model display order:
    uv run --locked --extra plots python -m evaluation.plot_comparison \
        --results path/to/model-a/eval_results.json path/to/model-b/eval_results.json

Outputs default to docs/assets/evaluation/codex-comparison-2026-10-26-*.{png,svg}.
Coding-v2 summaries require an explicit, distinct --prefix, such as coding-v2-2026-10-26.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from textwrap import fill

from .plot_results import (
    BLUE,
    DEFAULT_OUTPUT,
    ERROR,
    GRID,
    INK,
    MUTED,
    PURPLE,
    Patch,
    PercentFormatter,
    filename_prefix,
    plt,
    save,
    style,
)

CASE_IDS = ("case_001", "case_002")
COLORS = (BLUE, PURPLE)
DEFAULT_PREFIX = "codex-comparison-2026-10-26"
V2_SCORING = "coding-validation-v2"
V2_DATASET = "coding-v2-fy2027"
METRICS = (
    ("Top-1 diagnosis · strict alias", "diagnosis_top1_strict", 0),
    ("Top-1 diagnosis · either alias", "diagnosis_top1_any", 1),
    ("Top-3 diagnosis · either alias", "diagnosis_top3_any", 2),
    ("ICD-10 · exact label", "icd10_exact", 3),
    ("ICD-10 · category label", "icd10_category", 4),
    ("Treatment structure complete", "treatment_structure_complete", 6),
    ("DDI engine executed", "ddi_engine_checked", 7),
    ("Audit report produced", "audit_completed", 8),
)
V2_METRICS = (
    *METRICS[:3],
    ("Exact v2 complete-code reference", "icd10_exact", 3),
    ("ICD-10 · category label", "icd10_category", 4),
    ("Primary code valid & complete", "icd10_primary_catalog_valid", 6),
    ("All codes valid & complete", "icd10_all_codes_catalog_valid", 7),
    ("Treatment structure complete", "treatment_structure_complete", 9),
    ("DDI engine executed", "ddi_engine_checked", 10),
    ("Audit report produced", "audit_completed", 11),
)
MATCHED_FIELDS = (
    "fixture_sha256", "n_cases", "repeats", "reasoning_effort",
    "llm_timeout_seconds", "max_diagnosis_retries", "llm_backend",
)


def is_v2(record: dict) -> bool:
    return record["meta"].get("scoring_version") == V2_SCORING


def catalog_identity(meta: dict) -> tuple[str, str, str]:
    catalog = meta["coding_catalog"]
    code_set, release, source_hash = catalog["code_set"], catalog["release"], catalog["source"]["sha256"]
    if code_set != "ICD-10-CM" or release != "FY2027":
        raise ValueError("Coding-v2 requires the FY2027 ICD-10-CM catalog.")
    if not isinstance(source_hash, str) or len(source_hash) != 64 or any(
        char not in "0123456789abcdef" for char in source_hash.lower()
    ):
        raise ValueError("Coding-v2 requires a source-archive SHA-256 in coding_catalog.")
    return code_set, release, source_hash.lower()


def validate_records(records: list[dict]) -> None:
    """Reject unmatched protocols, pilot runs, or inconsistent summary counts."""
    if len(records) != 2:
        raise ValueError("Provide exactly two evaluation summaries.")
    versions = [record["meta"].get("scoring_version") for record in records]
    if versions[0] != versions[1] or versions[0] not in (None, V2_SCORING):
        raise ValueError("Both summaries must use the same supported scoring version.")
    v2 = is_v2(records[0])
    metrics = V2_METRICS if v2 else METRICS
    if v2:
        if any(record["meta"].get("dataset_version") != V2_DATASET for record in records):
            raise ValueError("Coding-v2 comparisons require dataset_version=coding-v2-fy2027.")
        if catalog_identity(records[0]["meta"]) != catalog_identity(records[1]["meta"]):
            raise ValueError("The summaries do not match on coding catalog release/source archive.")
    models = []
    for index, record in enumerate(records, 1):
        meta = record["meta"]
        for field in MATCHED_FIELDS:
            if field not in meta or meta[field] is None or meta[field] == "":
                raise ValueError(f"Summary {index} is missing protocol metadata: {field}.")
        model = meta["model"]
        if not isinstance(model, str) or not model.strip():
            raise ValueError(f"Summary {index} must name its requested model.")
        models.append(model)
        if meta["llm_backend"] != "codex":
            raise ValueError("Both summaries must use the Codex CLI backend.")
        if meta["n_cases"] != 2 or meta["repeats"] != 1:
            raise ValueError("Each summary must contain exactly two cases with one repeat.")
        rows = record["per_case"]
        if len(rows) != 2 or {(row["case_id"], row["repeat"]) for row in rows} != {
            (case_id, 1) for case_id in CASE_IDS
        }:
            raise ValueError("Each summary must contain case_001 and case_002 exactly once.")
        summary = record["aggregate"]
        if summary["n_runs"] != 2:
            raise ValueError(f"Summary {index} must report n_runs=2.")
        for _, key, _ in metrics:
            if any(type(row[key]) is not bool for row in rows):
                raise ValueError(f"Per-case {key} values must be booleans.")
            metric = summary[key]
            hits = sum(row[key] for row in rows)
            if metric["n"] != 2 or metric["hits"] != hits or metric["fraction"] != hits / 2:
                raise ValueError(f"Summary {index} has inconsistent counts for {key}.")
        for row in rows:
            value = row["latency_seconds"]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError("Latencies must be finite nonnegative numbers.")
        mean = sum(row["latency_seconds"] for row in rows) / 2
        if summary["mean_latency_seconds"] != round(mean, 2):
            raise ValueError(f"Summary {index} has an inconsistent mean latency.")
    if models[0] == models[1]:
        raise ValueError("Use two distinct requested model names.")
    for field in MATCHED_FIELDS:
        if records[0]["meta"][field] != records[1]["meta"][field]:
            raise ValueError(f"The summaries do not match on {field}.")
    for field in ("git_revision", "lockfile_sha256"):
        if records[0]["meta"].get(field) != records[1]["meta"].get(field):
            raise ValueError(f"The summaries do not match on {field}.")
    first = {row["case_id"]: row for row in records[0]["per_case"]}
    second = {row["case_id"]: row for row in records[1]["per_case"]}
    for case_id in CASE_IDS:
        for field in ("expected_diagnosis", "expected_icd10"):
            if first[case_id][field] != second[case_id][field]:
                raise ValueError(f"The summaries disagree on {case_id} {field}.")


def heading(fig, records: list[dict], title: str) -> None:
    meta = records[0]["meta"]
    dates = list(dict.fromkeys(record["meta"]["timestamp"][:10] for record in records))
    fig.text(0.04, 0.96, title, size=21, weight="bold", va="top")
    fig.text(0.04, 0.905,
             f"{meta['n_cases']} synthetic cases per model · 1 repeat · Codex CLI · "
             f"{meta['reasoning_effort']} reasoning", size=11.5, color=MUTED, va="top")
    fig.text(0.04, 0.865,
             f"{' / '.join(dates)} · {meta['llm_timeout_seconds']:g} s per-call timeout · "
             f"{meta['max_diagnosis_retries']} diagnosis loopbacks", size=10.5, color=MUTED, va="top")


def plot_metrics(records: list[dict], output_dir: Path, prefix: str) -> None:
    v2 = is_v2(records[0])
    metrics = V2_METRICS if v2 else METRICS
    fig, ax = plt.subplots(figsize=(12, 12 if v2 else 9.5))
    fig.subplots_adjust(left=0.34, right=0.94, top=0.76, bottom=0.17)
    heading(fig, records, "Matched coding v2 comparison · recorded checks" if v2
            else "Matched model comparison · recorded checks")
    for index, (record, color) in enumerate(zip(records, COLORS)):
        for _, key, y in metrics:
            metric = record["aggregate"][key]
            fraction = metric["hits"] / metric["n"]
            position = y + (index - 0.5) * 0.34
            ax.barh(position, 1, height=0.28, color="#F0F4F8")
            ax.barh(position, fraction, height=0.28, color=color)
            ax.text(fraction - 0.02 if fraction >= 0.3 else fraction + 0.02, position,
                    f"{metric['hits']}/{metric['n']} · {fraction:.0%}",
                    ha="right" if fraction >= 0.3 else "left", va="center",
                    color="white" if fraction >= 0.3 else INK, weight="bold", size=10.5)
    ax.set_yticks([item[2] for item in metrics], [item[0] for item in metrics])
    ax.tick_params(axis="both", length=0, pad=10)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(11.7 if v2 else 8.7, -1.1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(axis="x", color=GRID)
    ax.set_xlabel("Share of recorded runs meeting each check", labelpad=15)
    groups = (
        ((-0.8, "REFERENCE LABEL MATCHING"), (5.25, "FY2027 CATALOG COMPLETENESS"),
         (8.25, "OUTPUT STRUCTURE & STAGE EXECUTION")) if v2
        else ((-0.8, "LABEL MATCHING"), (5.25, "OUTPUT STRUCTURE & STAGE EXECUTION"))
    )
    for y, label in groups:
        ax.text(-0.47, y, label, transform=ax.get_yaxis_transform(), size=11,
                weight="bold", color=MUTED, va="center", clip_on=False)
    ax.legend(handles=[Patch(facecolor=color, label=record["meta"]["model"])
                       for record, color in zip(records, COLORS)],
              loc="lower left", bbox_to_anchor=(0, 1.045), frameon=False,
              ncol=2, fontsize=11, borderaxespad=0)
    fig.text(0.04, 0.075,
             "Catalog completeness and exact reference agreement do not adjudicate clinical correctness." if v2
             else "Substring aliases and fixture codes only; n=2 per model does not estimate uncertainty.",
             size=10.5, color=MUTED)
    fig.text(0.04, 0.045,
             "n=2 per model; no uncertainty estimate. Structure checks do not establish safety or compliance." if v2
             else "Structure/execution checks do not establish treatment safety or HIPAA compliance.",
             size=10.5, color=MUTED)
    save(fig, output_dir, f"{prefix}-metrics")


def plot_latency(records: list[dict], output_dir: Path, prefix: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 7.6))
    fig.subplots_adjust(left=0.085, right=0.96, top=0.75, bottom=0.28)
    heading(fig, records, "Matched coding v2 comparison · elapsed time" if is_v2(records[0])
            else "Matched model comparison · elapsed time")
    has_errors = False
    means = []
    for index, (record, color) in enumerate(zip(records, COLORS)):
        by_case = {row["case_id"]: row for row in record["per_case"]}
        rows = [by_case[case_id] for case_id in CASE_IDS]
        values = [row["latency_seconds"] for row in rows]
        positions = [x + (index - 0.5) * 0.34 for x in range(len(CASE_IDS))]
        bars = ax.bar(positions, values, width=0.30, color=color)
        for bar, row in zip(bars, rows):
            if row.get("pipeline_errors") or row.get("harness_error"):
                has_errors = True
                bar.set_hatch("///")
                bar.set_edgecolor(ERROR)
                bar.set_linewidth(1.5)
        ax.bar_label(bars, labels=[f"{value:.2f} s" for value in values], padding=6,
                     size=10.5, bbox={"facecolor": "white", "edgecolor": "none", "pad": 1})
        mean = sum(values) / len(values)
        means.append(mean)
        ax.axhline(mean, color=color, linewidth=1.4, linestyle=(0, (4, 3)))
    rows = {row["case_id"]: row for row in records[0]["per_case"]}
    labels = [case_id + "\n" + fill(rows[case_id]["expected_diagnosis"], width=30)
              for case_id in CASE_IDS]
    ax.set_xticks(range(len(CASE_IDS)), labels, size=11)
    ax.tick_params(axis="both", length=0, pad=10)
    maximum = max(row["latency_seconds"] for record in records for row in record["per_case"])
    ax.set_ylim(0, maximum * 1.22 if maximum else 1)
    ax.set_ylabel("Pipeline elapsed time (seconds)", labelpad=12)
    ax.grid(axis="y", color=GRID)
    handles = [Patch(facecolor=color, label=record["meta"]["model"])
               for record, color in zip(records, COLORS)]
    if has_errors:
        handles.append(Patch(facecolor="white", edgecolor=ERROR, hatch="///",
                             label="Pipeline / harness error"))
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1.035), frameon=False,
              ncol=len(handles), fontsize=11, borderaxespad=0)
    for index, (record, color, mean) in enumerate(zip(records, COLORS, means)):
        fig.text(0.04 + index * 0.47, 0.14, f"{record['meta']['model']} mean: {mean:.2f} s",
                 size=11.5, color=color, weight="bold")
    fig.text(0.04, 0.085, "Dashed lines show model means; failed observations remain included.",
             size=10.5, color=MUTED)
    fig.text(0.04, 0.05,
             "Codex CLI wall time includes startup/transport overhead; n=2 per model does not estimate uncertainty.",
             size=10.5, color=MUTED)
    save(fig, output_dir, f"{prefix}-latency")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", nargs=2, required=True, type=Path,
                        metavar=("MODEL_A_JSON", "MODEL_B_JSON"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--prefix", type=filename_prefix,
                        help=f"v1 default: {DEFAULT_PREFIX}; coding-v2 requires a distinct explicit prefix")
    args = parser.parse_args()
    try:
        records = [json.loads(path.read_text(encoding="utf-8")) for path in args.results]
        validate_records(records)
        if is_v2(records[0]) and (args.prefix is None or args.prefix == DEFAULT_PREFIX):
            raise ValueError("Coding-v2 requires a distinct explicit --prefix, such as coding-v2-2026-10-26.")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.prefix = args.prefix or DEFAULT_PREFIX
    style()
    plot_metrics(records, args.output_dir, args.prefix)
    plot_latency(records, args.output_dir, args.prefix)


if __name__ == "__main__":
    main()
