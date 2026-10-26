#!/usr/bin/env python
"""Run the five synthetic vignettes through a configured LLM pipeline.

Run from the repository root after installing the package:
    python -m evaluation.run_eval --repeats 2
    python -m evaluation.run_eval --cases case_001 --output-dir /tmp/clinical-eval

Scoring functions and --help need only the Python standard library. A live run
requires the medical_multi_agent package and a working configured LLM backend.
The alias rules are demonstration heuristics, not clinical judgments.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tempfile
import time
import uuid
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CASES_PATH = ROOT / "data" / "sample_patients.json"
RESULTS_DIR = ROOT / "evaluation" / "results"
HISTORICAL_DIR = RESULTS_DIR / "historical-2026-07-13"
METRICS = (
    "diagnosis_top1_strict", "diagnosis_top1_any", "diagnosis_top3_any",
    "icd10_exact", "icd10_category", "treatment_structure_complete",
    "ddi_engine_checked", "audit_completed",
)

# Keep the historical alias rules explicit for comparisons. "Strict" and
# "compatible" name these string-matching tiers; they do not certify diagnoses.
DX_ALIASES: dict[str, dict[str, list[str]]] = {
    "case_001": {  # Community-Acquired Pneumonia
        "strict": [
            "community-acquired pneumonia",
            "community acquired pneumonia",
            "bacterial pneumonia",
            "lobar pneumonia",
        ],
        "compatible": ["pneumonia"],
    },
    "case_002": {  # ST-Elevation Myocardial Infarction (STEMI)
        "strict": [
            "stemi",
            "st-elevation myocardial infarction",
            "st elevation myocardial infarction",
            "st segment elevation myocardial infarction",
        ],
        "compatible": [
            "myocardial infarction",
            "acute mi",
            "acute coronary syndrome",
        ],
    },
    "case_003": {  # Hypothyroidism (Hashimoto's Thyroiditis)
        "strict": [
            "hashimoto",
            "primary hypothyroidism",
            "hypothyroidism",
        ],
        "compatible": ["thyroiditis", "thyroid"],
    },
    "case_004": {  # Acute Decompensated Heart Failure
        "strict": [
            "acute decompensated heart failure",
            "decompensated heart failure",
            "acute heart failure",
            "adhf",
            "heart failure exacerbation",
            "acute on chronic heart failure",
            "acute-on-chronic heart failure",
        ],
        "compatible": ["heart failure", "chf", "cardiomyopathy"],
    },
    "case_005": {  # Acute Appendicitis
        "strict": ["acute appendicitis", "appendicitis"],
        "compatible": [],
    },
}



def _norm(text: str) -> str:
    text = re.sub(r"[^a-z0-9\s-]", " ", (text or "").lower())
    return re.sub(r"\s+", " ", text).strip()


def match_diagnosis(case_id: str, predicted: str) -> str:
    """Return an alias tier using substring matching without negation handling."""
    pred = _norm(predicted)
    aliases = DX_ALIASES.get(case_id, {})
    for tier in ("strict", "compatible"):
        if any(_norm(alias) in pred for alias in aliases.get(tier, [])):
            return tier
    return "miss"


def icd10_scores(expected: str, predicted: str) -> tuple[bool, bool]:
    """Return exact and category (prefix before the decimal) matches."""
    expected = (expected or "").strip().upper()
    predicted = (predicted or "").strip().upper()
    if not expected or not predicted:
        return False, False
    return expected == predicted, expected.split(".")[0] == predicted.split(".")[0]


def score_output(case: dict, result: dict, repeat: int, latency: float,
                 error: str | None = None) -> dict:
    """Score a pipeline result without invoking a model or writing files."""
    diagnosis = result.get("diagnosis") or {}
    coding = result.get("coding_result") or {}
    treatment = result.get("treatment_plan") or {}
    audit = result.get("audit_result") or {}
    primary = (diagnosis.get("primary_diagnosis") or {}).get("disease_name", "")
    differentials = [
        (item or {}).get("disease_name", "")
        for item in (diagnosis.get("differential_list") or [])
    ]
    top1 = match_diagnosis(case["id"], primary)
    predicted_code = (coding.get("primary_icd10") or {}).get("code", "")
    exact, category = icd10_scores(case["expected_icd10"], predicted_code)
    ddi = treatment.get("ddi_engine") or {}
    candidates = diagnosis.get("knowledge_graph_candidates") or []
    return {
        "case_id": case["id"],
        "repeat": repeat,
        "expected_diagnosis": case["expected_diagnosis"],
        "predicted_diagnosis": primary,
        "diagnosis_top1": top1,
        "diagnosis_top1_strict": top1 == "strict",
        "diagnosis_top1_any": top1 != "miss",
        "diagnosis_top3_any": any(
            match_diagnosis(case["id"], name) != "miss"
            for name in [primary] + differentials[:2]
        ),
        "expected_icd10": case["expected_icd10"],
        "predicted_icd10": predicted_code,
        "icd10_exact": exact,
        "icd10_category": category,
        "treatment_structure_complete": bool(treatment.get("medications") and ddi.get("checked") is True),
        "ddi_engine_checked": ddi.get("checked") is True,
        "ddi_engine_flags": ddi.get("engine_flagged_interactions", 0),
        "allergy_alerts": ddi.get("engine_flagged_allergy_alerts", 0),
        "graphrag_candidates_retrieved": len(candidates),
        "graphrag_top_candidate": candidates[0]["disease"] if candidates else "",
        "audit_completed": bool(audit),
        "diagnosis_attempts": result.get("diagnosis_attempts"),
        "pipeline_errors": result.get("errors") or [],
        "harness_error": error,
        "latency_seconds": round(latency, 2),
    }


def _validate_output_dir(output_dir: Path) -> Path:
    output_dir = output_dir.resolve()
    if output_dir.is_relative_to(HISTORICAL_DIR.resolve()):
        raise ValueError("The historical evaluation archive is read-only; choose another output directory.")
    return output_dir


def evaluate_case(pipeline, case: dict, repeat: int, output_dir: Path,
                  recursion_limit: int, capture_calls=None) -> dict:
    """Invoke a supplied pipeline and archive its result, including failures."""
    output_dir = _validate_output_dir(Path(output_dir))
    raw_dir = output_dir / "raw"
    for directory in (output_dir, raw_dir):
        directory.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryFile(dir=directory) as probe:
            probe.write(b"evaluation output write check")
            probe.flush()
    thread_id = f"eval_{case['id']}_r{repeat}_{uuid.uuid4().hex[:8]}"
    start = time.perf_counter()
    result = {}
    error = None
    with capture_calls() if capture_calls else nullcontext([]) as llm_calls:
        try:
            result = pipeline.invoke(
                {"raw_input": case["description"]},
                config={"configurable": {"thread_id": thread_id}, "recursion_limit": recursion_limit},
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
    latency = time.perf_counter() - start
    try:
        row = score_output(case, result, repeat, latency, error)
    except Exception as exc:
        error = f"Scoring error: {type(exc).__name__}: {exc}"
        row = score_output(case, {}, repeat, latency, error)
        row["pipeline_errors"] = result.get("errors") or []
        row["diagnosis_attempts"] = result.get("diagnosis_attempts")
    raw_path = raw_dir / f"{case['id']}_r{repeat}.json"
    with raw_path.open("x", encoding="utf-8") as stream:
        json.dump({
            "case": case,
            "pipeline_output": {key: result.get(key) for key in (
                "patient_info", "diagnosis", "treatment_plan", "coding_result",
                "audit_result", "errors", "diagnosis_attempts",
            )},
            "harness_error": error,
            "latency_seconds": row["latency_seconds"],
            **({"llm_calls": llm_calls} if capture_calls else {}),
        }, stream, indent=2, default=str)
    return row


def aggregate(rows: list[dict]) -> dict:
    """Aggregate run-level counts; repeats are not independent clinical cases."""
    count = len(rows)
    result = {"n_runs": count}
    for key in METRICS:
        hits = sum(bool(row[key]) for row in rows)
        result[key] = {"hits": hits, "n": count, "fraction": round(hits / count, 3) if count else None}
    result["mean_latency_seconds"] = round(
        sum(row["latency_seconds"] for row in rows) / count, 2
    ) if count else None
    return result


def to_markdown(meta: dict, rows: list[dict], summary: dict) -> str:
    lines = [
        "# Synthetic-vignette evaluation", "",
        f"- Run at: {meta['timestamp']}",
        f"- Model: `{meta['model']}`; backend: `{meta.get('backend', meta['base_url'] or 'OpenAI API')}`",
        f"- Cases: {meta['n_cases']}; repeats: {meta['repeats']}; runs: {summary['n_runs']}",
        f"- Source revision: `{meta['git_revision']}`; uncommitted changes: {meta['git_dirty']}",
        "- Raw pipeline results: `raw/`", "",
        "| Case | Repeat | Diagnosis alias tier | ICD exact | ICD category | Latency (s) |",
        "|---|---:|---|---|---|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['case_id']} | {row['repeat']} | {row['diagnosis_top1']} "
            f"| {row['icd10_exact']} | {row['icd10_category']} | {row['latency_seconds']} |"
        )
    lines += ["", "| Metric | Hits / runs | Fraction |", "|---|---:|---:|"]
    for key in METRICS:
        metric = summary[key]
        lines.append(f"| {key} | {metric['hits']} / {metric['n']} | {metric['fraction']} |")
    lines += [
        "", f"Mean pipeline latency: **{summary['mean_latency_seconds']} s**", "",
        "## Interpretation", "",
        "These are demonstration measurements on synthetic fixtures, not clinical validation.",
        "Diagnosis tiers use case-specific substring aliases without negation handling.",
        "Repeated runs reuse the same cases; they do not increase clinical case diversity.",
        "The small label set has not been independently reviewed by clinicians or coders.",
        "`treatment_structure_complete` checks for medications and a DDI execution stamp;",
        "it does not assess drug selection, dosage, completeness, or safety.",
        "`audit_completed` means a report exists; it does not establish HIPAA compliance.",
        "Latency depends on the backend and hardware; no throughput claim is made.",
    ]
    return "\n".join(lines) + "\n"


def _backend_metadata(settings) -> dict:
    """Record requested inference settings without credentials or CLI state."""
    backend = getattr(settings, "llm_backend", "openai")
    common = {
        "llm_backend": backend,
        "llm_timeout_seconds": getattr(settings, "llm_timeout_seconds", None),
        "max_diagnosis_retries": settings.max_diagnosis_retries,
        "lockfile_sha256": hashlib.sha256((ROOT / "uv.lock").read_bytes()).hexdigest(),
    }
    if backend == "codex":
        from medical_multi_agent.config.codex import get_codex_version

        return {
            **common,
            "model": settings.codex_model,
            "base_url": "",
            "backend": "Codex CLI (saved ChatGPT authentication)",
            "codex_cli_version": get_codex_version(settings.codex_cli_path),
            "reasoning_effort": settings.codex_reasoning_effort,
            "temperature": "unsupported by Codex CLI; original agent temperatures not applied",
            "prompt_transport": "original system text as model_instructions_file; original human text on stdin",
            "model_revision": "unavailable; model name is requested, backend resolution not exposed",
            "cli_retries": "no adapter retries; CLI transport recovery may occur internally",
            "isolation": "fresh ephemeral exec in a temporary directory per call; user config ignored",
        }
    return {
        **common,
        "model": settings.openai_model,
        "base_url": settings.openai_base_url,
        "backend": settings.openai_base_url or "OpenAI API",
        "temperature": {"intake": 0.1, "diagnosis": 0.2, "treatment": 0.2, "coding": 0.1},
        "client_max_retries": 1,
    }


def _positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=_positive_int, default=1)
    parser.add_argument("--cases", default="", help="comma-separated fixture IDs; default: all")
    parser.add_argument("--output-dir", type=Path, help="new output directory; default: a timestamped directory under evaluation/results")
    args = parser.parse_args(argv)
    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    if args.cases:
        ids = [value.strip() for value in args.cases.split(",")]
        known_ids = {case["id"] for case in cases}
        if any(not case_id or case_id not in known_ids for case_id in ids):
            parser.error("--cases must contain valid fixture IDs: " + ", ".join(sorted(known_ids)))
        if len(set(ids)) != len(ids):
            parser.error("--cases must not contain duplicate IDs; use --repeats instead")
        cases = [case for case in cases if case["id"] in ids]
    args.selected_cases = cases
    default_name = datetime.now(timezone.utc).strftime("run-%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:6]
    try:
        args.output_dir = _validate_output_dir(args.output_dir or RESULTS_DIR / "runs" / default_name)
    except ValueError as exc:
        parser.error(str(exc))
    if args.output_dir.exists() and (not args.output_dir.is_dir() or any(args.output_dir.iterdir())):
        parser.error("--output-dir must be a new or empty directory; existing results are preserved")
    return args


def _source_state() -> dict:
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL,
        ).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL,
        ).strip())
        return {"git_revision": revision, "git_dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"git_revision": "unavailable", "git_dirty": None}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    # Deliberately lazy: --help and scoring imports need no runtime or backend.
    from medical_multi_agent.config.settings import get_settings
    from medical_multi_agent.graph.clinical_pipeline import build_clinical_pipeline

    settings = get_settings()
    meta = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **_backend_metadata(settings),
        "n_cases": len(args.selected_cases),
        "repeats": args.repeats,
        "python_version": sys.version.split()[0],
        "fixture_sha256": hashlib.sha256(CASES_PATH.read_bytes()).hexdigest(),
        **_source_state(),
    }
    print(f"[eval] model={meta['model']} cases={meta['n_cases']} repeats={args.repeats}")
    pipeline = build_clinical_pipeline()
    capture_calls = None
    if meta["llm_backend"] == "codex":
        from medical_multi_agent.config.codex import capture_codex_calls

        capture_calls = capture_codex_calls
    rows = []
    for repeat in range(1, args.repeats + 1):
        for case in args.selected_cases:
            print(f"[eval] running {case['id']} r{repeat}", flush=True)
            rows.append(evaluate_case(
                pipeline, case, repeat, args.output_dir,
                recursion_limit=2 * settings.max_diagnosis_retries + 10,
                capture_calls=capture_calls,
            ))
            row = rows[-1]
            print(f"[eval] completed {case['id']} r{repeat}: {row['latency_seconds']}s; "
                  f"alias={row['diagnosis_top1']}; errors={len(row['pipeline_errors'])}; "
                  f"harness_error={bool(row['harness_error'])}", flush=True)
    meta["completed_at"] = datetime.now(timezone.utc).isoformat()
    summary = aggregate(rows)
    with (args.output_dir / "eval_results.json").open("x", encoding="utf-8") as stream:
        json.dump({"meta": meta, "aggregate": summary, "per_case": rows}, stream, indent=2)
    with (args.output_dir / "eval_results.md").open("x", encoding="utf-8") as stream:
        stream.write(to_markdown(meta, rows, summary))
    print(f"[eval] wrote {args.output_dir}")
    return int(any(row["harness_error"] or row["pipeline_errors"] for row in rows))


if __name__ == "__main__":
    sys.exit(main())
