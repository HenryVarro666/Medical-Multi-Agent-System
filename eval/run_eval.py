#!/usr/bin/env python
"""
Eval harness for the medical multi-agent clinical decision pipeline.

Runs every labeled vignette in python/data/sample_patients.json through the
full 5-agent LangGraph pipeline (intake -> diagnosis -> treatment -> coding
-> audit) against the LLM backend configured in python/.env, and scores each
case on:

  diagnosis_top1     primary diagnosis matches the expected label
                       "strict"     = same clinical entity (alias table below)
                       "compatible" = correct but coarser clinical entity
                       "miss"       = neither
  diagnosis_top3     expected label appears among primary + top differentials
  icd10_exact        coding agent's primary ICD-10 == expected code (exact)
  icd10_category     ICD-10 category (prefix before the dot) matches
  treatment_ok       plan exists, >=1 medication, and the deterministic DDI
                     engine verifiably ran (treatment_plan.ddi_engine.checked)
  audit_completed    audit agent produced a compliance report
  latency_seconds    wall-clock time for the full pipeline run

Every number in the output comes from an actual pipeline invocation — nothing
is estimated. Raw pipeline outputs are archived under eval/results/raw/ so
each scored cell can be traced back to the model's actual output.

Usage:
    python/.venv/bin/python eval/run_eval.py [--repeats N] [--cases id1,id2]

Outputs:
    eval/results/eval_results.json
    eval/results/eval_results.md
    eval/results/raw/<case_id>_r<repeat>.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY_DIR = ROOT / "python"
RESULTS_DIR = ROOT / "eval" / "results"
RAW_DIR = RESULTS_DIR / "raw"

# Make `src` importable and let pydantic-settings find python/.env
sys.path.insert(0, str(PY_DIR))
os.chdir(PY_DIR)

from src.config.settings import get_settings  # noqa: E402
from src.graph.clinical_pipeline import build_clinical_pipeline  # noqa: E402

# ---------------------------------------------------------------------------
# Diagnosis-label matching.
#
# Clinical labels have many surface forms ("CAP" vs "community-acquired
# pneumonia"). The alias table makes the matching rule explicit and auditable
# instead of hiding it in fuzzy string similarity:
#   strict     -> same clinical entity as the label
#   compatible -> clinically correct but coarser (e.g. "acute MI" for STEMI)
# ---------------------------------------------------------------------------
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
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9\s-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def match_diagnosis(case_id: str, predicted: str) -> str:
    """Return 'strict' | 'compatible' | 'miss' for a predicted dx name."""
    pred = _norm(predicted)
    if not pred:
        return "miss"
    aliases = DX_ALIASES.get(case_id, {"strict": [], "compatible": []})
    for tier in ("strict", "compatible"):
        for alias in aliases[tier]:
            if _norm(alias) in pred:
                return tier
    return "miss"


def icd10_scores(expected: str, predicted: str) -> tuple[bool, bool]:
    """(exact_match, category_match). Category = prefix before the dot."""
    exp = (expected or "").strip().upper()
    pred = (predicted or "").strip().upper()
    if not exp or not pred:
        return False, False
    exact = exp == pred
    category = exp.split(".")[0] == pred.split(".")[0]
    return exact, category


def evaluate_case(pipeline, case: dict, repeat: int) -> dict:
    case_id = case["id"]
    thread_id = f"eval_{case_id}_r{repeat}_{uuid.uuid4().hex[:8]}"

    t0 = time.perf_counter()
    error = None
    result: dict = {}
    try:
        result = pipeline.invoke(
            {"raw_input": case["description"]},
            config={"configurable": {"thread_id": thread_id}, "recursion_limit": 40},
        )
    except Exception as e:  # pipeline-level failure is itself a data point
        error = f"{type(e).__name__}: {e}"
    latency = time.perf_counter() - t0

    diagnosis = result.get("diagnosis") or {}
    coding = result.get("coding_result") or {}
    treatment = result.get("treatment_plan") or {}
    audit = result.get("audit_result") or {}

    primary = (diagnosis.get("primary_diagnosis") or {}).get("disease_name", "")
    differentials = [
        (d or {}).get("disease_name", "")
        for d in (diagnosis.get("differential_list") or [])
    ]
    top3_names = [primary] + differentials[:2]

    top1 = match_diagnosis(case_id, primary)
    top3_hit = any(
        match_diagnosis(case_id, name) != "miss" for name in top3_names
    )

    predicted_code = ((coding.get("primary_icd10") or {}).get("code")) or ""
    icd_exact, icd_cat = icd10_scores(case["expected_icd10"], predicted_code)

    ddi_stamp = (treatment.get("ddi_engine") or {})
    treatment_ok = bool(
        treatment
        and (treatment.get("medications") or [])
        and ddi_stamp.get("checked") is True
    )

    kg_candidates = diagnosis.get("knowledge_graph_candidates") or []

    row = {
        "case_id": case_id,
        "repeat": repeat,
        "expected_diagnosis": case["expected_diagnosis"],
        "predicted_diagnosis": primary,
        "diagnosis_top1": top1,                      # strict | compatible | miss
        "diagnosis_top1_strict": top1 == "strict",
        "diagnosis_top1_any": top1 != "miss",
        "diagnosis_top3_any": top3_hit,
        "expected_icd10": case["expected_icd10"],
        "predicted_icd10": predicted_code,
        "icd10_exact": icd_exact,
        "icd10_category": icd_cat,
        "treatment_ok": treatment_ok,
        "ddi_engine_checked": ddi_stamp.get("checked") is True,
        "ddi_engine_flags": ddi_stamp.get("engine_flagged_interactions", 0),
        "allergy_alerts": ddi_stamp.get("engine_flagged_allergy_alerts", 0),
        "graphrag_candidates_retrieved": len(kg_candidates),
        "graphrag_top_candidate": (kg_candidates[0]["disease"] if kg_candidates else ""),
        "audit_completed": bool(audit),
        "hipaa_compliant_flag": audit.get("hipaa_compliant"),
        "diagnosis_attempts": result.get("diagnosis_attempts"),
        "pipeline_errors": result.get("errors") or [],
        "harness_error": error,
        "latency_seconds": round(latency, 2),
    }

    # Archive raw output for full traceability.
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = RAW_DIR / f"{case_id}_r{repeat}.json"
    with open(raw_path, "w") as f:
        json.dump(
            {
                "case": case,
                "pipeline_output": {
                    k: result.get(k)
                    for k in (
                        "patient_info", "diagnosis", "treatment_plan",
                        "coding_result", "audit_result", "errors",
                        "diagnosis_attempts",
                    )
                },
                "harness_error": error,
                "latency_seconds": row["latency_seconds"],
            },
            f, indent=2, default=str,
        )
    return row


def aggregate(rows: list[dict]) -> dict:
    n = len(rows)

    def frac(key):
        hits = sum(1 for r in rows if r[key])
        return {"hits": hits, "n": n, "fraction": round(hits / n, 3) if n else None}

    return {
        "n_runs": n,
        "diagnosis_top1_strict": frac("diagnosis_top1_strict"),
        "diagnosis_top1_any": frac("diagnosis_top1_any"),
        "diagnosis_top3_any": frac("diagnosis_top3_any"),
        "icd10_exact": frac("icd10_exact"),
        "icd10_category": frac("icd10_category"),
        "treatment_ok": frac("treatment_ok"),
        "ddi_engine_checked": frac("ddi_engine_checked"),
        "audit_completed": frac("audit_completed"),
        "mean_latency_seconds": round(
            sum(r["latency_seconds"] for r in rows) / n, 2
        ) if n else None,
    }


def to_markdown(meta: dict, rows: list[dict], agg: dict) -> str:
    lines = [
        "# Eval results — medical multi-agent pipeline",
        "",
        f"- run at: {meta['timestamp']}",
        f"- backend: `{meta['base_url'] or 'OpenAI API'}`, model: `{meta['model']}`",
        f"- cases: {meta['n_cases']} labeled vignettes x {meta['repeats']} repeat(s)"
        f" = {agg['n_runs']} pipeline runs",
        "- every value below comes from an actual pipeline run"
        " (raw outputs in `eval/results/raw/`)",
        "",
        "## Per-case results",
        "",
        "| case | expected dx | predicted dx | top-1 | top-3 | expected ICD | predicted ICD | ICD exact | ICD cat | treat OK | DDI ran | audit | lat (s) |",
        "|------|-------------|--------------|-------|-------|--------------|---------------|-----------|---------|----------|---------|-------|---------|",
    ]
    for r in rows:
        lines.append(
            "| {case_id} r{repeat} | {exp_dx} | {pred_dx} | {top1} | {top3} | {exp_icd} | {pred_icd} | {icd_e} | {icd_c} | {tr} | {ddi} | {audit} | {lat} |".format(
                case_id=r["case_id"], repeat=r["repeat"],
                exp_dx=r["expected_diagnosis"],
                pred_dx=(r["predicted_diagnosis"] or "(none)"),
                top1=r["diagnosis_top1"],
                top3="yes" if r["diagnosis_top3_any"] else "no",
                exp_icd=r["expected_icd10"],
                pred_icd=(r["predicted_icd10"] or "(none)"),
                icd_e="yes" if r["icd10_exact"] else "no",
                icd_c="yes" if r["icd10_category"] else "no",
                tr="yes" if r["treatment_ok"] else "no",
                ddi="yes" if r["ddi_engine_checked"] else "no",
                audit="yes" if r["audit_completed"] else "no",
                lat=r["latency_seconds"],
            )
        )
    lines += [
        "",
        "## Aggregate",
        "",
        "| metric | hits / n | fraction |",
        "|--------|----------|----------|",
    ]
    for key in (
        "diagnosis_top1_strict", "diagnosis_top1_any", "diagnosis_top3_any",
        "icd10_exact", "icd10_category", "treatment_ok",
        "ddi_engine_checked", "audit_completed",
    ):
        m = agg[key]
        lines.append(f"| {key} | {m['hits']} / {m['n']} | {m['fraction']} |")
    lines += [
        "",
        f"Mean pipeline latency: **{agg['mean_latency_seconds']} s**",
        "",
        "## Honest caveats",
        "",
        f"- N = {meta['n_cases']} labeled cases: demo-scale. These fractions are NOT",
        "  population-level accuracy claims; with N=5, one case swings a metric by 20 points.",
        "- Diagnosis matching uses the explicit alias table in `eval/run_eval.py`",
        "  (`strict` = same clinical entity, `compatible` = clinically correct but coarser).",
        "- ICD-10 labels are taken from `python/data/sample_patients.json` as-is;",
        "  some labels are themselves debatable (e.g. case_003 is labeled E03.9 while",
        "  a coder might defend E06.3 for Hashimoto's thyroiditis).",
        "- `hipaa_compliant_flag` is usually false because the demo PHI name regex",
        "  flags capitalized bigrams (a known precision limitation, documented in code).",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--cases", type=str, default="",
                        help="comma-separated case ids (default: all)")
    args = parser.parse_args()

    with open(PY_DIR / "data" / "sample_patients.json") as f:
        cases = json.load(f)
    if args.cases:
        wanted = {c.strip() for c in args.cases.split(",")}
        cases = [c for c in cases if c["id"] in wanted]
    if not cases:
        sys.exit("No cases selected.")

    settings = get_settings()
    meta = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": settings.openai_model,
        "base_url": settings.openai_base_url,
        "n_cases": len(cases),
        "repeats": args.repeats,
    }
    print(f"[eval] backend={meta['base_url'] or 'OpenAI'} model={meta['model']}")
    print(f"[eval] {len(cases)} cases x {args.repeats} repeat(s)")

    pipeline = build_clinical_pipeline()

    rows: list[dict] = []
    for repeat in range(1, args.repeats + 1):
        for case in cases:
            print(f"[eval] running {case['id']} (repeat {repeat}) ...", flush=True)
            row = evaluate_case(pipeline, case, repeat)
            rows.append(row)
            print(
                f"[eval]   top1={row['diagnosis_top1']}"
                f" icd_exact={row['icd10_exact']}"
                f" pred_dx={row['predicted_diagnosis']!r}"
                f" pred_icd={row['predicted_icd10']!r}"
                f" lat={row['latency_seconds']}s",
                flush=True,
            )

    agg = aggregate(rows)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "eval_results.json", "w") as f:
        json.dump({"meta": meta, "aggregate": agg, "per_case": rows}, f, indent=2)
    with open(RESULTS_DIR / "eval_results.md", "w") as f:
        f.write(to_markdown(meta, rows, agg))

    print("\n[eval] aggregate:")
    print(json.dumps(agg, indent=2))
    print(f"\n[eval] wrote {RESULTS_DIR / 'eval_results.json'}")
    print(f"[eval] wrote {RESULTS_DIR / 'eval_results.md'}")


if __name__ == "__main__":
    main()
