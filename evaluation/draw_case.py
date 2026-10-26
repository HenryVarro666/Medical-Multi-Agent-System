"""Render the archived case_001 walkthrough, without running the pipeline.

    uv run --locked --extra plots python -m evaluation.draw_case
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from textwrap import wrap

from .draw_architecture import BLUE, MUTED, SURFACE, TINT, box, canvas, route, save, text

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "evaluation/results/historical-2026-07-13"
OUTPUT = ROOT / "docs/assets/examples"


def diagram(record: dict, score: dict, meta: dict):
    state = record["pipeline_output"]
    patient = state["patient_info"]
    diagnosis = state["diagnosis"]
    treatment = state["treatment_plan"]
    ddi = treatment["ddi_engine"]
    coding = state["coding_result"]
    audit = state["audit_result"]
    flag = json.dumps
    fig, ax = canvas(
        "case_001 · from input to final result",
        f"Historical run · {meta['timestamp'][:10]} · {meta['model']} · repeat 1 · observed path (no retry)",
    )

    # Seven nodes follow the archived sequence; this run took no retry branch.
    for start, end in ((304, 352), (616, 664), (928, 976)):
        route(ax, [(start, 264), (end, 264)])
    route(ax, [(1108, 372), (1108, 464)])
    route(ax, [(976, 568), (928, 568)])
    route(ax, [(664, 568), (616, 568)], color=BLUE)

    box(ax, 40, 156, 264, 216, "Clinical narrative", [
        f"Fever {patient['vital_signs']['temperature']:.1f}°C + cough",
        "Right-sided chest pain",
        "X-ray: RLL consolidation",
        "Diabetes / hypertension",
        "Penicillin allergy",
    ], tag="INPUT · SYNTHETIC", fill=SURFACE)
    box(ax, 352, 156, 264, 216, "Intake", [
        f"{patient['age']}-year-old {patient['gender']}",
        "Symptoms + labs structured",
        f"{len(patient['current_medications'])} current medications",
        "Penicillin allergy (rash)",
        "Output: patient_info",
    ], tag="01 · LLM")
    disease = diagnosis["primary_diagnosis"]
    candidates = diagnosis["knowledge_graph_candidates"]
    box(ax, 664, 156, 264, 216, "Diagnosis", [
        *wrap(disease["disease_name"], width=22),
        f"Confidence: {disease['confidence']} (model)",
        f"{len(candidates)} retrieval candidates",
        f"Top lookup: {candidates[0]['disease']}",
    ], tag="02 · LLM + RETRIEVAL")
    drugs = [item["generic_name"] for item in treatment["medications"]]
    box(ax, 976, 156, 264, 216, "Treatment", [
        drugs[0] + " +",
        drugs[1] + " (proposed)",
        f"DDI checked: {flag(ddi['checked'])}",
        f"{ddi['engine_flagged_interactions']} curated-rule matches",
        f"{len(treatment['drug_interactions'])} LLM interaction entries",
    ], tag="03 · LLM THEN RULES")
    box(ax, 976, 464, 264, 216, "Coding", [
        f"Primary code: {coding['primary_icd10']['code']}",
        f"DRG proposal: {coding['drg_group']['drg_code']}",
        f"{len(coding['secondary_icd10_codes'])} secondary code proposals",
        f"Model confidence: {coding['coding_confidence']:.2f}",
        "Output: coding_result",
    ], tag="04 · LLM")
    box(ax, 664, 464, 264, 216, "Audit", [
        f"hipaa_compliant: {flag(audit['hipaa_compliant'])}",
        "PHI pattern flagged: " + ", ".join(audit["phi_fields_found"]),
        "Risk label: " + audit["overall_risk_level"],
        "Masking / control claims:",
        "unsupported legacy output",
    ], tag="05 · LEGACY RULES", fill=SURFACE)
    box(ax, 40, 464, 576, 216,
        f"Final recorded result · {record['latency_seconds']:.2f} s", [
            f"Diagnosis alias tier: {score['diagnosis_top1']} · attempts: {state['diagnosis_attempts']}",
            f"Fixture {record['case']['expected_icd10']} → predicted {coding['primary_icd10']['code']}",
            f"Exact code: {flag(score['icd10_exact'])} · category: {flag(score['icd10_category'])}",
            f"Pipeline errors: {flag(state['errors'])} · harness error: {flag(record['harness_error'])}",
            "Five stage-result sections present",
        ], tag="END · ARCHIVED PIPELINE OUTPUT", fill=TINT, stroke=BLUE)

    text(ax, 40, 728, "Historical model output: medication, interaction, confidence, code and DRG proposals are unverified.",
         size=14, color=MUTED)
    text(ax, 40, 756, "Legacy audit masking/control claims are unsupported; current audit behavior has been corrected.",
         size=14, color=MUTED)
    ax.plot((40, 88), (808, 808), color=MUTED, linewidth=1.7)
    text(ax, 100, 813, "Observed sequential execution · case_001_r1.json · no new model run", size=13, color=MUTED)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    record = json.loads((ARCHIVE / "raw/case_001_r1.json").read_text(encoding="utf-8"))
    summary = json.loads((ARCHIVE / "eval_results.json").read_text(encoding="utf-8"))
    score = next(row for row in summary["per_case"] if row["case_id"] == "case_001" and row["repeat"] == 1)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Text and geometry are generated with the same offline font as the other diagrams.
    from .draw_architecture import plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.hashsalt": "medical-case-001"})
    save(diagram(record, score, summary["meta"]), args.output_dir, "case-001-walkthrough",
         "Historical case_001 end-to-end example",
         "A synthetic pneumonia narrative flows through Intake, Diagnosis, Treatment, Coding and "
         "legacy Audit, ending with a 27.10-second recorded result and an exact ICD-10 mismatch.")


if __name__ == "__main__":
    main()
