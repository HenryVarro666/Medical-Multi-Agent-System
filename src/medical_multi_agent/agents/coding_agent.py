"""
Coding Agent — ICD-10 automatic coding and DRGs grouping.

Responsibilities:
  - Preserve the original narrative and structured patient facts
  - Supply bounded candidates from the offline FY2027 ICD-10-CM catalog
  - Validate every proposed code's membership and completeness

Catalog validity does not establish clinical correctness. DRG proposals and
clinical code selection still require independent review.
"""

from __future__ import annotations

import json
import re

import structlog
from langchain_core.messages import HumanMessage, SystemMessage

from ..config.llm import get_llm
from ..services.icd10_catalog import (
    catalog_info,
    get_complete_codes,
    lookup_code,
    normalize_icd10_code,
    validate_code,
)

logger = structlog.get_logger(__name__)

CODING_CONTEXT = {
    "code_set": "ICD-10-CM",
    "release": "FY2027",
    "encounter_setting": "Unspecified synthetic research vignette; encounter setting is not documented.",
    "documentation_policy": (
        "Use the original narrative and structured patient facts. Do not infer an undocumented "
        "disease, culprit artery, causal relationship, or completed procedure from medications, "
        "test findings, a treatment proposal, or retrieval hints alone."
    ),
    "code_selection_policy": (
        "Choose the most specific documented, supported complete code. Catalog candidates are "
        "hints, not required answers or evidence that a condition is present. Document unresolved "
        "coding assumptions; a code's existence does not establish clinical correctness."
    ),
    "drg_policy": (
        "Do not assume an inpatient encounter or completed procedure. Use null for an unresolved "
        "DRG; any estimate is unverified and requires the applicable grouper and documentation."
    ),
}

CODING_SYSTEM_PROMPT = """You propose research ICD-10-CM coding from the supplied original narrative, complete patient information, diagnosis, treatment proposal, and coding contract.

Return a JSON object:
{
  "primary_icd10": {
    "code": "complete ICD-10-CM code supported by the documentation",
    "description": "official code description",
    "confidence": 0.92,
    "category": "category name"
  },
  "secondary_icd10_codes": [
    {
      "code": "ICD-10 code",
      "description": "description",
      "confidence": 0.85,
      "category": "category"
    }
  ],
  "drg_group": {
    "drg_code": "DRG number only if the encounter context supports one",
    "description": "DRG description",
    "weight": 1.2,
    "mean_los": 4.5
  },
  "coding_notes": "rationale for code selection",
  "coding_confidence": 0.90
}

Rules:
- Follow the supplied FY2027 coding contract and use complete catalog codes, never category headers.
- Select the most specific code supported by the documented facts; do not add unsupported specificity.
- Consider the original narrative and full patient history even when upstream summaries omit a fact.
- Include only documented, relevant secondary conditions; do not infer conditions from medications.
- Retrieval hints and candidate lists do not establish a diagnosis or force a code choice.
- Return secondary_icd10_codes as a list, including an empty list when none are supported.
- Do not assume the encounter setting, culprit artery, causal relationships, or completed procedures.
- drg_group may be null when the setting or grouping information is unresolved.
- Confidence is an uncalibrated model estimate, not validated coding accuracy.
- Return ONLY valid JSON, no markdown fences."""


def _catalog_candidates(diagnosis: dict, patient_info: dict | None) -> dict:
    """Offer bounded catalog families from observed hints and history mentions."""
    families = []
    primary = diagnosis.get("primary_diagnosis")
    hint = normalize_icd10_code(primary.get("icd10_hint")) if isinstance(primary, dict) else None
    prefixes = [(hint[:3], "diagnosis.primary_diagnosis.icd10_hint")] if hint else []
    history_prefixes = set()
    history = (patient_info or {}).get("medical_history") or []
    history_text = " ".join(item for item in history if isinstance(item, str)).lower()
    for pattern, prefix in (
        (r"\bhypertension\b", "I10"),
        (r"\btype\s*(?:2|ii)\s+diabet", "E11"),
        (r"\bhyperlipid(?:emia|aemia)\b", "E78"),
    ):
        if re.search(pattern, history_text):
            prefixes.append((prefix, "patient_info.medical_history text mention"))
            history_prefixes.add(prefix)
    seen = set()
    for prefix, basis in prefixes:
        if prefix in seen:
            continue
        seen.add(prefix)
        codes = get_complete_codes(prefix, limit=12)
        preferred = []
        generic = {"E11": "E11.9", "E78": "E78.5"}.get(prefix) if prefix in history_prefixes else None
        for code in (hint if hint and hint.startswith(prefix) else None, generic):
            entry = lookup_code(code) if code else None
            if entry and entry["is_complete_code"] and entry not in preferred:
                preferred.append(entry)
        preferred_codes = {entry["code"] for entry in preferred}
        codes = (preferred + [entry for entry in codes if entry["code"] not in preferred_codes])[:12]
        families.append({"prefix": prefix, "basis": basis, "codes": codes})
    return {
        "catalog_release": CODING_CONTEXT["release"],
        "policy": (
            "Incomplete candidate list; mentions and hints are not diagnoses or mandatory code choices. "
            "Do not infer complications or their absence from the candidate list; use documented facts."
        ),
        "families": families,
    }


def _validated_output(proposal, raw_output: str, parse_error: str | None = None) -> dict:
    """Keep model proposals while recording structural and catalog failures."""
    result = dict(proposal) if isinstance(proposal, dict) else {}
    schema_errors = []
    errors = []
    if parse_error:
        schema_errors.append(parse_error)
    if not isinstance(proposal, dict):
        schema_errors.append("Coding output must be a JSON object")

    def check_entry(entry, path):
        if not isinstance(entry, dict) or not isinstance(entry.get("code"), str) or not entry["code"].strip():
            schema_errors.append(f"{path} must be an object with a nonempty string code")
        proposed_code = entry.get("code") if isinstance(entry, dict) else None
        validation = validate_code(proposed_code)
        if not validation["valid"]:
            errors.append(f"{path}: {validation['reason']}")
        if isinstance(entry, dict):
            entry = dict(entry)
            if validation["code"] is not None:
                entry["code"] = validation["code"]
            if validation["valid"]:
                entry["model_description"] = entry.get("description")
                entry["description"] = validation["description"]
        else:
            entry = None
        return entry, validation

    source = proposal if isinstance(proposal, dict) else {}
    result["primary_icd10"], primary = check_entry(source.get("primary_icd10"), "primary_icd10")
    secondary = source.get("secondary_icd10_codes")
    secondary_validation = []
    if not isinstance(secondary, list):
        schema_errors.append("secondary_icd10_codes must be a list")
    else:
        normalized_secondary = []
        for index, entry in enumerate(secondary):
            normalized, validation = check_entry(entry, f"secondary_icd10_codes[{index}]")
            normalized_secondary.append(normalized)
            secondary_validation.append(validation)
        result["secondary_icd10_codes"] = normalized_secondary
    errors = schema_errors + errors
    valid = not errors
    result.update({
        "model_proposal": proposal,
        "raw_model_output": raw_output,
        "code_validation": {
            "catalog": catalog_info(),
            "schema_valid": not schema_errors,
            "valid": valid,
            "primary": primary,
            "secondary": secondary_validation,
            "errors": errors,
            "clinical_correctness": "not_adjudicated",
        },
        "requires_review": not valid,
        "clinical_review_required": True,
    })
    return result


def coding_agent(state) -> dict:
    """
    LangGraph node: Assign ICD-10 codes and DRGs grouping.
    Reads: state.raw_input, state.patient_info, state.diagnosis, state.treatment_plan
    Writes: state.coding_result, state.current_agent
    """
    logger.info("coding_agent.start")

    diagnosis = state.diagnosis
    treatment = state.treatment_plan

    if not diagnosis:
        return {
            "coding_result": None,
            "current_agent": "coding",
            "errors": state.errors + ["No diagnosis available for coding"],
        }

    llm = get_llm(temperature=0.1)

    context = json.dumps(
        {
            "raw_input": state.raw_input,
            "patient_info": state.patient_info,
            "diagnosis": diagnosis,
            "treatment_plan": treatment,
            "coding_contract": CODING_CONTEXT,
            "catalog_candidates": _catalog_candidates(diagnosis, state.patient_info),
        },
        indent=2,
        ensure_ascii=False,
    )

    messages = [
        SystemMessage(content=CODING_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Clinical data for coding:\n\n{context}\n\n"
                "Assign ICD-10 codes and DRGs group."
            )
        ),
    ]

    try:
        response = llm.invoke(messages)
        content = response.content
        try:
            proposal = json.loads(content)
            parse_error = None
        except json.JSONDecodeError as error:
            proposal = None
            parse_error = f"Coding JSON parse error: {error}"
        coding_data = _validated_output(proposal, content, parse_error)

        logger.info(
            "coding_agent.success",
            catalog_valid=coding_data["code_validation"]["valid"],
        )
        result = {
            "coding_result": coding_data,
            "current_agent": "coding",
        }
        if coding_data["requires_review"]:
            result["errors"] = state.errors + [
                "Coding validation failed: " + "; ".join(coding_data["code_validation"]["errors"])
            ]
        return result
    except Exception as e:
        logger.error("coding_agent.error", error=str(e))
        return {
            "coding_result": None,
            "current_agent": "coding",
            "errors": state.errors + [f"Coding error: {e}"],
        }
