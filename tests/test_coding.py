"""Coding preserves source facts and checks proposals against the offline catalog."""

import importlib
import json
from types import SimpleNamespace

import pytest

from evaluation.run_eval import score_output
from medical_multi_agent.graph.state import ClinicalState

coding = importlib.import_module("medical_multi_agent.agents.coding_agent")

NARRATIVE = (
    "Synthetic adult with fever and cough, documented hypertension and type 2 diabetes. "
    "Former smoker, quit 2 years ago."
)
PATIENT = {
    "medical_history": ["Hypertension", "Type 2 diabetes", "Former smoker, quit 2 years ago"],
    "current_medications": [{"name": "lisinopril"}],
}


@pytest.fixture
def invoke(monkeypatch):
    def run(proposal, *, state=None):
        prompts = []
        content = proposal if isinstance(proposal, str) else json.dumps(proposal)

        def respond(messages):
            prompts.append(messages)
            return SimpleNamespace(content=content)

        monkeypatch.setattr(coding, "get_llm", lambda temperature: SimpleNamespace(invoke=respond))
        state = state or ClinicalState(
            raw_input=NARRATIVE,
            patient_info=PATIENT,
            diagnosis={"primary_diagnosis": {"disease_name": "Pneumonia", "icd10_hint": "J18.9"}},
            treatment_plan={"medications": []},
        )
        result = coding.coding_agent(state)
        assert len(prompts) == 1
        context, _ = json.JSONDecoder().raw_decode(
            prompts[0][1].content.removeprefix("Clinical data for coding:\n\n")
        )
        return result, context

    return run


def test_coding_receives_original_patient_facts_and_explicit_contract(invoke):
    result, context = invoke({"primary_icd10": {"code": "J18.9"}, "secondary_icd10_codes": []})

    assert context["raw_input"] == NARRATIVE
    assert context["patient_info"] == PATIENT
    assert context["diagnosis"]["primary_diagnosis"]["disease_name"] == "Pneumonia"
    assert context["treatment_plan"] == {"medications": []}
    assert context["coding_contract"]["code_set"] == "ICD-10-CM"
    assert context["coding_contract"]["release"] == "FY2027"
    assert "unspecified" in context["coding_contract"]["encounter_setting"].lower()
    assert "expected_diagnosis" not in context and "expected_icd10" not in context
    assert result["coding_result"]["clinical_review_required"] is True


def test_raw_narrative_survives_omitted_upstream_history(invoke):
    state = ClinicalState(
        raw_input=NARRATIVE,
        patient_info={"medical_history": []},
        diagnosis={"primary_diagnosis": {"disease_name": "Pneumonia", "icd10_hint": "J18.9"}},
    )
    _, context = invoke(
        {"primary_icd10": {"code": "J18.9"}, "secondary_icd10_codes": []}, state=state,
    )

    assert context["raw_input"] == NARRATIVE
    assert context["patient_info"] == {"medical_history": []}


def test_complete_codes_normalize_and_keep_original_proposal(invoke):
    proposal = {
        "primary_icd10": {"code": " j189 ", "description": "Unverified model description"},
        "secondary_icd10_codes": [{"code": " i10 ", "description": "Another model description"}],
    }
    result, _ = invoke(proposal)
    output = result["coding_result"]

    assert output["primary_icd10"]["code"] == "J18.9"
    assert output["primary_icd10"]["description"] == "Pneumonia, unspecified organism"
    assert output["primary_icd10"]["model_description"] == "Unverified model description"
    assert output["secondary_icd10_codes"][0]["code"] == "I10"
    assert output["model_proposal"] == proposal
    assert json.loads(output["raw_model_output"]) == proposal
    assert output["code_validation"]["schema_valid"] is True
    assert output["code_validation"]["valid"] is True
    assert output["code_validation"]["primary"]["valid"] is True
    assert output["code_validation"]["secondary"][0]["valid"] is True
    assert output["requires_review"] is False
    assert output["clinical_review_required"] is True
    assert not result.get("errors")


@pytest.mark.parametrize(("code", "reason"), [
    ("I21.0", "non_reportable_header"),
    ("J18.99", "unknown_code"),
    ("I21..09", "malformed_code"),
])
def test_invalid_primary_codes_are_retained_and_flagged(invoke, code, reason):
    proposal = {"primary_icd10": {"code": code}, "secondary_icd10_codes": []}
    result, _ = invoke(proposal)
    output = result["coding_result"]

    assert output["model_proposal"] == proposal
    assert output["primary_icd10"]["code"] == code
    assert output["code_validation"]["valid"] is False
    assert output["code_validation"]["primary"]["reason"] == reason
    assert output["requires_review"] is True
    assert result["errors"]
    assert "primary_icd10" in result["errors"][-1]


def test_invalid_secondary_code_prevents_all_codes_passing(invoke):
    proposal = {
        "primary_icd10": {"code": "J18.9"},
        "secondary_icd10_codes": [{"code": "I21.0"}],
    }
    result, _ = invoke(proposal)
    output = result["coding_result"]

    assert output["code_validation"]["primary"]["valid"] is True
    assert output["code_validation"]["secondary"][0]["reason"] == "non_reportable_header"
    assert output["code_validation"]["valid"] is False
    assert output["secondary_icd10_codes"][0]["code"] == "I21.0"
    assert output["requires_review"] is True
    assert "secondary_icd10_codes[0]" in result["errors"][-1]


@pytest.mark.parametrize("proposal", [
    [],
    {"secondary_icd10_codes": []},
    {"primary_icd10": {"code": 123}, "secondary_icd10_codes": []},
    {"primary_icd10": {"code": "J18.9"}},
    {"primary_icd10": {"code": "J18.9"}, "secondary_icd10_codes": {}},
    {"primary_icd10": {"code": "J18.9"}, "secondary_icd10_codes": ["I10"]},
])
def test_json_structure_errors_remain_visible(invoke, proposal):
    result, _ = invoke(proposal)
    output = result["coding_result"]

    assert output["model_proposal"] == proposal
    assert output["code_validation"]["schema_valid"] is False
    assert output["code_validation"]["valid"] is False
    assert output["requires_review"] is True
    assert result["errors"]


def test_malformed_json_is_preserved_without_repair(invoke):
    raw = '{"primary_icd10": {"code": "J18.9",}}'
    result, _ = invoke(raw)
    output = result["coding_result"]

    assert output["raw_model_output"] == raw
    assert output["code_validation"]["valid"] is False
    assert output["code_validation"]["schema_valid"] is False
    assert output["requires_review"] is True
    assert any("JSON parse error" in error for error in result["errors"])


def test_candidate_catalog_is_bounded_and_derived_from_observed_facts(invoke):
    state = ClinicalState(
        raw_input=NARRATIVE,
        patient_info=PATIENT,
        diagnosis={"primary_diagnosis": {"disease_name": "Anterior STEMI", "icd10_hint": "I21.0"}},
    )
    _, context = invoke(
        {"primary_icd10": {"code": "I21.09"}, "secondary_icd10_codes": []}, state=state,
    )
    families = context["catalog_candidates"]["families"]
    candidates = [entry for family in families for entry in family["codes"]]

    assert {"I21", "I10", "E11"} <= {family["prefix"] for family in families}
    assert 0 < len(candidates) <= 48
    assert all(entry["is_complete_code"] for entry in candidates)
    assert "I21.09" in {entry["code"] for entry in candidates}
    assert "I21.0" not in {entry["code"] for entry in candidates}


@pytest.mark.parametrize("primary", ["I21.09", ["I21.09"]])
def test_nonobject_primary_is_preserved_but_normalized_for_safe_scoring(invoke, primary):
    proposal = {"primary_icd10": primary, "secondary_icd10_codes": []}
    result, _ = invoke(proposal)
    output = result["coding_result"]

    assert output["model_proposal"]["primary_icd10"] == primary
    assert output["primary_icd10"] is None
    assert output["code_validation"]["valid"] is False
    row = score_output(
        {"id": "synthetic-check", "expected_diagnosis": "Pneumonia", "expected_icd10": "J18.9"},
        {"coding_result": output, "errors": result["errors"]}, repeat=1, latency=0.1,
    )
    assert row["harness_error"] is None
    assert row["icd10_exact"] is False


def test_plain_history_candidates_include_complete_nonspecific_options(invoke):
    state = ClinicalState(
        raw_input="Synthetic adult with type 2 diabetes and hyperlipidemia.",
        patient_info={"medical_history": ["Type 2 diabetes", "Hyperlipidemia"]},
        diagnosis={"primary_diagnosis": {"disease_name": "Pneumonia", "icd10_hint": "J18.9"}},
    )
    _, context = invoke(
        {"primary_icd10": {"code": "J18.9"}, "secondary_icd10_codes": []}, state=state,
    )
    families = {family["prefix"]: family["codes"] for family in context["catalog_candidates"]["families"]}

    assert "E11.9" in {entry["code"] for entry in families["E11"]}
    assert "E78.5" in {entry["code"] for entry in families["E78"]}
    assert sum(len(entries) for entries in families.values()) <= 48
