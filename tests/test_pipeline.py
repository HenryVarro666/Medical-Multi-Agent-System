"""Exercise the real graph and agents with only the LLM transport replaced."""

import importlib
import json
from types import SimpleNamespace
from uuid import uuid4

import pytest

from medical_multi_agent.config.settings import get_settings
from medical_multi_agent.graph.clinical_pipeline import build_clinical_pipeline

PATIENT = {
    "name": "Synthetic",
    "age": 45,
    "gender": "male",
    "chief_complaint": "fever and cough",
    "symptoms": [{"name": "fever"}, {"name": "cough"}],
    "medical_history": ["Hypertension", "Former smoker, quit 2 years ago"],
    "current_medications": [{"name": "aspirin"}],
    "allergies": [{"substance": "penicillin"}],
}


@pytest.fixture
def fake_agents(monkeypatch):
    """Keep parsing, retrieval, DDI, routing, and auditing real and offline."""
    prompts = {}
    responses = {
        "intake": PATIENT,
        "diagnosis": {
            "primary_diagnosis": {"disease_name": "Pneumonia"},
            "needs_more_info": False,
        },
        "treatment": {
            "medications": [{"generic_name": "warfarin"}, {"generic_name": "amoxicillin"}],
            "drug_interactions": [],
        },
        "coding": {"primary_icd10": {"code": "J18.9"}, "secondary_icd10_codes": []},
    }

    def client(name):
        def invoke(messages):
            prompts.setdefault(name, []).append(messages[-1].content)
            value = responses[name]
            content = value if isinstance(value, str) else json.dumps(value)
            return SimpleNamespace(content=content)

        return SimpleNamespace(invoke=invoke)

    for name in responses:
        module = importlib.import_module(f"medical_multi_agent.agents.{name}_agent")
        monkeypatch.setattr(module, "get_llm", lambda temperature, name=name: client(name))
    return responses, prompts


def run_pipeline():
    return build_clinical_pipeline().invoke(
        {"raw_input": "Synthetic adult with fever and cough."},
        config={"configurable": {"thread_id": uuid4().hex}, "recursion_limit": 40},
    )


def test_complete_pipeline_keeps_retrieval_and_deterministic_checks(fake_agents):
    _, prompts = fake_agents
    result = run_pipeline()

    assert result.get("errors", []) == []
    assert result["current_agent"] == "audit"
    assert result["patient_info"]["chief_complaint"] == PATIENT["chief_complaint"]
    assert result["diagnosis"]["knowledge_graph_candidates"][0]["disease"] == "Pneumonia"
    assert "Pneumonia" in prompts["diagnosis"][0]
    plan = result["treatment_plan"]
    assert plan["ddi_engine"]["checked"] is True
    assert plan["ddi_engine"]["engine_flagged_interactions"] >= 1
    assert any(hit["source"] == "rule_engine" for hit in plan["drug_interactions"])
    assert plan["allergy_alerts"][0]["allergy"] == "penicillin"
    assert result["coding_result"]["primary_icd10"]["code"] == "J18.9"
    assert result["coding_result"]["code_validation"]["valid"] is True
    assert result["coding_result"]["clinical_review_required"] is True
    coding_context, _ = json.JSONDecoder().raw_decode(
        prompts["coding"][0].removeprefix("Clinical data for coding:\n\n")
    )
    assert coding_context["raw_input"] == "Synthetic adult with fever and cough."
    assert coding_context["patient_info"] == result["patient_info"]
    assert coding_context["patient_info"]["medical_history"] == PATIENT["medical_history"]
    assert coding_context["coding_contract"]["release"] == "FY2027"
    assert result["audit_result"]["audit_trail"]


@pytest.mark.parametrize("retries", [0, 2])
def test_retry_budget_terminates_real_graph(fake_agents, monkeypatch, retries):
    responses, prompts = fake_agents
    responses["diagnosis"]["needs_more_info"] = True
    monkeypatch.setenv("MAX_DIAGNOSIS_RETRIES", str(retries))
    get_settings.cache_clear()
    try:
        result = run_pipeline()
    finally:
        get_settings.cache_clear()

    assert result["diagnosis_attempts"] == retries + 1
    assert len(prompts["diagnosis"]) == retries + 1
    assert result["current_agent"] == "audit"
    assert result["treatment_plan"]["ddi_engine"]["checked"] is True


def test_invalid_intake_output_preserves_errors_and_finishes(fake_agents, monkeypatch):
    responses, prompts = fake_agents
    responses["intake"] = "not JSON"
    monkeypatch.setenv("MAX_DIAGNOSIS_RETRIES", "0")
    get_settings.cache_clear()
    try:
        result = run_pipeline()
    finally:
        get_settings.cache_clear()

    assert result["patient_info"] is None
    assert result["diagnosis"] is None
    assert result["treatment_plan"] is None
    assert result["coding_result"] is None
    assert any("Intake JSON parse error" in error for error in result["errors"])
    assert any("No patient info" in error for error in result["errors"])
    assert result["current_agent"] == "audit"
    assert "diagnosis" not in prompts


def test_invalid_coding_proposal_preserves_the_code_and_reaches_audit(fake_agents):
    responses, prompts = fake_agents
    responses["coding"]["primary_icd10"]["code"] = "I21.0"

    result = run_pipeline()

    coding = result["coding_result"]
    assert coding["primary_icd10"]["code"] == "I21.0"
    assert coding["model_proposal"] == responses["coding"]
    assert coding["code_validation"]["primary"]["reason"] == "non_reportable_header"
    assert coding["code_validation"]["valid"] is False
    assert coding["requires_review"] is True
    assert any("Coding validation failed" in error for error in result["errors"])
    assert len(prompts["coding"]) == 1
    assert result["current_agent"] == "audit"
    assert result["audit_result"]["audit_trail"]
