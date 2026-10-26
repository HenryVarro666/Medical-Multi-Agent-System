"""Offline API contract and request isolation tests."""

import asyncio
from importlib.metadata import version
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import MemorySaver

import medical_multi_agent
from medical_multi_agent.api import routes
from medical_multi_agent.api.main import app
from medical_multi_agent.config.settings import Settings
from medical_multi_agent.graph import clinical_pipeline


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(routes, "get_settings", lambda: Settings(_env_file=None), raising=False)
    return TestClient(app)


def test_offline_endpoints_work_without_llm_configuration(client, monkeypatch):
    def must_not_run():
        pytest.fail("Offline endpoints must not construct the LLM pipeline")

    monkeypatch.setattr(routes, "get_pipeline", must_not_run)
    assert client.get("/health").status_code == 200
    assert client.get("/api/v1/clinical/icd10/J18.9").json()["icd10"]["code"] == "J18.9"
    result = client.post(
        "/api/v1/clinical/ddi/check",
        json={"new_drugs": ["warfarin"], "current_drugs": ["aspirin"]},
    )
    assert result.status_code == 200
    assert result.json()["has_major_interaction"] is True


def test_default_request_labels_are_unique():
    first = routes.AnalyzeRequest(patient_description="A patient has a cough")
    second = routes.AnalyzeRequest(patient_description="Another patient has fever")
    assert first.thread_id != second.thread_id


def test_reused_request_label_runs_in_isolated_checkpoint_state(client, monkeypatch):
    checkpoints = {}

    def invoke(state, config):
        checkpoint = config["configurable"]["thread_id"]
        previous = checkpoints.get(checkpoint)
        checkpoints[checkpoint] = state["raw_input"]
        return {"errors": [previous] if previous else []}

    monkeypatch.setattr(routes, "get_pipeline", lambda: SimpleNamespace(invoke=invoke))
    for description in ("Patient one has a cough", "Patient two has a fever"):
        response = client.post(
            "/api/v1/clinical/analyze",
            json={"patient_description": description, "thread_id": "same-client-label"},
        )
        assert response.status_code == 200
        assert response.json()["errors"] == []
    assert len(checkpoints) == 2


def test_analysis_runs_outside_event_loop_thread(client, monkeypatch):
    def invoke(state, config):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return {"patient_info": {"runs_on_event_loop": False}}
        return {"patient_info": {"runs_on_event_loop": True}}

    monkeypatch.setattr(routes, "get_pipeline", lambda: SimpleNamespace(invoke=invoke))
    response = client.post(
        "/api/v1/clinical/analyze", json={"patient_description": "Patient has persistent fever"}
    )
    assert response.status_code == 200
    assert response.json()["patient_info"]["runs_on_event_loop"] is False


def test_retry_budget_sets_graph_recursion_limit(client, monkeypatch):
    monkeypatch.setattr(routes, "get_settings", lambda: Settings(_env_file=None, max_diagnosis_retries=20))

    def invoke(state, config):
        return {"patient_info": {"recursion_limit": config["recursion_limit"]}}

    monkeypatch.setattr(routes, "get_pipeline", lambda: SimpleNamespace(invoke=invoke))
    response = client.post(
        "/api/v1/clinical/analyze", json={"patient_description": "Patient has persistent fever"}
    )
    assert response.status_code == 200
    assert response.json()["patient_info"]["recursion_limit"] >= 50


def test_missing_remote_api_key_returns_configuration_error(client, monkeypatch):
    from medical_multi_agent.config import llm

    settings = Settings(_env_file=None, openai_api_key="", openai_base_url="")
    monkeypatch.setattr(llm, "get_settings", lambda: settings)
    response = client.post(
        "/api/v1/clinical/analyze", json={"patient_description": "Patient has persistent fever"}
    )
    assert response.status_code == 503
    assert "OPENAI_API_KEY" in response.json()["detail"]


def test_analysis_rejects_short_input(client):
    response = client.post("/api/v1/clinical/analyze", json={"patient_description": "short"})
    assert response.status_code == 422


def test_api_and_package_versions_match_project_metadata(client):
    expected = version("medical-multi-agent-system")
    assert client.get("/health").json()["version"] == expected
    assert app.version == expected
    assert getattr(medical_multi_agent, "__version__", None) == expected


@pytest.fixture
def stub_pipeline_nodes(monkeypatch):
    def diagnosis(state):
        return {"diagnosis_attempts": state.diagnosis_attempts + 1}

    for name in ("intake", "treatment", "coding", "audit"):
        monkeypatch.setattr(
            clinical_pipeline,
            f"{name}_agent",
            lambda state, name=name: {"current_agent": name},
        )
    monkeypatch.setattr(clinical_pipeline, "diagnosis_agent", diagnosis)


def test_default_pipeline_runs_independently_without_checkpoint_configuration(stub_pipeline_nodes):
    pipeline = clinical_pipeline.build_clinical_pipeline()
    for description in ("First synthetic patient", "Second synthetic patient"):
        result = pipeline.invoke({"raw_input": description})
        assert result["current_agent"] == "audit"
        assert result["diagnosis_attempts"] == 1
    assert pipeline.checkpointer is None


def test_explicit_pipeline_checkpointer_preserves_results(stub_pipeline_nodes):
    saver = MemorySaver()
    pipeline = clinical_pipeline.build_clinical_pipeline(checkpointer=saver)
    config = {"configurable": {"thread_id": "explicit-checkpoint"}}
    pipeline.invoke({"raw_input": "Synthetic patient for an explicit checkpoint"}, config=config)
    assert pipeline.checkpointer is saver
    snapshot = pipeline.get_state(config)
    assert snapshot.values["raw_input"] == "Synthetic patient for an explicit checkpoint"
    assert snapshot.values["current_agent"] == "audit"
