"""Offline checks for evaluation scoring, output isolation, and the CLI."""

import json
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from evaluation.run_eval import (
    METRICS,
    aggregate,
    evaluate_case,
    icd10_scores,
    main,
    match_diagnosis,
    parse_args,
    score_output,
)

CASE = {
    "id": "case_001",
    "description": "A synthetic patient narrative.",
    "expected_diagnosis": "Community-Acquired Pneumonia",
    "expected_icd10": "J18.1",
}


def test_scoring_distinguishes_alias_and_code_granularity():
    assert match_diagnosis("case_001", "Community-acquired pneumonia") == "strict"
    assert match_diagnosis("case_001", "Pneumonia") == "compatible"
    assert match_diagnosis("case_001", "Bronchitis") == "miss"
    assert match_diagnosis("unknown", "Pneumonia") == "miss"
    assert icd10_scores("J18.1", "j18.9") == (False, True)
    assert icd10_scores("J18.1", "J18.1") == (True, True)
    assert icd10_scores("", "") == (False, False)


def test_structure_completion_does_not_measure_treatment_safety():
    output = {
        "diagnosis": {"primary_diagnosis": {"disease_name": "Pneumonia"}},
        "treatment_plan": {
            "medications": [{"drug_name": "illustrative medication"}],
            "ddi_engine": {"checked": True, "engine_flagged_interactions": 1},
        },
        "audit_result": {"assessment": "requires review"},
    }
    row = score_output(CASE, output, repeat=1, latency=1.25)
    assert row["treatment_structure_complete"] is True
    assert row["ddi_engine_flags"] == 1
    assert row["audit_completed"] is True
    assert row["diagnosis_top1_strict"] is False
    assert row["diagnosis_top1_any"] is True


def test_mock_pipeline_outputs_are_archived_to_selected_directory(tmp_path):
    pipeline = Mock()
    pipeline.invoke.return_value = {
        "diagnosis": {"primary_diagnosis": {"disease_name": "Community-acquired pneumonia"}},
        "coding_result": {"primary_icd10": {"code": "J18.1"}},
        "errors": [],
    }
    row = evaluate_case(pipeline, CASE, repeat=2, output_dir=tmp_path, recursion_limit=14)
    archived = json.loads((tmp_path / "raw" / "case_001_r2.json").read_text())
    assert archived["case"] == CASE
    assert archived["pipeline_output"]["diagnosis"] == pipeline.invoke.return_value["diagnosis"]
    assert row["diagnosis_top1_strict"] is True
    assert row["icd10_exact"] is True
    assert pipeline.invoke.call_args.args == ({"raw_input": CASE["description"]},)


def test_pipeline_failure_is_a_scored_observation(tmp_path):
    pipeline = Mock()
    pipeline.invoke.side_effect = RuntimeError("backend unavailable")
    row = evaluate_case(pipeline, CASE, repeat=1, output_dir=tmp_path, recursion_limit=14)
    assert row["harness_error"] == "RuntimeError: backend unavailable"
    assert row["diagnosis_top1_any"] is False
    assert row["audit_completed"] is False
    summary = aggregate([row])
    assert summary["n_runs"] == 1
    assert summary["diagnosis_top1_any"] == {"hits": 0, "n": 1, "fraction": 0.0}
    assert aggregate([])["mean_latency_seconds"] is None


def test_scoring_failure_preserves_raw_pipeline_output(tmp_path):
    pipeline = Mock()
    pipeline.invoke.return_value = {
        "diagnosis": {
            "primary_diagnosis": {"disease_name": "Community-acquired pneumonia"},
            "differential_list": ["Bronchitis"],
        },
        "audit_result": {"assessment": "report exists"},
        "errors": ["Recorded agent warning"],
        "diagnosis_attempts": 1,
    }

    row = evaluate_case(pipeline, CASE, repeat=1, output_dir=tmp_path, recursion_limit=14)

    assert "Scoring error: AttributeError" in row["harness_error"]
    assert not any(row[key] for key in METRICS)
    assert row["pipeline_errors"] == ["Recorded agent warning"]
    assert row["diagnosis_attempts"] == 1
    archived = json.loads((tmp_path / "raw" / "case_001_r1.json").read_text())
    assert archived["pipeline_output"]["diagnosis"] == pipeline.invoke.return_value["diagnosis"]
    assert archived["pipeline_output"]["audit_result"] == pipeline.invoke.return_value["audit_result"]
    assert archived["pipeline_output"]["errors"] == ["Recorded agent warning"]
    assert archived["harness_error"] == row["harness_error"]
    pipeline.invoke.assert_called_once()


def test_unwritable_output_fails_before_pipeline_invocation(tmp_path, monkeypatch):
    pipeline = Mock()
    pipeline.invoke.return_value = {}

    def reject_probe(*args, **kwargs):
        raise PermissionError("output directory is read-only")

    monkeypatch.setattr(tempfile, "TemporaryFile", reject_probe)

    with pytest.raises(PermissionError, match="read-only"):
        evaluate_case(pipeline, CASE, repeat=1, output_dir=tmp_path, recursion_limit=14)

    pipeline.invoke.assert_not_called()
    assert not list(tmp_path.rglob("*.json"))


def test_evaluation_preserves_existing_raw_file(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    raw_path = raw_dir / "case_001_r1.json"
    raw_path.write_text("original recorded result")
    pipeline = Mock()
    pipeline.invoke.return_value = {}

    with pytest.raises(FileExistsError):
        evaluate_case(pipeline, CASE, repeat=1, output_dir=tmp_path, recursion_limit=14)

    assert raw_path.read_text() == "original recorded result"


@pytest.mark.parametrize("arguments", [
    ["--repeats", "0"],
    ["--repeats", "-1"],
    ["--cases", "case_unknown"],
    ["--cases", "case_001,"],
    ["--cases", "case_001,case_001"],
    ["--output-dir", "evaluation/results/historical-2026-07-13"],
    ["--output-dir", "evaluation/results/historical-2026-07-13/nested"],
])
def test_cli_rejects_invalid_selection_and_historical_overwrite(arguments):
    with pytest.raises(SystemExit) as error:
        parse_args(arguments)
    assert error.value.code == 2


def test_help_runs_with_only_python_standard_library():
    result = subprocess.run(
        [sys.executable, "-S", "evaluation/run_eval.py", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--output-dir" in result.stdout


def test_default_output_uses_ignored_runs_directory():
    args = parse_args([])
    assert args.output_dir.parent.name == "runs"
    assert args.output_dir.name.startswith("run-")


def test_main_writes_mock_run_and_uses_configured_retry_budget(tmp_path, monkeypatch):
    from medical_multi_agent.config import settings
    from medical_multi_agent.graph import clinical_pipeline

    pipeline = Mock()
    pipeline.invoke.return_value = {"errors": []}
    monkeypatch.setattr(settings, "get_settings", lambda: SimpleNamespace(
        openai_model="mock-model", openai_base_url="http://localhost:8001/v1",
        max_diagnosis_retries=4,
    ))
    monkeypatch.setattr(clinical_pipeline, "build_clinical_pipeline", lambda: pipeline)
    assert main(["--cases", "case_001", "--output-dir", str(tmp_path)]) == 0
    report = json.loads((tmp_path / "eval_results.json").read_text())
    assert report["meta"]["model"] == "mock-model"
    assert report["aggregate"]["n_runs"] == 1
    assert pipeline.invoke.call_args.kwargs["config"]["recursion_limit"] == 18
    assert (tmp_path / "eval_results.md").exists()
    with pytest.raises(SystemExit):
        parse_args(["--output-dir", str(tmp_path)])


def test_main_reports_scoring_failure_and_exits_nonzero(tmp_path, monkeypatch):
    from medical_multi_agent.config import settings
    from medical_multi_agent.graph import clinical_pipeline

    pipeline = Mock()
    pipeline.invoke.return_value = {
        "diagnosis": {
            "primary_diagnosis": {"disease_name": "Pneumonia"},
            "differential_list": ["Bronchitis"],
        },
    }
    monkeypatch.setattr(settings, "get_settings", lambda: SimpleNamespace(
        openai_model="mock-model", openai_base_url="http://localhost:8001/v1",
        max_diagnosis_retries=2,
    ))
    monkeypatch.setattr(clinical_pipeline, "build_clinical_pipeline", lambda: pipeline)

    assert main(["--cases", "case_001", "--output-dir", str(tmp_path)]) == 1

    report = json.loads((tmp_path / "eval_results.json").read_text())
    assert report["aggregate"]["n_runs"] == 1
    assert report["aggregate"]["diagnosis_top1_any"]["hits"] == 0
    assert "Scoring error: AttributeError" in report["per_case"][0]["harness_error"]
    assert (tmp_path / "eval_results.md").exists()
    assert (tmp_path / "raw" / "case_001_r1.json").exists()
