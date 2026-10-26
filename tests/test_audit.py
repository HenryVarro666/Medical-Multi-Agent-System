"""PHI screening must report only checks the application performs."""

from medical_multi_agent.agents.audit_agent import audit_agent
from medical_multi_agent.graph.state import ClinicalState
from medical_multi_agent.services.hipaa_service import detect_phi


def test_audit_does_not_certify_unverified_deployment_controls():
    result = audit_agent(ClinicalState(patient_info={"age": 30}))["audit_result"]
    checks = {check["check_name"]: check for check in result["compliance_checks"]}
    assert checks["phi_scan"]["passed"] is True
    for name in ("data_encryption_at_rest", "data_encryption_in_transit", "access_control_rbac", "data_retention_policy"):
        assert checks[name]["passed"] is False
        assert "not verified" in checks[name]["detail"].lower()
    assert result["hipaa_compliant"] is False


def test_audit_does_not_claim_to_mask_returned_patient_data():
    state = ClinicalState(patient_info={"name": "Jane Smith", "email": "jane@example.com"})
    result = audit_agent(state)["audit_result"]
    assert "name" in result["phi_fields_found"]
    assert "email" in result["phi_fields_found"]
    assert result["phi_fields_masked"] == []
    assert state.patient_info["email"] == "jane@example.com"
    assert not any(record["action"] == "data_masking" for record in result["audit_trail"])


def test_phi_name_pattern_does_not_match_every_pair_of_lowercase_words():
    assert "names" not in detect_phi("persistent cough and fever")
    assert "names" in detect_phi("Jane Smith")
