"""
Audit Agent — Demonstration PHI screening and deployment control checklist.

Responsibilities:
  - Scan all pipeline outputs for PHI (Protected Health Information)
  - List the identifier categories found by a limited regex scan
  - Return audit records as part of the pipeline output
  - Mark deployment controls as unverified

The scan can miss identifiers or flag ordinary clinical terminology. It does
not redact pipeline outputs or establish HIPAA compliance.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone

import structlog

from ..models.treatment import AuditRecord, AuditResult, ComplianceCheck

logger = structlog.get_logger(__name__)

# Demonstration patterns cover a subset of identifier categories.
PHI_PATTERNS = {
    "name": r"\b[A-Z][a-z]+\s[A-Z][a-z]+\b",
    "date_of_birth": r"\b\d{4}[-/]\d{2}[-/]\d{2}\b",
    "phone": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
    "email": r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b",
    "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
    "mrn": r"\bMRN[:\s]?\d+\b",
    "ip_address": r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b",
    "zip_code": r"\b\d{5}(-\d{4})?\b",
    "address": r"\b\d+\s[\w\s]+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Boulevard|Blvd)\b",
}

HIPAA_CHECKS = [
    "phi_scan",
    "data_encryption_at_rest",
    "data_encryption_in_transit",
    "access_control_rbac",
    "audit_logging",
    "minimum_necessary_rule",
    "breach_notification_ready",
    "data_retention_policy",
]


def _scan_for_phi(data: dict) -> list[str]:
    """Scan a dict (serialized to string) for PHI patterns."""
    text = json.dumps(data, ensure_ascii=False)
    found = []
    for phi_type, pattern in PHI_PATTERNS.items():
        if re.search(pattern, text):
            found.append(phi_type)
    return found


def _create_audit_record(action: str, resource_type: str, detail: str) -> dict:
    return AuditRecord(
        timestamp=datetime.now(timezone.utc).isoformat(),
        user_id="system",
        action=action,
        resource_type=resource_type,
        detail=detail,
        outcome="success",
    ).model_dump()


def audit_agent(state) -> dict:
    """
    LangGraph node: screen outputs for PHI and report unverified controls.
    Reads: all state fields
    Writes: state.audit_result, state.current_agent
    """
    logger.info("audit_agent.start")

    audit_trail = []
    compliance_checks = []
    phi_found = []

    # Aggregate all data for PHI scanning
    all_data = {}
    for field_name in ("patient_info", "diagnosis", "treatment_plan", "coding_result"):
        val = getattr(state, field_name, None)
        if val:
            all_data[field_name] = val

    # 1. PHI Scan
    phi_found = _scan_for_phi(all_data)
    compliance_checks.append(
        ComplianceCheck(
            check_name="phi_scan",
            passed=len(phi_found) == 0,
            detail=f"Found {len(phi_found)} PHI types: {', '.join(phi_found)}" if phi_found else "No PHI detected",
        ).model_dump()
    )
    audit_trail.append(_create_audit_record("phi_scan", "pipeline_output", f"Scanned {len(all_data)} sections"))

    # These deployment properties cannot be inferred from a pipeline output.
    for check_name in HIPAA_CHECKS[1:]:
        compliance_checks.append(
            ComplianceCheck(
                check_name=check_name,
                passed=False,
                detail="Not verified by this demonstration. Requires deployment-specific review.",
            ).model_dump()
        )

    # 4. Overall assessment
    all_passed = all(c["passed"] for c in compliance_checks)
    risk_level = "low" if all_passed else ("medium" if len(phi_found) <= 2 else "high")

    recommendations = []
    if phi_found:
        recommendations.append("Ensure all PHI is masked before external transmission")
    if not all_passed:
        recommendations.append("Review failed compliance checks and remediate")
    recommendations.append(
        "Review deployment access, encryption, retention and audit storage controls. "
        "This report does not establish HIPAA compliance."
    )

    audit_trail.append(
        _create_audit_record(
            "compliance_assessment",
            "pipeline",
            f"Overall: {'PASS' if all_passed else 'NEEDS_REVIEW'}, risk={risk_level}",
        )
    )

    result = AuditResult(
        hipaa_compliant=all_passed,
        compliance_checks=compliance_checks,
        phi_fields_found=phi_found,
        phi_fields_masked=[],
        audit_trail=audit_trail,
        recommendations=recommendations,
        overall_risk_level=risk_level,
    )

    logger.info("audit_agent.success", hipaa_compliant=all_passed, risk=risk_level)
    return {
        "audit_result": result.model_dump(),
        "current_agent": "audit",
    }
