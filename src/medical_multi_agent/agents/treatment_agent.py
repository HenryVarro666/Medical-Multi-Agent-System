"""
Treatment Agent — Evidence-based treatment recommendation.

Responsibilities:
  - Generate treatment plan based on confirmed diagnosis
  - Check drug-drug interactions (DDI) via the deterministic rule engine:
    appending curated rule findings to the LLM-proposed plan
  - Verify contraindications against patient allergies / history
  - Provide non-pharmacological treatment suggestions
  - Include evidence references for each recommendation
"""

from __future__ import annotations

import json

import structlog
from langchain_core.messages import HumanMessage, SystemMessage

from ..config.llm import get_llm
from ..services.drug_interaction import (
    check_allergy_contraindication,
    check_interactions,
)

logger = structlog.get_logger(__name__)

TREATMENT_SYSTEM_PROMPT = """You are an expert clinical pharmacologist and treatment specialist. Given a patient's diagnosis and clinical data, provide a comprehensive, evidence-based treatment plan.

Return a JSON object:
{
  "diagnosis_addressed": "the primary diagnosis being treated",
  "medications": [
    {
      "drug_name": "brand name",
      "generic_name": "generic name",
      "dosage": "e.g., 500mg",
      "route": "oral|iv|im|topical|etc",
      "frequency": "e.g., twice daily",
      "duration": "e.g., 7 days",
      "contraindications": ["list any relevant"],
      "side_effects": ["common side effects"]
    }
  ],
  "drug_interactions": [
    {
      "drug_a": "drug 1",
      "drug_b": "drug 2 (can be current medication)",
      "severity": "none|minor|moderate|major|contraindicated",
      "description": "interaction details",
      "recommendation": "what to do"
    }
  ],
  "non_drug_treatments": ["physical therapy", "dietary changes", "etc."],
  "lifestyle_recommendations": ["exercise", "sleep hygiene", "etc."],
  "follow_up_plan": "when and what to check",
  "warnings": ["critical warnings for this treatment"],
  "evidence_references": ["guideline or study reference"]
}

Rules:
- ALWAYS check the patient's current medications for interactions.
- ALWAYS check allergies before recommending any drug.
- Flag any major or contraindicated interactions prominently.
- Provide at least one non-drug treatment option.
- Return ONLY valid JSON, no markdown fences."""


def _run_ddi_engine(treatment_data: dict, patient_info: dict | None) -> dict:
    """
    Reconcile the LLM-proposed plan against the deterministic DDI engine.

    - Every interaction the rule engine flags is appended to
      treatment_data["drug_interactions"] tagged source="rule_engine".
    - Allergy contraindications go to treatment_data["allergy_alerts"].
    - A verification stamp is stored in treatment_data["ddi_engine"] so
      downstream consumers (audit, evals) can confirm the check really ran.
    """
    patient_info = patient_info or {}

    proposed: list[str] = []
    for med in treatment_data.get("medications") or []:
        if not isinstance(med, dict):
            continue
        for key in ("generic_name", "drug_name"):
            name = (med.get(key) or "").strip()
            if name and name.lower() not in [p.lower() for p in proposed]:
                proposed.append(name)

    current = [
        (m.get("name") or "").strip()
        for m in (patient_info.get("current_medications") or [])
        if isinstance(m, dict) and m.get("name")
    ]
    allergies = [
        (a.get("substance") or "").strip()
        for a in (patient_info.get("allergies") or [])
        if isinstance(a, dict) and a.get("substance")
    ]

    engine_hits = check_interactions(proposed, current)

    allergy_alerts = []
    for drug in proposed:
        alert = check_allergy_contraindication(drug, allergies)
        if alert:
            allergy_alerts.append(alert)

    interactions = treatment_data.setdefault("drug_interactions", [])
    for hit in engine_hits:
        interactions.append(
            {
                "drug_a": hit["drug_a"],
                "drug_b": hit["drug_b"],
                "severity": hit["severity"],
                "description": hit["description"],
                "recommendation": hit["recommendation"],
                "source": "rule_engine",
            }
        )

    treatment_data["allergy_alerts"] = allergy_alerts
    treatment_data["ddi_engine"] = {
        "checked": True,
        "proposed_drugs_checked": proposed,
        "current_medications_checked": current,
        "allergies_checked": allergies,
        "engine_flagged_interactions": len(engine_hits),
        "engine_flagged_allergy_alerts": len(allergy_alerts),
    }

    if engine_hits or allergy_alerts:
        logger.warning(
            "treatment_agent.ddi_engine_flags",
            interactions=len(engine_hits),
            allergy_alerts=len(allergy_alerts),
        )
    return treatment_data


def treatment_agent(state) -> dict:
    """
    LangGraph node: Generate treatment plan.
    Reads: state.patient_info, state.diagnosis
    Writes: state.treatment_plan, state.current_agent
    """
    logger.info("treatment_agent.start")

    diagnosis = state.diagnosis
    patient_info = state.patient_info

    if not diagnosis:
        return {
            "treatment_plan": None,
            "current_agent": "treatment",
            "errors": state.errors + ["No diagnosis available for treatment planning"],
        }

    llm = get_llm(temperature=0.2)

    context = json.dumps(
        {"patient_info": patient_info, "diagnosis": diagnosis},
        indent=2,
        ensure_ascii=False,
    )

    messages = [
        SystemMessage(content=TREATMENT_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Clinical context:\n\n{context}\n\n"
                "Provide a comprehensive treatment plan with drug interaction checks."
            )
        ),
    ]

    try:
        response = llm.invoke(messages)
        content = response.content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()

        treatment_data = json.loads(content)

        # Append findings from the limited curated interaction/allergy rules.
        treatment_data = _run_ddi_engine(treatment_data, patient_info)

        logger.info(
            "treatment_agent.success",
            medications_count=len(treatment_data.get("medications", [])),
            ddi_engine_checked=True,
        )
        return {
            "treatment_plan": treatment_data,
            "current_agent": "treatment",
        }
    except json.JSONDecodeError as e:
        logger.error("treatment_agent.json_error", error=str(e))
        return {
            "treatment_plan": None,
            "current_agent": "treatment",
            "errors": state.errors + [f"Treatment JSON parse error: {e}"],
        }
    except Exception as e:
        logger.error("treatment_agent.error", error=str(e))
        return {
            "treatment_plan": None,
            "current_agent": "treatment",
            "errors": state.errors + [f"Treatment error: {e}"],
        }
