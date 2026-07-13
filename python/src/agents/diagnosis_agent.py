"""
Diagnosis Agent — Differential diagnosis based on structured patient data.

Responsibilities:
  - Analyze symptoms + lab results against medical knowledge
  - Generate ranked differential diagnosis list with confidence scores
  - Provide evidence chains for each candidate diagnosis
  - Recommend additional tests if information is insufficient
  - Grounds the LLM with GraphRAG knowledge-graph retrieval: candidate
    diseases retrieved by symptom overlap are injected into the prompt,
    and the retrieval trace is preserved in the diagnosis output.
"""

from __future__ import annotations
import json
import structlog
from langchain_core.messages import HumanMessage, SystemMessage

from ..config.llm import get_llm
from ..services.graphrag_service import get_graphrag_service

logger = structlog.get_logger(__name__)

DIAGNOSIS_SYSTEM_PROMPT = """You are an expert diagnostician performing differential diagnosis. Given structured patient information, provide a comprehensive differential diagnosis.

Return a JSON object with this structure:
{
  "primary_diagnosis": {
    "disease_name": "most likely diagnosis",
    "icd10_hint": "approximate ICD-10 code (e.g., J18.9)",
    "confidence": 0.85,
    "evidence": ["supporting finding 1", "supporting finding 2"],
    "reasoning": "clinical reasoning explanation"
  },
  "differential_list": [
    {
      "disease_name": "alternative diagnosis",
      "icd10_hint": "ICD-10 code",
      "confidence": 0.6,
      "evidence": ["evidence 1"],
      "reasoning": "why this is considered"
    }
  ],
  "recommended_tests": ["test 1 to confirm/rule out", "test 2"],
  "clinical_notes": "overall clinical impression",
  "knowledge_sources": ["source 1", "source 2"],
  "needs_more_info": false
}

Rules:
- Confidence scores must be between 0 and 1.
- Provide at least 2-3 differential diagnoses.
- List evidence from the patient data that supports each diagnosis.
- If critical information is missing, set needs_more_info to true.
- Use standard medical terminology and ICD-10 code hints.
- Return ONLY valid JSON, no markdown fences."""


def _retrieve_kg_candidates(patient_info: dict) -> list[dict]:
    """
    GraphRAG retrieval step: query the knowledge-graph service with the
    patient's symptom list and return ranked candidate diseases.
    Deterministic, runs locally (Neo4j when configured, in-memory fallback).
    """
    try:
        symptoms = [
            s.get("name", "")
            for s in (patient_info.get("symptoms") or [])
            if isinstance(s, dict) and s.get("name")
        ]
        if not symptoms:
            return []
        candidates = get_graphrag_service().find_diseases_by_symptoms(symptoms)
        return candidates[:5]
    except Exception as e:  # retrieval must never break the diagnosis path
        logger.warning("diagnosis_agent.graphrag_failed", error=str(e))
        return []


def _format_kg_block(candidates: list[dict]) -> str:
    if not candidates:
        return ""
    lines = [
        f"- {c['disease']}"
        + (f" (ICD-10 {c['icd10_code']})" if c.get("icd10_code") else "")
        + f": matched {c['symptom_match_count']} of the patient's symptoms"
        for c in candidates
    ]
    return (
        "\n\nKnowledge-graph retrieval (deterministic symptom->disease lookup, "
        "ranked by symptom overlap). Treat these as grounding hints — you are "
        "NOT limited to this list and must weigh all clinical evidence:\n"
        + "\n".join(lines)
    )


def diagnosis_agent(state) -> dict:
    """
    LangGraph node: Generate differential diagnosis from patient info.
    Reads: state.patient_info, state.diagnosis_attempts
    Writes: state.diagnosis, state.needs_more_info, state.diagnosis_attempts,
            state.current_agent
    """
    attempts = getattr(state, "diagnosis_attempts", 0) + 1
    logger.info("diagnosis_agent.start", attempt=attempts)

    patient_info = state.patient_info
    if not patient_info:
        return {
            "diagnosis": None,
            "needs_more_info": True,
            "diagnosis_attempts": attempts,
            "current_agent": "diagnosis",
            "errors": state.errors + ["No patient info available for diagnosis"],
        }

    llm = get_llm(temperature=0.2)

    patient_summary = json.dumps(patient_info, indent=2, ensure_ascii=False)

    # --- GraphRAG grounding: inject retrieved candidates into the prompt ---
    kg_candidates = _retrieve_kg_candidates(patient_info)
    kg_block = _format_kg_block(kg_candidates)
    if kg_candidates:
        logger.info(
            "diagnosis_agent.graphrag_retrieval",
            candidates=[c["disease"] for c in kg_candidates],
        )

    messages = [
        SystemMessage(content=DIAGNOSIS_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Patient information:\n\n{patient_summary}"
                f"{kg_block}\n\nProvide your differential diagnosis."
            )
        ),
    ]

    try:
        response = llm.invoke(messages)
        content = response.content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()

        diagnosis_data = json.loads(content)
        needs_more = diagnosis_data.pop("needs_more_info", False)

        # Preserve the retrieval trace so downstream consumers (and evals)
        # can verify what the knowledge graph actually contributed.
        diagnosis_data["knowledge_graph_candidates"] = kg_candidates
        sources = diagnosis_data.get("knowledge_sources") or []
        if kg_candidates:
            kg_source = "GraphRAG knowledge graph (symptom->disease retrieval)"
            if kg_source not in sources:
                sources.append(kg_source)
        diagnosis_data["knowledge_sources"] = sources

        logger.info(
            "diagnosis_agent.success",
            primary=diagnosis_data.get("primary_diagnosis", {}).get("disease_name"),
            attempt=attempts,
        )
        return {
            "diagnosis": diagnosis_data,
            "needs_more_info": needs_more,
            "diagnosis_attempts": attempts,
            "current_agent": "diagnosis",
        }
    except json.JSONDecodeError as e:
        logger.error("diagnosis_agent.json_error", error=str(e))
        return {
            "diagnosis": None,
            "needs_more_info": False,
            "diagnosis_attempts": attempts,
            "current_agent": "diagnosis",
            "errors": state.errors + [f"Diagnosis JSON parse error: {e}"],
        }
    except Exception as e:
        logger.error("diagnosis_agent.error", error=str(e))
        return {
            "diagnosis": None,
            "needs_more_info": False,
            "diagnosis_attempts": attempts,
            "current_agent": "diagnosis",
            "errors": state.errors + [f"Diagnosis error: {e}"],
        }
