"""
LangGraph clinical decision pipeline orchestrator.

Wires the five agents into a sequential pipeline with conditional routing:

    Intake -> Diagnosis --(needs_more_info?)--> Intake (loop back, capped)
                         --(ready)----------> Treatment -> Coding -> Audit -> END

The diagnosis->intake loop-back is guarded by a retry counter
(state.diagnosis_attempts vs settings.max_diagnosis_retries). Once the retry
budget is exhausted the pipeline proceeds to Treatment on a low-confidence
path. Callers must supply a recursion limit that accommodates the retry budget.
"""

from __future__ import annotations

import structlog
from langgraph.graph import END, StateGraph

from ..agents.audit_agent import audit_agent
from ..agents.coding_agent import coding_agent
from ..agents.diagnosis_agent import diagnosis_agent
from ..agents.intake_agent import intake_agent
from ..agents.treatment_agent import treatment_agent
from ..config.settings import get_settings
from .state import ClinicalState

logger = structlog.get_logger(__name__)


def _route_after_diagnosis(state: ClinicalState) -> str:
    """
    Conditional edge after Diagnosis Agent.

    If the agent determined that more patient info is needed, route back to
    Intake — but only while the retry budget is not exhausted
    (settings.max_diagnosis_retries loop-backs, default 2). Once the cap is
    reached, proceed to Treatment anyway so the pipeline always terminates.
    """
    if state.needs_more_info:
        max_retries = get_settings().max_diagnosis_retries
        if state.diagnosis_attempts <= max_retries:
            logger.info(
                "route.diagnosis_retry",
                attempt=state.diagnosis_attempts,
                max_retries=max_retries,
            )
            return "intake"
        logger.warning(
            "route.diagnosis_retry_cap_reached",
            attempts=state.diagnosis_attempts,
            max_retries=max_retries,
        )
    return "treatment"


def build_clinical_pipeline(checkpointer=None):
    """
    Construct and compile the clinical decision LangGraph pipeline.

    State is not retained between invocations by default. Callers can opt into
    checkpoint retention by supplying an explicit checkpointer, then must also
    provide its required thread configuration when invoking the graph.

    Returns a compiled graph that can be invoked with:
        result = pipeline.invoke({"raw_input": "patient description ..."})
    """
    workflow = StateGraph(ClinicalState)

    # --- Register nodes ---
    workflow.add_node("intake", intake_agent)
    workflow.add_node("diagnosis", diagnosis_agent)
    workflow.add_node("treatment", treatment_agent)
    workflow.add_node("coding", coding_agent)
    workflow.add_node("audit", audit_agent)

    # --- Define edges ---
    workflow.set_entry_point("intake")
    workflow.add_edge("intake", "diagnosis")

    workflow.add_conditional_edges(
        "diagnosis",
        _route_after_diagnosis,
        {
            "intake": "intake",
            "treatment": "treatment",
        },
    )

    workflow.add_edge("treatment", "coding")
    workflow.add_edge("coding", "audit")
    workflow.add_edge("audit", END)

    return workflow.compile(checkpointer=checkpointer)


# Convenience singleton
_pipeline = None


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = build_clinical_pipeline()
    return _pipeline
