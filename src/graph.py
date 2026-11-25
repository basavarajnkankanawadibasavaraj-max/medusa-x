from typing import TypedDict, List, Optional
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END

from schemas import ClinicalEvent, AuditResult
from parser import extract_from_document
from auditor import run_clinical_audit
from memory import PatientDigitalTwin
from llm_client import generate_structured_output


class VerifiedClinicalBrief(BaseModel):
    patient_overview: str
    active_diagnoses: List[str] = Field(description="Verified diagnoses with [Doc_ID] citations")
    reconciled_medications: List[str] = Field(description="Verified active medications with dose and [Doc_ID] citations")
    discontinued_medications: List[str] = Field(description="Stopped medications and why, with [Doc_ID] citations")
    chronological_summary: List[str] = Field(description="Ordered clinical events with dates and [Doc_ID] citations")
    followup_plan: List[str]


class MedusaState(TypedDict):
    raw_records: List[dict]
    extracted_events: List[ClinicalEvent]
    audit_result: Optional[AuditResult]
    retrieval_attempts: int
    retrieved_evidence: List[str]
    agent_logs: List[str]
    final_mode: str  # "VERIFIED_SUMMARY" or "ESCALATED"
    verified_brief: Optional[VerifiedClinicalBrief]


def node_extract_and_index(state: MedusaState) -> MedusaState:
    logs = list(state.get("agent_logs", []))
    logs.append("📥 [Layer 1: Ingestion] Extracting atomic clinical facts & indexing into Hybrid Memory (NetworkX + ChromaDB)...")

    twin = PatientDigitalTwin()
    twin.ingest_raw_documents(state["raw_records"])

    all_events = []
    for doc in state["raw_records"]:
        res = extract_from_document(doc)
        all_events.extend(res.events)

    twin.build_temporal_graph(all_events)
    logs.append(f"✅ Extracted {len(all_events)} verified events across {len(state['raw_records'])} documents.")

    return {
        **state,
        "extracted_events": all_events,
        "agent_logs": logs,
        "retrieval_attempts": 0,
        "retrieved_evidence": [],
    }


def node_audit(state: MedusaState) -> MedusaState:
    logs = list(state["agent_logs"])
    logs.append("🔍 [Layer 2 & 3: Clinical Auditor] Checking timeline state transitions, drug allergies, and missing follow-ups...")

    audit = run_clinical_audit(state["extracted_events"])
    logs.append(f"📊 [Epistemic Judge] Trust Score: {audit.trust_score:.2f} | Conflicts Found: {len(audit.conflicts)} | Missing Items: {len(audit.missing_information)}")

    return {
        **state,
        "audit_result": audit,
        "agent_logs": logs,
    }


def node_active_retriever(state: MedusaState) -> MedusaState:
    """Autonomous Evidence Retriever: queries ChromaDB vector store when missing information is flagged."""
    logs = list(state["agent_logs"])
    audit = state["audit_result"]
    attempt = state["retrieval_attempts"] + 1

    query = "; ".join(audit.missing_information)
    logs.append(f"🔎 [Active Retriever Loop #{attempt}] Searching vector store for missing clinical evidence: '{query}'")

    twin = PatientDigitalTwin()
    twin.ingest_raw_documents(state["raw_records"])
    hits = twin.semantic_search(query, n_results=2)

    evidence_notes = [f"[{h['doc_id']} | {h['doc_date']}]: {h['text']}" for h in hits]
    logs.append(f"📄 Retrieved {len(hits)} supporting snippets for verification.")

    return {
        **state,
        "retrieval_attempts": attempt,
        "retrieved_evidence": evidence_notes,
        "agent_logs": logs,
    }


def route_after_audit(state: MedusaState) -> str:
    audit = state["audit_result"]
    if audit.missing_information and state["retrieval_attempts"] < 1:
        return "active_retrieve"
    if audit.must_escalate or audit.trust_score < 0.85:
        return "escalate"
    return "summarize"


def node_escalate(state: MedusaState) -> MedusaState:
    logs = list(state["agent_logs"])
    logs.append("🛑 [Safety Gatekeeper] HALTED! Uncertainty/Conflict threshold breached. Escalating to human clinician.")
    return {
        **state,
        "final_mode": "ESCALATED",
        "agent_logs": logs,
    }


def node_summarize(state: MedusaState) -> MedusaState:
    logs = list(state["agent_logs"])
    logs.append("✅ [Safety Gatekeeper] Trust threshold passed (>= 0.85). Generating cited clinical handoff summary...")

    events_dump = "\n".join([
        f"- [{ev.event_date}] ({ev.doc_id}) [{ev.category}] {ev.entity_name} | Status: {ev.status} | Dose/Val: {ev.value_or_dose}"
        for ev in sorted(state["extracted_events"], key=lambda e: e.event_date)
    ])

    brief = generate_structured_output(
        response_model=VerifiedClinicalBrief,
        messages=[
            {
                "role": "system",
                "content": "Generate a concise, structured clinical handoff summary using ONLY the verified events provided. Cite [Doc_ID] on every item. Output ONLY valid JSON.",
            },
            {"role": "user", "content": events_dump},
        ],
        max_tokens=4096,
        temperature=0.0,
    )

    return {
        **state,
        "final_mode": "VERIFIED_SUMMARY",
        "verified_brief": brief,
        "agent_logs": logs,
    }


def build_medusa_graph():
    workflow = StateGraph(MedusaState)

    workflow.add_node("extract_and_index", node_extract_and_index)
    workflow.add_node("audit", node_audit)
    workflow.add_node("active_retrieve", node_active_retriever)
    workflow.add_node("escalate", node_escalate)
    workflow.add_node("summarize", node_summarize)

    workflow.set_entry_point("extract_and_index")
    workflow.add_edge("extract_and_index", "audit")
    workflow.add_conditional_edges(
        "audit",
        route_after_audit,
        {
            "active_retrieve": "active_retrieve",
            "escalate": "escalate",
            "summarize": "summarize",
        },
    )
    workflow.add_edge("active_retrieve", "audit")
    workflow.add_edge("escalate", END)
    workflow.add_edge("summarize", END)

    return workflow.compile()