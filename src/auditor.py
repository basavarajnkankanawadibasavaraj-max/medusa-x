import os
import json
from collections import defaultdict
import instructor
from openai import OpenAI
from dotenv import load_dotenv

from schemas import ClinicalEvent, AuditResult
from parser import extract_from_document

load_dotenv()

client = instructor.from_openai(
    OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=os.getenv("GROQ_API_KEY"),
    ),
    mode=instructor.Mode.JSON,
)

def build_temporal_timeline(all_events: list[ClinicalEvent]) -> dict:
    """
    Symbolic Layer: Orders all events chronologically and detects deterministic
    medication state-machine violations (e.g., Discontinued -> Active without restart rationale).
    """
    sorted_events = sorted(all_events, key=lambda e: e.event_date)
    med_histories = defaultdict(list)
    deterministic_flags = []

    for ev in sorted_events:
        if ev.category == "Medication":
            key = ev.entity_name.strip().lower()
            med_histories[key].append(ev)

    for med_name, history in med_histories.items():
        for i in range(1, len(history)):
            prev_ev = history[i - 1]
            curr_ev = history[i]
            if prev_ev.status in ["Discontinued", "Held"] and curr_ev.status in ["Active", "Started"]:
                deterministic_flags.append(
                    f"STATE VIOLATION on '{curr_ev.entity_name}': Marked '{prev_ev.status}' on {prev_ev.event_date} "
                    f"({prev_ev.doc_id}: \"{prev_ev.exact_quote}\") but reappeared as '{curr_ev.status}' on "
                    f"{curr_ev.event_date} ({curr_ev.doc_id}: \"{curr_ev.exact_quote}\")."
                )

    return {
        "sorted_events": sorted_events,
        "deterministic_flags": deterministic_flags,
    }

AUDITOR_SYSTEM_PROMPT = """
You are the MEDUSA-X Epistemic Clinical Auditor & Safety Gatekeeper.
Your job is to verify reconstructed patient timelines and catch dangerous discrepancies BEFORE any clinical summary is trusted.

Analyze the chronological clinical events and the deterministic timeline flags to detect:
1. Outdated_Medication: Drugs copy-pasted or resumed after being discontinued/held without clinical justification.
2. Allergy_Mismatch: Prescribed drugs that cross-react with documented patient allergies (e.g., Penicillin allergy vs. Amoxicillin/Ampicillin).
3. Contradictory_Diagnosis: Conflicting diagnoses across notes.
4. Missing_Followup: Critical abnormal labs (e.g., low Hemoglobin, high INR) or suspected conditions lacking follow-up verification at discharge.

EPISTEMIC HUMILITY RULES:
- Calculate a 'trust_score' between 0.0 and 1.0 (deduct heavily for every unresolved conflict or missing critical lab).
- Set 'must_escalate' = True if there is ANY CRITICAL conflict, medication discrepancy, allergy risk, or if trust_score < 0.85.
- Every conflict MUST cite exact quotes and Document IDs in 'evidence_a' and 'evidence_b'.
"""

def run_clinical_audit(all_events: list[ClinicalEvent]) -> AuditResult:
    timeline_data = build_temporal_timeline(all_events)
    sorted_events = timeline_data["sorted_events"]
    deterministic_flags = timeline_data["deterministic_flags"]

    events_dump = "\n".join([
        f"- [{ev.event_date}] ({ev.doc_id}) [{ev.category}] {ev.entity_name} | Status: {ev.status} | Dose/Val: {ev.value_or_dose} | Quote: \"{ev.exact_quote}\""
        for ev in sorted_events
    ])

    flags_dump = "\n".join(deterministic_flags) if deterministic_flags else "None detected by rule engine."

    prompt = f"""
    === CHRONOLOGICAL PATIENT TIMELINE ===
    {events_dump}

    === DETERMINISTIC STATE-MACHINE FLAGS ===
    {flags_dump}

    Perform a complete forensic audit, assign a trust_score, and decide whether the system MUST STOP and escalate to a human clinician.
    """

    return client.chat.completions.create(
        model="openai/gpt-oss-120b",
        response_model=AuditResult,
        messages=[
            {"role": "system", "content": AUDITOR_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
    )

if __name__ == "__main__":
    with open("data/patient_trap_1.json", "r") as f:
        records = json.load(f)

    print("🔍 Step 1: Extracting events across all patient documents...")
    all_events = []
    for doc in records:
        extracted = extract_from_document(doc)
        all_events.extend(extracted.events)

    print(f"✅ Extracted {len(all_events)} clinical events. Running Layer 2 & 3 Forensic Audit...\n")
    audit = run_clinical_audit(all_events)

    print("=" * 65)
    print(f"📊 RECORD TRUST SCORE: {audit.trust_score:.2f} / 1.00")
    if audit.must_escalate:
        print("🛑 GATEKEEPER DECISION: SYSTEM HALTED — HUMAN ESCALATION REQUIRED")
        print(f"   Reason: {audit.escalation_reason}")
    else:
        print("✅ GATEKEEPER DECISION: SAFE TO SUMMARIZE")
    print("=" * 65)

    print("\n🚨 DETECTED CONFLICTS:")
    for idx, c in enumerate(audit.conflicts, 1):
        print(f"\n  {idx}. [{c.severity}] {c.conflict_type}")
        print(f"     Issue:      {c.description}")
        print(f"     Evidence A: {c.evidence_a}")
        print(f"     Evidence B: {c.evidence_b}")

    if audit.missing_information:
        print("\n❓ MISSING CLINICAL INFORMATION:")
        for item in audit.missing_information:
            print(f"  - {item}")