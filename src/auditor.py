from collections import defaultdict
from schemas import ClinicalEvent, AuditResult
from llm_client import generate_structured_output


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
1. Outdated_Medication: Drugs copy-pasted or resumed after being discontinued/held without clinical justification, or medications contraindicated by new critical lab results (e.g., Metformin with severe renal failure / low eGFR, or ACE inhibitors/Lisinopril with severe hyperkalemia) or pregnancy (e.g., Isotretinoin, Lisinopril).
2. Allergy_Mismatch: Prescribed drugs that cross-react with documented patient allergies (e.g., Penicillin allergy vs. Amoxicillin, or Sulfa allergy vs. Bactrim/Sulfamethoxazole).
3. Contradictory_Diagnosis: Conflicting diagnoses across notes, or dangerous drug-drug / duplicate therapy interactions (e.g., Warfarin + NSAID/Ibuprofen, duplicate DOACs like Apixaban + Rivaroxaban, or severe QTc prolongation > 500ms + Levofloxacin/Amiodarone).
4. Missing_Followup: Critical abnormal labs or acute organ injuries lacking follow-up verification at discharge.

EPISTEMIC HUMILITY RULES:
- Calculate a 'trust_score' between 0.0 and 1.0 (deduct heavily for every unresolved conflict or missing critical lab).
- Set 'must_escalate' = True if there is ANY CRITICAL conflict, medication discrepancy, allergy risk, or if trust_score < 0.85.
- Every conflict MUST cite exact quotes and Document IDs in 'evidence_a' and 'evidence_b'.
- Output ONLY valid JSON matching the requested schema.
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

    audit = generate_structured_output(
        response_model=AuditResult,
        messages=[
            {"role": "system", "content": AUDITOR_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        max_tokens=4096,
        temperature=0.0,
    )

    # Enforce deterministic trust score math based on verified conflict severities
    critical_count = sum(1 for c in audit.conflicts if c.severity == "CRITICAL")
    warning_count = sum(1 for c in audit.conflicts if c.severity == "WARNING")
    missing_count = len(audit.missing_information)

    computed_score = 1.0 - (0.30 * critical_count) - (0.10 * warning_count) - (0.05 * missing_count)
    audit.trust_score = round(max(0.05 if critical_count > 0 else 0.0, min(1.0, computed_score)), 2)
    audit.must_escalate = (critical_count > 0) or (audit.trust_score < 0.85)

    return audit