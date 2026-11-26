from collections import defaultdict
from schemas import ClinicalEvent, AuditResult
from llm_client import generate_structured_output


def build_temporal_timeline(all_events: list[ClinicalEvent]) -> dict:
    """
    Orders all extracted clinical events chronologically and flags obvious
    discontinued-to-active medication state transitions as context hints for the LLM auditor.
    """
    sorted_events = sorted(all_events, key=lambda e: e.event_date)
    med_histories: dict[str, list[ClinicalEvent]] = defaultdict(list)
    state_hints = []

    for ev in sorted_events:
        if ev.category == "Medication":
            key = ev.entity_name.strip().lower().split("(")[0].strip()
            med_histories[key].append(ev)

    for med_key, history in med_histories.items():
        for i in range(1, len(history)):
            prev_ev = history[i - 1]
            curr_ev = history[i]
            if prev_ev.status in ["Discontinued", "Held"] and curr_ev.status in ["Active", "Started"]:
                state_hints.append(
                    f"- '{curr_ev.entity_name}' was marked '{prev_ev.status}' on {prev_ev.event_date} "
                    f"({prev_ev.doc_id}: \"{prev_ev.exact_quote}\") and reappeared as '{curr_ev.status}' on "
                    f"{curr_ev.event_date} ({curr_ev.doc_id}: \"{curr_ev.exact_quote}\")."
                )

    return {
        "sorted_events": sorted_events,
        "state_hints": state_hints,
    }


AUDITOR_SYSTEM_PROMPT = """
You are the MEDUSA-X Epistemic Clinical Auditor & Safety Gatekeeper.
Analyze the complete chronological patient timeline across all source documents and evaluate clinical safety, drug interactions, allergy contraindications, and unresolved critical findings.

Flag CRITICAL conflicts ONLY when there is a genuine clinical safety hazard:
1. Outdated_Medication: A medication explicitly discontinued/held for an adverse event or contraindication is restarted without justification.
2. Allergy_Mismatch: An active medication conflicts with or cross-reacts with a documented patient allergy (e.g., Penicillin -> Amoxicillin, Sulfa -> Bactrim/Sulfamethoxazole).
3. Contradictory_Diagnosis / Unsafe Drug Combination:
   - Anticoagulant (Warfarin, Apixaban, Rivaroxaban) + NSAID (Ibuprofen, Diclofenac, Naproxen) bleeding risk.
   - Duplicate systemic anticoagulants active simultaneously (e.g., Apixaban + Rivaroxaban).
   - Known teratogen (Isotretinoin, Warfarin, Methotrexate, Lisinopril) active in a pregnant patient.
   - Severe QTc prolongation (> 500 ms) with active QT-prolonging medications (Amiodarone, Levofloxacin, Haloperidol, Sotalol).
   - Severe renal impairment (eGFR < 30 or Creatinine > 2.0) with active Metformin, or Hyperkalemia (K+ > 5.2) with potassium-raising drugs (Lisinopril, Losartan, Spironolactone).

Important Clinical Context Guardrails:
- Do NOT flag Missing_Followup as CRITICAL. Missing follow-ups are WARNING severity at most.
- If an outpatient or discharge report documents an appropriate follow-up plan (e.g., repeat labs in 3 months, next visit scheduled, or condition marked "Improving" on active therapy), do NOT flag Missing_Followup at all.
- Legitimate post-PCI dual antiplatelet therapy (Aspirin + Clopidogrel) and documented IV-to-PO antibiotic or insulin step-downs are SAFE.

Every conflict MUST cite exact verbatim quotes and Doc_IDs from the timeline in `evidence_a` and `evidence_b`.
Output ONLY valid JSON matching the requested schema.
"""


def _apply_deterministic_score(audit: AuditResult) -> AuditResult:
    # Ensure Missing_Followup is never treated as a CRITICAL halt
    for c in audit.conflicts:
        if c.conflict_type == "Missing_Followup" and c.severity == "CRITICAL":
            c.severity = "WARNING"

    critical_count = sum(1 for c in audit.conflicts if c.severity == "CRITICAL")
    warning_count = sum(1 for c in audit.conflicts if c.severity == "WARNING")
    missing_count = len(audit.missing_information)

    missing_penalty = (
        (0.05 * missing_count)
        if (critical_count + warning_count) > 0
        else min(0.10, 0.05 * missing_count)
    )
    computed_score = 1.0 - (0.30 * critical_count) - (0.10 * warning_count) - missing_penalty
    audit.trust_score = round(max(0.05 if critical_count > 0 else 0.0, min(1.0, computed_score)), 2)
    audit.must_escalate = (critical_count > 0) or (audit.trust_score < 0.85)
    return audit


def run_clinical_audit(all_events: list[ClinicalEvent]) -> AuditResult:
    timeline_data = build_temporal_timeline(all_events)
    sorted_events = timeline_data["sorted_events"]
    state_hints = timeline_data["state_hints"]

    events_dump = "\n".join([
        f"- [{ev.event_date}] ({ev.doc_id}) [{ev.category}] {ev.entity_name} | Status: {ev.status} | Dose/Val: {ev.value_or_dose} | Quote: \"{ev.exact_quote}\""
        for ev in sorted_events
    ])

    hints_block = (
        "\n=== TIMELINE STATE TRANSITIONS TO VERIFY ===\n" + "\n".join(state_hints)
        if state_hints
        else ""
    )

    prompt = f"""
    === CHRONOLOGICAL PATIENT TIMELINE ===
    {events_dump}
    {hints_block}

    Perform a thorough, context-aware clinical safety and consistency audit of this patient record.
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

    return _apply_deterministic_score(audit)