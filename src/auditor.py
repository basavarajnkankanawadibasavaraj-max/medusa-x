import os
import re
import time
from collections import defaultdict
import instructor
from instructor.core import InstructorRetryException
from openai import OpenAI, RateLimitError, BadRequestError
from dotenv import load_dotenv

from schemas import ClinicalEvent, AuditResult

load_dotenv()

client = instructor.from_openai(
    OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=os.getenv("GROQ_API_KEY"),
    ),
    mode=instructor.Mode.JSON,
)

FALLBACK_MODELS = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "meta-llama/llama-4-maverick-17b-128e-instruct",
    "meta-llama/llama-4-scout-17b-16e-instruct",
]

EXHAUSTED_MODELS: set[str] = set()

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
1. Outdated_Medication: Drugs copy-pasted or resumed after being discontinued/held without clinical justification, or medications contraindicated by new critical lab results (e.g., Metformin with severe renal failure / low eGFR, or ACE inhibitors/Lisinopril with severe hyperkalemia).
2. Allergy_Mismatch: Prescribed drugs that cross-react with documented patient allergies (e.g., Penicillin allergy vs. Amoxicillin, or Sulfa allergy vs. Bactrim/Sulfamethoxazole).
3. Contradictory_Diagnosis: Conflicting diagnoses across notes.
4. Missing_Followup: Critical abnormal labs or acute organ injuries lacking follow-up verification at discharge.

EPISTEMIC HUMILITY RULES:
- Calculate a 'trust_score' between 0.0 and 1.0 (deduct heavily for every unresolved conflict or missing critical lab).
- Set 'must_escalate' = True if there is ANY CRITICAL conflict, medication discrepancy, allergy risk, or if trust_score < 0.85.
- Every conflict MUST cite exact quotes and Document IDs in 'evidence_a' and 'evidence_b'.
"""

def _sleep_for_rate_limit(err: Exception, attempt: int):
    msg = str(err)
    match = re.search(r"try again in (\d+(?:\.\d+)?)s", msg)
    wait_s = float(match.group(1)) + 1.0 if match else 3.0 * (attempt + 1)
    time.sleep(min(wait_s, 15.0))

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

    for model_name in FALLBACK_MODELS:
        if model_name in EXHAUSTED_MODELS:
            continue
        for attempt in range(3):
            try:
                return client.chat.completions.create(
                    model=model_name,
                    response_model=AuditResult,
                    messages=[
                        {"role": "system", "content": AUDITOR_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.0,
                    max_tokens=8192,
                    max_retries=1,
                )
            except (RateLimitError, BadRequestError, InstructorRetryException) as e:
                err_str = str(e).lower()
                if "tokens per day" in err_str or "tpd" in err_str or "model_not_found" in err_str:
                    EXHAUSTED_MODELS.add(model_name)
                    break
                if "json_validate_failed" in err_str or "400" in err_str:
                    if attempt >= 1:
                        break
                    time.sleep(1.0)
                    continue
                if "429" not in err_str and "rate_limit" not in err_str:
                    raise
                if attempt == 2:
                    break
                _sleep_for_rate_limit(e, attempt)

    raise RuntimeError("All Groq fallback models reached their rate limits or failed validation.")