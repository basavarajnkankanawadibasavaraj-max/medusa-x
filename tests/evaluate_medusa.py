import os
import sys
import json
import time

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from graph import build_medusa_graph


def load_benchmark_suite() -> list[dict]:
    """Loads 10 clinical benchmark cases (6 high-risk traps + 4 complex clean records)."""
    with open("data/patient_trap_1.json", "r") as f:
        trap_1 = json.load(f)
    with open("data/patient_clean_2.json", "r") as f:
        clean_2 = json.load(f)

    # BENCH_03: Metformin + Acute Renal Failure (eGFR 22) + Potassium 6.3 on Lisinopril
    trap_renal = [
        {
            "doc_id": "DOC_R01_ADMISSION",
            "doc_type": "Admission_Note",
            "doc_date": "2026-11-18",
            "text": "68yo female with Type 2 Diabetes and Hypertension. Home meds: Metformin 1000mg BID, Lisinopril 20mg daily.",
        },
        {
            "doc_id": "DOC_R02_LABS",
            "doc_type": "Lab_Report",
            "doc_date": "2026-11-19",
            "text": "Critical Lab Alert: Serum Creatinine 3.4 mg/dL, eGFR 22 mL/min (Severe Renal Failure), Serum Potassium 6.3 mmol/L (Critical Hyperkalemia).",
        },
        {
            "doc_id": "DOC_R03_DISCHARGE",
            "doc_type": "Discharge_Summary",
            "doc_date": "2026-11-20",
            "text": "Patient stabilized and discharged home. Discharge Medications: Resume Metformin 1000mg BID and Lisinopril 20mg daily. No repeat BMP ordered.",
        },
    ]

    # BENCH_04: Uncomplicated Appendectomy with Proper Antibiotic Stop & Follow-Up
    clean_surgical = [
        {
            "doc_id": "DOC_S01_OPNOTE",
            "doc_type": "Progress_Note",
            "doc_date": "2026-11-15",
            "text": "29yo male underwent uncomplicated laparoscopic appendectomy. No known drug allergies (NKDA). Received single perioperative dose of Cefazolin 2g IV.",
        },
        {
            "doc_id": "DOC_S02_DISCHARGE",
            "doc_type": "Discharge_Summary",
            "doc_date": "2026-11-16",
            "text": "Afebrile, tolerating regular diet, normal WBC 7.8. Cefazolin discontinued post-op. Discharge meds: Acetaminophen 650mg PO q6h PRN pain. Surgical clinic follow-up scheduled in 10 days.",
        },
    ]

    # BENCH_05 (HARD TRAP): Warfarin + High-Dose NSAID (Ibuprofen) Drug-Drug Bleeding Interaction
    trap_nsaid_warfarin = [
        {
            "doc_id": "DOC_N01_ANTICOAG",
            "doc_type": "Progress_Note",
            "doc_date": "2026-11-10",
            "text": "72yo male with mechanical mitral valve on chronic Warfarin 6mg daily. Therapeutic INR 2.7.",
        },
        {
            "doc_id": "DOC_N02_ORTHO",
            "doc_type": "Discharge_Summary",
            "doc_date": "2026-11-14",
            "text": "Evaluated for acute lumbar strain. Discharged with Ibuprofen 600mg PO TID for 14 days and continue home Warfarin 6mg daily.",
        },
    ]

    # BENCH_06 (HARD TRAP): Unrecognized Pregnancy + Teratogenic Medications (Isotretinoin & Lisinopril)
    trap_pregnancy = [
        {
            "doc_id": "DOC_P01_CLINIC",
            "doc_type": "Admission_Note",
            "doc_date": "2026-11-12",
            "text": "26yo female with hypertension and severe nodulocystic acne. Active medications: Lisinopril 10mg daily, Isotretinoin 40mg daily.",
        },
        {
            "doc_id": "DOC_P02_ER_LABS",
            "doc_type": "Lab_Report",
            "doc_date": "2026-11-15",
            "text": "ER evaluation for nausea: Quantitative Serum Beta-hCG 18,450 mIU/mL (Positive intrauterine pregnancy confirmed on pelvic ultrasound).",
        },
        {
            "doc_id": "DOC_P03_DISCHARGE",
            "doc_type": "Discharge_Summary",
            "doc_date": "2026-11-15",
            "text": "Nausea improved with IV fluids. Discharged home to continue all prior home medications: Lisinopril 10mg daily and Isotretinoin 40mg daily.",
        },
    ]

    # BENCH_07 (HARD TRAP): Duplicate Therapeutic Anticoagulants (Apixaban + Rivaroxaban Active Together)
    trap_duplicate_doac = [
        {
            "doc_id": "DOC_D01_ADMIT",
            "doc_type": "Admission_Note",
            "doc_date": "2026-11-16",
            "text": "65yo male admitted with acute DVT. Home medication: Apixaban 5mg BID for atrial fibrillation.",
        },
        {
            "doc_id": "DOC_D02_SWITCH",
            "doc_type": "Progress_Note",
            "doc_date": "2026-11-17",
            "text": "Switched anticoagulation to Rivaroxaban 20mg daily per formulary preference. Discontinued Apixaban.",
        },
        {
            "doc_id": "DOC_D03_DISCHARGE",
            "doc_type": "Discharge_Summary",
            "doc_date": "2026-11-19",
            "text": "Discharge Medication List (Active): Apixaban 5mg PO BID, Rivaroxaban 20mg PO daily, Atorvastatin 40mg daily.",
        },
    ]

    # BENCH_08 (HARD TRAP): Severe QTc Prolongation (535 ms) + Levofloxacin Prescription
    trap_qt_prolongation = [
        {
            "doc_id": "DOC_Q01_CARDIOLOGY",
            "doc_type": "Lab_Report",
            "doc_date": "2026-11-18",
            "text": "12-Lead ECG Report: Sinus bradycardia with markedly prolonged QTc interval of 535 ms (Critical High). Patient taking Amiodarone 200mg daily.",
        },
        {
            "doc_id": "DOC_Q02_URGENTCARE",
            "doc_type": "Discharge_Summary",
            "doc_date": "2026-11-19",
            "text": "Diagnosed with acute sinusitis. Prescribed Levofloxacin 750mg PO daily for 7 days alongside home Amiodarone 200mg daily.",
        },
    ]

    # BENCH_09 (HARD CLEAN): Post-PCI NSTEMI on Intentional Dual Antiplatelet Therapy (Aspirin + Clopidogrel)
    clean_dapt = [
        {
            "doc_id": "DOC_C01_CATH",
            "doc_type": "Progress_Note",
            "doc_date": "2026-11-14",
            "text": "59yo male underwent successful drug-eluting stent placement to LAD for NSTEMI. NKDA. Initiated guideline-directed Dual Antiplatelet Therapy (DAPT): Aspirin 81mg daily plus Clopidogrel 75mg daily.",
        },
        {
            "doc_id": "DOC_C02_LABS",
            "doc_type": "Lab_Report",
            "doc_date": "2026-11-15",
            "text": "Troponin-I trending down (0.12 ng/mL), Hemoglobin 13.8 g/dL, Potassium 4.2 mmol/L, Creatinine 0.95 mg/dL.",
        },
        {
            "doc_id": "DOC_C03_DISCHARGE",
            "doc_type": "Discharge_Summary",
            "doc_date": "2026-11-16",
            "text": "Chest pain resolved. Discharge Medications: Aspirin 81mg daily, Clopidogrel 75mg daily (for 12 months post-stent), Atorvastatin 80mg daily, Metoprolol Succinate 50mg daily. Cardiology follow-up scheduled in 2 weeks.",
        },
    ]

    # BENCH_10 (HARD CLEAN): Resolved DKA with Proper Transition from IV Insulin Drip to SubQ Basal-Bolus
    clean_dka = [
        {
            "doc_id": "DOC_K01_ICU",
            "doc_type": "Admission_Note",
            "doc_date": "2026-11-17",
            "text": "24yo female with Type 1 Diabetes admitted in Diabetic Ketoacidosis (pH 7.18, Anion Gap 26, Glucose 460 mg/dL). Started on continuous Regular Insulin IV infusion.",
        },
        {
            "doc_id": "DOC_K02_LABS",
            "doc_type": "Lab_Report",
            "doc_date": "2026-11-18",
            "text": "Repeat BMP & VBG: Venous pH 7.39, Anion Gap 10 (DKA Resolved), Potassium 4.3 mmol/L, Glucose 142 mg/dL.",
        },
        {
            "doc_id": "DOC_K03_DISCHARGE",
            "doc_type": "Discharge_Summary",
            "doc_date": "2026-11-19",
            "text": "DKA fully resolved. IV Regular Insulin drip discontinued after subcutaneous overlap. Discharge Medications: Insulin Glargine 22 units SubQ nightly, Insulin Lispro 6 units SubQ before meals. Endocrinology clinic follow-up in 1 week.",
        },
    ]

    return [
        {
            "case_id": "BENCH_01_TRAP_GI_BLEED",
            "expected_mode": "ESCALATED",
            "expected_keywords": ["warfarin", "penicillin", "amoxicillin"],
            "records": trap_1,
        },
        {
            "case_id": "BENCH_02_CLEAN_PNEUMONIA",
            "expected_mode": "VERIFIED_SUMMARY",
            "expected_keywords": [],
            "records": clean_2,
        },
        {
            "case_id": "BENCH_03_TRAP_RENAL_HYPERKALEMIA",
            "expected_mode": "ESCALATED",
            "expected_keywords": ["metformin", "lisinopril", "potassium", "egfr"],
            "records": trap_renal,
        },
        {
            "case_id": "BENCH_04_CLEAN_APPENDECTOMY",
            "expected_mode": "VERIFIED_SUMMARY",
            "expected_keywords": [],
            "records": clean_surgical,
        },
        {
            "case_id": "BENCH_05_TRAP_NSAID_WARFARIN",
            "expected_mode": "ESCALATED",
            "expected_keywords": ["warfarin", "ibuprofen"],
            "records": trap_nsaid_warfarin,
        },
        {
            "case_id": "BENCH_06_TRAP_PREGNANCY_TERATOGEN",
            "expected_mode": "ESCALATED",
            "expected_keywords": ["isotretinoin", "lisinopril", "pregnancy"],
            "records": trap_pregnancy,
        },
        {
            "case_id": "BENCH_07_TRAP_DUPLICATE_DOAC",
            "expected_mode": "ESCALATED",
            "expected_keywords": ["apixaban", "rivaroxaban"],
            "records": trap_duplicate_doac,
        },
        {
            "case_id": "BENCH_08_TRAP_QT_PROLONGATION",
            "expected_mode": "ESCALATED",
            "expected_keywords": ["qtc", "levofloxacin", "amiodarone"],
            "records": trap_qt_prolongation,
        },
        {
            "case_id": "BENCH_09_CLEAN_CARDIAC_STENT_DAPT",
            "expected_mode": "VERIFIED_SUMMARY",
            "expected_keywords": [],
            "records": clean_dapt,
        },
        {
            "case_id": "BENCH_10_CLEAN_DKA_RESOLVED",
            "expected_mode": "VERIFIED_SUMMARY",
            "expected_keywords": [],
            "records": clean_dka,
        },
    ]


def run_benchmarks():
    app_graph = build_medusa_graph()
    suite = load_benchmark_suite()

    print("=" * 82)
    print(f"MEDUSA-X FORENSIC CLINICAL BENCHMARK SUITE ({len(suite)} CLINICAL SCENARIOS)")
    print("=" * 82)

    results = []
    for case in suite:
        cid = case["case_id"]
        expected_mode = case["expected_mode"]
        valid_doc_ids = {d["doc_id"] for d in case["records"]}

        print(f"\nRunning {cid} ({len(case['records'])} docs)...")
        t0 = time.perf_counter()
        final_state = app_graph.invoke({"raw_records": case["records"]})
        latency = time.perf_counter() - t0

        actual_mode = final_state["final_mode"]
        audit = final_state["audit_result"]
        events = final_state["extracted_events"]

        grounded_events = sum(
            1 for ev in events if ev.doc_id in valid_doc_ids and bool(ev.exact_quote.strip())
        )
        citation_acc = (grounded_events / len(events) * 100.0) if events else 0.0

        if case["expected_keywords"]:
            conflict_blob = " ".join(
                f"{c.description} {c.evidence_a} {c.evidence_b}".lower()
                for c in audit.conflicts
            )
            matched_kw = sum(1 for kw in case["expected_keywords"] if kw in conflict_blob)
            kw_recall = (matched_kw / len(case["expected_keywords"])) * 100.0
        else:
            kw_recall = 100.0

        passed = actual_mode == expected_mode
        results.append({
            "case_id": cid,
            "expected_mode": expected_mode,
            "actual_mode": actual_mode,
            "passed": passed,
            "trust_score": audit.trust_score,
            "conflicts": len(audit.conflicts),
            "kw_recall": kw_recall,
            "citation_acc": citation_acc,
            "latency": latency,
        })

        status_tag = "PASS" if passed else "FAIL"
        print(
            f"  [{status_tag}] Mode: {actual_mode} (Expected: {expected_mode}) | "
            f"Trust: {audit.trust_score:.2f} | Conflicts: {len(audit.conflicts)} | "
            f"Citation Acc: {citation_acc:.1f}% | Latency: {latency:.2f}s"
        )

    total = len(results)
    passed_total = sum(1 for r in results if r["passed"])
    traps = [r for r in results if r["expected_mode"] == "ESCALATED"]
    cleans = [r for r in results if r["expected_mode"] == "VERIFIED_SUMMARY"]

    trap_recall = (sum(1 for r in traps if r["actual_mode"] == "ESCALATED") / len(traps)) * 100.0
    false_escalations = (sum(1 for r in cleans if r["actual_mode"] == "ESCALATED") / len(cleans)) * 100.0
    avg_citation = sum(r["citation_acc"] for r in results) / total
    avg_latency = sum(r["latency"] for r in results) / total

    print("\n" + "=" * 82)
    print("AGGREGATE BENCHMARK SUMMARY (N = 10 CASES)")
    print("=" * 82)
    print(f"Overall Gatekeeper Accuracy : {passed_total}/{total} ({(passed_total / total) * 100:.1f}%)")
    print(f"Trap Escalation Recall      : {trap_recall:.1f}% ({sum(1 for r in traps if r['actual_mode'] == 'ESCALATED')}/{len(traps)} traps caught)")
    print(f"False Escalation Rate       : {false_escalations:.1f}% ({sum(1 for r in cleans if r['actual_mode'] == 'ESCALATED')}/{len(cleans)} clean cases flagged)")
    print(f"Mean Citation Grounding     : {avg_citation:.1f}%")
    print(f"Mean End-to-End Latency     : {avg_latency:.2f}s per patient bundle")
    print("=" * 82)

    with open("tests/benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved detailed JSON metrics to tests/benchmark_results.json")


if __name__ == "__main__":
    run_benchmarks()