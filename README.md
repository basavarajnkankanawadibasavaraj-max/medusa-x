# MEDUSA-X — Neuro-Symbolic Clinical Forensic Auditor & Safety Gatekeeper

**MEDUSA-X** is an automated clinical reconciliation and safety gatekeeper designed to eliminate preventable post-discharge medication errors and diagnostic oversights. It ingests multi-format patient records (digital PDFs, lab reports, discharge summaries, and handwritten scans), reconstructs a chronological patient timeline, and audits clinical consistency using first-principles medical invariants and verbatim dual-quote grounding.

---

## Key Features

- **Multimodal Clinical Ingestion & Vision OCR (`src/pdf_loader.py`):** Parses digital PDFs, structured JSON/TXT bundles, and scanned/handwritten outpatient records (`.png`, `.jpg`) with automatic Vision OCR failover.
- **Parallel Event Extraction (`src/graph.py`, `src/parser.py`):** Uses `ThreadPoolExecutor` and strict Pydantic schemas (`ClinicalEvent`) to extract timestamped diagnoses, medications, dosages, lab values, and allergies concurrently.
- **First-Principles Epistemic Auditor (`src/auditor.py`):** Evaluates the complete chronological timeline across four universal clinical safety invariants without brittle specialty-specific lookup tables:
  1. **Mechanism-of-Action Contraindications** (allergy cross-reactivity, drug–disease, and drug–lab conflicts)
  2. **Temporal State & Reversal Integrity** (catching medications stopped/held for adverse events that are accidentally restarted at discharge)
  3. **Acute Pathophysiology & Trajectory Resolution** (flagging unaddressed critical lab drops or acute conditions cleared prematurely)
  4. **Outpatient & Discharge Follow-Up Integrity** (preventing false alarms when valid monitoring plans are documented)
- **Zero-Trust Deterministic Gatekeeper:** Computes an auditable Trust Score (`0.00–1.00`), halting unsafe discharges (`SYSTEM HALTED - HUMAN ESCALATION REQUIRED`) with exact verbatim quotes (`Evidence A` vs. `Evidence B`) while clearing safe charts (`VERIFIED CLINICAL BRIEF`) with zero false escalations.
- **Task-Aware Multi-Model Routing (`src/llm_client.py`):** Routes parallel document extraction to fast lightweight models (`Groq llama-3.1-8b-instant`) and full-timeline clinical reasoning to high-context models (`Google Gemini Flash`) with automatic zero-sleep failover.
- **Interactive Clinical Dashboard & A4 PDF Export (`app.py`, `src/report_exporter.py`):** Streamlit Digital Twin UI with longitudinal timeline views and single-page A4 executive PDF audit reports.

---

## Benchmark Performance

Evaluated across a 10-scenario clinical benchmark suite spanning acute inpatient traps, allergy mismatches, drug state reversals, and legitimate chronic outpatient care:

| Metric | Score |
| :--- | :--- |
| **Overall Gatekeeper Accuracy** | **100.0% (10/10)** |
| **Critical Clinical Trap Recall** | **100.0% (6/6)** |
| **False Escalation Rate (Clean Charts)** | **0.0% (0/4)** |
| **Mean End-to-End Bundle Latency** | **~21.8s – 24.5s** |

---

## System Architecture

```text
[Patient Documents: PDF / Images / JSON / Handwritten Scans]
                           │
                           ▼
        ┌─────────────────────────────────────┐
        │  1. Multimodal Ingestion & OCR      │  (PyMuPDF + Vision OCR)
        └─────────────────────────────────────┘
                           │
                           ▼
        ┌─────────────────────────────────────┐
        │  2. Parallel Atomic Extraction      │  (ThreadPoolExecutor + Pydantic)
        └─────────────────────────────────────┘
                           │
                           ▼
        ┌─────────────────────────────────────┐
        │  3. Chronological State Machine     │  (Temporal Ordering + State Hints)
        └─────────────────────────────────────┘
                           │
                           ▼
        ┌─────────────────────────────────────┐
        │  4. First-Principles LLM Auditor    │  (4 Clinical Invariants + Verbatim Quotes)
        └─────────────────────────────────────┘
                           │
                           ▼
        ┌─────────────────────────────────────┐
        │  5. Deterministic Safety Gatekeeper │  (Trust Score 0.00–1.00 -> HALT or VERIFY)
        └─────────────────────────────────────┘