import sys
import os
import json
import streamlit as st

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from graph import build_medusa_graph
from pdf_loader import parse_uploaded_pdfs
from report_exporter import generate_single_page_pdf
from graph_visualizer import build_knowledge_graph_svg

st.set_page_config(
    page_title="MEDUSA-X | Epistemic Clinical Intelligence",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for clean clinical cards, KPI metrics, and typography
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }
    .medusa-hero {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.25rem 1.6rem;
        margin-bottom: 1.25rem;
    }
    .medusa-hero h1 {
        color: #f8fafc;
        font-size: 1.75rem;
        margin: 0 0 0.3rem 0;
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    .medusa-hero p {
        color: #94a3b8;
        font-size: 0.92rem;
        margin: 0;
    }
    .kpi-card {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 10px;
        padding: 0.9rem 1.1rem;
    }
    .kpi-label {
        color: #9ca3af;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.25rem;
    }
    .kpi-value {
        color: #f9fafb;
        font-size: 1.45rem;
        font-weight: 700;
    }
    .conflict-card {
        background: rgba(127, 29, 29, 0.16);
        border: 1px solid rgba(239, 68, 68, 0.35);
        border-left: 4px solid #ef4444;
        border-radius: 8px;
        padding: 1rem 1.15rem;
        margin-bottom: 0.85rem;
    }
    .conflict-header {
        color: #fca5a5;
        font-weight: 700;
        font-size: 0.98rem;
        margin-bottom: 0.35rem;
    }
    .conflict-desc {
        color: #f3f4f6;
        font-size: 0.92rem;
        margin-bottom: 0.55rem;
    }
    .conflict-ev {
        color: #cbd5e1;
        font-size: 0.82rem;
        font-family: monospace;
        background: rgba(15, 23, 42, 0.65);
        padding: 0.35rem 0.6rem;
        border-radius: 5px;
        margin-top: 0.25rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header Banner
st.markdown(
    """
    <div class="medusa-hero">
        <h1>🩺 MEDUSA-X — Epistemic Clinical Intelligence & Reconciliation</h1>
        <p>Neuro-Symbolic Patient Timeline Reconstruction • Medication State-Machine Audit • Automated Safety Gatekeeper</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Sidebar Controls
with st.sidebar:
    st.header("📂 Patient Record Selector")
    case_choice = st.radio(
        "Select Clinical Bundle:",
        [
            "Trap Case 1: GI Bleed + Outdated Warfarin + Penicillin Allergy",
            "Clean Case 2: Resolved Pneumonia (Safe Reconciled Record)",
            "Upload Clinical Files (PDF, PNG, JPG, TXT)",
            "Custom JSON Input (Paste Your Own Records)",
        ],
    )

    records = []
    if "Trap Case 1" in case_choice:
        with open("data/patient_trap_1.json", "r") as f:
            records = json.load(f)
    elif "Clean Case 2" in case_choice:
        with open("data/patient_clean_2.json", "r") as f:
            records = json.load(f)
    elif "Upload Clinical Files" in case_choice or "Upload Hospital PDFs" in case_choice:
        uploaded_pdfs = st.file_uploader(
            "Upload clinical reports (PDF, PNG, JPG, JPEG, TXT):",
            type=["pdf", "png", "jpg", "jpeg", "txt"],
            accept_multiple_files=True,
        )
        if uploaded_pdfs:
            with st.spinner("Extracting text & running Vision OCR on handwritten/scanned images..."):
                records = parse_uploaded_pdfs(uploaded_pdfs)
            st.success(f"Loaded {len(records)} clinical document(s) ready for audit.")
        else:
            st.info("Upload PDF, PNG, JPG, or TXT clinical records above to begin.")
    else:
        raw_input = st.text_area(
            "Paste JSON array of documents:",
            value='[{"doc_id": "DOC_01", "doc_type": "Admission_Note", "doc_date": "2026-11-20", "text": "Patient allergic to Sulfa drugs. Prescribed Bactrim (Sulfamethoxazole) 800mg."}]',
            height=220,
        )
        try:
            records = json.loads(raw_input)
        except Exception as e:
            st.error(f"Invalid JSON: {e}")
            records = []

    st.divider()
    run_clicked = st.button("🚀 Run MEDUSA-X Forensic Audit", type="primary", use_container_width=True)

if run_clicked and records:
    with st.spinner("Running LangGraph Neuro-Symbolic Verification Loop..."):
        app_graph = build_medusa_graph()
        st.session_state["final_state"] = app_graph.invoke({"raw_records": records})

final_state = st.session_state.get("final_state")

# Main Workspace
if final_state:
    audit = final_state["audit_result"]
    events = sorted(final_state["extracted_events"], key=lambda e: e.event_date)
    is_escalated = final_state["final_mode"] == "ESCALATED"

    # Top KPI Row + Full-Width Download Button
    k1, k2, k3, k4, k5 = st.columns([1, 1, 1, 1.3, 1.35])
    with k1:
        score_color = "#ef4444" if audit.trust_score < 0.85 else "#10b981"
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-label">Record Trust Score</div>'
            f'<div class="kpi-value" style="color:{score_color};">{audit.trust_score:.2f} / 1.00</div></div>',
            unsafe_allow_html=True,
        )
    with k2:
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-label">Extracted Events</div>'
            f'<div class="kpi-value">{len(events)}</div></div>',
            unsafe_allow_html=True,
        )
    with k3:
        conf_color = "#ef4444" if audit.conflicts else "#10b981"
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-label">Safety Conflicts</div>'
            f'<div class="kpi-value" style="color:{conf_color};">{len(audit.conflicts)}</div></div>',
            unsafe_allow_html=True,
        )
    with k4:
        gate_label = "HALTED (ESCALATE)" if is_escalated else "VERIFIED SAFE"
        gate_color = "#ef4444" if is_escalated else "#10b981"
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-label">Gatekeeper Status</div>'
            f'<div class="kpi-value" style="color:{gate_color}; font-size:1.15rem; padding-top:0.25rem;">{gate_label}</div></div>',
            unsafe_allow_html=True,
        )
    with k5:
        st.markdown("<div style='height: 0.45rem;'></div>", unsafe_allow_html=True)
        pdf_bytes = generate_single_page_pdf(final_state)
        st.download_button(
            label="📥 Download 1-Page Executive PDF",
            data=pdf_bytes,
            file_name="MEDUSA_X_Audit_Summary.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

    st.markdown("<div style='height: 0.8rem;'></div>", unsafe_allow_html=True)

    # 4-Tab Clinical Workspace
    tab_audit, tab_graph, tab_timeline, tab_sources = st.tabs(
        [
            f"🛡️ Safety Gatekeeper & Audit ({len(audit.conflicts)} Conflicts)",
            "🕸️ Patient Digital Twin Graph",
            f"🕒 Reconstructed Timeline ({len(events)} Events)",
            f"📄 Source Records ({len(records)}) & Agent Trace",
        ]
    )

    with tab_audit:
        if is_escalated:
            st.error(f"🛑 **SYSTEM HALTED — HUMAN ESCALATION REQUIRED:** {audit.escalation_reason}")
            st.markdown("### 🚨 Detected Clinical Conflicts")
            for idx, c in enumerate(audit.conflicts, 1):
                st.markdown(
                    f"""
                    <div class="conflict-card">
                        <div class="conflict-header">{idx}. [{c.severity}] {c.conflict_type}</div>
                        <div class="conflict-desc">{c.description}</div>
                        <div class="conflict-ev"><strong>Evidence A:</strong> {c.evidence_a}</div>
                        <div class="conflict-ev"><strong>Evidence B:</strong> {c.evidence_b}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            if audit.missing_information:
                st.markdown("### ❓ Unresolved / Missing Follow-Up Information")
                for m in audit.missing_information:
                    st.warning(f"• {m}")
        else:
            st.success("✅ **VERIFIED CLINICAL BRIEF — SAFE TO PROCEED**")
            brief = final_state["verified_brief"]
            st.info(brief.patient_overview)

            c_left, c_right = st.columns(2)
            with c_left:
                st.markdown("#### 🩺 Verified Active Diagnoses")
                for d in brief.active_diagnoses:
                    st.markdown(f"- {d}")

                if brief.discontinued_medications:
                    st.markdown("#### 🚫 Discontinued / Historical Medications")
                    for dm in brief.discontinued_medications:
                        st.markdown(f"- {dm}")

            with c_right:
                st.markdown("#### 💊 Reconciled Active Medications")
                for m in brief.reconciled_medications:
                    st.markdown(f"- {m}")

                st.markdown("#### 📅 Discharge Follow-Up Plan")
                for f_item in brief.followup_plan:
                    st.markdown(f"- {f_item}")

    with tab_graph:
        st.markdown("### 🕸️ Neuro-Symbolic Patient Digital Twin Knowledge Graph")
        st.caption("Hover over any node to inspect exact dates, clinical status transitions, and verbatim source quotes.")
        svg_html, g_stats = build_knowledge_graph_svg(records, events, audit)

        g1, g2, g3, g4 = st.columns(4)
        g1.metric("Graph Nodes", g_stats["total_nodes"])
        g2.metric("Directed Edges", g_stats["total_edges"])
        g3.metric("Verified Nodes (Emerald)", g_stats["verified_count"])
        g4.metric("Flagged / Conflict Nodes (Crimson)", g_stats["flagged_count"])

        st.markdown(svg_html, unsafe_allow_html=True)

    with tab_timeline:
        st.markdown("### 🕒 Chronological Patient Digital Twin Events")
        timeline_rows = [
            {
                "Date": ev.event_date,
                "Source Doc": ev.doc_id,
                "Category": ev.category,
                "Clinical Entity": ev.entity_name,
                "Status": ev.status,
                "Dose / Value": ev.value_or_dose or "—",
                "Verbatim Quote": f'"{ev.exact_quote}"',
            }
            for ev in events
        ]
        st.dataframe(timeline_rows, use_container_width=True, hide_index=True)

    with tab_sources:
        col_s1, col_s2 = st.columns([1.2, 1])
        with col_s1:
            st.markdown("### 📄 Fragmented Source Documents")
            for doc in records:
                with st.expander(f"📄 {doc['doc_id']} — {doc['doc_type']} ({doc['doc_date']})", expanded=False):
                    st.write(doc["text"])
        with col_s2:
            st.markdown("### 🤖 LangGraph Execution Trace")
            for log in final_state["agent_logs"]:
                st.info(log)

else:
    # Pre-run preview of loaded documents
    st.markdown("### 📄 Loaded Clinical Records Ready for Audit")
    if not records:
        st.info("Select a sample case, upload PDF/PNG/JPG/TXT clinical records in the left sidebar, and click **Run MEDUSA-X Forensic Audit**.")
    else:
        cols = st.columns(2)
        for idx, doc in enumerate(records):
            with cols[idx % 2]:
                with st.expander(f"📄 {doc['doc_id']} — {doc['doc_type']} ({doc['doc_date']})", expanded=True):
                    st.write(doc["text"])