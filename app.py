import sys
import os
import json
import streamlit as st

# Ensure src/ is importable
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from graph import build_medusa_graph

st.set_page_config(page_title="MEDUSA-X | Epistemic Clinical Intelligence", page_icon="🩺", layout="wide")

st.title("🩺 MEDUSA-X: Epistemic Clinical Intelligence & Reconciliation")
st.caption("Agentic Patient Timeline Reconstruction • Medication Reconciliation • Automated Safety Gatekeeper")

# Sidebar Controls
st.sidebar.header("📂 Patient Record Selector")
case_choice = st.sidebar.radio(
    "Choose a Clinical Record Bundle:",
    [
        "Trap Case 1: GI Bleed + Outdated Warfarin + Penicillin Allergy",
        "Clean Case 2: Resolved Pneumonia (Safe Reconciled Record)",
        "Custom JSON Input (Paste Your Own Records)",
    ],
)

if "Trap Case 1" in case_choice:
    with open("data/patient_trap_1.json", "r") as f:
        records = json.load(f)
elif "Clean Case 2" in case_choice:
    with open("data/patient_clean_2.json", "r") as f:
        records = json.load(f)
else:
    raw_input = st.sidebar.text_area(
        "Paste JSON array of documents:",
        value='[{"doc_id": "DOC_01", "doc_type": "Admission_Note", "doc_date": "2026-11-20", "text": "Patient allergic to Sulfa drugs. Prescribed Bactrim (Sulfamethoxazole) 800mg."}]',
        height=250,
    )
    try:
        records = json.loads(raw_input)
    except Exception as e:
        st.sidebar.error(f"Invalid JSON: {e}")
        records = []

# Layout: 3 Columns
col_docs, col_agent, col_output = st.columns([1.1, 1.1, 1.4])

with col_docs:
    st.subheader("1️⃣ Fragmented Source Records")
    for doc in records:
        with st.expander(f"📄 {doc['doc_id']} ({doc['doc_date']})", expanded=True):
            st.markdown(f"**Type:** `{doc['doc_type']}`")
            st.write(doc["text"])

run_clicked = st.sidebar.button("🚀 Run MEDUSA-X Forensic Audit", type="primary", use_container_width=True)

if run_clicked and records:
    with st.spinner("Running LangGraph Agentic Verification Loop..."):
        app_graph = build_medusa_graph()
        final_state = app_graph.invoke({"raw_records": records})

    with col_agent:
        st.subheader("2️⃣ Agent Reasoning & Timeline")
        for log in final_state["agent_logs"]:
            st.info(log)

        st.markdown("#### 🕒 Reconstructed Chronological Events")
        sorted_evs = sorted(final_state["extracted_events"], key=lambda e: e.event_date)
        for ev in sorted_evs:
            badge = "🔴" if ev.status in ["Discontinued", "Abnormal"] else "🟢"
            st.markdown(
                f"{badge} **`{ev.event_date}`** | **{ev.entity_name}** (`{ev.status}`) "
                f"{f'— *{ev.value_or_dose}*' if ev.value_or_dose else ''} `[{ev.doc_id}]`"
            )

    with col_output:
        st.subheader("3️⃣ Safety Gatekeeper Decision")
        audit = final_state["audit_result"]
        st.metric("Record Trust Score", f"{audit.trust_score:.2f} / 1.00")

        if final_state["final_mode"] == "ESCALATED":
            st.error("🛑 **SYSTEM HALTED — HUMAN ESCALATION REQUIRED**")
            st.markdown(f"**Escalation Reason:** {audit.escalation_reason}")
            st.warning("⚠️ Automated clinical summary blocked to prevent medical error.")

            st.markdown("### 🚨 Detected Clinical Conflicts")
            for c in audit.conflicts:
                with st.container(border=True):
                    st.markdown(f"**[{c.severity}] {c.conflict_type}**")
                    st.write(c.description)
                    st.caption(f"**Evidence A:** {c.evidence_a}")
                    st.caption(f"**Evidence B:** {c.evidence_b}")

            if audit.missing_information:
                st.markdown("### ❓ Missing Clinical Information")
                for m in audit.missing_information:
                    st.markdown(f"- {m}")
        else:
            st.success("✅ **VERIFIED CLINICAL BRIEF (SAFE TO PROCEED)**")
            brief = final_state["verified_brief"]
            st.write(brief.patient_overview)

            st.markdown("#### 🩺 Active Diagnoses")
            for d in brief.active_diagnoses:
                st.markdown(f"- {d}")

            st.markdown("#### 💊 Reconciled Active Medications")
            for m in brief.reconciled_medications:
                st.markdown(f"- {m}")

            if brief.discontinued_medications:
                st.markdown("#### 🚫 Discontinued / Historical Medications")
                for dm in brief.discontinued_medications:
                    st.markdown(f"- {dm}")

            st.markdown("#### 📅 Chronological Summary & Follow-Up")
            for item in brief.chronological_summary:
                st.markdown(f"- {item}")