import html
import math
import networkx as nx


def build_knowledge_graph_svg(raw_records: list[dict], extracted_events: list, audit_result) -> tuple[str, dict]:
    """
    Constructs a directed NetworkX Patient Digital Twin graph and renders it as a
    high-contrast interactive SVG with hover tooltips—zero external plotting libraries required.
    """
    G = nx.DiGraph()
    G.add_node("PATIENT", kind="patient", label="Patient Digital Twin", detail="Unified Longitudinal Record")

    # Identify entity names involved in any detected conflict
    conflicted_terms = set()
    if audit_result and getattr(audit_result, "conflicts", None):
        for c in audit_result.conflicts:
            combined = f"{c.description} {c.evidence_a} {c.evidence_b}".lower()
            for ev in extracted_events:
                if ev.entity_name.lower() in combined:
                    conflicted_terms.add(ev.entity_name.lower())

    doc_ids = []
    for doc in raw_records:
        did = doc["doc_id"]
        doc_ids.append(did)
        G.add_node(
            did,
            kind="document",
            label=did,
            detail=f"{doc['doc_type']} ({doc['doc_date']})",
        )
        G.add_edge("PATIENT", did, relation="HAS_RECORD")

    event_nodes = []
    for idx, ev in enumerate(extracted_events):
        ev_node_id = f"EV_{idx}_{ev.entity_name}"
        is_flagged = (
            ev.status in ["Discontinued", "Held", "Abnormal"]
            or ev.entity_name.lower() in conflicted_terms
        )
        G.add_node(
            ev_node_id,
            kind="flagged" if is_flagged else "verified",
            label=ev.entity_name,
            status=ev.status,
            date=ev.event_date,
            quote=ev.exact_quote,
            dose=ev.value_or_dose or "",
        )
        source_doc = ev.doc_id if G.has_node(ev.doc_id) else (doc_ids[0] if doc_ids else "PATIENT")
        G.add_edge(source_doc, ev_node_id, relation=ev.status.upper())
        event_nodes.append(ev_node_id)

    # Radial Concentric Layout (Patient -> Documents -> Clinical Entities)
    width, height = 1040, 620
    cx, cy = width / 2, height / 2
    positions = {"PATIENT": (cx, cy)}

    # Ring 1: Documents
    r_docs = 145
    for i, did in enumerate(doc_ids):
        angle = (2 * math.pi * i / max(1, len(doc_ids))) - (math.pi / 2)
        positions[did] = (cx + r_docs * math.cos(angle), cy + r_docs * math.sin(angle))

    # Ring 2: Clinical Events (elliptical outer ring for readable labels)
    rx_ev, ry_ev = 415, 255
    for i, ev_id in enumerate(event_nodes):
        angle = (2 * math.pi * i / max(1, len(event_nodes))) - (math.pi / 2)
        positions[ev_id] = (cx + rx_ev * math.cos(angle), cy + ry_ev * math.sin(angle))

    svg_parts = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" height="600" '
        f'style="background:#0b1120; border:1px solid #1e293b; border-radius:12px; font-family:Inter,system-ui,sans-serif;">',
        "<defs>"
        '<marker id="arrow-muted" markerWidth="8" markerHeight="8" refX="18" refY="3" orient="auto">'
        '<path d="M0,0 L0,6 L7,3 z" fill="#475569" /></marker>'
        '<marker id="arrow-red" markerWidth="8" markerHeight="8" refX="18" refY="3" orient="auto">'
        '<path d="M0,0 L0,6 L7,3 z" fill="#ef4444" /></marker>'
        '<marker id="arrow-green" markerWidth="8" markerHeight="8" refX="18" refY="3" orient="auto">'
        '<path d="M0,0 L0,6 L7,3 z" fill="#10b981" /></marker>'
        "</defs>",
    ]

    # Draw Edges
    for u, v, data in G.edges(data=True):
        x1, y1 = positions[u]
        x2, y2 = positions[v]
        v_kind = G.nodes[v].get("kind")
        if v_kind == "flagged":
            stroke, marker = "#ef4444", "url(#arrow-red)"
            opacity = "0.78"
        elif v_kind == "verified":
            stroke, marker = "#10b981", "url(#arrow-green)"
            opacity = "0.60"
        else:
            stroke, marker = "#475569", "url(#arrow-muted)"
            opacity = "0.65"

        svg_parts.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{stroke}" stroke-width="1.5" stroke-opacity="{opacity}" marker-end="{marker}" />'
        )

    # Draw Nodes
    for node_id, attrs in G.nodes(data=True):
        x, y = positions[node_id]
        kind = attrs["kind"]
        raw_label = attrs["label"]
        short_label = html.escape(raw_label[:20] + ("…" if len(raw_label) > 20 else ""))

        if kind == "patient":
            fill, border, radius = "#1d4ed8", "#93c5fd", 26
            tooltip = "Unified Patient Digital Twin"
        elif kind == "document":
            fill, border, radius = "#1e293b", "#60a5fa", 18
            tooltip = html.escape(f"{raw_label}: {attrs.get('detail', '')}")
        elif kind == "flagged":
            fill, border, radius = "#991b1b", "#fca5a5", 13
            tooltip = html.escape(
                f"[FLAGGED] {raw_label} | Status: {attrs.get('status')} | Date: {attrs.get('date')} | Quote: \"{attrs.get('quote')}\""
            )
        else:
            fill, border, radius = "#065f46", "#6ee7b7", 12
            tooltip = html.escape(
                f"[VERIFIED] {raw_label} | Status: {attrs.get('status')} | Date: {attrs.get('date')} | Quote: \"{attrs.get('quote')}\""
            )

        svg_parts.append(f"<g><title>{tooltip}</title>")
        svg_parts.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius}" fill="{fill}" stroke="{border}" stroke-width="2" />'
        )
        label_y = y + radius + 13
        svg_parts.append(
            f'<text x="{x:.1f}" y="{label_y:.1f}" fill="#e2e8f0" font-size="10.5" '
            f'font-weight="600" text-anchor="middle">{short_label}</text>'
        )
        svg_parts.append("</g>")

    # Legend Overlay
    svg_parts.append(
        '<g transform="translate(18, 18)">'
        '<rect x="0" y="0" width="235" height="96" rx="8" fill="#0f172a" stroke="#334155" />'
        '<circle cx="16" cy="20" r="6" fill="#1d4ed8" stroke="#93c5fd" stroke-width="1.5" />'
        '<text x="30" y="24" fill="#cbd5e1" font-size="11">Patient Digital Twin Hub</text>'
        '<circle cx="16" cy="42" r="6" fill="#1e293b" stroke="#60a5fa" stroke-width="1.5" />'
        '<text x="30" y="46" fill="#cbd5e1" font-size="11">Clinical Source Document</text>'
        '<circle cx="16" cy="64" r="6" fill="#065f46" stroke="#6ee7b7" stroke-width="1.5" />'
        '<text x="30" y="68" fill="#cbd5e1" font-size="11">Verified Active Entity</text>'
        '<circle cx="16" cy="84" r="6" fill="#991b1b" stroke="#fca5a5" stroke-width="1.5" />'
        '<text x="30" y="88" fill="#fca5a5" font-size="11">Discontinued / Flagged / Conflict</text>'
        "</g>"
    )

    svg_parts.append("</svg>")

    stats = {
        "total_nodes": G.number_of_nodes(),
        "total_edges": G.number_of_edges(),
        "flagged_count": sum(1 for _, d in G.nodes(data=True) if d.get("kind") == "flagged"),
        "verified_count": sum(1 for _, d in G.nodes(data=True) if d.get("kind") == "verified"),
    }
    return "".join(svg_parts), stats