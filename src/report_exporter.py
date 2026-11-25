import re
from datetime import datetime
import pymupdf


def _clean_pdf_text(text: str) -> str:
    """
    Replaces Unicode punctuation (non-breaking hyphens, smart quotes, em-dashes, bullets)
    with clean Latin-1 ASCII equivalents so PyMuPDF Base-14 fonts never render '?' marks.
    """
    if not text:
        return ""
    replacements = {
        "\u2010": "-",  # hyphen
        "\u2011": "-",  # non-breaking hyphen (causes follow?up)
        "\u2012": "-",  # figure dash
        "\u2013": "-",  # en-dash
        "\u2014": " - ",  # em-dash
        "\u2015": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2022": "-",  # bullet
        "\u00a0": " ",  # non-breaking space
        "\u2265": ">=",
        "\u2264": "<=",
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    # Strip any remaining non-Latin-1 characters safely
    return text.encode("latin-1", errors="ignore").decode("latin-1")


def _wrap_lines(text: str, max_chars: int = 92) -> list[str]:
    """Word-wraps clean text cleanly to fit exact card widths."""
    words = _clean_pdf_text(text).split()
    lines, current = [], ""
    for w in words:
        if len(current) + len(w) + 1 <= max_chars:
            current = f"{current} {w}".strip()
        else:
            if current:
                lines.append(current)
            current = w
    if current:
        lines.append(current)
    return lines or [""]


def generate_single_page_pdf(final_state: dict) -> bytes:
    """
    Generates an executive, single-page A4 clinical reconciliation PDF report
    with vector cards, severity badges, trust score gauge, and clean typography.
    """
    audit = final_state["audit_result"]
    mode = final_state["final_mode"]
    events = sorted(final_state["extracted_events"], key=lambda e: e.event_date)
    is_escalated = mode == "ESCALATED"

    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)  # A4 Portrait

    # Palette
    navy = (0.07, 0.13, 0.24)
    crimson = (0.76, 0.14, 0.16)
    crimson_bg = (0.99, 0.94, 0.94)
    emerald = (0.07, 0.48, 0.27)
    emerald_bg = (0.93, 0.98, 0.95)
    slate_bg = (0.96, 0.97, 0.98)
    border_gray = (0.84, 0.87, 0.91)
    text_dark = (0.11, 0.14, 0.19)
    text_muted = (0.38, 0.43, 0.50)

    accent = crimson if is_escalated else emerald
    accent_bg = crimson_bg if is_escalated else emerald_bg

    # 1. Top Navy Header Bar
    page.draw_rect(pymupdf.Rect(0, 0, 595, 58), color=navy, fill=navy)
    page.insert_text(
        (32, 26),
        "MEDUSA-X  |  CLINICAL RECONCILIATION & SAFETY AUDIT REPORT",
        fontsize=12.5,
        fontname="hebo",
        color=(1, 1, 1),
    )
    page.insert_text(
        (32, 43),
        f"Neuro-Symbolic Epistemic Verification Engine   |   Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        fontsize=8,
        fontname="helv",
        color=(0.78, 0.84, 0.92),
    )

    # 2. Gatekeeper Status Banner + Trust Score Box
    y = 68
    page.draw_rect(pymupdf.Rect(32, y, 563, y + 48), color=accent, fill=accent_bg, width=1.0)
    page.draw_rect(pymupdf.Rect(32, y, 38, y + 48), color=accent, fill=accent)

    banner_title = (
        "GATEKEEPER DECISION: SYSTEM HALTED - HUMAN ESCALATION REQUIRED"
        if is_escalated
        else "GATEKEEPER DECISION: VERIFIED CLINICAL BRIEF (SAFE TO PROCEED)"
    )
    page.insert_text((48, y + 18), banner_title, fontsize=9.5, fontname="hebo", color=accent)

    rationale = _clean_pdf_text(
        audit.escalation_reason if is_escalated else final_state["verified_brief"].patient_overview
    )
    rat_lines = _wrap_lines(rationale, max_chars=74)
    page.insert_text((48, y + 32), rat_lines[0], fontsize=8, fontname="helv", color=text_dark)
    if len(rat_lines) > 1:
        page.insert_text((48, y + 42), rat_lines[1][:78], fontsize=8, fontname="helv", color=text_dark)

    # Trust Score Pill on Right Side of Banner
    page.draw_rect(pymupdf.Rect(468, y + 6, 555, y + 42), color=(1, 1, 1), fill=(1, 1, 1))
    page.insert_text((480, y + 18), "TRUST SCORE", fontsize=6.5, fontname="hebo", color=text_muted)
    page.insert_text(
        (478, y + 35),
        f"{audit.trust_score:.2f} / 1.00",
        fontsize=11.5,
        fontname="hebo",
        color=accent,
    )

    y += 62

    # 3. Main Findings Section (Escalation Conflicts OR Verified Brief)
    if is_escalated:
        page.insert_text(
            (32, y + 10),
            f"DETECTED CLINICAL CONFLICTS & SAFETY RISKS ({len(audit.conflicts)})",
            fontsize=9.5,
            fontname="hebo",
            color=navy,
        )
        y += 16

        for idx, c in enumerate(audit.conflicts[:4], 1):
            desc_lines = _wrap_lines(c.description, max_chars=86)
            ev_a = _clean_pdf_text(c.evidence_a)[:84]
            ev_b = _clean_pdf_text(c.evidence_b)[:84]
            card_h = 38 + (len(desc_lines) * 10)

            page.draw_rect(pymupdf.Rect(32, y, 563, y + card_h), color=border_gray, fill=slate_bg, width=0.7)
            # Severity Badge
            page.draw_rect(pymupdf.Rect(40, y + 6, 102, y + 18), color=crimson, fill=crimson)
            page.insert_text((46, y + 14.5), _clean_pdf_text(c.severity.upper()), fontsize=7, fontname="hebo", color=(1, 1, 1))
            page.insert_text(
                (110, y + 15),
                f"{idx}. {_clean_pdf_text(c.conflict_type)}",
                fontsize=8.5,
                fontname="hebo",
                color=text_dark,
            )

            cy = y + 27
            for dl in desc_lines[:2]:
                page.insert_text((40, cy), dl, fontsize=8, fontname="helv", color=text_dark)
                cy += 10

            page.insert_text((40, cy + 1), f"Evidence A: {ev_a}", fontsize=7.5, fontname="helv", color=text_muted)
            page.insert_text((40, cy + 11), f"Evidence B: {ev_b}", fontsize=7.5, fontname="helv", color=text_muted)
            y += card_h + 6

        if audit.missing_information:
            page.insert_text(
                (32, y + 10),
                "UNRESOLVED / MISSING CLINICAL FOLLOW-UP ITEMS",
                fontsize=9,
                fontname="hebo",
                color=navy,
            )
            y += 16
            for m in audit.missing_information[:3]:
                page.insert_text(
                    (40, y + 8),
                    f"- {_clean_pdf_text(m)[:92]}",
                    fontsize=8,
                    fontname="helv",
                    color=text_dark,
                )
                y += 12
            y += 4
    else:
        brief = final_state["verified_brief"]
        sections = [
            ("VERIFIED ACTIVE DIAGNOSES", brief.active_diagnoses[:5]),
            ("RECONCILED ACTIVE MEDICATIONS", brief.reconciled_medications[:6]),
            ("DISCONTINUED / HISTORICAL MEDICATIONS", brief.discontinued_medications[:4]),
        ]
        for title, items in sections:
            if not items:
                continue
            page.insert_text((32, y + 10), title, fontsize=9, fontname="hebo", color=navy)
            y += 15
            for item in items:
                for wl in _wrap_lines(f"- {item}", max_chars=90)[:2]:
                    page.insert_text((40, y + 8), wl, fontsize=8, fontname="helv", color=text_dark)
                    y += 11
            y += 5

    # 4. Compact Chronological Timeline Table (Fills Remaining Space Cleanly)
    y = max(y + 4, 465)
    page.insert_text(
        (32, y + 10),
        f"RECONSTRUCTED CHRONOLOGICAL TIMELINE ({len(events)} ATOMIC CLINICAL EVENTS)",
        fontsize=9,
        fontname="hebo",
        color=navy,
    )
    y += 15

    # Table Header
    page.draw_rect(pymupdf.Rect(32, y, 563, y + 16), color=navy, fill=navy)
    page.insert_text((38, y + 11), "DATE", fontsize=7, fontname="hebo", color=(1, 1, 1))
    page.insert_text((102, y + 11), "SOURCE DOC", fontsize=7, fontname="hebo", color=(1, 1, 1))
    page.insert_text((215, y + 11), "CLINICAL ENTITY", fontsize=7, fontname="hebo", color=(1, 1, 1))
    page.insert_text((385, y + 11), "STATUS / VALUE", fontsize=7, fontname="hebo", color=(1, 1, 1))
    y += 16

    max_rows = min(len(events), int((808 - y) / 14))
    for idx, ev in enumerate(events[:max_rows]):
        row_bg = (1, 1, 1) if idx % 2 == 0 else slate_bg
        page.draw_rect(pymupdf.Rect(32, y, 563, y + 14), color=border_gray, fill=row_bg, width=0.3)

        status_str = _clean_pdf_text(ev.status)
        val_str = _clean_pdf_text(ev.value_or_dose) if ev.value_or_dose else ""
        combined_status = f"{status_str} ({val_str})" if val_str else status_str
        status_color = crimson if ev.status in ["Discontinued", "Abnormal", "Held"] else text_dark

        page.insert_text((38, y + 10), _clean_pdf_text(ev.event_date), fontsize=7.3, fontname="helv", color=text_dark)
        page.insert_text((102, y + 10), _clean_pdf_text(ev.doc_id)[:20], fontsize=7.3, fontname="helv", color=text_muted)
        page.insert_text((215, y + 10), _clean_pdf_text(ev.entity_name)[:30], fontsize=7.3, fontname="hebo", color=text_dark)
        page.insert_text((385, y + 10), combined_status[:34], fontsize=7.3, fontname="helv", color=status_color)
        y += 14

    if len(events) > max_rows:
        page.insert_text(
            (38, y + 10),
            f"+ {len(events) - max_rows} additional verified events omitted for single-page executive fit.",
            fontsize=7,
            fontname="helv",
            color=text_muted,
        )

    # Footer
    page.draw_line(pymupdf.Point(32, 816), pymupdf.Point(563, 816), color=border_gray, width=0.6)
    page.insert_text(
        (32, 828),
        "MEDUSA-X Epistemic Clinical Intelligence  |  Deterministic State Machine + LangGraph Safety Gatekeeper",
        fontsize=7.2,
        fontname="helv",
        color=text_muted,
    )
    page.insert_text((515, 828), "Page 1 of 1", fontsize=7.2, fontname="hebo", color=text_muted)

    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes