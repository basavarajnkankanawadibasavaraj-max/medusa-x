import re
from datetime import date
import pymupdf

def infer_doc_type(header_or_name: str, text: str) -> str:
    """Infers clinical document type from section header, filename, or text."""
    combined = f"{header_or_name} {text[:300]}".lower()
    if "discharge" in combined:
        return "Discharge_Summary"
    if "lab" in combined or "hemoglobin" in combined or "inr" in combined or "egfr" in combined or "potassium" in combined:
        return "Lab_Report"
    if "progress" in combined or "consult" in combined or "imaging" in combined or "cardiology" in combined:
        return "Progress_Note"
    if "prescription" in combined or "rx" in combined:
        return "Prescription"
    return "Admission_Note"

def extract_date_from_text(text: str, fallback_date: str) -> str:
    """Finds an ISO date (YYYY-MM-DD) in the text if present, otherwise uses fallback."""
    match = re.search(r"\b(20\d{2}-\d{2}-\d{2})\b", text[:500])
    if match:
        return match.group(1)
    return fallback_date

def parse_uploaded_pdfs(uploaded_files) -> list[dict]:
    """
    Converts uploaded PDFs into structured MEDUSA-X document records.
    Splits multi-section dossiers into clean clinical notes while filtering out
    tiny header/footer fragments (< 180 chars).
    """
    records = []
    today_str = date.today().isoformat()

    for file_idx, uploaded_file in enumerate(uploaded_files, 1):
        pdf_bytes = uploaded_file.read()
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        pages_text = [page.get_text("text").strip() for page in doc if page.get_text("text").strip()]
        doc.close()

        full_text = "\n".join(pages_text)
        if not full_text.strip():
            continue

        clean_name = re.sub(r"[^A-Za-z0-9_]+", "_", uploaded_file.name.rsplit(".", 1)[0]).upper()

        # Split on 'Section 1:', 'SECTION 1:', or 'DOC_01_' and ignore tiny fragments
        section_splits = re.split(
            r"(?i)(?=\bDOC_\d+_[A-Z_]+|\bSection\s+\d+\s*:)",
            full_text,
        )
        section_splits = [s.strip() for s in section_splits if len(s.strip()) >= 180]

        if len(section_splits) <= 1 and len(pages_text) > 1:
            section_splits = [p for p in pages_text if len(p) >= 100]

        if len(section_splits) > 1:
            for sec_idx, sec_text in enumerate(section_splits, 1):
                first_line = sec_text.splitlines()[0][:45]
                sec_tag = re.sub(r"[^A-Za-z0-9_]+", "_", first_line).strip("_").upper()
                doc_id = f"SEC_{sec_idx:02d}_{sec_tag[:22]}"
                doc_type = infer_doc_type(first_line, sec_text)
                doc_date = extract_date_from_text(sec_text, today_str)

                records.append({
                    "doc_id": doc_id,
                    "doc_type": doc_type,
                    "doc_date": doc_date,
                    "text": sec_text,
                })
        else:
            doc_id = f"PDF_{file_idx:02d}_{clean_name[:22]}"
            doc_type = infer_doc_type(uploaded_file.name, full_text)
            doc_date = extract_date_from_text(full_text, today_str)

            records.append({
                "doc_id": doc_id,
                "doc_type": doc_type,
                "doc_date": doc_date,
                "text": full_text,
            })

    return records