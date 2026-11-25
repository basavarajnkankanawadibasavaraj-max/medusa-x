import os
import re
import base64
import pymupdf
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

HANDWRITING_PROMPT = """
You are a forensic medical transcription specialist.
Transcribe ALL printed and handwritten text from this clinical document image accurately.
CRITICAL SAFETY RULES:
1. Preserve exact drug names, dosages, units (mg, mcg, mL, units), routes (PO, IV, SubQ), and frequencies (OD, BD/BID, TID, QID, PRN, HS).
2. Preserve dates, patient allergies, diagnosis notes, and lab values verbatim.
3. If any handwritten word or dosage number is smudged, ambiguous, or unreadable, write [ILLEGIBLE_HANDWRITING] instead of guessing.
4. Output clean, structured plain text without commentary.
"""


def _get_vision_routes() -> list[tuple[OpenAI, str]]:
    """Builds live Vision-capable routes across Groq and Google Gemini free tier."""
    routes = []

    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        groq_client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=groq_key)
        try:
            live_groq = [
                m.id
                for m in groq_client.models.list().data
                if "vision" in m.id.lower() or "scout" in m.id.lower() or "maverick" in m.id.lower()
            ]
            for m in live_groq:
                routes.append((groq_client, m))
        except Exception:
            routes.append((groq_client, "meta-llama/llama-4-scout-17b-16e-instruct"))

    gemini_key = os.getenv("GEMINI_API_KEY")
    if gemini_key:
        gemini_client = OpenAI(
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key=gemini_key,
        )
        try:
            live_gemini = [
                m.id.replace("models/", "")
                for m in gemini_client.models.list().data
                if "flash" in m.id.lower()
                and "image" not in m.id.lower()
                and "audio" not in m.id.lower()
            ]
            for m in sorted(live_gemini, reverse=True):
                routes.append((gemini_client, m))
        except Exception:
            routes.append((gemini_client, "gemini-3.8-flash"))

    return routes


def _transcribe_image_bytes(img_bytes: bytes, mime_type: str = "image/png") -> str:
    """Uses Multimodal Vision models (Groq -> Gemini) to transcribe handwritten/scanned pages."""
    b64_img = base64.b64encode(img_bytes).decode("utf-8")
    data_url = f"data:{mime_type};base64,{b64_img}"

    for client, model_name in _get_vision_routes():
        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": HANDWRITING_PROMPT},
                            {"type": "image_url", "image_url": {"url": data_url}},
                        ],
                    }
                ],
                temperature=0.0,
                max_tokens=2048,
            )
            text = (resp.choices[0].message.content or "").strip()
            if text:
                return text
        except Exception:
            continue

    return "[VISION_OCR_FAILED: Unable to transcribe handwritten or scanned image]"


def _infer_doc_metadata(text: str, filename: str, index: int) -> tuple[str, str]:
    """Extracts document type and ISO date from transcribed, digital, or plain text."""
    combined = f"{filename} {text}".lower()
    if "discharge" in combined:
        doc_type = "Discharge_Summary"
    elif "lab" in combined or "hemoglobin" in combined or "inr" in combined or "creatinine" in combined:
        doc_type = "Lab_Report"
    elif "rx" in combined or "prescription" in combined or "tab." in combined or "cap." in combined:
        doc_type = "Handwritten_Prescription"
    elif "progress" in combined or "ward" in combined:
        doc_type = "Progress_Note"
    else:
        doc_type = "Admission_Note"

    date_match = re.search(r"\b(20\d{2}-\d{2}-\d{2})\b", f"{filename} {text}")
    if date_match:
        doc_date = date_match.group(1)
    else:
        slash_match = re.search(r"\b(\d{1,2})[/-](\d{1,2})[/-](20\d{2})\b", text)
        if slash_match:
            d, m, y = slash_match.groups()
            doc_date = f"{y}-{int(m):02d}-{int(d):02d}"
        else:
            doc_date = f"2026-11-{10 + min(index, 18):02d}"

    return doc_type, doc_date


def parse_uploaded_pdfs(uploaded_files) -> list[dict]:
    """
    Processes uploaded clinical files (.pdf, .png, .jpg, .jpeg, .txt).
    - .txt files: decoded directly as UTF-8/Latin-1 clinical notes.
    - .png / .jpg / .jpeg: transcribed via Multimodal Vision OCR.
    - .pdf files: extracts selectable digital text, or routes scanned/handwritten pages (< 40 chars) to Vision OCR.
    """
    records = []
    record_counter = 1

    for file_obj in uploaded_files:
        fname = file_obj.name
        raw_bytes = file_obj.read()
        ext = os.path.splitext(fname)[1].lower()
        clean_stem = re.sub(r"[^A-Za-z0-9_]", "_", os.path.splitext(fname)[0])[:18]

        # Case A: Plain-text clinical report (.txt)
        if ext == ".txt":
            try:
                txt_content = raw_bytes.decode("utf-8").strip()
            except UnicodeDecodeError:
                txt_content = raw_bytes.decode("latin-1", errors="ignore").strip()

            if txt_content:
                doc_type, doc_date = _infer_doc_metadata(txt_content, fname, record_counter)
                records.append(
                    {
                        "doc_id": f"DOC_{record_counter:02d}_{clean_stem}_TXT",
                        "doc_type": doc_type,
                        "doc_date": doc_date,
                        "text": txt_content,
                    }
                )
                record_counter += 1
            continue

        # Case B: Image upload (.png, .jpg, .jpeg)
        if ext in [".png", ".jpg", ".jpeg", ".webp"]:
            mime = "image/png" if ext == ".png" else "image/jpeg"
            transcribed = _transcribe_image_bytes(raw_bytes, mime_type=mime)
            doc_type, doc_date = _infer_doc_metadata(transcribed, fname, record_counter)
            records.append(
                {
                    "doc_id": f"DOC_{record_counter:02d}_{clean_stem}_OCR",
                    "doc_type": doc_type,
                    "doc_date": doc_date,
                    "text": transcribed,
                }
            )
            record_counter += 1
            continue

        # Case C: PDF file (digital, scanned, or mixed pages)
        pdf_doc = pymupdf.open(stream=raw_bytes, filetype="pdf")
        for page_idx, page in enumerate(pdf_doc, start=1):
            digital_text = page.get_text("text").strip()

            if len(digital_text) >= 40:
                page_text = digital_text
                tag = ""
            else:
                pix = page.get_pixmap(dpi=200)
                page_text = _transcribe_image_bytes(pix.tobytes("png"), mime_type="image/png")
                tag = "_OCR"

            if not page_text.strip():
                continue

            doc_type, doc_date = _infer_doc_metadata(page_text, fname, record_counter)
            records.append(
                {
                    "doc_id": f"DOC_{record_counter:02d}_{clean_stem}_P{page_idx}{tag}",
                    "doc_type": doc_type,
                    "doc_date": doc_date,
                    "text": page_text,
                }
            )
            record_counter += 1

        pdf_doc.close()

    return records