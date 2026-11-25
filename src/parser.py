from schemas import DocumentExtraction
from llm_client import generate_structured_output

SYSTEM_PROMPT = """
You are a forensic clinical data extraction engine.
Extract EVERY medication, diagnosis, abnormal or key lab result, allergy, and procedure from the provided medical document.
STRICT RULES:
1. Never guess or infer unstated facts.
2. Every extracted event MUST include the exact 'doc_id' and a short verbatim 'exact_quote' (under 15 words) from the text.
3. Pay close attention to status changes: if a medication is stopped or held, mark status as 'Discontinued' or 'Held'.
4. Output ONLY valid JSON matching the requested schema.
"""


def extract_from_document(doc: dict) -> DocumentExtraction:
    prompt = f"""
    Document ID: {doc['doc_id']}
    Document Type: {doc['doc_type']}
    Document Date: {doc['doc_date']}
    
    Clinical Text:
    {doc['text']}
    """
    return generate_structured_output(
        response_model=DocumentExtraction,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        max_tokens=4096,
        temperature=0.0,
    )