import os
import json
import instructor
from openai import OpenAI
from dotenv import load_dotenv
from schemas import DocumentExtraction

load_dotenv()

client = instructor.from_openai(
    OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=os.getenv("GROQ_API_KEY"),
    ),
    mode=instructor.Mode.JSON,
)

SYSTEM_PROMPT = """
You are a forensic clinical data extraction engine.
Extract EVERY medication, diagnosis, lab result, allergy, and procedure from the provided medical document.
STRICT RULES:
1. Never guess or infer unstated facts.
2. Every extracted event MUST include the exact 'doc_id' and a verbatim 'exact_quote' from the text.
3. Pay close attention to status changes: if a medication is stopped or held, mark status as 'Discontinued' or 'Held'.
"""

def extract_from_document(doc: dict) -> DocumentExtraction:
    prompt = f"""
    Document ID: {doc['doc_id']}
    Document Type: {doc['doc_type']}
    Document Date: {doc['doc_date']}

    Clinical Text:
    {doc['text']}
    """
    return client.chat.completions.create(
        model="openai/gpt-oss-120b",
        response_model=DocumentExtraction,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
    )

if __name__ == "__main__":
    with open("data/patient_trap_1.json", "r") as f:
        records = json.load(f)

    print("🔍 Running Layer 1 Forensic Extraction (via Groq)...\n")
    for doc in records:
        extracted = extract_from_document(doc)
        print(f"=== {extracted.doc_id} ({extracted.doc_date}) ===")
        for ev in extracted.events:
            print(f"  [{ev.category}] {ev.entity_name} -> Status: {ev.status} | Dose/Val: {ev.value_or_dose}")
            print(f"     Proof: \"{ev.exact_quote}\"")
        print()