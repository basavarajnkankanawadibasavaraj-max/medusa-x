import os
import json
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import ServerError, ClientError
from schemas import DocumentExtraction

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Exact verified models from your API key across separate serving pools
MODEL_FALLBACKS = [
    "gemini-4-argon",
    "gemini-4-carbon",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.8-flash",
    "gemini-2.5-pro",
    "gemma-4-31b-it",
]

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
    
    last_error = None
    for model_name in MODEL_FALLBACKS:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    response_mime_type="application/json",
                    response_json_schema=DocumentExtraction.model_json_schema(),
                    temperature=0.0,
                ),
            )
            return DocumentExtraction.model_validate_json(response.text)
        except (ServerError, ClientError) as e:
            print(f"  ⚠️ {model_name} unavailable ({e.code}), switching...")
            last_error = e
            time.sleep(0.5)
            
    raise RuntimeError(f"All models failed. Last error: {last_error}")

if __name__ == "__main__":
    with open("data/patient_trap_1.json", "r") as f:
        records = json.load(f)

    print("🔍 Running Layer 1 Forensic Extraction...\n")
    for doc in records:
        extracted = extract_from_document(doc)
        print(f"=== {extracted.doc_id} ({extracted.doc_date}) ===")
        for ev in extracted.events:
            print(f"  [{ev.category}] {ev.entity_name} -> Status: {ev.status} | Dose/Val: {ev.value_or_dose}")
            print(f"     Proof: \"{ev.exact_quote}\"")
        print()