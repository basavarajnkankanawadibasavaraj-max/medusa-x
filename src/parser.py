import os
import re
import time
import instructor
from instructor.core import InstructorRetryException
from openai import OpenAI, RateLimitError, BadRequestError
from dotenv import load_dotenv
from schemas import DocumentExtraction

load_dotenv()

raw_openai = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY"),
)

client = instructor.from_openai(raw_openai, mode=instructor.Mode.JSON)

FALLBACK_MODELS = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "meta-llama/llama-4-maverick-17b-128e-instruct",
    "meta-llama/llama-4-scout-17b-16e-instruct",
]

EXHAUSTED_MODELS: set[str] = set()

SYSTEM_PROMPT = """
You are a forensic clinical data extraction engine.
Extract EVERY medication, diagnosis, abnormal or key lab result, allergy, and procedure from the provided medical document.
STRICT RULES:
1. Never guess or infer unstated facts.
2. Every extracted event MUST include the exact 'doc_id' and a short verbatim 'exact_quote' (under 15 words) from the text.
3. Pay close attention to status changes: if a medication is stopped or held, mark status as 'Discontinued' or 'Held'.
"""

def _sleep_for_rate_limit(err: Exception, attempt: int):
    msg = str(err)
    match = re.search(r"try again in (\d+(?:\.\d+)?)s", msg)
    wait_s = float(match.group(1)) + 1.0 if match else 3.0 * (attempt + 1)
    time.sleep(min(wait_s, 15.0))

def extract_from_document(doc: dict) -> DocumentExtraction:
    prompt = f"""
    Document ID: {doc['doc_id']}
    Document Type: {doc['doc_type']}
    Document Date: {doc['doc_date']}
    
    Clinical Text:
    {doc['text']}
    """
    for model_name in FALLBACK_MODELS:
        if model_name in EXHAUSTED_MODELS:
            continue
        for attempt in range(3):
            try:
                return client.chat.completions.create(
                    model=model_name,
                    response_model=DocumentExtraction,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.0,
                    max_tokens=8192,
                    max_retries=1,
                )
            except (RateLimitError, BadRequestError, InstructorRetryException) as e:
                err_str = str(e).lower()
                if "tokens per day" in err_str or "tpd" in err_str or "model_not_found" in err_str:
                    EXHAUSTED_MODELS.add(model_name)
                    break
                if "json_validate_failed" in err_str or "400" in err_str:
                    if attempt >= 1:
                        break
                    time.sleep(1.0)
                    continue
                if "429" not in err_str and "rate_limit" not in err_str:
                    raise
                if attempt == 2:
                    break
                _sleep_for_rate_limit(e, attempt)

    raise RuntimeError("All Groq fallback models reached their rate limits or failed validation.")