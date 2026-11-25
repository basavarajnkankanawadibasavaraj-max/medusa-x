import os
import re
import time
from typing import Type, TypeVar
import instructor
from instructor.core import InstructorRetryException
from openai import OpenAI, RateLimitError, BadRequestError, NotFoundError, APIConnectionError
from pydantic import BaseModel, ValidationError
from dotenv import load_dotenv

load_dotenv()

T = TypeVar("T", bound=BaseModel)

EXHAUSTED_ROUTES: set[tuple[str, str]] = set()
_CLIENTS: dict[str, instructor.Instructor] = {}
_AUTO_WATERFALL: list[dict] | None = None

# Exclude non-chat, audio, guard, and tiny 7B models that cannot follow nested JSON schemas
_EXCLUDED_KEYWORDS = (
    "whisper",
    "tts",
    "guard",
    "prompt-guard",
    "safeguard",
    "embed",
    "image",
    "audio",
    "vision",
    "allam",
)


def _discover_all_routes() -> list[dict]:
    """
    Automatically queries configured API providers (Groq, and Gemini if set)
    and builds a prioritized list of all capable, live text models.
    """
    global _AUTO_WATERFALL
    if _AUTO_WATERFALL is not None:
        return _AUTO_WATERFALL

    routes = []

    # 1. Auto-discover all capable Groq chat models
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        raw_groq = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=groq_key)
        _CLIENTS["groq"] = instructor.from_openai(raw_groq, mode=instructor.Mode.JSON)
        try:
            all_groq = [
                m.id
                for m in raw_groq.models.list().data
                if not any(bad in m.id.lower() for bad in _EXCLUDED_KEYWORDS)
            ]
            priority_order = [
                "openai/gpt-oss-120b",
                "openai/gpt-oss-20b",
                "qwen/qwen3-32b",
                "llama-3.3-70b-versatile",
                "llama-3.1-8b-instant",
            ]
            ordered = [m for m in priority_order if m in all_groq] + [
                m for m in sorted(all_groq) if m not in priority_order
            ]
            for m in ordered:
                routes.append({"provider": "groq", "model": m})
        except Exception:
            routes.extend([
                {"provider": "groq", "model": "openai/gpt-oss-120b"},
                {"provider": "groq", "model": "openai/gpt-oss-20b"},
            ])

    # 2. Auto-discover live Gemini models if GEMINI_API_KEY is present
    gemini_key = os.getenv("GEMINI_API_KEY")
    if gemini_key:
        raw_gemini = OpenAI(
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key=gemini_key,
        )
        _CLIENTS["gemini"] = instructor.from_openai(raw_gemini, mode=instructor.Mode.JSON)
        try:
            all_gemini = [
                m.id.replace("models/", "")
                for m in raw_gemini.models.list().data
                if "flash" in m.id.lower() and not any(bad in m.id.lower() for bad in _EXCLUDED_KEYWORDS)
            ]
            for m in sorted(all_gemini, reverse=True):
                routes.append({"provider": "gemini", "model": m})
        except Exception:
            pass

    _AUTO_WATERFALL = routes
    return routes


def _sleep_for_rate_limit(err: Exception, attempt: int):
    msg = str(err)
    match = re.search(r"try again in (\d+(?:\.\d+)?)s", msg)
    wait_s = float(match.group(1)) + 1.0 if match else 2.5 * (attempt + 1)
    time.sleep(min(wait_s, 10.0))


def generate_structured_output(
    response_model: Type[T],
    messages: list[dict],
    max_tokens: int = 4096,
    temperature: float = 0.0,
) -> T:
    """
    Executes a structured Pydantic LLM call with full auto-discovery and
    automatic failover across every capable model on your configured API keys.
    """
    last_error = None
    routes = _discover_all_routes()

    for route in routes:
        provider = route["provider"]
        model_name = route["model"]

        if (provider, model_name) in EXHAUSTED_ROUTES:
            continue

        client = _CLIENTS.get(provider)
        if client is None:
            continue

        extra = {"reasoning_effort": "none"} if (provider == "groq" and "qwen" in model_name.lower()) else {}

        for attempt in range(2):
            try:
                return client.chat.completions.create(
                    model=model_name,
                    response_model=response_model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    max_retries=1,
                    extra_body=extra,
                )
            except (
                RateLimitError,
                BadRequestError,
                NotFoundError,
                APIConnectionError,
                ValidationError,
                InstructorRetryException,
            ) as e:
                last_error = e
                err_str = str(e).lower()

                # Permanently retire route for this session if TPD exhausted, unsupported, or oversized
                if (
                    "tokens per day" in err_str
                    or "tpd" in err_str
                    or "quota" in err_str
                    or "not_found" in err_str
                    or "not found" in err_str
                    or "no longer available" in err_str
                    or "does not exist" in err_str
                    or "not supported" in err_str
                    or "request too large" in err_str
                    or "413" in err_str
                    or "404" in err_str
                ):
                    EXHAUSTED_ROUTES.add((provider, model_name))
                    break

                # Schema / JSON validation failure: retry once, then retire weak model and move on
                if "validation error" in err_str or "json_validate_failed" in err_str or "400" in err_str:
                    if attempt >= 1:
                        EXHAUSTED_ROUTES.add((provider, model_name))
                        break
                    time.sleep(1.0)
                    continue

                # Per-minute TPM rate limit: brief pause once, then immediately try next model
                if "429" in err_str or "rate_limit" in err_str:
                    if attempt >= 1:
                        break
                    _sleep_for_rate_limit(e, attempt)
                    continue

                raise

    raise RuntimeError(f"All auto-discovered fallback routes exhausted. Last error: {last_error}")