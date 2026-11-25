import os
import time
from typing import Type, TypeVar
import instructor
from instructor.core import InstructorRetryException, IncompleteOutputException
from openai import OpenAI, RateLimitError, BadRequestError, NotFoundError, APIConnectionError
from pydantic import BaseModel, ValidationError
from dotenv import load_dotenv

load_dotenv()

T = TypeVar("T", bound=BaseModel)

EXHAUSTED_ROUTES: set[tuple[str, str]] = set()
_CLIENTS: dict[str, instructor.Instructor] = {}
_AUTO_WATERFALL: list[dict] | None = None

_EXCLUDED_KEYWORDS = (
    "whisper",
    "tts",
    "orpheus",
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
    """Builds a latency-optimized list of fast, structured-JSON-capable models."""
    global _AUTO_WATERFALL
    if _AUTO_WATERFALL is not None:
        return _AUTO_WATERFALL

    routes = []

    # 1. Groq Fast Models First (sub-2s structured JSON)
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
            # Fast, high-accuracy models first; heavy reasoning models as backup
            speed_priority = [
                "llama-3.3-70b-versatile",
                "qwen/qwen3-32b",
                "openai/gpt-oss-20b",
                "openai/gpt-oss-120b",
                "llama-3.1-8b-instant",
            ]
            ordered = [m for m in speed_priority if m in all_groq] + [
                m for m in sorted(all_groq) if m not in speed_priority
            ]
            for m in ordered:
                routes.append({"provider": "groq", "model": m})
        except Exception:
            routes.append({"provider": "groq", "model": "llama-3.3-70b-versatile"})

    # 2. Google Gemini Flash Models (High quota + fast JSON)
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


def generate_structured_output(
    response_model: Type[T],
    messages: list[dict],
    max_tokens: int = 4096,
    temperature: float = 0.0,
) -> T:
    """Zero-sleep instant failover across fast Groq and Gemini endpoints."""
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

        extra = {}
        effective_max_tokens = max_tokens
        if provider == "groq":
            if "qwen" in model_name.lower():
                extra = {"reasoning_effort": "none"}
            elif "gpt-oss" in model_name.lower():
                extra = {"reasoning_effort": "low"}
                effective_max_tokens = min(max_tokens, 5500)
        elif provider == "gemini":
            effective_max_tokens = 8192

        try:
            return client.chat.completions.create(
                model=model_name,
                response_model=response_model,
                messages=messages,
                temperature=temperature,
                max_tokens=effective_max_tokens,
                max_retries=1,
                extra_body=extra,
            )
        except (
            RateLimitError,
            BadRequestError,
            NotFoundError,
            APIConnectionError,
            ValidationError,
            IncompleteOutputException,
            InstructorRetryException,
        ) as e:
            last_error = e
            err_str = str(e).lower()

            if (
                "tokens per day" in err_str
                or "tpd" in err_str
                or "quota" in err_str
                or "not_found" in err_str
                or "not found" in err_str
                or "no longer available" in err_str
                or "does not exist" in err_str
                or "not supported" in err_str
                or "terms acceptance" in err_str
                or "request too large" in err_str
                or "413" in err_str
                or "404" in err_str
            ):
                EXHAUSTED_ROUTES.add((provider, model_name))

            # Fail over immediately (0s sleep) to the next live route
            continue

    raise RuntimeError(f"All auto-discovered fallback routes exhausted. Last error: {last_error}")