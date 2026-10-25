"""Gemini Developer API client for `CloudAdapter.generate` (Lab 5 Part B).

Why Gemini on an Azure project: an Azure for Students subscription cannot deploy a managed
Azure OpenAI model within this course's budget, and Gemini's free tier returns exact
per-request token counts at no cost. That the LLM provider differs from the cloud
provider is itself a portability finding: `generate` is the one seam method whose natural
home is not the cloud you deploy on.

Uses the generateContent REST endpoint with only the standard library, so it runs in the
same minimal environments as the drift detector. Google now labels generateContent
"legacy" in favour of the Interactions API; it still serves every model used here, and
moving would touch this file only.

Token counts come from the response's own `usageMetadata`, never from counting words:
  input_tokens         promptTokenCount
  output_tokens        candidatesTokenCount + thoughtsTokenCount   (thinking is billed as output)
  thinking_tokens      thoughtsTokenCount                          (reported separately)
  cached_input_tokens  cachedContentTokenCount                     (implicit cache hits)

Environment:
  GEMINI_API_KEY   required; never logged, never written to a report
  GEMINI_MODEL     default gemini-3.5-flash-lite (free tier; cheapest paid rate)
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

ENDPOINT = os.environ.get(
    "GEMINI_ENDPOINT",  # override only for offline tests against a fake server
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent")
DEFAULT_MODEL = "gemini-3.5-flash-lite"
RETRY_STATUSES = {429, 500, 503}


def generate(prompt: str, params: dict[str, Any]) -> dict[str, Any]:
    """One call. `params` keys: system, max_output_tokens, temperature, thinking_level,
    json (bool), model. Returns response text plus usage read from the provider."""
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY is not set")
    model = params.get("model") or os.environ.get("GEMINI_MODEL", DEFAULT_MODEL)

    config: dict[str, Any] = {"maxOutputTokens": int(params.get("max_output_tokens", 256))}
    if params.get("temperature") is not None:
        config["temperature"] = float(params["temperature"])
    if params.get("thinking_level"):
        config["thinkingConfig"] = {"thinkingLevel": params["thinking_level"]}
    if params.get("json", True):
        config["responseMimeType"] = "application/json"

    body: dict[str, Any] = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": config,
    }
    if params.get("system"):
        body["systemInstruction"] = {"parts": [{"text": params["system"]}]}

    data = json.dumps(body).encode()
    url = ENDPOINT.format(model=model)
    for attempt in range(6):
        req = urllib.request.Request(url, data=data, method="POST", headers={
            "Content-Type": "application/json", "x-goog-api-key": key})
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=90) as r:
                payload = json.loads(r.read())
            latency_ms = round((time.perf_counter() - started) * 1000)
            break
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:300]
            if exc.code in RETRY_STATUSES and attempt < 5:
                # Free tier: a few requests per minute. Back off rather than fail the run.
                time.sleep(min(60, 5 * 2 ** attempt) * float(os.environ.get("GEMINI_BACKOFF_SCALE", "1")))
                continue
            raise RuntimeError(f"Gemini {exc.code}: {detail}") from None

    candidate = (payload.get("candidates") or [{}])[0]
    text = "".join(p.get("text", "") for p in candidate.get("content", {}).get("parts", [])
                   if not p.get("thought"))
    usage = payload.get("usageMetadata", {})
    thinking = int(usage.get("thoughtsTokenCount", 0))
    return {
        "response": text,
        "input_tokens": int(usage.get("promptTokenCount", 0)),
        "output_tokens": int(usage.get("candidatesTokenCount", 0)) + thinking,
        "thinking_tokens": thinking,
        "cached_input_tokens": int(usage.get("cachedContentTokenCount", 0)),
        "latency_ms": latency_ms,
        "finish_reason": candidate.get("finishReason", "UNKNOWN"),
        "model": payload.get("modelVersion", model),
    }
