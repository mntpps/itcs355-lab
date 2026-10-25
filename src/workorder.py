"""Lab 5 Part B — the LLM step: turn a prediction into a work-order decision.

    record line  ->  [system prompt + model]  ->  raw JSON  ->  guard()  ->  work order

Provider-neutral (Layer 1): the model call goes through the adapter's `generate`, and
everything here is plain Python. The prompt shapes what the model does; the guard is
code, and enforces what must hold whatever the model does.

The guard is deliberately narrow. It owns the decisions where a wrong answer is
expensive and the rule is mechanical:

  unusable_output    reply is not JSON, or the decision is not one of DECISIONS
                     -> review_by_engineer (a human looks; nothing is auto-closed)
  probability_floor  failure_probability >= 0.70 but the model said no_action
                     -> schedule_urgent. This is the injection that matters: a note
                     talking the model out of an urgent work order is a missed failure.
  invalid_input      failure_probability missing or outside 0..1, yet the model decided
                     -> cannot_decide
  implausible_reading  a reading outside src.data.PLAUSIBLE_RANGES (the same bounds as
                     the API schema and the data contract tests), yet the model decided
                     -> cannot_decide. 176 "temp_c" is a Fahrenheit feed, and a work order
                     must never auto-close on it: the Lab 4 incident, at the LLM step.
  ungrounded_part    a part number in `parts` that is not in the input -> removed

Each correction is recorded in the output's "guard" list, so a guarded answer is never
mistaken for a model that behaved, and the golden set can assert the model did not need
the guard.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from src.data import PLAUSIBLE_RANGES

PROMPTS = Path(__file__).resolve().parent / "prompts"
DECISIONS = {"schedule_urgent", "review_by_engineer", "no_action", "cannot_decide", "refuse"}
URGENT_AT = 0.70
_FIELD = re.compile(r'(\w+)=("([^"]*)"|\S+)')
_PART = re.compile(r"\b[A-Z]{2,4}-\d{3,6}\b")


def system_prompt(version: str = "v1") -> str:
    matches = sorted(PROMPTS.glob(f"workorder-{version}*.txt"))
    if not matches:
        raise FileNotFoundError(f"no prompt workorder-{version}*.txt in {PROMPTS}")
    return matches[0].read_text()


def parse_fields(line: str) -> dict[str, str]:
    """key=value and key="quoted value" pairs from one input line."""
    return {m.group(1): (m.group(3) if m.group(3) is not None else m.group(2))
            for m in _FIELD.finditer(line)}


def probability(fields: dict[str, str]) -> float | None:
    try:
        p = float(fields["failure_probability"])
    except (KeyError, ValueError):
        return None
    return p if 0.0 <= p <= 1.0 else None


def implausible(fields: dict[str, str]) -> list[str]:
    """Readings present in the input that fall outside their plausible range."""
    bad = []
    for name, (lo, hi) in PLAUSIBLE_RANGES.items():
        try:
            value = float(fields[name])
        except (KeyError, ValueError):
            continue
        if not lo <= value <= hi:
            bad.append(name)
    return bad


def guard(raw: str, line: str) -> tuple[str, list[str]]:
    """Return (final JSON text, flags). Never raises on model output."""
    fields = parse_fields(line)
    p = probability(fields)
    flags: list[str] = []
    try:
        out = json.loads(raw)
        if not isinstance(out, dict) or out.get("decision") not in DECISIONS:
            raise ValueError
    except (ValueError, TypeError):
        out = {"decision": "review_by_engineer",
               "rationale": "Model output was unusable; routed to an engineer.", "parts": []}
        flags.append("unusable_output")

    if p is not None and p >= URGENT_AT and out["decision"] == "no_action":
        out["decision"] = "schedule_urgent"
        flags.append("probability_floor")
    decided = {"schedule_urgent", "review_by_engineer", "no_action"}
    if p is None and out["decision"] in decided:
        out["decision"] = "cannot_decide"
        flags.append("invalid_input")
    if out["decision"] in decided and implausible(fields):
        out["decision"] = "cannot_decide"
        flags.append("implausible_reading")

    parts = out.get("parts") if isinstance(out.get("parts"), list) else []
    grounded = [x for x in parts if not _PART.fullmatch(str(x)) or str(x) in line]
    if len(grounded) != len(parts):
        flags.append("ungrounded_part")
    out["parts"] = grounded

    if flags:
        out["guard"] = flags
    return json.dumps(out, ensure_ascii=False), flags


def run(line: str, generate, params: dict) -> dict:
    """One step through any `generate(prompt, params)` — the adapter's, or a stub."""
    result = generate(line, params)
    final, flags = guard(result["response"], line)
    return {**result, "raw_response": result["response"], "response": final, "guard": flags}
