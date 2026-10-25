"""Evaluation harness for an LLM step. Session 5, Lab 5 Part B.

    python scripts/llm_eval.py --out reports/llm_eval.json
    python scripts/llm_eval.py --responses evals/fixtures/triage-regressed.jsonl \
        --baseline reports/llm_eval.json

An LLM has no accuracy number you can watch. Change the prompt, change the model version,
change the temperature, and nothing fails — the answers just quietly get worse. That is
the whole operational problem, and a golden set with a gate is the cheapest honest answer
to it. This is Lab 4's evaluation gate applied to text instead of a metric.

Responses come from a fixture file by default, so the harness runs offline, free, and
deterministically — you can develop the gate without spending a satang, and CI can run it
on every commit. `--live` routes through YOUR adapter's `generate`, which is the Lab 5
implementation task.

The gate is deliberately stricter than a pass-rate threshold: a case that passed in the
baseline and fails now is a regression, even if the overall rate went up. Averages hide
exactly the failure you care about.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import llmcost  # noqa: E402

DEFAULT_GOLDEN = ROOT / "evals" / "golden" / "triage.jsonl"
DEFAULT_RESPONSES = ROOT / "evals" / "fixtures" / "triage-baseline.jsonl"


def shown(path: Path) -> str:
    """Repo-relative when it can be, verbatim otherwise. --out is free to point anywhere."""
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"missing file: {path}")
    rows = []
    for lineno, line in enumerate(path.read_text().splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{path}:{lineno} is not valid JSON — {exc}") from exc
    return rows


def check_one(check: dict, response: str) -> tuple[bool, str]:
    """Return (passed, explanation). Explanations end up in the report, so make them
    sayable out loud — you will be reading them back to a room in Session 5."""
    kind = check.get("type")
    value = check.get("value")

    if kind == "contains":
        ok = str(value).lower() in response.lower()
        return ok, f"expected to contain {value!r}"
    if kind == "not_contains":
        ok = str(value).lower() not in response.lower()
        return ok, f"expected NOT to contain {value!r}"
    if kind == "regex":
        ok = re.search(str(value), response) is not None
        return ok, f"expected to match /{value}/"
    if kind == "not_regex":
        match = re.search(str(value), response)
        detail = f" but found {match.group(0)!r}" if match else ""
        return match is None, f"expected NOT to match /{value}/{detail}"
    if kind == "max_words":
        words = len(response.split())
        return words <= int(value), f"expected at most {value} words, got {words}"
    if kind == "json_field":
        field = check.get("field")
        try:
            parsed = json.loads(response)
        except json.JSONDecodeError:
            return False, f"expected valid JSON with {field}={value!r}, got unparseable text"
        actual = parsed.get(field)
        return actual == value, f"expected {field}={value!r}, got {actual!r}"

    raise SystemExit(f"unknown check type {kind!r} in the golden set")


def score(golden: list[dict], responses: list[dict]) -> list[dict]:
    by_id = {r["id"]: r for r in responses}
    results = []
    for case in golden:
        cid = case["id"]
        got = by_id.get(cid)
        if got is None:
            results.append({"id": cid, "passed": False, "failures": ["no response recorded"]})
            continue
        text = got.get("response", "")
        failures = [why for check in case.get("checks", [])
                    for ok, why in [check_one(check, text)] if not ok]
        results.append({
            "id": cid,
            "passed": not failures,
            "failures": failures,
            "input_tokens": got.get("input_tokens", 0),
            "output_tokens": got.get("output_tokens", 0),
            "cached_input_tokens": got.get("cached_input_tokens", 0),
            "thinking_tokens": got.get("thinking_tokens", 0),
            "finish_reason": got.get("finish_reason", ""),
            "guard": got.get("guard", []),
            "latency_ms": got.get("latency_ms", 0),
        })
    return results


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(int(round((pct / 100) * (len(ordered) - 1))), len(ordered) - 1)
    return ordered[idx]


def summarise(results: list[dict], provider: str, model: str) -> dict:
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    in_tok = sum(r.get("input_tokens", 0) for r in results)
    out_tok = sum(r.get("output_tokens", 0) for r in results)
    cached = sum(r.get("cached_input_tokens", 0) for r in results)
    avg = llmcost.Usage(
        input_tokens=round(in_tok / total) if total else 0,
        output_tokens=round(out_tok / total) if total else 0,
        cached_input_tokens=min(round(cached / total), round(in_tok / total)) if total else 0,
    )
    return {
        "provider": provider,
        "model": model,
        "cases": total,
        "passed": passed,
        "pass_rate": round(passed / total, 4) if total else 0.0,
        "p95_latency_ms": percentile([r.get("latency_ms", 0) for r in results], 95),
        "avg_input_tokens": avg.input_tokens,
        "avg_output_tokens": avg.output_tokens,
        "avg_cached_input_tokens": avg.cached_input_tokens,
        "max_output_tokens_seen": max((r.get("output_tokens", 0) for r in results), default=0),
        "cache_hit_rate": round(cached / in_tok, 4) if in_tok else 0.0,
        "guard_fired": sorted({f for r in results for f in r.get("guard", [])}),
        "thb_per_1k_requests": round(llmcost.cost_per_1k_requests(provider, model, avg), 4),
        "results": results,
    }


def gate(report: dict, baseline_path: Path) -> int:
    """Fail on any case that passed in the baseline and fails now."""
    baseline = json.loads(baseline_path.read_text())
    was = {r["id"]: r["passed"] for r in baseline.get("results", [])}
    now = {r["id"]: r["passed"] for r in report.get("results", [])}

    regressions = sorted(cid for cid, ok in now.items() if was.get(cid) and not ok)
    fixed = sorted(cid for cid, ok in now.items() if ok and was.get(cid) is False)

    print(f"\nbaseline pass rate  {baseline.get('pass_rate')}")
    print(f"current  pass rate  {report.get('pass_rate')}")
    if fixed:
        print(f"newly passing       {', '.join(fixed)}")
    if regressions:
        print(f"\nGATE FAILED — {len(regressions)} regression(s): {', '.join(regressions)}")
        for cid in regressions:
            for why in next(r for r in report["results"] if r["id"] == cid)["failures"]:
                print(f"    {cid}: {why}")
        return 1
    print("\nGATE PASSED — no case that passed before fails now")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN)
    ap.add_argument("--responses", type=Path, default=DEFAULT_RESPONSES,
                    help="recorded responses; omit and pass --live to call your adapter")
    ap.add_argument("--live", action="store_true",
                    help="generate responses through cloudlayer instead of a fixture")
    ap.add_argument("--provider", default="gcp",
                    help="price table to cost the run with (gemini for live Gemini runs)")
    ap.add_argument("--model", default="small",
                    help="model name in the price table; with --live also the model called")
    ap.add_argument("--prompt", default=None,
                    help="system prompt version under src/prompts (v1, v2-short, v0-naive). "
                         "Runs the work-order step: prompt + model + guard")
    ap.add_argument("--no-guard", action="store_true", help="score the raw model output")
    ap.add_argument("--max-output-tokens", type=int, default=1024)
    ap.add_argument("--thinking", default=None, help="thinkingLevel, e.g. minimal or low")
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--record", type=Path,
                    help="with --live: also write the responses as a fixture for offline runs")
    ap.add_argument("--out", type=Path, default=ROOT / "reports" / "llm_eval.json")
    ap.add_argument("--baseline", type=Path,
                    help="a previous report; fail if any case regresses against it")
    args = ap.parse_args()

    golden = read_jsonl(args.golden)
    if args.live:
        responses = generate_live(golden, args)
        if args.record:
            args.record.parent.mkdir(parents=True, exist_ok=True)
            args.record.write_text("".join(json.dumps(r) + "\n" for r in responses))
            print(f"recorded {shown(args.record)}")
    else:
        responses = read_jsonl(args.responses)
        if args.no_guard:
            responses = [{**r, "response": r.get("raw_response", r["response"])} for r in responses]

    report = summarise(score(golden, responses), args.provider, args.model)

    print(f"golden set  {args.golden.name}  ({report['cases']} cases)")
    for r in report["results"]:
        mark = "PASS" if r["passed"] else "FAIL"
        print(f"  [{mark}] {r['id']}")
        for why in r["failures"]:
            print(f"         {why}")
    print(f"\npass rate           {report['passed']}/{report['cases']}"
          f"  ({report['pass_rate'] * 100:.1f}%)")
    print(f"p95 latency         {report['p95_latency_ms']} ms")
    print(f"cost per 1k reqs    {report['thb_per_1k_requests']} THB"
          f"  ({report['provider']}/{report['model']})")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {shown(args.out)}")

    if args.baseline:
        return gate(report, args.baseline)
    return 0


def generate_live(golden: list[dict], args) -> list[dict]:
    """Route each prompt through the adapter's `generate` (Gemini on this project).

    Token counts come from the provider's usage fields via cloudlayer/gemini.py — never
    from counting words. With --prompt, each case runs through the full work-order step
    (system prompt + model + guard), and both the raw and the guarded reply are kept.
    """
    from cloudlayer.factory import get_adapter
    from src import config, workorder

    adapter = get_adapter(config.load(strict=False))
    params = {"max_output_tokens": args.max_output_tokens, "model": args.model,
              "thinking_level": args.thinking, "temperature": args.temperature}
    if args.prompt:
        params["system"] = workorder.system_prompt(args.prompt)
    rows = []
    for i, case in enumerate(golden, 1):
        if args.prompt and not args.no_guard:
            result = workorder.run(case["prompt"], adapter.generate, params)
        else:
            result = adapter.generate(case["prompt"], params)
            result["raw_response"] = result["response"]
        print(f"  {i:>2}/{len(golden)} {case['id']}  in={result['input_tokens']} "
              f"out={result['output_tokens']} {result.get('finish_reason', '')}"
              f"{'  guard=' + ','.join(result.get('guard', [])) if result.get('guard') else ''}")
        rows.append({"id": case["id"], **result})
    return rows

if __name__ == "__main__":
    sys.exit(main())
