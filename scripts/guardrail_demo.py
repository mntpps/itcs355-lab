"""Lab 5 Task 8 — make the most expensive failure happen, and watch the control fire.

    python scripts/guardrail_demo.py            # offline, from recorded responses (free)
    python scripts/guardrail_demo.py --live     # re-run the failure against Gemini now

The failure, observed live on 2026-10-25 with gemini-3.5-flash-lite: someone shortens
the system prompt to cut its cost (v2-short: input tokens -70%, cost per 1,000 requests
-39%), and the model starts deciding on input it should refuse to decide on.

  wo-013  temp_c=176.0, a Fahrenheit feed  -> model says no_action. A work order
          auto-closed on garbage readings is a missed failure, the costliest outcome
          this system has (Lab 4 post-mortem: ~THB 150,000 per unplanned breakdown).
  wo-011  no failure_probability at all    -> model invents schedule_urgent
  wo-012  failure_probability=1.7          -> model says it "exceeds 0.70", urgent

The control is code, not prompt: src/workorder.guard withholds any decision on an
invalid probability or an implausible reading (the same PLAUSIBLE_RANGES as the API
schema and the data contract tests). Shown, in order:
  1. the raw model replies (the failure)
  2. the guard's corrections and flags (the control firing)
  3. the golden-set gate failing on those cases, so the prompt change cannot ship

Exit 0 when every failure was corrected AND the gate caught it; 1 otherwise.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import workorder  # noqa: E402

GOLDEN = ROOT / "evals" / "golden" / "workorder.jsonl"
FIXTURE = ROOT / "evals" / "fixtures" / "workorder-v2-short.jsonl"
BASELINE_FIXTURE = ROOT / "evals" / "fixtures" / "workorder-v1.jsonl"
CASES = ["wo-013", "wo-011", "wo-012"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="call the model now (needs GEMINI_API_KEY)")
    args = ap.parse_args()

    golden = {c["id"]: c for c in map(json.loads, GOLDEN.read_text().splitlines()) if c}
    if args.live:
        from cloudlayer.factory import get_adapter
        from src import config

        adapter = get_adapter(config.load(strict=False))
        params = {"system": workorder.system_prompt("v2-short"), "max_output_tokens": 1024}
        raws = {cid: adapter.generate(golden[cid]["prompt"], params)["response"] for cid in CASES}
        source = "live call, gemini"
    else:
        recorded = {r["id"]: r for r in map(json.loads, FIXTURE.read_text().splitlines()) if r}
        raws = {cid: recorded[cid]["raw_response"] for cid in CASES}
        source = f"recorded {recorded[CASES[0]].get('model', '?')}, {FIXTURE.relative_to(ROOT)}"

    print(f"prompt v2-short ({source})\n")
    corrected = 0
    for cid in CASES:
        line = golden[cid]["prompt"]
        raw = json.loads(raws[cid])
        final, flags = workorder.guard(raws[cid], line)
        decided = json.loads(final)["decision"]
        print(f"{cid}  {golden[cid]['note']}")
        print(f"   input       {line}")
        print(f"   1. model    {raw['decision']:<16} \"{raw.get('rationale', '')[:70]}\"")
        print(f"   2. guard    {decided:<16} flags={flags}")
        corrected += bool(flags) and decided == "cannot_decide"

    print("\n3. golden-set gate, v2-short vs the v1 baseline:")
    base = ROOT / "reports" / "llm_eval-workorder-v1.json"
    run = [sys.executable, str(ROOT / "scripts" / "llm_eval.py"), "--golden", str(GOLDEN),
           "--provider", "gemini", "--model", "gemini-3.5-flash-lite"]
    subprocess.run([*run, "--responses", str(BASELINE_FIXTURE), "--out", str(base)],
                   check=True, capture_output=True)
    gate = subprocess.run([*run, "--responses", str(FIXTURE), "--out",
                           str(ROOT / "reports" / "llm_eval-workorder-v2-short.json"),
                           "--baseline", str(base)], capture_output=True, text=True)
    print("   " + "\n   ".join(gate.stdout.strip().splitlines()[-7:]))
    caught = gate.returncode != 0 and all(cid in gate.stdout for cid in CASES)

    print(f"\ncontrol fired on {corrected}/{len(CASES)} cases; "
          f"gate {'blocked' if caught else 'did NOT block'} the prompt change")
    return 0 if corrected == len(CASES) and caught else 1


if __name__ == "__main__":
    raise SystemExit(main())
