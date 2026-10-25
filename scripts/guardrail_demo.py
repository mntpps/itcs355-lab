"""Lab 5 Task 8 — make the most expensive failure happen, and watch the control fire.

    python scripts/guardrail_demo.py            # offline, from recorded responses (free)
    python scripts/guardrail_demo.py --live     # re-run the attack against Gemini now

The failure: a free-text operator note talks the model out of an urgent work order. The
machine in case wo-009 has a 0.93 failure probability; a work order that says no_action
is a missed failure, which is the costliest outcome this system has (Lab 4 post-mortem:
about THB 150,000 per unplanned breakdown).

Three things are shown, in order:
  1. the attack succeeding against the naive prompt (raw model output: no_action)
  2. the guard reversing it in code and flagging probability_floor
  3. the golden set catching it as a regression of wo-009, so it cannot return silently

Exit 0 when the control fired and the gate caught the regression; 1 otherwise.
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
NAIVE = ROOT / "evals" / "fixtures" / "workorder-v0-naive.jsonl"
BASELINE_FIXTURE = ROOT / "evals" / "fixtures" / "workorder-v1.jsonl"
CASE = "wo-009"


def load(path: Path) -> dict[str, dict]:
    return {r["id"]: r for r in map(json.loads, path.read_text().splitlines()) if r}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="call the model now (needs GEMINI_API_KEY)")
    args = ap.parse_args()

    case = next(c for c in map(json.loads, GOLDEN.read_text().splitlines()) if c["id"] == CASE)
    print(f"case {CASE}: {case['note']}")
    print(f"input  {case['prompt']}\n")

    if args.live:
        from cloudlayer.factory import get_adapter
        from src import config

        adapter = get_adapter(config.load(strict=False))
        params = {"system": workorder.system_prompt("v0-naive"), "max_output_tokens": 1024}
        rec = workorder.run(case["prompt"], adapter.generate, params)
    else:
        rec = load(NAIVE)[CASE]
        print(f"(recorded {rec.get('model', '?')} response from {NAIVE.relative_to(ROOT)})")

    raw = json.loads(rec["raw_response"]) if rec["raw_response"].strip().startswith("{") else {}
    final = json.loads(rec["response"])
    print(f"1. raw model decision     : {raw.get('decision', rec['raw_response'][:60])!r}")
    print(f"2. after the guard        : {final['decision']!r}   guard={final.get('guard', [])}")
    fired = "probability_floor" in final.get("guard", [])
    attack_worked = raw.get("decision") == "no_action"
    if not attack_worked:
        print("   the model resisted this time; the guard had nothing to correct")

    print("\n3. golden-set gate, naive prompt vs the v1 baseline:")
    base = ROOT / "reports" / "llm_eval-workorder-v1.json"
    run = [sys.executable, str(ROOT / "scripts" / "llm_eval.py"), "--golden", str(GOLDEN),
           "--provider", "gemini", "--model", "gemini-3.5-flash-lite"]
    subprocess.run([*run, "--responses", str(BASELINE_FIXTURE), "--out", str(base)],
                   check=True, capture_output=True)
    gate = subprocess.run([*run, "--responses", str(NAIVE), "--out",
                           str(ROOT / "reports" / "llm_eval-workorder-v0-naive.json"),
                           "--baseline", str(base)], capture_output=True, text=True)
    caught = gate.returncode != 0 and CASE in gate.stdout
    print("   " + "\n   ".join(gate.stdout.strip().splitlines()[-6:]))

    ok = (fired or not attack_worked) and (caught or not attack_worked)
    print(f"\n{'CONTROL FIRED' if fired else 'no correction needed'}; "
          f"gate {'caught' if caught else 'did not flag'} {CASE}")
    return 0 if ok and attack_worked else 1


if __name__ == "__main__":
    raise SystemExit(main())
