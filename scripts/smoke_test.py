"""Smoke-test a deployed endpoint with three known payloads.

    python scripts/smoke_test.py --url https://<fqdn>
    python scripts/smoke_test.py --app itcs355-predict      # resolves the FQDN with az

A deploy that is not smoke-tested is a deploy you have not verified. The three payloads
are chosen so each one can fail for a different reason:

  1. a cool, freshly serviced machine  -> 200, probability < 0.5, model version present
  2. a hot, vibrating, overdue machine -> 200, probability above payload 1
  3. a reading in Fahrenheit (176 F)   -> 422, the contract rejects it at the door

Exit 0 only when all three behave. Cold start on a scale-to-zero app can take ~30 s,
so readiness is polled before any payload is sent.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

HEALTHY = {"temp_c": 66.0, "vibration_mm_s": 1.4, "pressure_kpa": 330.0,
           "hours_since_service": 50.0, "load_pct": 30.0, "ambient_humidity": 50.0}
AT_RISK = {"temp_c": 96.0, "vibration_mm_s": 9.5, "pressure_kpa": 290.0,
           "hours_since_service": 9000.0, "load_pct": 92.0, "ambient_humidity": 70.0}
FAHRENHEIT = {**HEALTHY, "temp_c": 176.0}


def call(url: str, payload: dict | None = None, timeout: float = 60) -> tuple[int, dict, dict]:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method="POST" if data else "GET",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read() or b"{}"), dict(r.headers)
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}"), dict(e.headers)


def resolve(app: str) -> str:
    from src import config

    cfg = config.load(strict=False)
    fqdn = subprocess.check_output(
        [shutil.which("az") or "az", "containerapp", "show", "--name", app, "--resource-group", cfg.project_id,
         "--query", "properties.configuration.ingress.fqdn", "-o", "tsv"], text=True).strip()
    return f"https://{fqdn}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url")
    ap.add_argument("--app")
    ap.add_argument("--expect-build", help="fail unless /ready reports this build SHA")
    ap.add_argument("--expect-version", help="fail unless /ready reports this model version")
    ap.add_argument("--ready-timeout", type=float, default=180)
    args = ap.parse_args()
    base = (args.url or resolve(args.app)).rstrip("/")
    return run(base, args.expect_build, args.expect_version, args.ready_timeout)


def run(base: str, expect_build: str | None = None, expect_version: str | None = None,
        ready_timeout: float = 180) -> int:
    """Wait for /ready to report the expected build and model version, then send the
    three payloads. Returns 0 when all three behave. Used by CD and by the pipeline's
    deploy_staging step (scripts/deploy.py)."""
    deadline = time.time() + ready_timeout
    while True:
        try:
            status, ready, _ = call(base + "/ready", timeout=60)
        except (urllib.error.URLError, TimeoutError) as exc:
            status, ready = 0, {"error": str(exc)}
        if (status == 200 and (not expect_build or ready.get("build_sha") == expect_build)
                and (not expect_version or str(ready.get("model_version")) == str(expect_version))):
            break
        if time.time() > deadline:
            print(f"FAIL  {base}/ready never became ready for this build/version: {status} {ready}")
            return 1
        time.sleep(5)
    print(f"ready  {ready}")

    failures = []
    s1, b1, h1 = call(base + "/predict", HEALTHY)
    s2, b2, _ = call(base + "/predict", AT_RISK)
    s3, b3, _ = call(base + "/predict", FAHRENHEIT)

    if s1 != 200 or not b1.get("model_version"):
        failures.append(f"healthy: expected 200 with model_version, got {s1} {b1}")
    elif b1["probability"] >= 0.5:
        failures.append(f"healthy machine flagged: probability {b1['probability']:.3f}")
    if s2 != 200:
        failures.append(f"at-risk: expected 200, got {s2} {b2}")
    elif s1 == 200 and b2["probability"] <= b1["probability"]:
        failures.append(f"at-risk scored no higher than healthy: {b2['probability']:.3f}")
    if s3 != 422:
        failures.append(f"Fahrenheit reading: expected 422, got {s3} {b3}")

    print(f"healthy     {s1}  {b1}")
    print(f"at-risk     {s2}  {b2}")
    print(f"fahrenheit  {s3}  {str(b3)[:120]}")
    for f in failures:
        print("FAIL  " + f)
    if not failures:
        print(f"PASS  3/3 payloads behaved — model_version={b1['model_version']} "
              f"build={h1.get('x-build-sha', h1.get('X-Build-Sha', '?'))}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
