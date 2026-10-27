"""Lab 5 Task 5 item 6 — cost per prediction on the AWS serving function, measured.

    python scripts/lambda_rightsize.py --env cloud.aws.env --memory 1024 --label before
    python scripts/lambda_rightsize.py --env cloud.aws.env --memory 512  --label smaller-memory
    python scripts/lambda_rightsize.py --env cloud.aws.env --memory 1024 --batch 100 --label batched

For one memory size: sets the function's memory (if it differs), takes one cold start,
then sends --n warm predictions. Every invocation returns its own Lambda REPORT line
(LogType=Tail), so the numbers are AWS's own: billed duration, memory size, max memory
used. Nothing here is estimated except the price per GB-second, which is a parameter.

Results append to reports/lab5-lambda-rightsize.json. Lambda bills memory x billed time,
and CPU scales with memory, so halving memory can halve the cost or can stretch the
duration enough to eat the saving. That is why this measures instead of assuming.
"""
from __future__ import annotations

import argparse
import base64
import json
import re
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from smoke_test import AT_RISK, HEALTHY  # noqa: E402

from cloudlayer.factory import get_adapter  # noqa: E402
from src import config  # noqa: E402

REPORT = re.compile(r"REPORT RequestId: \S+\s+Duration: ([\d.]+) ms\s+Billed Duration: (\d+) ms\s+"
                    r"Memory Size: (\d+) MB\s+Max Memory Used: (\d+) MB(?:\s+Init Duration: ([\d.]+) ms)?")
THB_PER_USD = 33.76          # 23 Oct 2026, as in reports/lab5-cost.md


def event(payload: dict, path: str) -> bytes:
    body = json.dumps(payload)
    return json.dumps({
        "version": "2.0", "routeKey": "$default", "rawPath": path, "rawQueryString": "",
        "headers": {"content-type": "application/json", "host": "lambda"},
        "requestContext": {"http": {"method": "POST", "path": path, "protocol": "HTTP/1.1",
                                    "sourceIp": "127.0.0.1", "userAgent": "itcs355"},
                           "domainName": "lambda", "stage": "$default"},
        "body": body, "isBase64Encoded": False}).encode()


def call(lam, fn: str, payload: dict, batch: int = 1) -> dict:
    """One invocation: a single reading on /predict, or `batch` readings on /predict/batch."""
    path = "/predict"
    if batch > 1:
        path = "/predict/batch"
        payload = {"rows": [HEALTHY if i % 2 else AT_RISK for i in range(batch)]}
    started = time.perf_counter()
    r = lam.invoke(FunctionName=fn, Payload=event(payload, path), LogType="Tail")
    client_ms = (time.perf_counter() - started) * 1000
    out = json.loads(r["Payload"].read())
    if r.get("FunctionError") or int(out.get("statusCode", 500)) >= 400:
        raise RuntimeError(f"invoke failed: {out}")
    m = REPORT.search(base64.b64decode(r["LogResult"]).decode(errors="replace"))
    if not m:
        raise RuntimeError("no REPORT line in the tail log")
    return {"client_ms": client_ms, "duration_ms": float(m[1]), "billed_ms": int(m[2]),
            "memory_mb": int(m[3]), "max_used_mb": int(m[4]),
            "init_ms": float(m[5]) if m[5] else None,
            "predictions": len(json.loads(out["body"]).get("probabilities", [0]))}


def pct(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, int(round(q / 100 * (len(values) - 1))))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", type=Path, required=True)
    ap.add_argument("--memory", type=int, required=True, help="MB")
    ap.add_argument("--label", required=True)
    ap.add_argument("--n", type=int, default=50, help="warm invocations")
    ap.add_argument("--batch", type=int, default=1,
                    help="readings per invocation (1 = /predict; up to 100 = /predict/batch)")
    ap.add_argument("--usd-per-gb-s", type=float, default=0.0000166667,
                    help="x86 on-demand; AWS's published example rate (US East). Sydney's list "
                         "price was not confirmed; the before/after ratio does not depend on it")
    ap.add_argument("--usd-per-million-requests", type=float, default=0.20)
    args = ap.parse_args()

    cfg = config.load_file(args.env)
    fn = config.read_env_file(args.env)["SERVING_ENDPOINT"]
    lam = get_adapter(cfg)._client("lambda")

    current = lam.get_function_configuration(FunctionName=fn)["MemorySize"]
    if current != args.memory:
        print(f"{fn}: memory {current} -> {args.memory} MB")
        lam.update_function_configuration(FunctionName=fn, MemorySize=args.memory)
        lam.get_waiter("function_updated_v2").wait(FunctionName=fn)

    cold = call(lam, fn, HEALTHY, args.batch)
    print(f"first call after the change: {cold['client_ms']:.0f} ms at the client, "
          f"init {cold['init_ms']} ms, billed {cold['billed_ms']} ms")
    warm = [call(lam, fn, HEALTHY if i % 2 else AT_RISK, args.batch) for i in range(args.n)]
    assert all(w["predictions"] == args.batch for w in warm), "a batch came back short"

    billed = [w["billed_ms"] for w in warm]
    gb_s = statistics.mean(billed) / 1000 * args.memory / 1024      # per invocation
    calls_per_1k = 1000 / args.batch
    usd_1k = calls_per_1k * (gb_s * args.usd_per_gb_s + args.usd_per_million_requests / 1e6)
    row = {
        "label": args.label, "memory_mb": args.memory, "batch": args.batch,
        "n_warm": len(warm),
        "measured_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "cold_start_client_ms": round(cold["client_ms"]), "cold_init_ms": cold["init_ms"],
        "warm_duration_ms_p50": round(pct([w["duration_ms"] for w in warm], 50), 2),
        "warm_duration_ms_p95": round(pct([w["duration_ms"] for w in warm], 95), 2),
        "warm_client_ms_p95": round(pct([w["client_ms"] for w in warm], 95)),
        "billed_ms_mean": round(statistics.mean(billed), 2),
        "max_memory_used_mb": max(w["max_used_mb"] for w in warm),
        "gb_s_per_invocation": round(gb_s, 6),
        "gb_s_per_prediction": round(gb_s / args.batch, 7),
        "usd_per_1k_predictions": round(usd_1k, 7),
        "thb_per_1k_predictions": round(usd_1k * THB_PER_USD, 6),
        "usd_per_gb_s": args.usd_per_gb_s,
    }
    out = ROOT / "reports" / "lab5-lambda-rightsize.json"
    rows = json.loads(out.read_text()) if out.exists() else []
    rows = [r for r in rows if r["label"] != args.label] + [row]
    out.write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps(row, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
