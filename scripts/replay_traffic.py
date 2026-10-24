"""Lab 4 — replay sensor readings against the service, as the fleet would send them.

    # normal traffic, to fill the window and give the dashboard a baseline
    python scripts/replay_traffic.py --url http://localhost:8080 --source data/raw/sensors.csv

    # the injected-drift exercise: replay the shifted file and stamp the injection time
    python scripts/inject_drift.py --feature temp_c --mode shift --magnitude 6
    python scripts/replay_traffic.py --url https://<staging> --source data/current.csv --mark-injection

Rows are shuffled, because a fleet reports interleaved; replaying the file in order would
send one machine's 25 readings back to back, and a window of a few machines is far
noisier than the threshold was calibrated for (see reports/lab4-threshold.json,
"clustered"). --bad-rate sends a small share of invalid payloads so the 4xx panel has
something to show.

--mark-injection appends the start time to reports/lab4-timeline.md. Detection latency
is the gap between that line and the alert.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.data import FEATURES

BANGKOK = dt.timezone(dt.timedelta(hours=7))


def post(url: str, body: dict) -> int:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code
    except (urllib.error.URLError, TimeoutError):
        return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True, help="service base URL")
    ap.add_argument("--source", type=Path, default=Path("data/raw/sensors.csv"))
    ap.add_argument("--rate", type=float, default=8.0, help="requests per second")
    ap.add_argument("--count", type=int, default=1500, help="readings to send (0 = whole file)")
    ap.add_argument("--bad-rate", type=float, default=0.01, help="share of invalid payloads")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--mark-injection", action="store_true")
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    rows = pd.read_csv(args.source)[FEATURES].sample(frac=1.0, random_state=args.seed)
    if args.count:
        rows = rows.head(args.count)
    records = rows.to_dict("records")
    rng = random.Random(args.seed)
    target = args.url.rstrip("/") + "/predict"

    started = dt.datetime.now(BANGKOK)
    if args.mark_injection:
        timeline = Path("reports/lab4-timeline.md")
        timeline.parent.mkdir(parents=True, exist_ok=True)
        with timeline.open("a") as fh:
            fh.write(f"- {started.isoformat(timespec='seconds')}  INJECTION START  "
                     f"source={args.source} {args.label}\n")
    print(f"{started:%H:%M:%S} ICT  sending {len(records)} readings from {args.source} "
          f"at {args.rate}/s to {target}")

    statuses: dict[int, int] = {}
    interval = 1.0 / args.rate
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = []
        next_at = time.perf_counter()
        for i, rec in enumerate(records):
            body = dict(rec)
            if rng.random() < args.bad_rate:
                body.pop(rng.choice(FEATURES))          # a malformed caller -> 422
            futures.append(pool.submit(post, target, body))
            next_at += interval
            time.sleep(max(0.0, next_at - time.perf_counter()))
            if (i + 1) % 250 == 0:
                print(f"  {dt.datetime.now(BANGKOK):%H:%M:%S}  sent {i + 1}")
        for f in futures:
            code = f.result()
            statuses[code] = statuses.get(code, 0) + 1
    print(f"done {dt.datetime.now(BANGKOK):%H:%M:%S} ICT  status counts: {statuses}")
    return 0 if statuses.get(200, 0) > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
