"""Lab 5 Task 5 item 6 — the applied optimisation: score the fleet in batches.

    python scripts/score_fleet.py --env cloud.aws.env --data data/raw/sensors.csv
    python scripts/score_fleet.py --data data/raw/sensors.csv            # primary cloud (Azure)

The daily job scores one reading per machine, the latest, for all 240 machines. It used
to send them one call at a time to /predict: 240 invocations a day. It now sends them to
/predict/batch, 100 readings per call: 3 invocations. Measured on the Lambda
(reports/lab5-lambda-rightsize.json, 27 Oct): one call costs about the same whether it
scores 1 row or 100 (6.6 vs 12.0 billed ms), so cost per prediction falls about 78-fold
and the whole fleet's scores come back in 3 round trips instead of 240.

Writes reports/fleet-scores.csv (machine_id, probability, model_version).
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cloudlayer.factory import get_adapter  # noqa: E402
from src import config, data  # noqa: E402

BATCH = 100   # service/schemas.py BatchRequest max_length


def latest_per_machine(df):
    """The fleet as of now: each machine's most recent reading."""
    return df.sort_values(data.ID).groupby(data.GROUP).tail(1).sort_values(data.GROUP)


def score(adapter, endpoint: str, rows: list[dict], batch: int = BATCH) -> tuple[list[float], str, int]:
    probs, version, calls = [], "?", 0
    for i in range(0, len(rows), batch):
        out = adapter.invoke(endpoint, {"rows": rows[i:i + batch]}, path="/predict/batch")
        probs += out["probabilities"]
        version = out.get("model_version", version)
        calls += 1
    return probs, version, calls


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", type=Path, help="a second provider's env file; default: cloud.env")
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "raw" / "sensors.csv")
    ap.add_argument("--endpoint", help="default: SERVING_ENDPOINT (AWS) or STAGING_URL (Azure)")
    ap.add_argument("--out", type=Path, default=ROOT / "reports" / "fleet-scores.csv")
    args = ap.parse_args()

    if args.env:
        cfg = config.load_file(args.env)
        endpoint = args.endpoint or config.read_env_file(args.env)["SERVING_ENDPOINT"]
    else:
        import os

        cfg = config.load(strict=False)
        endpoint = args.endpoint or os.environ["STAGING_URL"]

    fleet = latest_per_machine(data.load_raw(args.data))
    rows = fleet[data.FEATURES].to_dict("records")
    started = time.perf_counter()
    probs, version, calls = score(get_adapter(cfg), endpoint, rows)
    elapsed = time.perf_counter() - started

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["machine_id", "probability", "model_version"])
        for mid, p in zip(fleet["machine_id"], probs):
            w.writerow([mid, round(p, 6), version])
    flagged = sum(p >= 0.40 for p in probs)
    print(f"scored {len(probs)} machines in {calls} calls ({elapsed:.1f} s) with model {version}; "
          f"{flagged} at >= 0.40 (engineer review or urgent). Wrote {args.out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
