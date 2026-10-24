"""Lab 4 — drift detection.

PSI and Kolmogorov–Smirnov, implemented directly (src/drift_stats.py) so you can see
what the numbers mean, and shared with the service so the dashboard and this detector
score a window identically.

    # a file of inputs, e.g. after `make inject-drift`
    python -m monitoring.drift --current data/current.csv

    # the live service's rolling window — what the scheduled job runs
    python -m monitoring.drift --current-url https://<staging-fqdn> --emit --notify

Exit codes: 0 no drift, 2 drift alert, 3 the inputs themselves are broken (schema,
nulls, out-of-range values), 1 the check itself failed (e.g. metric emission).

Threshold: PSI >= 0.10 on any physical sensor, or >= 0.29 on load_pct, over a
500-reading window. Neither number is the credit-scoring default; both come from this
problem's own no-drift noise floor (scripts/drift_threshold_study.py,
reports/lab4-threshold.json), and the README argues them.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config, data, seeds  # noqa: E402
from src.drift_stats import (  # noqa: E402,F401  (psi/ks re-exported for callers and tests)
    FLEET_MIX_THRESHOLDS,
    SENSOR_PSI_THRESHOLD,
    WINDOW_SIZE,
    ks_statistic,
    psi,
    psi_from_profile,
    reference_profile,
    threshold_for,
)

BANGKOK = dt.timezone(dt.timedelta(hours=7))
# Above this share of nulls or out-of-range values, the window is a pipeline problem,
# not a distribution problem — and it must not be read as drift. See the decision tree
# in the Lab 4 handout: schema or null-rate change -> do NOT retrain.
MAX_BAD_RATE = 0.01


@dataclass
class FeatureDrift:
    feature: str
    psi: float
    ks_statistic: float
    ref_mean: float
    cur_mean: float
    threshold: float
    verdict: str


def verdict_for(score: float, threshold: float) -> str:
    if score >= threshold:
        return "DRIFT"
    if score >= threshold / 2:
        return "watch"
    return "stable"


def compare(reference: pd.DataFrame, current: pd.DataFrame, features: list[str],
            sensor_threshold: float = SENSOR_PSI_THRESHOLD) -> list[FeatureDrift]:
    results = []
    for feature in features:
        ref = reference[feature].to_numpy(dtype=float)
        cur = current[feature].dropna().to_numpy(dtype=float)
        # --threshold tunes the sensors; fleet-mix features keep their own limit.
        limit = FLEET_MIX_THRESHOLDS.get(feature, sensor_threshold)
        score = psi_from_profile(reference_profile(ref), cur)
        results.append(FeatureDrift(
            feature=feature,
            psi=round(score, 5),
            ks_statistic=round(ks_statistic(ref, cur), 5),
            ref_mean=round(float(ref.mean()), 4),
            cur_mean=round(float(cur.mean()), 4) if cur.size else float("nan"),
            threshold=limit,
            verdict=verdict_for(score, limit),
        ))
    return sorted(results, key=lambda r: r.psi / r.threshold, reverse=True)


def integrity(current: pd.DataFrame) -> dict:
    """Is this window even valid input? Checked BEFORE drift, because a broken producer
    and a shifted world look identical on a PSI chart and demand opposite responses."""
    missing = [f for f in data.FEATURES if f not in current.columns]
    report = {"missing_columns": missing, "null_rate": {}, "out_of_range_rate": {}}
    for f in data.FEATURES:
        if f in missing:
            continue
        col = pd.to_numeric(current[f], errors="coerce")
        report["null_rate"][f] = round(float(col.isna().mean()), 4)
        lo, hi = data.PLAUSIBLE_RANGES[f]
        report["out_of_range_rate"][f] = round(float(((col < lo) | (col > hi)).mean()), 4)
    report["broken"] = bool(missing) or any(
        v > MAX_BAD_RATE for v in [*report["null_rate"].values(),
                                   *report["out_of_range_rate"].values()])
    return report


def fetch_window(base_url: str) -> tuple[pd.DataFrame, dict]:
    req = urllib.request.Request(base_url.rstrip("/") + "/monitoring/window")
    token = os.environ.get("MONITORING_TOKEN")
    if token:
        req.add_header("x-monitoring-token", token)
    # A scale-to-zero app takes ~30 s to wake; give it room before calling it down.
    with urllib.request.urlopen(req, timeout=120) as r:
        payload = json.loads(r.read())
    meta = {k: v for k, v in payload.items() if k != "rows"}
    return pd.DataFrame(payload["rows"]), meta


def load_reference(path: Path, split: str) -> pd.DataFrame:
    df = data.load_raw(path)
    if split == "train":
        df, _, _ = data.split(df, seed=seeds.DEFAULT_SEED)
    return df


def notify(text: str) -> bool:
    """Post to ALERT_WEBHOOK_URL. Discord and Slack take the same idea, different keys."""
    url = os.environ.get("ALERT_WEBHOOK_URL")
    if not url:
        print("notify: ALERT_WEBHOOK_URL not set — alert printed only", file=sys.stderr)
        return False
    if "hooks.slack.com" in url:
        # Slack's mrkdwn: *bold*, not Discord's **bold**.
        body = {"text": text.replace("**", "*")}
    else:
        body = {"content": text[:1900]}
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json",
                                          "User-Agent": "itcs355-drift/1.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.status < 300


def alert_text(results: list[FeatureDrift], meta: dict, detected_at: dt.datetime,
               integrity_report: dict) -> str:
    breached = [r for r in results if r.verdict == "DRIFT"]
    lines = [
        f"**ITCS355 drift alert** — {len(breached)} feature(s) over threshold",
        f"detected {detected_at.astimezone(BANGKOK):%Y-%m-%d %H:%M:%S} ICT "
        f"({detected_at:%H:%M:%S} UTC)",
    ]
    if meta.get("oldest"):
        oldest = dt.datetime.fromtimestamp(meta["oldest"], BANGKOK)
        newest = dt.datetime.fromtimestamp(meta["newest"], BANGKOK)
        lines.append(f"window: {meta.get('n')} readings, {oldest:%H:%M:%S}–{newest:%H:%M:%S} ICT, "
                     f"model v{meta.get('model_version')} build {str(meta.get('build_sha'))[:7]}")
    for r in breached:
        lines.append(f"• `{r.feature}` PSI {r.psi:.3f} (≥ {r.threshold}), KS {r.ks_statistic:.3f}, "
                     f"mean {r.ref_mean:.2f} → {r.cur_mean:.2f}")
    lines.append("schema/nulls/ranges: " + ("BROKEN — pipeline problem, do not retrain"
                                             if integrity_report["broken"] else "clean"))
    lines.append("Before retraining: is this the world changing, or a producer breaking?")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", type=Path, default=Path("data/raw/sensors.csv"))
    ap.add_argument("--reference-split", choices=["train", "all"], default="train",
                    help="compare against what the model learned from (default) or the whole file")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--current", type=Path, help="CSV of recent inputs")
    src.add_argument("--current-url", help="base URL of the service; reads /monitoring/window")
    ap.add_argument("--threshold", type=float, default=SENSOR_PSI_THRESHOLD,
                    help="PSI alert threshold for the physical sensors. Justified in the README.")
    ap.add_argument("--min-samples", type=int, default=WINDOW_SIZE,
                    help="the thresholds are calibrated for this window size; fewer is no verdict")
    ap.add_argument("--out", type=Path, default=Path("reports/drift.json"))
    ap.add_argument("--emit", action="store_true", help="send scores as cloud metrics")
    ap.add_argument("--notify", action="store_true", help="post alerts to ALERT_WEBHOOK_URL")
    args = ap.parse_args()

    reference = load_reference(args.reference, args.reference_split)
    if args.current_url:
        current, meta = fetch_window(args.current_url)
    else:
        current, meta = pd.read_csv(args.current), {"n": None}
        meta["n"] = len(current)
    detected_at = dt.datetime.now(dt.timezone.utc)

    integ = integrity(current)
    record: dict = {"checked_at": detected_at.isoformat(), "window": meta, "integrity": integ}
    args.out.parent.mkdir(parents=True, exist_ok=True)

    if len(current) < args.min_samples:
        record["status"] = "insufficient"
        args.out.write_text(json.dumps(record, indent=2))
        print(f"NO VERDICT  window has {len(current)} readings; thresholds are calibrated for "
              f"{args.min_samples}. A window that never fills is itself worth a look "
              f"(traffic stopped? replica restarted?).")
        return 0

    results = compare(reference, current, [f for f in data.FEATURES if f in current.columns],
                      sensor_threshold=args.threshold)
    record["features"] = [asdict(r) for r in results]
    breached = [r for r in results if r.verdict == "DRIFT"]
    record["status"] = "broken" if integ["broken"] else ("drift" if breached else "ok")
    args.out.write_text(json.dumps(record, indent=2, default=str))

    print(f"{'feature':<22}{'psi':>9}{'limit':>7}{'ks':>9}{'ref mean':>11}{'cur mean':>11}  verdict")
    for r in results:
        print(f"{r.feature:<22}{r.psi:>9.4f}{r.threshold:>7.2f}{r.ks_statistic:>9.4f}"
              f"{r.ref_mean:>11.2f}{r.cur_mean:>11.2f}  {r.verdict}")
    print(f"integrity: {'BROKEN' if integ['broken'] else 'clean'}  "
          f"(missing={integ['missing_columns']}, worst null rate="
          f"{max(integ['null_rate'].values(), default=0):.3f}, worst out-of-range rate="
          f"{max(integ['out_of_range_rate'].values(), default=0):.3f})")

    failed_check = False
    if args.emit:
        # Best effort per metric, but a failure fails the run: a detector whose metric
        # silently stopped arriving is a dashboard that silently went flat.
        from cloudlayer.factory import get_adapter
        try:
            adapter = get_adapter(config.load(strict=False))
            for r in results:
                adapter.emit_metric(f"drift.psi.{r.feature}", r.psi)
            adapter.emit_metric("drift.psi.max_ratio", max(r.psi / r.threshold for r in results))
            print("emitted drift metrics")
        except Exception as exc:  # noqa: BLE001 — report, then fail the run below
            print(f"emit failed: {exc}", file=sys.stderr)
            failed_check = True

    if integ["broken"] or breached:
        text = alert_text(results, meta, detected_at, integ)
        print("\n" + text)
        if args.notify:
            try:
                sent = notify(text)
                print(f"notify: {'sent' if sent else 'not sent'}")
            except Exception as exc:  # noqa: BLE001
                print(f"notify failed: {exc}", file=sys.stderr)
                failed_check = True
        return 3 if integ["broken"] else 2

    print(f"\nOK  no feature above its threshold (sensors {args.threshold}, "
          f"load_pct {threshold_for('load_pct')})")
    return 1 if failed_check else 0


if __name__ == "__main__":
    raise SystemExit(main())
