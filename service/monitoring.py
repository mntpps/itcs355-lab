"""Lab 4 — what the running service tells you about itself.

Prometheus exposition on /metrics, scraped by the stack in monitoring/stack/. Five
signals, matching the dashboard panels one for one:

  http_requests_total{path,method,status,status_class}   request rate, 4xx vs 5xx
  request_latency_ms{path}  (histogram)                  p50 / p95 / p99
  feature_window_psi{feature}, feature_window_mean{...}  rolling feature distribution
  model_version_info{model_version,build_sha}            what is serving right now

The rolling window is the last WINDOW_SIZE accepted readings. It is in process memory,
which is a deliberate simplification with two consequences worth knowing before you are
asked about them:
  * it empties on restart, including every scale-from-zero, so a fresh replica reports
    "window filling" rather than a PSI;
  * each replica has its own, so staging runs with max one replica (and one uvicorn
    worker) to keep a single, scrapeable window.
A production version would write accepted inputs to a log or table and compute the
window there.

Only ACCEPTED readings enter the window. A Fahrenheit feed is rejected with 422 by the
schema and shows up as a 4xx spike, never as drift; a subtler fault that stays inside
the plausible ranges passes validation and is exactly what the window is for.
"""
from __future__ import annotations

import hmac
import json
import os
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram
from prometheus_client.core import GaugeMetricFamily

from src.data import FEATURES
from src.drift_stats import WINDOW_SIZE, psi_from_profile, threshold_for

PROFILE_PATH = Path(os.environ.get(
    "REFERENCE_PROFILE", Path(__file__).with_name("reference_profile.json")))

REGISTRY = CollectorRegistry()

REQUESTS = Counter(
    "http_requests", "HTTP requests served, by route and status class.",
    ["path", "method", "status", "status_class"], registry=REGISTRY)

# Buckets bracket the 200 ms p95 target closely, because a percentile estimated from a
# histogram is only as precise as the buckets either side of it.
LATENCY = Histogram(
    "request_latency_ms", "Server-side request latency in milliseconds.", ["path"],
    buckets=(5, 10, 25, 50, 75, 100, 150, 200, 250, 300, 500, 750, 1000, 2000, 5000),
    registry=REGISTRY)

MODEL_INFO = Gauge(
    "model_version_info", "1 for the model version and image build this replica serves.",
    ["model_version", "build_sha"], registry=REGISTRY)


class RollingWindow:
    """The last `size` accepted readings, per feature, with arrival times."""

    def __init__(self, size: int = WINDOW_SIZE) -> None:
        self.size = size
        self._lock = threading.Lock()
        self._rows: dict[str, deque] = {f: deque(maxlen=size) for f in FEATURES}
        self._times: deque = deque(maxlen=size)

    def add(self, rows: list[dict]) -> None:
        now = time.time()
        with self._lock:
            for row in rows:
                for f in FEATURES:
                    self._rows[f].append(float(row[f]))
                self._times.append(now)

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "n": len(self._times),
                "size": self.size,
                "oldest": self._times[0] if self._times else None,
                "newest": self._times[-1] if self._times else None,
                "rows": {f: list(self._rows[f]) for f in FEATURES},
            }


WINDOW = RollingWindow()


def _load_profile() -> dict | None:
    try:
        return json.loads(PROFILE_PATH.read_text())["features"]
    except (OSError, KeyError, ValueError):
        return None


class WindowCollector:
    """Computes the rolling statistics at scrape time, not per request."""

    def __init__(self, window: RollingWindow) -> None:
        self.window = window
        self.profile = _load_profile()

    def collect(self):
        snap = self.window.snapshot()
        size = GaugeMetricFamily("feature_window_size", "Readings in the rolling drift window.")
        size.add_metric([], snap["n"])
        yield size
        if not self.profile:
            return
        psi = GaugeMetricFamily(
            "feature_window_psi", "PSI of the rolling window against the training reference.",
            labels=["feature"])
        mean = GaugeMetricFamily(
            "feature_window_mean", "Mean of the rolling window.", labels=["feature"])
        ref_mean = GaugeMetricFamily(
            "feature_reference_mean", "Mean of the training reference.", labels=["feature"])
        threshold = GaugeMetricFamily(
            "feature_psi_threshold", "Alert threshold for this feature's PSI.", labels=["feature"])
        # A partial window would report PSI against a threshold derived for a full one.
        full = snap["n"] >= self.window.size
        for f in FEATURES:
            values = np.asarray(snap["rows"][f])
            ref_mean.add_metric([f], self.profile[f]["ref_mean"])
            threshold.add_metric([f], threshold_for(f))
            if values.size:
                mean.add_metric([f], float(values.mean()))
            if full:
                psi.add_metric([f], psi_from_profile(self.profile[f], values))
        yield from (psi, mean, ref_mean, threshold)


REGISTRY.register(WindowCollector(WINDOW))


def record_request(path: str, method: str, status: int, latency_ms: float) -> None:
    REQUESTS.labels(path, method, str(status), f"{status // 100}xx").inc()
    LATENCY.labels(path).observe(latency_ms)


def set_model_info(model_version: str, build_sha: str) -> None:
    MODEL_INFO.clear()
    MODEL_INFO.labels(model_version, build_sha).set(1)


def window_access_allowed(supplied: str | None) -> bool:
    """The window holds raw inputs, so staging protects it with a shared token.

    No MONITORING_TOKEN configured means a local or CI run, where it is open.
    """
    expected = os.environ.get("MONITORING_TOKEN")
    if not expected:
        return True
    return supplied is not None and hmac.compare_digest(supplied, expected)
