"""Lab 4 — integration test against a RUNNING container.

CI builds the serving image, starts it, and runs:

    SERVICE_URL=http://localhost:8080 EXPECT_MODEL_VERSION=<sha> EXPECT_BUILD_SHA=<sha> \\
        pytest -q tests/test_integration.py

Skipped when SERVICE_URL is unset, so `make test` stays offline. This is the only test
that exercises the image as built: the Dockerfile, the installed dependencies, the
entrypoint, and the port.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

import pytest

BASE = os.environ.get("SERVICE_URL", "").rstrip("/")
pytestmark = pytest.mark.skipif(not BASE, reason="SERVICE_URL not set — no running container")

VALID = {"temp_c": 78.4, "vibration_mm_s": 3.1, "pressure_kpa": 315.2,
         "hours_since_service": 4200.0, "load_pct": 68.0, "ambient_humidity": 55.0}


def call(path: str, body: dict | None = None):
    req = urllib.request.Request(BASE + path, method="POST" if body is not None else "GET",
                                 data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, r.read().decode(), dict(r.headers)
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(), dict(e.headers)


def test_health_and_ready():
    assert call("/health")[0] == 200
    status, body, _ = call("/ready")
    assert status == 200, body
    expected = os.environ.get("EXPECT_BUILD_SHA")
    if expected:
        assert json.loads(body)["build_sha"] == expected


def test_predict_response_shape_and_version():
    status, body, headers = call("/predict", VALID)
    assert status == 200, body
    payload = json.loads(body)
    assert set(payload) == {"probability", "model_version"}, payload
    assert isinstance(payload["probability"], float) and 0.0 <= payload["probability"] <= 1.0
    expected = os.environ.get("EXPECT_MODEL_VERSION")
    if expected:
        assert payload["model_version"] == expected
    lower = {k.lower(): v for k, v in headers.items()}
    assert lower.get("x-request-id")


def test_batch_endpoint():
    status, body, _ = call("/predict/batch", {"rows": [VALID, {**VALID, "temp_c": 95.0}]})
    assert status == 200, body
    assert len(json.loads(body)["probabilities"]) == 2


def test_invalid_input_rejected_by_the_image():
    assert call("/predict", {**VALID, "temp_c": 176.0})[0] == 422


def test_metrics_exposed_with_model_version():
    status, body, _ = call("/metrics")
    assert status == 200
    assert 'http_requests_total{method="POST",path="/predict"' in body
    assert "model_version_info{" in body
    assert "request_latency_ms_bucket" in body


def test_accepted_inputs_reach_the_drift_window():
    before = json.loads(call("/monitoring/window")[1])["n"]
    call("/predict", VALID)
    after = json.loads(call("/monitoring/window")[1])["n"]
    assert after == min(before + 1, 500)
