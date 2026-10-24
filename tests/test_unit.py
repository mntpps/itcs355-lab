"""Lab 4 — unit tests. Fast, no network, no cloud.

These test CODE: the drift statistics, the rolling window, the integrity check, the
split logic, the schema bounds, the secret scanner. They fail when someone changes the
code, not when the data changes — that is test_data.py's job.
"""
from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
import pytest

from src import config, data
from src.drift_stats import (
    FLEET_MIX_THRESHOLDS,
    SENSOR_PSI_THRESHOLD,
    ks_statistic,
    psi,
    psi_from_profile,
    reference_profile,
    threshold_for,
)

RAW = config.REPO_ROOT / "data" / "raw" / "sensors.csv"
RNG = np.random.default_rng(0)


# --- drift statistics ----------------------------------------------------------------

def test_psi_is_near_zero_for_same_distribution():
    ref = RNG.normal(80, 10, 20000)
    cur = RNG.normal(80, 10, 20000)
    assert psi(ref, cur) < 0.01


def test_psi_detects_a_mean_shift():
    ref = RNG.normal(80, 10, 20000)
    assert psi(ref, ref + 6) > 0.25


def test_psi_detects_a_spread_change_the_mean_hides():
    """The scale fault: mean untouched, shape changed. PSI must see it."""
    ref = RNG.normal(80, 10, 20000)
    cur = 80 + (RNG.normal(80, 10, 20000) - 80) * 1.5
    assert abs(cur.mean() - ref.mean()) < 0.5
    assert psi(ref, cur) > SENSOR_PSI_THRESHOLD


def test_profile_round_trip_matches_direct_psi():
    """The service scores against a stored profile; the detector against raw data.
    They must give the same number or the dashboard and the alert disagree."""
    ref = RNG.normal(0, 1, 5000)
    cur = RNG.normal(0.3, 1.2, 500)
    assert psi_from_profile(reference_profile(ref), cur) == pytest.approx(psi(ref, cur))


def test_psi_survives_values_outside_the_reference_range():
    ref = RNG.uniform(0, 1, 1000)
    assert np.isfinite(psi(ref, np.full(200, 1e6)))


def test_ks_extremes():
    a = np.arange(100.0)
    assert ks_statistic(a, a) == 0.0
    assert ks_statistic(a, a + 1000) == 1.0


def test_load_pct_has_its_own_threshold_and_sensors_share_one():
    assert threshold_for("load_pct") == FLEET_MIX_THRESHOLDS["load_pct"] > SENSOR_PSI_THRESHOLD
    for f in data.FEATURES:
        if f not in FLEET_MIX_THRESHOLDS:
            assert threshold_for(f) == SENSOR_PSI_THRESHOLD


# --- detector logic ------------------------------------------------------------------

def _window(n=600, **overrides):
    rows = {f: RNG.uniform(*data.PLAUSIBLE_RANGES[f], n) for f in data.FEATURES}
    rows.update(overrides)
    return pd.DataFrame(rows)


def test_integrity_passes_a_clean_window():
    from monitoring.drift import integrity
    assert not integrity(_window())["broken"]


def test_integrity_flags_a_missing_column():
    from monitoring.drift import integrity
    assert integrity(_window().drop(columns=["pressure_kpa"]))["broken"]


def test_integrity_flags_nulls_above_threshold():
    from monitoring.drift import integrity
    w = _window()
    w.loc[: len(w) // 10, "temp_c"] = np.nan
    assert integrity(w)["broken"]


def test_integrity_flags_a_unit_change():
    """Fahrenheit is out of range, so it is a pipeline break, never 'drift'."""
    from monitoring.drift import integrity
    w = _window()
    w["temp_c"] = w["temp_c"] * 9 / 5 + 32
    assert integrity(w)["broken"]


def test_rolling_window_keeps_only_the_most_recent():
    from service.monitoring import RollingWindow
    win = RollingWindow(size=3)
    win.add([{f: float(i) for f in data.FEATURES} for i in range(5)])
    snap = win.snapshot()
    assert snap["n"] == 3
    assert snap["rows"]["temp_c"] == [2.0, 3.0, 4.0]


def test_monitoring_token_is_enforced_when_set(monkeypatch):
    from service.monitoring import window_access_allowed
    monkeypatch.setenv("MONITORING_TOKEN", "s3cret-value")
    assert not window_access_allowed(None)
    assert not window_access_allowed("wrong")
    assert window_access_allowed("s3cret-value")
    monkeypatch.delenv("MONITORING_TOKEN")
    assert window_access_allowed(None)


# --- contracts between modules -------------------------------------------------------

def test_request_schema_bounds_match_the_data_contract():
    """service/schemas.py says it mirrors PLAUSIBLE_RANGES. Hold it to that: widening one
    without the other lets the API accept what the contract tests call corrupt."""
    from service.schemas import PredictRequest
    for f, (lo, hi) in data.PLAUSIBLE_RANGES.items():
        meta = {type(m).__name__: m for m in PredictRequest.model_fields[f].metadata}
        assert meta["Ge"].ge == lo and meta["Le"].le == hi, f


def test_serving_forces_single_threaded_prediction():
    from sklearn.ensemble import RandomForestClassifier

    from service.app import prepare_for_serving
    model = RandomForestClassifier(n_jobs=-1)
    assert prepare_for_serving(model).n_jobs == 1


# --- split logic (property tests: about the CODE, not the data) ----------------------

@pytest.fixture(scope="module")
def df():
    if not RAW.exists():
        pytest.skip("run `make data` first")
    return data.load_raw(RAW)


def test_split_is_deterministic_given_seed(df):
    a = data.split(df, seed=7)
    b = data.split(df, seed=7)
    for pa, pb in zip(a, b):
        assert pa[data.ID].tolist() == pb[data.ID].tolist()


def test_split_changes_with_seed(df):
    a, _, _ = data.split(df, seed=1)
    b, _, _ = data.split(df, seed=2)
    assert a[data.ID].tolist() != b[data.ID].tolist(), "seed has no effect — split is not random"


def test_every_row_lands_in_exactly_one_split(df):
    train, val, test = data.split(df, seed=13)
    total = len(train) + len(val) + len(test)
    assert total == len(df), f"rows lost or duplicated: {total} vs {len(df)}"


def test_data_fingerprint_is_stable():
    if not RAW.exists():
        pytest.skip("no raw data")
    assert data.data_fingerprint(RAW) == data.data_fingerprint(RAW)


# --- the secret scanner ---------------------------------------------------------------

def _scanner():
    spec = importlib.util.spec_from_file_location(
        "scan_secrets", config.REPO_ROOT / "scripts" / "scan_secrets.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_scanner_catches_a_real_shaped_key_and_ignores_placeholders():
    scan = _scanner()
    fake_key = "AKIA" + "Q" * 16                      # built at runtime: never in history
    lines = [
        ("c1", "a.py", f"key = '{fake_key}'"),
        ("c2", "b.yml", "client-id: ${{ secrets.AZURE_CLIENT_ID }}"),
        ("c3", "c.env", 'password = "<your-password>"'),
    ]
    hits = scan.findings(lines)
    assert [h[2] for h in hits] == ["aws-access-key-id"]
