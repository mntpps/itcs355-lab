"""Lab 4 — model behaviour tests.

These assert what the model DOES, independent of its metric. A model can score 0.85 and
still be broken in ways these catch, and unlike a metric threshold they do not drift out
of relevance as the data changes.

Quiz 4 asks you to distinguish these from data contract tests. The difference: a contract
test fails when an upstream producer changes something; a behaviour test fails when the
model changes.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from src import config, data, seeds

RAW = config.REPO_ROOT / "data" / "raw" / "sensors.csv"

# loadtest/k6.js sets p95 < 200 ms end to end. Lab 3 measured ~56 ms p50 at one user,
# almost all of it network, TLS and the framework, and at 10 users queueing on a
# 0.25 vCPU replica dominated. So the model gets a tenth of the budget: 20 ms per
# single-row prediction, leaving 180 ms for everything it does not control. Served
# single-threaded it takes ~8 ms; with training's n_jobs=-1 left on it takes ~35 ms and
# this test fails — which is the regression it exists to catch.
P95_TARGET_MS = 200.0
LATENCY_BUDGET_MS = P95_TARGET_MS * 0.10


@pytest.fixture(scope="module")
def fitted():
    if not RAW.exists():
        pytest.skip("run `make data` first")
    seed = seeds.set_all()
    df = data.load_raw(RAW)
    train_df, _, test_df = data.split(df, seed=seed)
    # Trained exactly as src/train.py trains (n_jobs=-1), then prepared exactly as the
    # service prepares it. Behaviour is asserted on the model AS SERVED.
    model = RandomForestClassifier(n_estimators=120, max_depth=8, min_samples_leaf=5,
                                   random_state=seed, n_jobs=-1)
    model.fit(train_df[data.FEATURES], train_df[data.TARGET])
    from service.app import prepare_for_serving
    return prepare_for_serving(model), test_df


def test_predictions_are_valid_probabilities(fitted):
    model, test_df = fitted
    proba = model.predict_proba(test_df[data.FEATURES])[:, 1]
    assert np.all((proba >= 0) & (proba <= 1))
    assert not np.isnan(proba).any()


def test_known_healthy_machine_scores_low(fitted):
    """A cool, new, lightly loaded machine must not be flagged as about to fail.

    This is a domain assertion. If it fails, the model has learned something backwards,
    and no aggregate metric would have told you.
    """
    model, _ = fitted
    healthy = pd.DataFrame([{
        "temp_c": 66.0, "vibration_mm_s": 1.4, "pressure_kpa": 330.0,
        "hours_since_service": 50.0, "load_pct": 30.0, "ambient_humidity": 50.0,
    }])[data.FEATURES]
    assert model.predict_proba(healthy)[0, 1] < 0.5


def test_risk_increases_with_wear(fitted):
    """Monotonicity the domain requires: more hours since service, higher risk.

    Tree ensembles are not monotonic by construction, so this can genuinely fail. If it
    does, that is information — not a reason to delete the test.
    """
    model, _ = fitted
    base = {
        "temp_c": 82.0, "vibration_mm_s": 4.0, "pressure_kpa": 310.0,
        "load_pct": 75.0, "ambient_humidity": 60.0,
    }
    rows = pd.DataFrame([{**base, "hours_since_service": h} for h in (100, 3000, 8500)])[data.FEATURES]
    proba = model.predict_proba(rows)[:, 1]
    assert proba[0] <= proba[-1], f"risk fell as wear rose: {proba}"


def test_risk_increases_with_temperature(fitted):
    """Monotonicity the domain requires: a hotter machine, all else equal, is not safer."""
    model, _ = fitted
    base = {"vibration_mm_s": 4.0, "pressure_kpa": 310.0, "hours_since_service": 4000.0,
            "load_pct": 75.0, "ambient_humidity": 58.0}
    rows = pd.DataFrame([{**base, "temp_c": t} for t in (65.0, 80.0, 95.0, 110.0)])[data.FEATURES]
    proba = model.predict_proba(rows)[:, 1]
    assert proba[0] <= proba[-1], f"risk fell as temperature rose: {proba}"


def test_ambient_humidity_does_not_drive_the_decision(fitted):
    """Invariance: humidity has no physical link to bearing failure here. If sweeping it
    from 20% to 95% moves risk by more than 0.10, the model has learned a spurious cue
    that a humid week would turn into a wave of false alarms."""
    model, _ = fitted
    base = {"temp_c": 82.0, "vibration_mm_s": 4.0, "pressure_kpa": 310.0,
            "hours_since_service": 4000.0, "load_pct": 75.0}
    rows = pd.DataFrame([{**base, "ambient_humidity": h} for h in (20, 40, 60, 80, 95)])[data.FEATURES]
    proba = model.predict_proba(rows)[:, 1]
    assert proba.max() - proba.min() < 0.10, f"humidity swings risk: {proba.round(3)}"


def test_prediction_latency_within_budget(fitted):
    model, test_df = fitted
    sample = test_df[data.FEATURES].head(100)
    model.predict_proba(sample)  # warm up; the first call includes lazy setup
    started = time.perf_counter()
    for _ in range(20):
        model.predict_proba(sample.head(1))
    elapsed_ms = (time.perf_counter() - started) * 1000 / 20
    assert elapsed_ms < LATENCY_BUDGET_MS, f"single prediction took {elapsed_ms:.2f} ms"


def test_model_is_not_constant(fitted):
    """A model that always predicts the majority class can post a respectable accuracy
    and is useless. Catch it here rather than in production."""
    model, test_df = fitted
    proba = model.predict_proba(test_df[data.FEATURES])[:, 1]
    assert proba.std() > 0.01, "predictions have almost no variance"
