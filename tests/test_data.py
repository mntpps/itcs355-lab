"""Data contract tests — assertions about the DATA, not the code.

They fail when an upstream producer changes something even though our code is
untouched, which is exactly when nobody thinks to look. Each one names the incident it
exists to catch; the README's Lab 4 section lists them together.

Property tests about the splitting LOGIC moved to tests/test_unit.py in Lab 4, except
the leakage test, which stays here: a split that leaks is a contract the training data
must satisfy before any model is trained on it.
"""
from __future__ import annotations

import json

import pytest

from src import config, data

RAW = config.REPO_ROOT / "data" / "raw" / "sensors.csv"
PROFILE = config.REPO_ROOT / "service" / "reference_profile.json"

# Stated null-rate thresholds. Identifiers and the label must be complete: a row without
# them cannot be split or learned from. A feature may lose up to 1% to sensor dropouts
# before the batch is rejected — above that, imputation is inventing the data.
MAX_NULL_RATE = {data.ID: 0.0, data.GROUP: 0.0, data.TARGET: 0.0,
                 **{f: 0.01 for f in data.FEATURES}}

# A new batch may differ from the training reference, but not by much without a human
# deciding it should. Mean within a quarter of a reference standard deviation, spread
# within a third. A unit change (kPa -> psi, C -> F) moves the mean by many SDs.
MAX_MEAN_SHIFT_SD = 0.25
SPREAD_RATIO = (0.75, 1.33)


@pytest.fixture(scope="module")
def df():
    if not RAW.exists():
        pytest.skip("data/raw/sensors.csv missing — run `make data` or `dvc pull` first")
    return data.load_raw(RAW)


def test_schema_columns_present_and_typed(df):
    """Incident: upstream renames temp_c -> temperature, or ships vibration as a string
    with units ("3.1 mm/s"). Training would KeyError or silently coerce."""
    missing = set(data.SCHEMA) - set(df.columns)
    assert not missing, f"missing columns: {sorted(missing)}"
    unexpected = set(df.columns) - set(data.SCHEMA)
    assert not unexpected, f"unexpected columns: {sorted(unexpected)}"
    for col, expected in data.SCHEMA.items():
        assert str(df[col].dtype) == expected, f"{col}: expected {expected}, got {df[col].dtype}"


def test_null_rate_below_stated_threshold(df):
    """Incident: a gateway outage writes blank readings for an afternoon; mean
    imputation fills them with a healthy-looking average and hides failing machines."""
    rates = df[list(MAX_NULL_RATE)].isna().mean()
    offenders = {c: round(float(r), 4) for c, r in rates.items() if r > MAX_NULL_RATE[c]}
    assert not offenders, f"null rate above threshold: {offenders} (limits: {MAX_NULL_RATE})"


def test_features_within_plausible_ranges(df):
    """Incident: a sensor vendor switches the feed to Fahrenheit. 80 C becomes 176 F, the
    model sees every machine as overheating, and maintenance gets a false-alarm storm."""
    for col, (lo, hi) in data.PLAUSIBLE_RANGES.items():
        assert df[col].min() >= lo, f"{col} below plausible floor {lo}: {df[col].min()}"
        assert df[col].max() <= hi, f"{col} above plausible ceiling {hi}: {df[col].max()}"


def test_feature_distributions_match_training_reference(df):
    """Incident: pressure sensors are replaced with units reporting psi. 315 kPa becomes
    45.7 psi — inside the 0-600 plausible range, so the range test passes, but the
    distribution is nowhere near what the model learned."""
    profile = json.loads(PROFILE.read_text())["features"]
    for f in data.FEATURES:
        ref_mean, ref_std = profile[f]["ref_mean"], profile[f]["ref_std"]
        shift = abs(df[f].mean() - ref_mean) / ref_std
        assert shift <= MAX_MEAN_SHIFT_SD, (
            f"{f}: mean {df[f].mean():.2f} is {shift:.2f} SD from the training reference "
            f"{ref_mean:.2f} (limit {MAX_MEAN_SHIFT_SD})")
        ratio = df[f].std() / ref_std
        assert SPREAD_RATIO[0] <= ratio <= SPREAD_RATIO[1], (
            f"{f}: spread is {ratio:.2f}x the training reference (allowed {SPREAD_RATIO})")


def test_serving_reference_profile_is_current(df):
    """Incident: the model is retrained on new data but the drift reference baked into
    the serving image is not regenerated, so the dashboard compares live traffic against
    the wrong baseline — showing drift that is not there, or hiding drift that is.
    Fix with `python scripts/build_reference_profile.py`."""
    committed = json.loads(PROFILE.read_text())
    assert committed["fingerprint"] == data.data_fingerprint(RAW), (
        "service/reference_profile.json was built from different data — regenerate it")


def test_target_is_binary_and_not_degenerate(df):
    """Incident: the labelling job that joins maintenance tickets fails and writes 0 for
    every row; a model trained on it predicts 'never fails' with 100% accuracy."""
    values = set(df[data.TARGET].unique().tolist())
    assert values <= {0, 1}, f"target has values outside 0/1: {values}"
    rate = df[data.TARGET].mean()
    assert 0.01 < rate < 0.99, f"target is degenerate, positive rate = {rate:.4f}"


def test_identifier_is_unique(df):
    """Incident: an ingestion retry replays a batch, so readings appear twice — double
    weight in training, and the same reading can land on both sides of a split."""
    assert df[data.ID].is_unique, "reading_id is not unique"


def test_no_machine_leaks_across_splits(df):
    """Incident (Lab 1): a row-wise split lets readings from one machine sit in train and
    test, the model memorises machines, and a 0.85 validation score never survives
    contact with new machines."""
    train, val, test = data.split(df, seed=42)
    tr, va, te = (set(p[data.GROUP]) for p in (train, val, test))
    assert not tr & va, f"machines in both train and val: {sorted(tr & va)[:5]}"
    assert not tr & te, f"machines in both train and test: {sorted(tr & te)[:5]}"
    assert not va & te, f"machines in both val and test: {sorted(va & te)[:5]}"
