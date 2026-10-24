"""Distribution-shift statistics shared by the service and the scheduled detector.

One implementation, two callers: the service computes a rolling PSI over its recent
inputs for the dashboard, and monitoring/drift.py computes the same number on a schedule
to decide whether to alert. If they used different code, the dashboard and the alert
could disagree about the same window, and you would spend the incident arguing about
which one is right.

PSI bins are fixed by the REFERENCE (quantiles of the training inputs), so a reference
profile — bin edges plus reference counts — is everything needed to score a new window.
That profile is small enough to ship inside the serving image.
"""
from __future__ import annotations

import numpy as np

DEFAULT_BINS = 10


def reference_profile(reference: np.ndarray, bins: int = DEFAULT_BINS) -> dict:
    """Quantile bin edges of the reference and the reference count in each bin.

    Only the inner edges are stored; the outer bins are open-ended so a value beyond
    anything seen in training still lands in a bin rather than being dropped.
    """
    reference = np.asarray(reference, dtype=float)
    edges = np.quantile(reference, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    edges = np.unique(edges)
    counts, _ = np.histogram(reference, bins=edges)
    return {
        "inner_edges": [float(e) for e in edges[1:-1]],
        "ref_counts": [int(c) for c in counts],
        "ref_mean": float(reference.mean()),
        "ref_std": float(reference.std()),
        "n": int(reference.size),
    }


def psi_from_profile(profile: dict, current: np.ndarray) -> float:
    """Population Stability Index of `current` against a stored reference profile.

    Laplace smoothing (+1 per bin) keeps an empty bin from making the log term
    infinite. With few samples that smoothing is not negligible, which is one reason
    PSI on a small window is biased upwards — see reports/lab4-threshold.md.
    """
    current = np.asarray(current, dtype=float)
    if current.size == 0 or len(profile["ref_counts"]) < 2:
        return 0.0
    edges = np.concatenate([[-np.inf], profile["inner_edges"], [np.inf]])
    ref_counts = np.asarray(profile["ref_counts"], dtype=float)
    cur_counts, _ = np.histogram(current, bins=edges)
    ref_prop = (ref_counts + 1) / (ref_counts.sum() + len(ref_counts))
    cur_prop = (cur_counts + 1) / (cur_counts.sum() + len(cur_counts))
    return float(np.sum((cur_prop - ref_prop) * np.log(cur_prop / ref_prop)))


def psi(reference: np.ndarray, current: np.ndarray, bins: int = DEFAULT_BINS) -> float:
    """PSI of `current` against `reference`, binning on the reference's quantiles.

    Sensitive to changes in shape, not only in mean — which is why a feature can drift
    badly while its average looks untouched.
    """
    return psi_from_profile(reference_profile(reference, bins), current)


def ks_statistic(reference: np.ndarray, current: np.ndarray) -> float:
    """Two-sample Kolmogorov–Smirnov statistic: the largest gap between the two CDFs.

    Complements PSI. KS is more sensitive to a shift in location; PSI to a change in
    shape. Reporting both, and noticing when they disagree, is worth more than either.
    """
    ref = np.sort(np.asarray(reference, dtype=float))
    cur = np.sort(np.asarray(current, dtype=float))
    if ref.size == 0 or cur.size == 0:
        return 0.0
    pooled = np.concatenate([ref, cur])
    cdf_ref = np.searchsorted(ref, pooled, side="right") / len(ref)
    cdf_cur = np.searchsorted(cur, pooled, side="right") / len(cur)
    return float(np.max(np.abs(cdf_ref - cdf_cur)))


# --- The alerting rule ------------------------------------------------------------------
# Derived in scripts/drift_threshold_study.py (results: reports/lab4-threshold.json) and
# argued in the README. Shared here so the dashboard's threshold line and the scheduled
# detector can never disagree.
WINDOW_SIZE = 500
"""Readings per drift window. The thresholds below are only valid at this size: PSI's
no-drift noise floor roughly halves each time the window doubles."""

SENSOR_PSI_THRESHOLD = 0.10
"""Physical sensors. The no-drift PSI that a 15-minute check would exceed once a month
(max over the five sensors, 500-reading windows of unseen machines) is 0.094."""

FLEET_MIX_THRESHOLDS = {"load_pct": 0.29}
"""Features that are per-machine traits. load_pct is set by each machine's duty cycle,
so its PSI mostly measures WHICH machines are reporting: its no-drift p99.9 is 0.27.
Holding it to the sensor threshold would page ~2,700 times a month."""


def threshold_for(feature: str) -> float:
    return FLEET_MIX_THRESHOLDS.get(feature, SENSOR_PSI_THRESHOLD)
