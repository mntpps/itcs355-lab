"""The registration gate's rule. Provider-neutral: plain numbers in, a decision out.

scripts/evaluation_gate.py does the I/O (load both models, score them on the candidate's
held-out machines) and calls `decide`. The thresholds are derived, not guessed:
scripts/gate_threshold_study.py measures them over 100 fresh batches, and the results
are in reports/lab5-gate.json.

Why the comparison must be PAIRED. The test set is 48 held-out machines (1,200
readings), so its ROC-AUC moves a lot from batch to batch: the incumbent's recipe,
retrained, scores anywhere from 0.79 to 0.88 depending on which machines land in
the test split. Comparing a candidate's score on this week's test set against the 0.8533
the incumbent (v2) logged on a different test set measures which machines were drawn, not
which model is better. Scoring both models on the same machines removes that: the paired
delta of a no-change retrain has a standard deviation of 0.006, against 0.019 unpaired.
"""
from __future__ import annotations

from sklearn.metrics import roc_auc_score

MIN_IMPROVEMENT = 0.010
"""Paired ROC-AUC gain the candidate needs over production. Retraining production's own
recipe on a fresh batch, where no real improvement is possible, gives over 100 batches:

  v2's random forest        paired delta p90 0.008, p95 0.010, max 0.014: 4% pass at 0.010
  v3's logistic regression  paired delta p95 0.002, max 0.004: none pass

So on a weekly schedule a no-change retrain is promoted about twice a year while the
forest is in production, and essentially never while the logistic regression is; either
way it swaps in a model statistically equal to the one it replaces. The scaffold's 0.002
passes 46% of the forest retrains.

Raising the bar costs power. Logistic regression over v2 is a real improvement (mean
paired gain +0.012). A single run passes 63% of the time at 0.010, 45% at 0.0125 and 30%
at 0.015. A real improvement gets another chance every week, so at 0.010 it is promoted
within four weeks 98% of the time (76% at 0.015). The full sweep is in
reports/lab5-gate.json."""

MIN_ABSOLUTE = 0.78
"""Floor on the candidate's own score, whatever production scores. A healthy retrain of
v2's recipe never scored below 0.79 on any of 100 batches, so 0.78 never
blocks a good model on an unlucky split, and does block a model that has broken. The
degraded recipe in the study (5 stumps) never passed."""

BASELINE_MARGIN = 0.0
"""The candidate must also beat the best single raw feature on the same test set (sorting
machines by vibration alone scores 0.81 on average). A model that cannot outrank one
sensor has learned nothing worth deploying, whatever the absolute floor says. It is a
real constraint: v2's random forest beats that baseline by only 0.03 on average, and on
1 batch in 100 it did not beat it at all."""


def best_single_feature_auc(df, features, target) -> float:
    """ROC-AUC of the best one-feature ranking, in whichever direction ranks better."""
    best = 0.5
    for f in features:
        a = float(roc_auc_score(df[target], df[f]))
        best = max(best, a, 1.0 - a)
    return best


def decide(candidate: float, incumbent: float | None, baseline: float | None = None,
           min_improvement: float = MIN_IMPROVEMENT, min_absolute: float = MIN_ABSOLUTE,
           ) -> tuple[str, list[str]]:
    """Return ("pass" | "fail", reasons). `incumbent` None means no production model."""
    reasons = []
    if candidate < min_absolute:
        reasons.append(f"candidate {candidate:.4f} is below the absolute floor {min_absolute}")
    if baseline is not None and candidate <= baseline + BASELINE_MARGIN:
        reasons.append(f"candidate {candidate:.4f} does not beat the best single feature "
                       f"{baseline:.4f}")
    if incumbent is not None:
        delta = candidate - incumbent
        if delta < min_improvement:
            reasons.append(f"paired gain {delta:+.4f} over production {incumbent:.4f} is below "
                           f"the required {min_improvement:+.3f} (noise level)")
    return ("fail" if reasons else "pass"), reasons
