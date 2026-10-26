"""Lab 5 Task 1 — derive the evaluation gate's thresholds instead of guessing them.

    python scripts/gate_threshold_study.py            # ~3 minutes, writes reports/lab5-gate.json

The gate compares a candidate against the production model ON THE SAME TEST SET (a
paired comparison; see scripts/evaluation_gate.py). This script measures, over many
fresh batches of data from the same process, what that comparison looks like when:

  null      the candidate is the incumbent's own recipe retrained on a new batch: no real
            improvement exists, so every positive delta is noise. MIN_IMPROVEMENT must sit
            above this distribution, or the gate promotes noise. Measured for both
            production recipes this project has had: v2's random forest and v3's
            logistic regression.
  logreg    a genuinely different model (standardised logistic regression). How often
            the gate lets a real improvement through is its power.
  degraded  a broken recipe (5 stumps). The gate must stop it every time.

It also records the score of the best single raw feature on each test set: a model that
cannot beat "sort machines by vibration" has learned nothing worth deploying, and that is
the floor the gate enforces alongside the absolute one.

The random-forest incumbent is rebuilt exactly as registered model v2 was trained: seed
20260101, 100 trees, depth 4, leaf 5 (Lab 2's selected configuration). That is checked,
not assumed: on the two batches the live pipeline evaluated, it reproduces the scores
the live gate measured for v2 to four decimals (0.8224 on week 43, 0.8123 on week 44).
An earlier version of this study assumed the training script's defaults (200 trees,
depth 8), which v2 is not; its thresholds survived the correction (reports/lab5-gate.json).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from make_dataset import build  # noqa: E402

from src import data, gate  # noqa: E402
from src.train import build_model  # noqa: E402

SEED = 20260101
V2 = dict(n_estimators=100, max_depth=4, min_samples_leaf=5)   # registered model v2
F, T = data.FEATURES, data.TARGET


class Recipe(argparse.Namespace):
    def __init__(self, n_estimators=200, max_depth=8, min_samples_leaf=5):
        super().__init__(n_estimators=n_estimators, max_depth=max_depth,
                         min_samples_leaf=min_samples_leaf)


def fit(kind: str, train_df, recipe=None):
    return build_model(kind, recipe or Recipe(), SEED).fit(train_df[F], train_df[T])


def auc(model, df) -> float:
    return float(roc_auc_score(df[T], model.predict_proba(df[F])[:, 1]))


def pct(values, q) -> float:
    return round(float(np.percentile(values, q)), 4)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=int, default=100)
    ap.add_argument("--out", type=Path, default=ROOT / "reports" / "lab5-gate.json")
    args = ap.parse_args()

    started = time.time()
    base_train, _, base_test = data.split(build(SEED), seed=SEED)
    incumbent = fit("rf", base_train, Recipe(**V2))
    incumbent_lr = fit("logreg", base_train)
    incumbent_auc = auc(incumbent, base_test)
    print(f"incumbent (v2 recipe) test ROC-AUC {incumbent_auc:.4f}")

    rows = []
    for k in range(1, args.batches + 1):
        train_df, _, test_df = data.split(build(SEED + k), seed=SEED)
        inc = auc(incumbent, test_df)
        same = auc(fit("rf", train_df, Recipe(**V2)), test_df)
        logreg = auc(fit("logreg", train_df), test_df)
        broken = auc(fit("rf", train_df, Recipe(n_estimators=5, max_depth=1)), test_df)
        rows.append({
            "batch_seed": SEED + k, "incumbent": inc, "same_recipe": same,
            "logreg": logreg, "degraded": broken,
            "logreg_incumbent": auc(incumbent_lr, test_df),
            "best_single_feature": gate.best_single_feature_auc(test_df, F, T),
        })
        if k % 20 == 0:
            print(f"  {k}/{args.batches} batches  ({time.time() - started:.0f} s)")

    null = np.array([r["same_recipe"] - r["incumbent"] for r in rows])
    real = np.array([r["logreg"] - r["incumbent"] for r in rows])
    bad = np.array([r["degraded"] - r["incumbent"] for r in rows])
    null_lr = np.array([r["logreg"] - r["logreg_incumbent"] for r in rows])
    same = np.array([r["same_recipe"] for r in rows])
    single = np.array([r["best_single_feature"] for r in rows])
    unpaired = same - incumbent_auc   # what comparing to the incumbent's own logged score measures

    def decisions(cand_key, inc_key="incumbent", m=gate.MIN_IMPROVEMENT):
        return [gate.decide(r[cand_key], r[inc_key], r["best_single_feature"],
                            min_improvement=m)[0] for r in rows]

    def rate(cand_key, inc_key="incumbent", m=gate.MIN_IMPROVEMENT):
        return round(decisions(cand_key, inc_key, m).count("pass") / len(rows), 3)

    result = {
        "batches": args.batches,
        "incumbent_logged_test_roc_auc": round(incumbent_auc, 4),
        "thresholds": {"MIN_ABSOLUTE": gate.MIN_ABSOLUTE, "MIN_IMPROVEMENT": gate.MIN_IMPROVEMENT},
        "null_delta": {"mean": round(float(null.mean()), 4), "sd": round(float(null.std()), 4),
                       "p90": pct(null, 90), "p95": pct(null, 95), "p99": pct(null, 99),
                       "max": round(float(null.max()), 4),
                       "share_at_least_0.002": round(float((null >= 0.002).mean()), 3)},
        "null_delta_logreg": {"mean": round(float(null_lr.mean()), 4),
                              "sd": round(float(null_lr.std()), 4),
                              "p90": pct(null_lr, 90), "p95": pct(null_lr, 95),
                              "max": round(float(null_lr.max()), 4)},
        "unpaired_delta_vs_logged": {"sd": round(float(unpaired.std()), 4),
                                     "min": round(float(unpaired.min()), 4),
                                     "max": round(float(unpaired.max()), 4)},
        "same_recipe_auc": {"min": round(float(same.min()), 4), "p5": pct(same, 5),
                            "mean": round(float(same.mean()), 4)},
        "best_single_feature_auc": {"mean": round(float(single.mean()), 4),
                                    "max": round(float(single.max()), 4)},
        "margin_over_single_feature_same_recipe_min": round(float((same - single).min()), 4),
        "logreg_delta": {"mean": round(float(real.mean()), 4), "min": round(float(real.min()), 4),
                         "max": round(float(real.max()), 4)},
        "degraded_delta": {"mean": round(float(bad.mean()), 4), "max": round(float(bad.max()), 4)},
        "pass_rate": {
            "null (false promotion)": rate("same_recipe"),
            "null, logreg retrained vs logreg (false promotion)": rate("logreg", "logreg_incumbent"),
            "logreg (power)": rate("logreg"),
            "degraded (must be 0)": rate("degraded"),
        },
        # How the choice of MIN_IMPROVEMENT trades false promotions against power
        # (paired delta only; the floors are applied as configured).
        "margin_sweep": [
            {"min_improvement": m, "null_pass": rate("same_recipe", m=m),
             "null_logreg_pass": rate("logreg", "logreg_incumbent", m=m),
             "logreg_pass": rate("logreg", m=m)}
            for m in (0.002, 0.005, 0.0075, 0.010, 0.0125, 0.015)
        ],
        "same_recipe_below_single_feature": round(float((same <= single).mean()), 3),
        "runtime_s": round(time.time() - started, 1),
        "per_batch": [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}
                      for r in rows],
    }
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "per_batch"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
