"""Lab 5 Task 3 — what happens when retraining is triggered by drift that is really a
corrupted upstream feed.

    python scripts/retrain_trigger_study.py      # ~1 minute -> reports/lab5-retrain-triggers.json

The Lab 4 incident: every temp_c reading arrives offset by a constant (a calibration
fault), while the machines are unchanged. For each offset, over 20 fresh weekly batches:

  contract   does validate (tests/test_data.py's mean-shift rule, 0.25 reference SD) stop it?
  drift      does the Lab 4 detector (PSI >= 0.10 on a 500-reading window) fire on it?
  gate       does a logistic-regression candidate trained on the corrupted batch pass the
             paired gate, with v2 (forest) or v3 (logistic regression) in production?
  damage     the share of truly failing machines the candidate would flag for at least an
             engineer's review (probability >= 0.40, the work-order rule) on CLEAN data,
             against production's share.

The point the numbers make: ROC-AUC only ranks machines, and a constant offset does not
change the ranking, so the gate cannot see this corruption. It only sees whether the
recipe is better. With v2 in production, a better recipe trained on corrupted data passes
the gate and ships the corruption with it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from gate_threshold_study import V2, Recipe, auc  # noqa: E402
from make_dataset import build  # noqa: E402

from src import data, gate  # noqa: E402
from src.drift_stats import SENSOR_PSI_THRESHOLD, WINDOW_SIZE, psi  # noqa: E402
from src.train import build_model  # noqa: E402

SEED = 20260101
F, T = data.FEATURES, data.TARGET
MAX_MEAN_SHIFT_SD = 0.25   # tests/test_data.py
REVIEW_AT = 0.40           # src/prompts/workorder-v1.txt: review_by_engineer from 0.40


def flagged(model, df) -> float:
    p = model.predict_proba(df[F])[:, 1]
    return float((p[df[T].values == 1] >= REVIEW_AT).mean())


def main() -> int:
    ref = build(SEED)
    profile = json.loads((ROOT / "service" / "reference_profile.json").read_text())["features"]["temp_c"]
    v2 = build_model("rf", Recipe(**V2), SEED).fit(*(lambda d: (d[F], d[T]))(data.split(ref, seed=SEED)[0]))
    wk44 = data.split(build(202644), seed=SEED)[0]
    v3 = build_model("logreg", Recipe(), SEED).fit(wk44[F], wk44[T])
    rng = np.random.default_rng(0)

    out = []
    for offset in (0.0, 1.0, 2.0, 3.0, 4.0, 6.0):
        rows = []
        for k in range(20):
            clean = build(202645 + k)
            bad = clean.copy()
            bad["temp_c"] = bad["temp_c"] + offset
            window = rng.choice(bad["temp_c"].values, WINDOW_SIZE, replace=False)
            tr, _, te = data.split(bad, seed=SEED)
            te_clean = data.split(clean, seed=SEED)[2]
            cand = build_model("logreg", Recipe(), SEED).fit(tr[F], tr[T])
            base = gate.best_single_feature_auc(te, F, T)
            rows.append({
                "contract_passes": abs(bad["temp_c"].mean() - profile["ref_mean"]) / profile["ref_std"]
                <= MAX_MEAN_SHIFT_SD,
                "drift_fires": psi(ref["temp_c"].values, window) >= SENSOR_PSI_THRESHOLD,
                "gate_passes_vs_v2": gate.decide(auc(cand, te), auc(v2, te), base)[0] == "pass",
                "gate_passes_vs_v3": gate.decide(auc(cand, te), auc(v3, te), base)[0] == "pass",
                "candidate_flags": flagged(cand, te_clean),
                "v2_flags": flagged(v2, te_clean), "v3_flags": flagged(v3, te_clean),
            })
        mean = {k: round(float(np.mean([r[k] for r in rows])), 3) for k in rows[0]}
        out.append({"temp_offset_c": offset, **mean})
        print(f"+{offset:.0f} C  contract passes {mean['contract_passes']:.0%}  drift fires "
              f"{mean['drift_fires']:.0%}  gate passes vs v2 {mean['gate_passes_vs_v2']:.0%} / "
              f"vs v3 {mean['gate_passes_vs_v3']:.0%}  failing machines flagged on clean data: "
              f"candidate {mean['candidate_flags']:.1%}, v2 {mean['v2_flags']:.1%}, "
              f"v3 {mean['v3_flags']:.1%}")
    (ROOT / "reports" / "lab5-retrain-triggers.json").write_text(json.dumps(out, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
