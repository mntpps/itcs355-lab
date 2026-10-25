"""Lab 5 — the registration gate.

Scores the candidate AND the production model on the same held-out machines (the
candidate's test split of this batch), then applies src.gate.decide. A paired
comparison, because the test set is small enough that comparing against the score the
incumbent logged on a different test set mostly measures which machines were drawn
(src/gate.py has the numbers).

    # the pipeline step: always exits 0 on a decision; the decision is the output
    python scripts/evaluation_gate.py --metrics m/metrics.json --model m/model \\
        --data d/sensors.csv --decision-out gate/

    # by hand: exits 1 when the candidate should NOT be registered
    python scripts/evaluation_gate.py --metrics m/metrics.json --model m/model --data d/sensors.csv

The production model is read from the registry: the version tagged role=production
(src/registry.py). None means a first registration, which only the absolute floor and
the single-feature baseline apply to. If the registry cannot be reached, the gate fails
closed: an error, never a pass.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config, data, gate, registry  # noqa: E402

# Kept here for the scaffold's readers; the values and their derivation live in src/gate.py.
MIN_ABSOLUTE = gate.MIN_ABSOLUTE
MIN_IMPROVEMENT = gate.MIN_IMPROVEMENT


def auc_on(model, df) -> float:
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(df[data.TARGET], model.predict_proba(df[data.FEATURES])[:, 1]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", type=Path, required=True, help="the train step's metrics.json")
    ap.add_argument("--model", type=Path, required=True, help="the candidate's MLflow model folder")
    ap.add_argument("--data", type=Path, required=True, help="the batch the candidate was trained on")
    ap.add_argument("--incumbent-model", default=None,
                    help="override: an MLflow model URI to compare against instead of production")
    ap.add_argument("--decision-out", type=Path, default=None,
                    help="write decision + gate.json here and exit 0 either way (pipeline mode)")
    args = ap.parse_args()

    import mlflow.sklearn

    cfg = config.load(strict=False)
    payload = json.loads(args.metrics.read_text())
    seed = int(payload["seed"])
    _, _, test_df = data.split(data.load_raw(args.data), seed=seed)

    candidate = auc_on(mlflow.sklearn.load_model(str(args.model)), test_df)
    baseline = gate.best_single_feature_auc(test_df, data.FEATURES, data.TARGET)
    print(f"test split: {test_df[data.GROUP].nunique()} machines, {len(test_df)} readings "
          f"(seed {seed})")
    print(f"candidate ({payload.get('model', '?')})   ROC-AUC {candidate:.4f}")
    print(f"best single feature   ROC-AUC {baseline:.4f}")

    incumbent_uri, incumbent_version = args.incumbent_model, None
    if incumbent_uri is None:
        c = registry.client(cfg.mlflow_tracking_uri)
        prod = registry.production_version(c, cfg.model_registry_name)
        if prod is not None:
            incumbent_version = str(prod.version)
            incumbent_uri = f"models:/{cfg.model_registry_name}/{prod.version}"
    incumbent = None
    if incumbent_uri:
        mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
        incumbent = auc_on(mlflow.sklearn.load_model(incumbent_uri), test_df)
        print(f"production ({incumbent_uri})  ROC-AUC {incumbent:.4f} on the same machines  "
              f"paired gain {candidate - incumbent:+.4f}")
    else:
        print(f"no version of {cfg.model_registry_name} is tagged role=production: "
              "first registration, floors only")

    decision, reasons = gate.decide(candidate, incumbent, baseline)
    print(f"GATE {decision.upper()}")
    for r in reasons:
        print(f"  - {r}")
    if decision == "pass":
        print(f"  requirements: >= {gate.MIN_ABSOLUTE}, beats single feature, "
              f"gain >= {gate.MIN_IMPROVEMENT:+.3f}")

    if args.decision_out:
        args.decision_out.mkdir(parents=True, exist_ok=True)
        (args.decision_out / "decision").write_text(decision + "\n")
        (args.decision_out / "gate.json").write_text(json.dumps({
            "decision": decision, "reasons": reasons,
            "candidate_roc_auc": round(candidate, 4),
            "incumbent_roc_auc": None if incumbent is None else round(incumbent, 4),
            "incumbent_version": incumbent_version,
            "paired_gain": None if incumbent is None else round(candidate - incumbent, 4),
            "best_single_feature_roc_auc": round(baseline, 4),
            "min_improvement": gate.MIN_IMPROVEMENT, "min_absolute": gate.MIN_ABSOLUTE,
            "run_id": payload.get("run_id"), "seed": seed,
            "pipeline_run": os.environ.get("PIPELINE_RUN_ID"),
        }, indent=2) + "\n")
        return 0
    return 0 if decision == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
